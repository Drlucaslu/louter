#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) training with LoRA.

Implements the clipped surrogate objective with KL penalty:
  L = -E[ min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A) ] + β * KL(π_θ || π_ref)

Where:
  r(θ) = π_θ(a|s) / π_old(a|s)   (probability ratio)
  A    = group-relative advantage  (from reward_rollouts.py)
  ε    = clip range (default 0.2)
  β    = KL penalty weight (default 0.02)

Usage:
    # Train from scored rollouts
    python train_grpo.py --data scored_rollouts.jsonl --base-model Qwen/Qwen2.5-1.5B-Instruct

    # Start from existing SFT adapter
    python train_grpo.py --data scored_rollouts.jsonl --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter ../output

    # Merge after training
    python train_grpo.py --merge --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter ./rl_output --output-dir ./rl_merged
"""

import argparse
import json
import math
import sys
from pathlib import Path


def load_scored_rollouts(path: str) -> list[dict]:
    """Load scored rollouts from JSONL."""
    rollouts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rollouts.append(json.loads(line))
    return rollouts


def prepare_grpo_pairs(rollouts: list[dict], tokenizer) -> list[dict]:
    """Convert scored rollouts to (prompt, completion, advantage) training pairs.

    Filters out error completions and those with near-zero advantage.
    """
    pairs = []
    for rollout in rollouts:
        messages = rollout["prompt_messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)

        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            continue

        for comp in rollout.get("completions", []):
            if comp.get("finish_reason") == "error":
                continue
            content = comp.get("content", "")
            if not content.strip():
                continue

            advantage = comp.get("advantage", 0.0)

            # Skip near-zero advantages (uninformative)
            if abs(advantage) < 0.01:
                continue

            pairs.append({
                "prompt": prompt_text,
                "completion": content,
                "advantage": advantage,
                "reward": comp.get("reward", 0.0),
            })

    return pairs


def train(args):
    """Run GRPO training with LoRA."""
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {args.base_model}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine device
    if torch.cuda.is_available():
        dtype = torch.bfloat16
        device = torch.device("cuda")
        print("Using CUDA with bfloat16", file=sys.stderr)
    elif torch.backends.mps.is_available():
        dtype = torch.float16
        device = torch.device("mps")
        print("Using MPS with float16", file=sys.stderr)
    else:
        dtype = torch.float32
        device = torch.device("cpu")
        print("Using CPU with float32", file=sys.stderr)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )

    # Load or create LoRA adapter
    if args.adapter and Path(args.adapter).exists():
        print(f"Loading existing LoRA adapter: {args.adapter}", file=sys.stderr)
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
        print("Creating new LoRA adapter", file=sys.stderr)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Load reference model for KL computation (frozen copy)
    print("Loading reference model for KL penalty...", file=sys.stderr)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Load and prepare training data
    print(f"Loading scored rollouts: {args.data}", file=sys.stderr)
    rollouts = load_scored_rollouts(args.data)
    pairs = prepare_grpo_pairs(rollouts, tokenizer)
    print(f"Prepared {len(pairs)} training pairs from {len(rollouts)} rollout groups", file=sys.stderr)

    if not pairs:
        print("No training pairs — check that rollouts have non-zero advantages", file=sys.stderr)
        sys.exit(1)

    # Training loop
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    # Cosine scheduler
    total_steps = min(args.max_steps, math.ceil(len(pairs) / args.batch_size) * args.epochs)
    warmup_steps = int(total_steps * 0.1)

    def get_lr(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    model.train()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_eps = args.clip_epsilon
    kl_beta = args.kl_beta
    kl_cap = args.kl_cap
    max_seq_len = args.max_seq_len

    global_step = 0
    total_loss = 0.0
    total_kl = 0.0
    log_interval = 10
    save_interval = args.save_steps

    print(f"Training for {total_steps} steps (eps={clip_eps}, beta={kl_beta})", file=sys.stderr)

    for epoch in range(args.epochs):
        # Shuffle pairs each epoch
        import random
        random.shuffle(pairs)

        for batch_start in range(0, len(pairs), args.batch_size):
            if global_step >= args.max_steps:
                break

            batch = pairs[batch_start:batch_start + args.batch_size]

            batch_loss = torch.tensor(0.0, device=device, dtype=dtype)
            batch_kl = torch.tensor(0.0, device=device, dtype=dtype)
            valid_samples = 0

            for pair in batch:
                prompt = pair["prompt"]
                completion = pair["completion"]
                advantage = pair["advantage"]

                full_text = prompt + completion
                tokens = tokenizer(
                    full_text, return_tensors="pt", truncation=True, max_length=max_seq_len,
                ).to(device)
                prompt_tokens = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=max_seq_len,
                )
                prompt_len = prompt_tokens["input_ids"].shape[1]

                if tokens["input_ids"].shape[1] <= prompt_len:
                    continue  # completion was truncated entirely

                # Forward pass: current policy
                outputs = model(**tokens)
                logits = outputs.logits[:, prompt_len - 1:-1, :]  # shift for next-token prediction
                target_ids = tokens["input_ids"][:, prompt_len:]

                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

                # Reference model log-probs (for KL penalty and ratio)
                with torch.no_grad():
                    ref_outputs = ref_model(**tokens)
                    ref_logits = ref_outputs.logits[:, prompt_len - 1:-1, :]
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                    ref_token_log_probs = ref_log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

                # Per-token probability ratio — avoids numerical overflow/underflow
                # that occurs when exponentiating the sum of all token log-probs
                per_token_ratio = torch.exp(token_log_probs - ref_token_log_probs)

                # KL divergence (approximate: mean per-token KL)
                per_token_kl = (token_log_probs - ref_token_log_probs).mean()

                # Per-token clipped surrogate loss, averaged across completion tokens
                adv = torch.tensor(advantage, device=device, dtype=dtype)
                surr1 = per_token_ratio * adv
                surr2 = torch.clamp(per_token_ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # KL penalty
                kl_loss = kl_beta * per_token_kl

                batch_loss += policy_loss + kl_loss
                batch_kl += per_token_kl.detach()
                valid_samples += 1

            if valid_samples == 0:
                continue

            # Average over batch and accumulate gradients
            loss = batch_loss / valid_samples

            # Gradient accumulation
            scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()

            if (global_step + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            total_kl += (batch_kl / valid_samples).item()
            global_step += 1

            # Logging
            if global_step % log_interval == 0:
                avg_loss = total_loss / log_interval
                avg_kl = total_kl / log_interval
                lr = scheduler.get_last_lr()[0] * args.lr
                print(
                    f"  step {global_step}/{total_steps}  loss={avg_loss:.4f}  kl={avg_kl:.4f}  lr={lr:.2e}",
                    file=sys.stderr,
                )
                total_loss = 0.0
                total_kl = 0.0

                # KL cap safety check
                if abs(avg_kl) > kl_cap:
                    print(
                        f"  WARNING: KL divergence ({avg_kl:.4f}) exceeds cap ({kl_cap}). "
                        f"Stopping early to prevent mode collapse.",
                        file=sys.stderr,
                    )
                    break

            # Save checkpoint
            if global_step % save_interval == 0:
                ckpt_dir = output_dir / f"checkpoint-{global_step}"
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                print(f"  Saved checkpoint: {ckpt_dir}", file=sys.stderr)

        if global_step >= args.max_steps:
            break

    # Save final adapter
    print(f"Saving final adapter to {output_dir}", file=sys.stderr)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metadata
    meta = {
        "base_model": args.base_model,
        "starting_adapter": args.adapter,
        "total_steps": global_step,
        "clip_epsilon": clip_eps,
        "kl_beta": kl_beta,
        "learning_rate": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "num_pairs": len(pairs),
        "num_rollout_groups": len(rollouts),
    }
    (output_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"GRPO training complete. {global_step} steps.", file=sys.stderr)


def merge_and_export(args):
    """Merge LoRA adapter with base model and export."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {args.base_model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True,
    )

    adapter_path = args.adapter or "./rl_output"
    print(f"Loading adapter: {adapter_path}", file=sys.stderr)
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging adapter with base model...", file=sys.stderr)
    model = model.merge_and_unload()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {output_dir}", file=sys.stderr)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Create Ollama Modelfile
    modelfile_path = output_dir / "Modelfile"
    modelfile_path.write_text(f"FROM {output_dir}\n")
    print(f"Created Ollama Modelfile: {modelfile_path}", file=sys.stderr)
    print(f"\nTo import: ollama create louter-rl -f {modelfile_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="GRPO training with LoRA for Louter RL")
    parser.add_argument("--merge", action="store_true", help="Merge adapter with base model")

    # Model
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", help="Starting LoRA adapter path (reuse SFT weights)")
    parser.add_argument("--data", help="Scored rollouts JSONL file")
    parser.add_argument("--output-dir", default="./rl_output")

    # LoRA
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)

    # GRPO hyperparameters
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clip range (default: 0.2)")
    parser.add_argument("--kl-beta", type=float, default=0.02, help="KL penalty weight (default: 0.02)")
    parser.add_argument("--kl-cap", type=float, default=0.1, help="KL divergence cap — stop if exceeded (default: 0.1)")

    # Training
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--save-steps", type=int, default=50)

    args = parser.parse_args()

    if args.merge:
        merge_and_export(args)
    else:
        if not args.data:
            print("Error: --data is required for training", file=sys.stderr)
            sys.exit(1)
        train(args)


if __name__ == "__main__":
    main()
