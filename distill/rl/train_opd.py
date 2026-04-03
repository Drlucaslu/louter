#!/usr/bin/env python3
"""
On-Policy Distillation (OPD) training with LoRA.

When the local model fails and the cloud model succeeds on the same prompt,
OPD provides token-level gradient signal: tokens where the local model assigns
low probability to the correct cloud token get stronger gradient updates.

This is stronger than SFT (sequence-level) and GRPO (scalar reward) because
it gives per-token directional signal.

Can be combined with GRPO: L_total = W_RL * L_grpo + W_OPD * L_opd

Usage:
    # Train OPD from cloud reference pairs
    python train_opd.py --data scored_rollouts.jsonl --base-model Qwen/Qwen2.5-1.5B-Instruct

    # Combined GRPO + OPD
    python train_opd.py --data scored_rollouts.jsonl --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --w-rl 1.0 --w-opd 1.0

    # OPD only (no GRPO loss)
    python train_opd.py --data scored_rollouts.jsonl --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --w-rl 0.0 --w-opd 1.0
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path


def load_scored_rollouts(path: str) -> list[dict]:
    rollouts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rollouts.append(json.loads(line))
    return rollouts


def extract_opd_pairs(rollouts: list[dict], tokenizer) -> list[dict]:
    """Extract pairs where we have a reference (cloud) completion to distill from.

    Each pair has:
      - prompt: tokenized prompt text
      - student_completion: the model's own completion (from rollout)
      - teacher_completion: the cloud reference completion
      - advantage: from GRPO scoring
    """
    pairs = []
    for rollout in rollouts:
        reference = rollout.get("reference_completion")
        if not reference:
            continue

        # Extract teacher text
        if isinstance(reference, str):
            try:
                reference = json.loads(reference)
            except json.JSONDecodeError:
                teacher_text = reference
                reference = None
        if isinstance(reference, dict):
            teacher_text = reference.get("content", "")
            if isinstance(teacher_text, list):
                teacher_text = " ".join(
                    p.get("text", "") for p in teacher_text
                    if isinstance(p, dict) and p.get("type") == "text"
                )
        else:
            teacher_text = str(reference) if reference else ""

        if not teacher_text.strip():
            continue

        messages = rollout["prompt_messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)

        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            continue

        # Use each rollout completion as a student sample
        for comp in rollout.get("completions", []):
            if comp.get("finish_reason") == "error":
                continue
            student_text = comp.get("content", "")
            if not student_text.strip():
                continue

            pairs.append({
                "prompt": prompt_text,
                "student_completion": student_text,
                "teacher_completion": teacher_text,
                "advantage": comp.get("advantage", 0.0),
                "reward": comp.get("reward", 0.0),
            })

    return pairs


def train(args):
    """Run combined GRPO + OPD training."""
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {args.base_model}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        dtype = torch.bfloat16
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dtype = torch.float16
        device = torch.device("mps")
    else:
        dtype = torch.float32
        device = torch.device("cpu")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )

    if args.adapter and Path(args.adapter).exists():
        print(f"Loading existing LoRA adapter: {args.adapter}", file=sys.stderr)
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Reference model for KL
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Load data
    rollouts = load_scored_rollouts(args.data)
    pairs = extract_opd_pairs(rollouts, tokenizer)
    print(f"Extracted {len(pairs)} OPD pairs from {len(rollouts)} rollout groups", file=sys.stderr)

    if not pairs:
        print("No OPD pairs found (need rollouts with reference_completion).", file=sys.stderr)
        sys.exit(1)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

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

    w_rl = args.w_rl
    w_opd = args.w_opd
    clip_eps = args.clip_epsilon
    kl_beta = args.kl_beta
    max_seq_len = args.max_seq_len

    global_step = 0
    total_rl_loss = 0.0
    total_opd_loss = 0.0
    log_interval = 10

    print(f"Training {total_steps} steps (W_RL={w_rl}, W_OPD={w_opd})", file=sys.stderr)

    for epoch in range(args.epochs):
        random.shuffle(pairs)

        for batch_start in range(0, len(pairs), args.batch_size):
            if global_step >= args.max_steps:
                break

            batch = pairs[batch_start:batch_start + args.batch_size]
            batch_rl = torch.tensor(0.0, device=device, dtype=dtype)
            batch_opd = torch.tensor(0.0, device=device, dtype=dtype)
            valid = 0

            for pair in batch:
                prompt = pair["prompt"]
                student_text = pair["student_completion"]
                teacher_text = pair["teacher_completion"]
                advantage = pair["advantage"]

                # ── GRPO loss (on student completion) ──
                rl_loss = torch.tensor(0.0, device=device, dtype=dtype)
                if w_rl > 0:
                    full_text = prompt + student_text
                    tokens = tokenizer(
                        full_text, return_tensors="pt", truncation=True, max_length=max_seq_len,
                    ).to(device)
                    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
                    p_len = prompt_ids["input_ids"].shape[1]

                    if tokens["input_ids"].shape[1] > p_len:
                        outputs = model(**tokens)
                        logits = outputs.logits[:, p_len - 1:-1, :]
                        target = tokens["input_ids"][:, p_len:]

                        log_probs = F.log_softmax(logits, dim=-1)
                        tok_lp = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1).sum()

                        with torch.no_grad():
                            ref_out = ref_model(**tokens)
                            ref_logits = ref_out.logits[:, p_len - 1:-1, :]
                            ref_lp = F.log_softmax(ref_logits, dim=-1)
                            ref_tok_lp = ref_lp.gather(2, target.unsqueeze(-1)).squeeze(-1).sum()

                        ratio = torch.exp(tok_lp - ref_tok_lp)
                        adv_t = torch.tensor(advantage, device=device, dtype=dtype)
                        surr1 = ratio * adv_t
                        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
                        rl_loss = -torch.min(surr1, surr2) + kl_beta * (tok_lp - ref_tok_lp).mean()

                # ── OPD loss (token-level distillation from teacher) ──
                opd_loss = torch.tensor(0.0, device=device, dtype=dtype)
                if w_opd > 0:
                    teacher_full = prompt + teacher_text
                    teacher_tokens = tokenizer(
                        teacher_full, return_tensors="pt", truncation=True, max_length=max_seq_len,
                    ).to(device)
                    prompt_ids_t = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
                    p_len_t = prompt_ids_t["input_ids"].shape[1]

                    if teacher_tokens["input_ids"].shape[1] > p_len_t:
                        # Student log-probs on teacher tokens
                        student_out = model(**teacher_tokens)
                        student_logits = student_out.logits[:, p_len_t - 1:-1, :]
                        teacher_target = teacher_tokens["input_ids"][:, p_len_t:]

                        student_lp = F.log_softmax(student_logits, dim=-1)
                        per_token_lp = student_lp.gather(2, teacher_target.unsqueeze(-1)).squeeze(-1)

                        # OPD loss: negative log-likelihood of teacher tokens under student
                        # Tokens where student assigns low prob get stronger gradient
                        opd_loss = -per_token_lp.mean()

                combined = w_rl * rl_loss + w_opd * opd_loss

                if combined.requires_grad:
                    batch_rl += rl_loss.detach()
                    batch_opd += opd_loss.detach()
                    scaled = combined / (args.batch_size * args.gradient_accumulation)
                    scaled.backward()
                    valid += 1

            if valid == 0:
                continue

            if (global_step + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_rl_loss += (batch_rl / max(valid, 1)).item()
            total_opd_loss += (batch_opd / max(valid, 1)).item()
            global_step += 1

            if global_step % log_interval == 0:
                avg_rl = total_rl_loss / log_interval
                avg_opd = total_opd_loss / log_interval
                print(
                    f"  step {global_step}/{total_steps}  rl_loss={avg_rl:.4f}  opd_loss={avg_opd:.4f}",
                    file=sys.stderr,
                )
                total_rl_loss = 0.0
                total_opd_loss = 0.0

            if global_step % args.save_steps == 0:
                ckpt = output_dir / f"checkpoint-{global_step}"
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)

        if global_step >= args.max_steps:
            break

    # Save final
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    meta = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "total_steps": global_step,
        "w_rl": w_rl, "w_opd": w_opd,
        "num_pairs": len(pairs),
        "mode": "combined" if w_rl > 0 and w_opd > 0 else ("opd_only" if w_rl == 0 else "grpo_only"),
    }
    (output_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"OPD training complete. {global_step} steps.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="On-Policy Distillation (OPD) training")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", help="Starting LoRA adapter path")
    parser.add_argument("--data", required=True, help="Scored rollouts JSONL (must have reference_completion)")
    parser.add_argument("--output-dir", default="./rl_output")

    # LoRA
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)

    # Loss weights
    parser.add_argument("--w-rl", type=float, default=1.0, help="GRPO loss weight (default: 1.0)")
    parser.add_argument("--w-opd", type=float, default=1.0, help="OPD loss weight (default: 1.0)")

    # GRPO params
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.02)

    # Training
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--save-steps", type=int, default=50)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
