#!/usr/bin/env python3
"""
Evaluate RL model vs SFT baseline before deployment.

Holds out a subset of episodes, generates completions from both models,
scores them with the judge, and only recommends deployment if the RL model
wins on average reward.

Usage:
    python evaluate.py --episodes episodes.jsonl --rl-model ./rl_merged --sft-model ./sft_merged
    python evaluate.py --episodes episodes.jsonl --rl-model ./rl_merged --sft-model louter-distilled \
        --backend ollama
    python evaluate.py --report output/eval_report.json  # View previous report
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path


def load_episodes(path: str, holdout_ratio: float = 0.1, max_eval: int = 50) -> list[dict]:
    """Load episodes and select a holdout eval set."""
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))

    # Deterministic shuffle for reproducibility
    random.seed(42)
    random.shuffle(episodes)

    n_eval = max(1, min(max_eval, int(len(episodes) * holdout_ratio)))
    return episodes[:n_eval]


def generate_completion(messages: list[dict], backend: str, model: str, hf_model=None, hf_tokenizer=None) -> str:
    """Generate a single completion from a model."""
    if backend == "transformers":
        import torch
        prompt = hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = hf_tokenizer(prompt, return_tensors="pt").to(hf_model.device)
        with torch.no_grad():
            outputs = hf_model.generate(
                **inputs, max_new_tokens=1024, temperature=0.1, do_sample=True,
                pad_token_id=hf_tokenizer.pad_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return hf_tokenizer.decode(generated, skip_special_tokens=True)

    elif backend == "ollama":
        import requests
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        resp = requests.post(
            f"{base_url}/api/chat",
            json={"model": model, "messages": messages, "stream": False,
                  "options": {"temperature": 0.1}},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    elif backend == "vllm":
        import requests
        base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json={"model": model, "messages": messages, "temperature": 0.1, "max_tokens": 1024},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    raise ValueError(f"Unknown backend: {backend}")


def score_with_judge(prompt_summary: str, completion: str, provider: str, model: str) -> float | None:
    """Score using judge model. Returns [-1, 1] or None."""
    judge_prompt = (
        f"Rate this response 1-5 on correctness and helpfulness.\n\n"
        f"Request: {prompt_summary[:1000]}\n\nResponse: {completion[:2000]}\n\n"
        f'Respond ONLY with JSON: {{"score": <1-5>}}'
    )

    try:
        if provider == "anthropic":
            import anthropic
            resp = anthropic.Anthropic().messages.create(
                model=model, max_tokens=64,
                messages=[{"role": "user", "content": judge_prompt}],
            )
            raw = resp.content[0].text
        elif provider == "openai":
            import openai
            resp = openai.OpenAI().chat.completions.create(
                model=model, max_tokens=64,
                messages=[{"role": "user", "content": judge_prompt}],
            )
            raw = resp.choices[0].message.content
        elif provider == "ollama":
            import requests
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            resp = requests.post(
                f"{base_url}/api/chat",
                json={"model": model, "messages": [{"role": "user", "content": judge_prompt}],
                      "stream": False, "options": {"temperature": 0.1}},
                timeout=120,
            )
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]
        else:
            return None

        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(raw[start:end])
            score = int(parsed.get("score", 3))
            return (score - 3.0) / 2.0
    except Exception as e:
        print(f"    Judge error: {e}", file=sys.stderr)

    return None


def get_prompt_summary(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        if msg.get("role") in ("user", "system"):
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            if content:
                parts.append(content)
    return "\n".join(parts)[:1500]


def load_hf_model(model_path: str):
    """Load a HuggingFace model for evaluation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL model vs SFT baseline")
    parser.add_argument("--episodes", help="Episodes JSONL file")
    parser.add_argument("--rl-model", help="RL model path or name")
    parser.add_argument("--sft-model", help="SFT model path or name (baseline)")

    parser.add_argument("--backend", default="transformers", choices=["transformers", "ollama", "vllm"])
    parser.add_argument("--holdout-ratio", type=float, default=0.1, help="Fraction of episodes for eval")
    parser.add_argument("--max-eval", type=int, default=50, help="Max eval episodes")

    # Judge
    parser.add_argument("--judge-provider", default="anthropic", choices=["anthropic", "openai", "ollama"])
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--rate-limit", type=float, default=1.0)

    # Output
    parser.add_argument("--report", help="Path to save/view eval report JSON")

    args = parser.parse_args()

    # View existing report
    if args.report and not args.episodes:
        report_path = Path(args.report)
        if report_path.exists():
            report = json.loads(report_path.read_text())
            print(json.dumps(report, indent=2))
            return
        print(f"Report not found: {report_path}", file=sys.stderr)
        sys.exit(1)

    if not args.episodes or not args.rl_model or not args.sft_model:
        print("Error: --episodes, --rl-model, and --sft-model are required", file=sys.stderr)
        sys.exit(1)

    default_judge = {"anthropic": "claude-sonnet-4-20250514", "openai": "gpt-4o-mini", "ollama": "qwen2.5:7b"}
    judge_model = args.judge_model or default_judge[args.judge_provider]

    # Load eval episodes
    eval_episodes = load_episodes(args.episodes, args.holdout_ratio, args.max_eval)
    print(f"Evaluating on {len(eval_episodes)} held-out episodes", file=sys.stderr)

    # Load models
    rl_hf, rl_tok, sft_hf, sft_tok = None, None, None, None
    if args.backend == "transformers":
        print(f"Loading RL model: {args.rl_model}", file=sys.stderr)
        rl_hf, rl_tok = load_hf_model(args.rl_model)
        print(f"Loading SFT model: {args.sft_model}", file=sys.stderr)
        sft_hf, sft_tok = load_hf_model(args.sft_model)

    rl_scores = []
    sft_scores = []
    results = []

    for i, ep in enumerate(eval_episodes):
        messages = ep["prompt_messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)

        print(f"  [{i+1}/{len(eval_episodes)}] ", end="", file=sys.stderr)

        # Generate from both models
        try:
            rl_completion = generate_completion(messages, args.backend, args.rl_model, rl_hf, rl_tok)
        except Exception as e:
            print(f"RL generation failed: {e}", file=sys.stderr)
            continue

        try:
            sft_completion = generate_completion(messages, args.backend, args.sft_model, sft_hf, sft_tok)
        except Exception as e:
            print(f"SFT generation failed: {e}", file=sys.stderr)
            continue

        # Score both
        prompt_summary = get_prompt_summary(messages)

        rl_score = score_with_judge(prompt_summary, rl_completion, args.judge_provider, judge_model)
        if args.rate_limit > 0:
            time.sleep(args.rate_limit)
        sft_score = score_with_judge(prompt_summary, sft_completion, args.judge_provider, judge_model)
        if args.rate_limit > 0:
            time.sleep(args.rate_limit)

        if rl_score is not None and sft_score is not None:
            rl_scores.append(rl_score)
            sft_scores.append(sft_score)
            winner = "rl" if rl_score > sft_score else ("sft" if sft_score > rl_score else "tie")
            print(f"RL={rl_score:+.2f} SFT={sft_score:+.2f} → {winner}", file=sys.stderr)

            results.append({
                "episode_id": ep.get("id", ""),
                "rl_score": rl_score,
                "sft_score": sft_score,
                "winner": winner,
            })
        else:
            print("scoring failed", file=sys.stderr)

    # Summary
    if not rl_scores:
        print("\nNo episodes scored successfully.", file=sys.stderr)
        sys.exit(1)

    rl_mean = statistics.mean(rl_scores)
    sft_mean = statistics.mean(sft_scores)
    rl_wins = sum(1 for r in results if r["winner"] == "rl")
    sft_wins = sum(1 for r in results if r["winner"] == "sft")
    ties = sum(1 for r in results if r["winner"] == "tie")
    deploy = rl_mean > sft_mean

    report = {
        "n_episodes": len(results),
        "rl_model": args.rl_model,
        "sft_model": args.sft_model,
        "rl_mean_score": round(rl_mean, 4),
        "sft_mean_score": round(sft_mean, 4),
        "rl_wins": rl_wins,
        "sft_wins": sft_wins,
        "ties": ties,
        "improvement": round(rl_mean - sft_mean, 4),
        "recommend_deploy": deploy,
        "judge": f"{args.judge_provider}/{judge_model}",
        "results": results,
    }

    print(f"\n{'='*50}", file=sys.stderr)
    print(f"  RL  mean: {rl_mean:+.4f}  (wins: {rl_wins})", file=sys.stderr)
    print(f"  SFT mean: {sft_mean:+.4f}  (wins: {sft_wins})", file=sys.stderr)
    print(f"  Ties: {ties}", file=sys.stderr)
    print(f"  Improvement: {rl_mean - sft_mean:+.4f}", file=sys.stderr)
    print(f"  Recommend deploy: {'YES' if deploy else 'NO'}", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)

    # Save report
    report_path = Path(args.report) if args.report else Path("output/eval_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {report_path}", file=sys.stderr)

    # Exit code: 0 if RL wins, 1 if not (for use in pipeline scripts)
    sys.exit(0 if deploy else 1)


if __name__ == "__main__":
    main()
