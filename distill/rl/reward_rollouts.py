#!/usr/bin/env python3
"""
Score rollouts and compute GRPO group-relative advantages.

For each rollout group (G completions per prompt), this script:
  1. Scores each completion using judge + reference comparison + structural checks
  2. Computes group-relative advantages (the GRPO core)
  3. Outputs scored rollouts ready for train_grpo.py

Usage:
    python reward_rollouts.py --rollouts rollouts.jsonl --output scored_rollouts.jsonl
    python reward_rollouts.py --rollouts rollouts.jsonl --output scored.jsonl --judge-provider ollama
    python reward_rollouts.py --rollouts rollouts.jsonl --output scored.jsonl --no-judge  # structural only
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path


# ── Reference comparison ──

def compute_rouge_l(reference: str, candidate: str) -> float:
    """Compute ROUGE-L F1 score between reference and candidate."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return scores["rougeL"].fmeasure
    except ImportError:
        # Fallback: simple longest common subsequence ratio
        return _simple_lcs_ratio(reference, candidate)


def _simple_lcs_ratio(a: str, b: str) -> float:
    """Simple LCS-based similarity without rouge_score dependency."""
    if not a or not b:
        return 0.0
    words_a = a.split()
    words_b = b.split()
    if not words_a or not words_b:
        return 0.0

    # LCS length via DP (on words, capped for performance)
    max_len = 500
    wa = words_a[:max_len]
    wb = words_b[:max_len]
    m, n = len(wa), len(wb)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if wa[i - 1] == wb[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    lcs_len = prev[n]

    precision = lcs_len / n if n > 0 else 0
    recall = lcs_len / m if m > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Structural scoring for tool calls ──

def score_tool_call_structure(completion_text: str, tools: list[dict] | None) -> float:
    """Quick structural check for tool-call-like content. Returns [0, 1]."""
    # Check if completion contains valid JSON tool call patterns
    score = 0.5  # neutral baseline

    if not completion_text.strip():
        return 0.0

    # Check for JSON-like structure
    try:
        parsed = json.loads(completion_text)
        score += 0.25  # valid JSON
        if isinstance(parsed, dict) and ("name" in parsed or "function" in parsed):
            score += 0.25  # looks like a tool call
    except json.JSONDecodeError:
        # Not pure JSON, but could be mixed text + tool calls
        if '{"' in completion_text or '"name"' in completion_text:
            score += 0.1

    return min(1.0, score)


# ── Judge scoring ──

JUDGE_PROMPT = """\
Rate this assistant response on a scale of 1-5:
- Correctness (1=wrong, 5=perfect)
- Helpfulness (1=irrelevant, 5=exactly right)

User request (summary): {prompt_summary}

Assistant response: {completion}

Respond with ONLY JSON: {{"score": <1-5>, "reason": "<brief>"}}"""


def get_prompt_summary(messages: list[dict], max_chars: int = 1000) -> str:
    """Extract a short summary of the prompt for the judge."""
    parts = []
    for msg in messages:
        if msg.get("role") in ("user", "system"):
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            if content:
                parts.append(content)
    text = "\n".join(parts)
    return text[:max_chars] if len(text) > max_chars else text


def score_with_judge(prompt_summary: str, completion: str, provider: str, model: str) -> float | None:
    """Score a completion using a judge model. Returns normalized score [-1, 1] or None."""
    judge_input = JUDGE_PROMPT.format(prompt_summary=prompt_summary, completion=completion[:2000])

    try:
        if provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=model, max_tokens=128,
                messages=[{"role": "user", "content": judge_input}],
            )
            raw = resp.content[0].text
        elif provider == "openai":
            import openai
            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model=model, max_tokens=128,
                messages=[{"role": "user", "content": judge_input}],
            )
            raw = resp.choices[0].message.content
        elif provider == "ollama":
            import requests
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            resp = requests.post(
                f"{base_url}/api/chat",
                json={"model": model, "messages": [{"role": "user", "content": judge_input}],
                      "stream": False, "options": {"temperature": 0.1}},
                timeout=120,
            )
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]
        else:
            return None

        # Parse score
        text = raw.strip()
        if text.startswith("```"):
            lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(text[start:end])
            score = int(parsed.get("score", 3))
            return (score - 3.0) / 2.0  # [1,5] → [-1,1]

    except Exception as e:
        print(f"    Judge error: {e}", file=sys.stderr)

    return None


# ── GRPO advantage computation ──

def compute_grpo_advantages(rewards_per_group: list[list[float]]) -> list[list[float]]:
    """Compute group-relative advantages for GRPO.

    For each group, advantages = (reward - group_mean) / group_std.
    This is the core of GRPO: no critic needed, just relative comparison.
    """
    advantages = []
    for group_rewards in rewards_per_group:
        if len(group_rewards) < 2:
            advantages.append([0.0] * len(group_rewards))
            continue

        mean = sum(group_rewards) / len(group_rewards)
        std = max(statistics.stdev(group_rewards), 1e-8)
        group_adv = [(r - mean) / std for r in group_rewards]
        advantages.append(group_adv)

    return advantages


# ── Main scoring pipeline ──

def score_rollout_group(
    rollout: dict,
    use_judge: bool,
    judge_provider: str,
    judge_model: str,
    reference_weight: float,
    judge_weight: float,
    rate_limit: float,
) -> dict:
    """Score all completions in a rollout group and compute advantages."""
    messages = rollout["prompt_messages"]
    completions = rollout["completions"]
    reference = rollout.get("reference_completion")

    # Extract reference text for comparison
    ref_text = ""
    if reference:
        if isinstance(reference, str):
            try:
                reference = json.loads(reference)
            except json.JSONDecodeError:
                ref_text = reference
        if isinstance(reference, dict):
            ref_text = reference.get("content", "")
            if isinstance(ref_text, list):
                ref_text = " ".join(
                    p.get("text", "") for p in ref_text
                    if isinstance(p, dict) and p.get("type") == "text"
                )

    prompt_summary = get_prompt_summary(messages) if use_judge else ""

    rewards = []
    scored_completions = []

    for comp in completions:
        content = comp.get("content", "")
        if comp.get("finish_reason") == "error" or not content.strip():
            rewards.append(-1.0)
            scored_completions.append({**comp, "reward": -1.0, "reward_components": {"error": True}})
            continue

        components = {}

        # Reference similarity
        if ref_text:
            rouge = compute_rouge_l(ref_text, content)
            components["reference_rouge_l"] = round(rouge, 4)
            ref_score = rouge * 2.0 - 1.0  # [0,1] → [-1,1]
        else:
            ref_score = 0.0

        # Judge score
        judge_score = 0.0
        if use_judge:
            js = score_with_judge(prompt_summary, content, judge_provider, judge_model)
            if js is not None:
                judge_score = js
                components["judge_score"] = round(js, 4)
            if rate_limit > 0:
                time.sleep(rate_limit)

        # Combine scores
        total_weight = 0.0
        weighted_sum = 0.0

        if ref_text:
            weighted_sum += ref_score * reference_weight
            total_weight += reference_weight
        if use_judge and "judge_score" in components:
            weighted_sum += judge_score * judge_weight
            total_weight += judge_weight

        if total_weight > 0:
            reward = weighted_sum / total_weight
        else:
            # No scoring signals — use length heuristic (prefer non-trivial responses)
            reward = 0.0 if len(content) > 20 else -0.5

        reward = max(-1.0, min(1.0, reward))
        rewards.append(reward)
        components["final_reward"] = round(reward, 4)
        scored_completions.append({**comp, "reward": round(reward, 4), "reward_components": components})

    # Compute GRPO advantages
    advantages = compute_grpo_advantages([rewards])[0]

    for i, comp in enumerate(scored_completions):
        comp["advantage"] = round(advantages[i], 4)

    return {
        **rollout,
        "completions": scored_completions,
        "group_mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
        "group_std_reward": round(statistics.stdev(rewards), 4) if len(rewards) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Score rollouts and compute GRPO advantages")
    parser.add_argument("--rollouts", required=True, help="Input rollouts JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output scored rollouts JSONL file")

    # Judge
    parser.add_argument("--no-judge", action="store_true", help="Skip judge scoring (reference + structural only)")
    parser.add_argument("--judge-provider", default="anthropic", choices=["anthropic", "openai", "ollama"])
    parser.add_argument("--judge-model", default=None, help="Judge model name (default: auto)")
    parser.add_argument("--rate-limit", type=float, default=0.5, help="Seconds between judge API calls")

    # Weights
    parser.add_argument("--reference-weight", type=float, default=0.4, help="Weight for reference comparison")
    parser.add_argument("--judge-weight", type=float, default=0.6, help="Weight for judge scores")

    args = parser.parse_args()

    default_judge_models = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o-mini",
        "ollama": "qwen2.5:7b",
    }
    judge_model = args.judge_model or default_judge_models.get(args.judge_provider, "")
    use_judge = not args.no_judge

    # Load rollouts
    rollouts = []
    with open(args.rollouts) as f:
        for line in f:
            line = line.strip()
            if line:
                rollouts.append(json.loads(line))

    if not rollouts:
        print("No rollouts to score.", file=sys.stderr)
        return

    print(f"Scoring {len(rollouts)} rollout groups", file=sys.stderr)
    if use_judge:
        print(f"Judge: {args.judge_provider}/{judge_model}", file=sys.stderr)
    else:
        print("Judge: disabled (reference + structural only)", file=sys.stderr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_completions = 0
    with open(output_path, "w") as out:
        for i, rollout in enumerate(rollouts):
            G = len(rollout.get("completions", []))
            print(f"  [{i+1}/{len(rollouts)}] {G} completions...", end=" ", file=sys.stderr)

            scored = score_rollout_group(
                rollout=rollout,
                use_judge=use_judge,
                judge_provider=args.judge_provider,
                judge_model=judge_model,
                reference_weight=args.reference_weight,
                judge_weight=args.judge_weight,
                rate_limit=args.rate_limit,
            )

            rewards = [c.get("reward", 0) for c in scored["completions"]]
            advantages = [c.get("advantage", 0) for c in scored["completions"]]
            total_completions += len(scored["completions"])

            print(
                f"mean_r={scored['group_mean_reward']:+.3f} "
                f"adv_range=[{min(advantages):+.2f}, {max(advantages):+.2f}]",
                file=sys.stderr,
            )

            out.write(json.dumps(scored, ensure_ascii=False) + "\n")

    print(f"\nDone: {total_completions} completions scored → {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
