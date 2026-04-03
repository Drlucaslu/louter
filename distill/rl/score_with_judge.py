#!/usr/bin/env python3
"""
Score RL episodes using a judge model (cloud or local).

The judge rates each assistant response on correctness, helpfulness, and
completeness. Scores are normalized to [-1.0, 1.0] and written back to the
rl_episodes table.

Supported judge backends:
  - anthropic: Claude via Anthropic API (ANTHROPIC_API_KEY)
  - openai:    GPT via OpenAI API (OPENAI_API_KEY)
  - ollama:    Local model via Ollama REST API

Usage:
    python score_with_judge.py --db ../louter.db                       # Score all unscored
    python score_with_judge.py --db ../louter.db --provider ollama     # Use local judge
    python score_with_judge.py --db ../louter.db --source local        # Only score local episodes
    python score_with_judge.py --db ../louter.db --dry-run --limit 5   # Preview without writing
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

JUDGE_PROMPT = """\
You are an expert evaluator. Rate the assistant's response to the user's request.

Score each dimension from 1 to 5:
- **Correctness**: Is the response factually and logically correct? (1=wrong, 5=perfect)
- **Helpfulness**: Does it address the user's actual request? (1=irrelevant, 5=exactly what was asked)
- **Completeness**: Is anything important missing? (1=incomplete, 5=thorough)

<user_request>
{prompt}
</user_request>

<assistant_response>
{completion}
</assistant_response>

Respond with ONLY a JSON object, no other text:
{{"correctness": <1-5>, "helpfulness": <1-5>, "completeness": <1-5>, "reason": "<one sentence>"}}"""


def connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_unscored_episodes(
    conn: sqlite3.Connection,
    source_filter: str | None = None,
    limit: int = 0,
) -> list[dict]:
    """Get episodes that haven't been scored by a judge yet."""
    query = """
        SELECT * FROM rl_episodes
        WHERE (reward_source = 'implicit' OR reward_source IS NULL)
    """
    params: list = []

    if source_filter:
        query += " AND source = ?"
        params.append(source_filter)

    query += " ORDER BY created_at ASC"

    if limit > 0:
        query += " LIMIT ?"
        params.append(limit)

    cursor = conn.execute(query, params)
    return [dict(row) for row in cursor.fetchall()]


def extract_prompt_text(messages_json: str, max_chars: int = 4000) -> str:
    """Extract a readable prompt from the messages JSON."""
    try:
        messages = json.loads(messages_json) if isinstance(messages_json, str) else messages_json
    except (json.JSONDecodeError, TypeError):
        return "(unable to parse messages)"

    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
            )
        if content:
            parts.append(f"[{role}]: {content}")

    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...(truncated)"
    return text


def extract_completion_text(completion_json: str, max_chars: int = 2000) -> str:
    """Extract readable text from the completion JSON."""
    try:
        completion = json.loads(completion_json) if isinstance(completion_json, str) else completion_json
    except (json.JSONDecodeError, TypeError):
        return "(unable to parse completion)"

    # Handle OpenAI chat format
    content = completion.get("content", "")
    if isinstance(content, list):
        content = " ".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )

    # Include tool calls if present
    tool_calls = completion.get("tool_calls", [])
    if tool_calls:
        tc_parts = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            tc_parts.append(f"tool_call: {fn.get('name', '?')}({fn.get('arguments', '')})")
        if content:
            content += "\n" + "\n".join(tc_parts)
        else:
            content = "\n".join(tc_parts)

    if not content:
        content = json.dumps(completion, ensure_ascii=False)

    if len(content) > max_chars:
        content = content[:max_chars] + "\n...(truncated)"
    return content


def normalize_score(correctness: int, helpfulness: int, completeness: int) -> float:
    """Normalize 1-5 scores to [-1.0, 1.0] range.

    Average of three dimensions, mapped from [1,5] to [-1,1].
    Score of 3 maps to 0.0 (neutral).
    """
    avg = (correctness + helpfulness + completeness) / 3.0
    return (avg - 3.0) / 2.0  # [1,5] -> [-1.0, 1.0]


def parse_judge_response(text: str) -> dict | None:
    """Parse the judge's JSON response, handling markdown fences."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        result = json.loads(text)
        if all(k in result for k in ("correctness", "helpfulness", "completeness")):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            result = json.loads(text[start:end])
            if all(k in result for k in ("correctness", "helpfulness", "completeness")):
                return result
        except json.JSONDecodeError:
            pass

    return None


# ── Judge backends ──

def score_with_anthropic(prompt: str, model: str) -> str:
    """Score using Anthropic API."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def score_with_openai(prompt: str, model: str) -> str:
    """Score using OpenAI API."""
    import openai

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def score_with_ollama(prompt: str, model: str, base_url: str = "http://localhost:11434") -> str:
    """Score using Ollama REST API."""
    import requests

    response = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def get_scorer(provider: str):
    """Return the appropriate scoring function for the provider."""
    scorers = {
        "anthropic": score_with_anthropic,
        "openai": score_with_openai,
        "ollama": score_with_ollama,
    }
    if provider not in scorers:
        print(f"Error: unknown provider '{provider}'. Use: {', '.join(scorers)}", file=sys.stderr)
        sys.exit(1)
    return scorers[provider]


def score_episode(
    episode: dict,
    scorer,
    model: str,
    provider: str,
) -> tuple[float, dict] | None:
    """Score a single episode. Returns (reward, details) or None on failure."""
    prompt_text = extract_prompt_text(episode["prompt_messages"])
    completion_text = extract_completion_text(episode["completion"])

    judge_input = JUDGE_PROMPT.format(prompt=prompt_text, completion=completion_text)

    try:
        if provider == "ollama":
            raw = scorer(judge_input, model, os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
        else:
            raw = scorer(judge_input, model)
    except Exception as e:
        print(f"  Error calling {provider}: {e}", file=sys.stderr)
        return None

    parsed = parse_judge_response(raw)
    if parsed is None:
        print(f"  Failed to parse judge response: {raw[:200]}", file=sys.stderr)
        return None

    correctness = int(parsed["correctness"])
    helpfulness = int(parsed["helpfulness"])
    completeness = int(parsed["completeness"])
    reward = normalize_score(correctness, helpfulness, completeness)

    details = {
        "correctness": correctness,
        "helpfulness": helpfulness,
        "completeness": completeness,
        "reward": round(reward, 4),
        "reason": parsed.get("reason", ""),
        "judge_provider": provider,
        "judge_model": model,
    }

    return reward, details


def main():
    parser = argparse.ArgumentParser(description="Score RL episodes with a judge model")
    parser.add_argument("--db", default="louter.db", help="Path to louter.db")
    parser.add_argument(
        "--provider", default="anthropic",
        choices=["anthropic", "openai", "ollama"],
        help="Judge model provider (default: anthropic)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Judge model name (default: auto per provider)",
    )
    parser.add_argument("--source", choices=["local", "cloud"], help="Filter by source")
    parser.add_argument("--limit", type=int, default=0, help="Max episodes to score (0 = all)")
    parser.add_argument("--rate-limit", type=float, default=1.0, help="Seconds between API calls (default: 1.0)")
    parser.add_argument("--dry-run", action="store_true", help="Score but don't write to DB")

    args = parser.parse_args()

    # Default models per provider
    default_models = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o-mini",
        "ollama": "qwen2.5:7b",
    }
    model = args.model or default_models[args.provider]

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = connect_db(str(db_path))
    episodes = get_unscored_episodes(conn, source_filter=args.source, limit=args.limit)

    if not episodes:
        print("No unscored episodes found.", file=sys.stderr)
        conn.close()
        return

    print(f"Scoring {len(episodes)} episodes with {args.provider}/{model}", file=sys.stderr)

    scorer = get_scorer(args.provider)
    scored = 0
    failed = 0

    for i, ep in enumerate(episodes):
        print(f"  [{i+1}/{len(episodes)}] Episode {ep['id'][:8]}... ", end="", file=sys.stderr)

        result = score_episode(ep, scorer, model, args.provider)

        if result is None:
            failed += 1
            print("FAILED", file=sys.stderr)
        else:
            reward, details = result
            scored += 1
            print(f"reward={reward:+.3f} ({details.get('reason', '')[:50]})", file=sys.stderr)

            if not args.dry_run:
                conn.execute(
                    "UPDATE rl_episodes SET reward = ?, reward_source = ?, reward_details = ? WHERE id = ?",
                    (reward, "judge", json.dumps(details, ensure_ascii=False), ep["id"]),
                )
                conn.commit()

        # Rate limiting
        if i < len(episodes) - 1 and args.rate_limit > 0:
            time.sleep(args.rate_limit)

    conn.close()

    print(f"\nDone: {scored} scored, {failed} failed", file=sys.stderr)
    if args.dry_run:
        print("(dry run — no changes written to database)", file=sys.stderr)


if __name__ == "__main__":
    main()
