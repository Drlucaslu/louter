#!/usr/bin/env python3
"""
Structural reward scoring for tool-call RL episodes.

Checks tool-call responses for:
  1. Valid JSON in function arguments
  2. Function name exists in the provided tools
  3. Required parameters are present
  4. No error patterns in subsequent tool results

These rewards are combined with (or replace) judge-based scores for
tool_call task types, providing fast, deterministic signal without
needing an API call.

Usage:
    python score_tool_calls.py --db ../louter.db                # Score all tool-call episodes
    python score_tool_calls.py --db ../louter.db --dry-run      # Preview without writing
    python score_tool_calls.py --db ../louter.db --stats        # Show scoring stats
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_tool_call_episodes(
    conn: sqlite3.Connection,
    only_unscored: bool = True,
    limit: int = 0,
) -> list[dict]:
    """Get rl_episodes that involve tool calls."""
    # Join with training_samples to get tool definitions and task_type
    query = """
        SELECT e.*, t.request_tools, t.task_type, t.has_tool_calls, t.request_messages
        FROM rl_episodes e
        JOIN training_samples t ON e.sample_id = t.id
        WHERE t.task_type = 'tool_call' OR t.has_tool_calls = 1
    """
    params: list = []

    if only_unscored:
        query += " AND (e.reward_source != 'environment' OR e.reward_source IS NULL)"

    query += " ORDER BY e.created_at ASC"

    if limit > 0:
        query += " LIMIT ?"
        params.append(limit)

    cursor = conn.execute(query, params)
    return [dict(row) for row in cursor.fetchall()]


def parse_tool_definitions(tools_json: str | None) -> list[dict]:
    """Parse tool/function definitions from the request."""
    if not tools_json:
        return []
    try:
        tools = json.loads(tools_json)
        if isinstance(tools, list):
            return tools
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def get_available_function_names(tools: list[dict]) -> set[str]:
    """Extract function names from tool definitions."""
    names = set()
    for tool in tools:
        # OpenAI format: {"type": "function", "function": {"name": "..."}}
        if isinstance(tool, dict):
            fn = tool.get("function", {})
            if isinstance(fn, dict) and "name" in fn:
                names.add(fn["name"])
            elif "name" in tool:
                names.add(tool["name"])
    return names


def get_required_params(tools: list[dict], function_name: str) -> set[str]:
    """Get required parameters for a function from tool definitions."""
    for tool in tools:
        fn = tool.get("function", {})
        if isinstance(fn, dict) and fn.get("name") == function_name:
            params = fn.get("parameters", {})
            return set(params.get("required", []))
    return set()


def check_error_in_tool_results(messages_json: str) -> bool:
    """Check if any tool result messages contain error patterns."""
    try:
        messages = json.loads(messages_json) if isinstance(messages_json, str) else messages_json
    except (json.JSONDecodeError, TypeError):
        return False

    error_patterns = [
        "error", "Error", "ERROR",
        "exception", "Exception",
        "failed", "Failed", "FAILED",
        "traceback", "Traceback",
        "not found", "Not Found",
        "invalid", "Invalid",
        "denied", "Denied",
        "permission", "Permission",
    ]

    for msg in messages:
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        if any(p in content for p in error_patterns):
            return True

    return False


def score_tool_call_episode(episode: dict) -> tuple[float, dict]:
    """Score a single tool-call episode on structural criteria.

    Returns (reward, details) where reward is in [-1.0, 1.0].
    """
    completion_json = episode["completion"]
    try:
        completion = json.loads(completion_json) if isinstance(completion_json, str) else completion_json
    except (json.JSONDecodeError, TypeError):
        return -1.0, {"reason": "unparseable_completion", "checks": {}}

    tools = parse_tool_definitions(episode.get("request_tools"))
    available_names = get_available_function_names(tools)

    tool_calls = completion.get("tool_calls", [])
    checks = {
        "has_tool_calls": len(tool_calls) > 0,
        "valid_json_args": True,
        "known_functions": True,
        "required_params_present": True,
        "no_tool_errors": True,
        "function_details": [],
    }

    if not tool_calls:
        # No tool calls in response — could be valid (model decided not to call)
        # or could be a failure (model should have called a tool). Neutral score.
        content = completion.get("content", "")
        if content:
            return 0.0, {"reason": "no_tool_calls_text_response", "checks": checks}
        return -0.5, {"reason": "no_tool_calls_no_content", "checks": checks}

    total_score = 0.0
    num_calls = len(tool_calls)

    for tc in tool_calls:
        fn = tc.get("function", {})
        fn_name = fn.get("name", "")
        fn_args_raw = fn.get("arguments", "")
        tc_detail = {"name": fn_name}

        # Check 1: Valid JSON arguments
        try:
            if isinstance(fn_args_raw, str):
                fn_args = json.loads(fn_args_raw)
            else:
                fn_args = fn_args_raw
            tc_detail["valid_json"] = True
            total_score += 0.25
        except (json.JSONDecodeError, TypeError):
            tc_detail["valid_json"] = False
            checks["valid_json_args"] = False
            total_score -= 0.5
            fn_args = {}

        # Check 2: Function name exists in tools
        if available_names:
            if fn_name in available_names:
                tc_detail["known_function"] = True
                total_score += 0.25
            else:
                tc_detail["known_function"] = False
                checks["known_functions"] = False
                total_score -= 0.5
        else:
            # No tool definitions to check against — skip
            tc_detail["known_function"] = None

        # Check 3: Required parameters present
        if available_names and fn_name in available_names and isinstance(fn_args, dict):
            required = get_required_params(tools, fn_name)
            present = set(fn_args.keys())
            missing = required - present
            if not missing:
                tc_detail["required_params"] = True
                total_score += 0.25
            else:
                tc_detail["required_params"] = False
                tc_detail["missing_params"] = list(missing)
                checks["required_params_present"] = False
                total_score -= 0.25

        checks["function_details"].append(tc_detail)

    # Check 4: Error patterns in tool results
    if check_error_in_tool_results(episode.get("request_messages", "[]")):
        checks["no_tool_errors"] = False
        total_score -= 0.25

    # Normalize to [-1.0, 1.0]
    # Max possible per call: 0.75 (json + name + params)
    # Normalize by number of calls
    max_possible = num_calls * 0.75
    if max_possible > 0:
        reward = max(-1.0, min(1.0, total_score / max_possible))
    else:
        reward = 0.0

    reason = "all_checks_passed" if all(
        checks[k] for k in ("valid_json_args", "known_functions", "required_params_present", "no_tool_errors")
    ) else "structural_issues"

    return round(reward, 4), {"reason": reason, "checks": checks}


def print_stats(conn: sqlite3.Connection):
    """Print tool-call scoring statistics."""
    cursor = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN e.reward_source = 'environment' THEN 1 ELSE 0 END) as scored,
            ROUND(AVG(CASE WHEN e.reward_source = 'environment' THEN e.reward END), 3) as avg_reward
        FROM rl_episodes e
        JOIN training_samples t ON e.sample_id = t.id
        WHERE t.task_type = 'tool_call' OR t.has_tool_calls = 1
    """)

    row = dict(cursor.fetchone())
    print("\n=== Tool-Call Scoring Stats ===\n", file=sys.stderr)
    print(f"Total tool-call episodes:  {row['total']}", file=sys.stderr)
    print(f"Scored (environment):      {row['scored']}", file=sys.stderr)
    if row["avg_reward"] is not None:
        print(f"Average reward:            {row['avg_reward']:+.3f}", file=sys.stderr)
    print(file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Score tool-call RL episodes structurally")
    parser.add_argument("--db", default="louter.db", help="Path to louter.db")
    parser.add_argument("--limit", type=int, default=0, help="Max episodes to score (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Score but don't write to DB")
    parser.add_argument("--rescore", action="store_true", help="Re-score already scored episodes")
    parser.add_argument("--stats", action="store_true", help="Print stats and exit")

    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = connect_db(str(db_path))

    if args.stats:
        print_stats(conn)
        conn.close()
        return

    episodes = get_tool_call_episodes(
        conn,
        only_unscored=not args.rescore,
        limit=args.limit,
    )

    if not episodes:
        print("No tool-call episodes to score.", file=sys.stderr)
        conn.close()
        return

    print(f"Scoring {len(episodes)} tool-call episodes", file=sys.stderr)

    scored = 0
    for i, ep in enumerate(episodes):
        reward, details = score_tool_call_episode(ep)
        scored += 1

        status = "OK" if details["reason"] == "all_checks_passed" else details["reason"]
        print(
            f"  [{i+1}/{len(episodes)}] {ep['id'][:8]}... reward={reward:+.4f} ({status})",
            file=sys.stderr,
        )

        if not args.dry_run:
            conn.execute(
                "UPDATE rl_episodes SET reward = ?, reward_source = ?, reward_details = ? WHERE id = ?",
                (reward, "environment", json.dumps(details, ensure_ascii=False), ep["id"]),
            )

    if not args.dry_run:
        conn.commit()

    conn.close()
    print(f"\nDone: {scored} episodes scored", file=sys.stderr)
    if args.dry_run:
        print("(dry run — no changes written to database)", file=sys.stderr)


if __name__ == "__main__":
    main()
