#!/usr/bin/env python3
"""
Export training samples from Louter's SQLite database as RL episodes
with implicit reward signals.

Reward assignment:
  +1.0  — successful response, no retry detected
  -1.0  — failed response (is_successful=0, i.e. retry detected)
  -0.5  — local response that was followed by a cloud fallback
  +1.0  — cloud response (used as positive reference)

Usage:
    python export_episodes.py                             # Export all unexported
    python export_episodes.py --db ../louter.db --stats   # Show episode stats
    python export_episodes.py --output episodes.jsonl     # Export to JSONL file
    python export_episodes.py --source local              # Only local episodes
"""

import argparse
import json
import sqlite3
import sys
import uuid
from pathlib import Path


def connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_rl_table(conn: sqlite3.Connection):
    """Create rl_episodes table if it doesn't exist (for standalone use)."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS rl_episodes (
            id TEXT PRIMARY KEY,
            sample_id TEXT NOT NULL,
            prompt_messages TEXT NOT NULL,
            completion TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'cloud',
            reward REAL,
            reward_source TEXT,
            reward_details TEXT,
            is_used_for_training INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_rl_sample_id ON rl_episodes(sample_id);
        CREATE INDEX IF NOT EXISTS idx_rl_source ON rl_episodes(source);
        CREATE INDEX IF NOT EXISTS idx_rl_reward ON rl_episodes(reward);
        CREATE INDEX IF NOT EXISTS idx_rl_is_used ON rl_episodes(is_used_for_training);
    """)


def find_fallback_pairs(conn: sqlite3.Connection) -> set[str]:
    """Find sample IDs where local failed and cloud took over.

    A fallback is detected when a local sample with is_successful=0 is followed
    by a cloud sample with similar timing (within 60s) for the same task type.
    """
    cursor = conn.execute("""
        SELECT l.id
        FROM training_samples l
        JOIN training_samples c ON c.task_type = l.task_type
            AND c.source = 'cloud'
            AND c.is_successful = 1
            AND abs(julianday(c.created_at) - julianday(l.created_at)) * 86400 < 60
        WHERE l.source = 'local' AND l.is_successful = 0
    """)
    return {row["id"] for row in cursor.fetchall()}


def get_already_exported_sample_ids(conn: sqlite3.Connection) -> set[str]:
    """Get sample_ids that already have rl_episodes."""
    cursor = conn.execute("SELECT DISTINCT sample_id FROM rl_episodes")
    return {row["sample_id"] for row in cursor.fetchall()}


def assign_reward(sample: dict, fallback_ids: set[str]) -> tuple[float, str, dict]:
    """Assign implicit reward to a training sample.

    Returns (reward, reward_source, reward_details).
    """
    source = sample["source"]
    is_successful = bool(sample["is_successful"])
    sample_id = sample["id"]

    details = {
        "source": source,
        "is_successful": is_successful,
        "task_type": sample["task_type"],
    }

    if not is_successful:
        # Failed response (retry detected by FeedbackTracker)
        details["reason"] = "retry_detected"
        return -1.0, "implicit", details

    if source == "local" and sample_id in fallback_ids:
        # Local response that led to cloud fallback
        details["reason"] = "fallback_to_cloud"
        return -0.5, "implicit", details

    if source == "cloud":
        # Successful cloud response — positive reference
        details["reason"] = "cloud_success"
        return 1.0, "implicit", details

    # Successful local response — positive
    details["reason"] = "local_success"
    return 1.0, "implicit", details


def export_episodes(
    conn: sqlite3.Connection,
    source_filter: str | None = None,
    limit: int = 0,
    write_to_db: bool = False,
) -> list[dict]:
    """Export training samples as RL episodes with implicit rewards."""

    already_exported = get_already_exported_sample_ids(conn)
    fallback_ids = find_fallback_pairs(conn)

    query = "SELECT * FROM training_samples WHERE 1=1"
    params: list = []

    if source_filter:
        query += " AND source = ?"
        params.append(source_filter)

    query += " ORDER BY created_at ASC"

    if limit > 0:
        query += " LIMIT ?"
        params.append(limit)

    cursor = conn.execute(query, params)
    rows = cursor.fetchall()

    episodes = []
    new_rows_for_db = []

    for row in rows:
        row = dict(row)
        if row["id"] in already_exported:
            continue

        # Parse messages and response
        try:
            messages = json.loads(row["request_messages"])
            response = json.loads(row["response_content"])
        except (json.JSONDecodeError, TypeError):
            continue

        reward, reward_source, reward_details = assign_reward(row, fallback_ids)

        episode = {
            "id": str(uuid.uuid4()),
            "sample_id": row["id"],
            "prompt_messages": messages,
            "completion": response,
            "source": row["source"],
            "reward": reward,
            "reward_source": reward_source,
            "reward_details": reward_details,
        }
        episodes.append(episode)

        if write_to_db:
            new_rows_for_db.append(episode)

    if write_to_db and new_rows_for_db:
        ensure_rl_table(conn)
        for ep in new_rows_for_db:
            conn.execute(
                "INSERT OR IGNORE INTO rl_episodes "
                "(id, sample_id, prompt_messages, completion, source, reward, reward_source, reward_details) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ep["id"],
                    ep["sample_id"],
                    json.dumps(ep["prompt_messages"], ensure_ascii=False),
                    json.dumps(ep["completion"], ensure_ascii=False),
                    ep["source"],
                    ep["reward"],
                    ep["reward_source"],
                    json.dumps(ep["reward_details"], ensure_ascii=False),
                ),
            )
        conn.commit()
        print(f"Wrote {len(new_rows_for_db)} episodes to rl_episodes table", file=sys.stderr)

    return episodes


def print_stats(conn: sqlite3.Connection):
    """Print statistics about RL episodes."""
    ensure_rl_table(conn)

    cursor = conn.execute("""
        SELECT
            source,
            COUNT(*) as total,
            SUM(CASE WHEN reward IS NOT NULL THEN 1 ELSE 0 END) as scored,
            SUM(CASE WHEN is_used_for_training = 1 THEN 1 ELSE 0 END) as used,
            ROUND(AVG(reward), 3) as avg_reward,
            SUM(CASE WHEN reward > 0 THEN 1 ELSE 0 END) as positive,
            SUM(CASE WHEN reward < 0 THEN 1 ELSE 0 END) as negative
        FROM rl_episodes
        GROUP BY source
    """)

    rows = cursor.fetchall()
    if not rows:
        print("No RL episodes found.", file=sys.stderr)

        # Show training_samples count as hint
        ts_count = conn.execute("SELECT COUNT(*) FROM training_samples").fetchone()[0]
        if ts_count > 0:
            print(
                f"Found {ts_count} training samples. "
                f"Run without --stats to export them as episodes.",
                file=sys.stderr,
            )
        return

    print("\n=== RL Episode Statistics ===\n", file=sys.stderr)
    print(
        f"{'Source':<10} {'Total':>8} {'Scored':>8} {'Used':>8} "
        f"{'Avg Reward':>11} {'Positive':>9} {'Negative':>9}",
        file=sys.stderr,
    )
    print("-" * 70, file=sys.stderr)

    grand_total = 0
    for row in rows:
        row = dict(row)
        avg_r = f"{row['avg_reward']:.3f}" if row["avg_reward"] is not None else "N/A"
        print(
            f"{row['source']:<10} {row['total']:>8} {row['scored']:>8} {row['used']:>8} "
            f"{avg_r:>11} {row['positive']:>9} {row['negative']:>9}",
            file=sys.stderr,
        )
        grand_total += row["total"]

    print("-" * 70, file=sys.stderr)
    print(f"{'TOTAL':<10} {grand_total:>8}", file=sys.stderr)

    # Reward source breakdown
    cursor2 = conn.execute("""
        SELECT reward_source, COUNT(*) as cnt, ROUND(AVG(reward), 3) as avg_r
        FROM rl_episodes
        WHERE reward IS NOT NULL
        GROUP BY reward_source
        ORDER BY cnt DESC
    """)
    rs_rows = cursor2.fetchall()
    if rs_rows:
        print("\nBy reward source:", file=sys.stderr)
        for r in rs_rows:
            r = dict(r)
            print(
                f"  {r['reward_source'] or 'none':<15} {r['cnt']:>6} episodes, avg reward: {r['avg_r']}",
                file=sys.stderr,
            )
    print(file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Export Louter training samples as RL episodes")
    parser.add_argument(
        "--db", default="louter.db", help="Path to louter.db (default: louter.db)"
    )
    parser.add_argument(
        "--source", choices=["local", "cloud"],
        help="Filter by source (default: all)",
    )
    parser.add_argument(
        "--output", "-o", help="Output JSONL file (default: stdout)"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max episodes to export (0 = all)"
    )
    parser.add_argument(
        "--write-db", action="store_true",
        help="Write episodes to rl_episodes table in the database",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Print episode statistics and exit"
    )

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

    episodes = export_episodes(
        conn,
        source_filter=args.source,
        limit=args.limit,
        write_to_db=args.write_db,
    )
    conn.close()

    if not episodes:
        print("No new episodes to export.", file=sys.stderr)
        return

    # Write JSONL
    output = open(args.output, "w") if args.output else sys.stdout
    for ep in episodes:
        output.write(json.dumps(ep, ensure_ascii=False) + "\n")

    if args.output:
        output.close()

    # Summary
    pos = sum(1 for e in episodes if (e.get("reward") or 0) > 0)
    neg = sum(1 for e in episodes if (e.get("reward") or 0) < 0)
    print(
        f"Exported {len(episodes)} episodes ({pos} positive, {neg} negative)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
