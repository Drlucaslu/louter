#!/usr/bin/env python3
"""
Ingest external conversation JSONL files into Louter's training_samples table.

Supports: OpenAI chat format, OpenClaw session format, ShareGPT format,
          and simple prompt/response format.

Usage:
    python ingest_conversations.py --input-dir ./test_data --db ./louter.db
    python ingest_conversations.py --input-dir ./test_data --db ./louter.db --source cloud
    python ingest_conversations.py --input-dir ./test_data --db ./louter.db --dry-run
"""

import argparse
import glob
import json
import sqlite3
import sys
import uuid
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS training_samples (
    id TEXT PRIMARY KEY,
    request_messages TEXT NOT NULL,
    request_tools TEXT,
    response_content TEXT NOT NULL,
    request_model TEXT NOT NULL DEFAULT '',
    actual_model TEXT NOT NULL DEFAULT '',
    provider_type TEXT NOT NULL DEFAULT '',
    task_type TEXT NOT NULL DEFAULT 'general',
    has_tool_calls INTEGER NOT NULL DEFAULT 0,
    is_successful INTEGER NOT NULL DEFAULT 1,
    source TEXT NOT NULL DEFAULT 'cloud',
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    is_exported INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_ts_task_type ON training_samples(task_type);
CREATE INDEX IF NOT EXISTS idx_ts_created_at ON training_samples(created_at);
CREATE INDEX IF NOT EXISTS idx_ts_is_successful ON training_samples(is_successful);
CREATE INDEX IF NOT EXISTS idx_ts_source ON training_samples(source);
CREATE INDEX IF NOT EXISTS idx_ts_is_exported ON training_samples(is_exported);
"""


def detect_and_convert(record: dict) -> dict | None:
    """Auto-detect format and convert to normalized form.

    Returns {"messages": [...], "response": {...}, "tools": [...] | None,
             "has_tool_calls": int, "is_successful": int}
    or None if unparseable.
    """
    # Format A: OpenAI chat {"messages": [...]}
    if "messages" in record and isinstance(record["messages"], list):
        messages = record["messages"]
        # If last message is assistant, split it off as the response
        if messages and messages[-1].get("role") == "assistant":
            prompt_messages = messages[:-1]
            response = messages[-1]
        # OpenClaw variant: messages + response_text
        elif "response_text" in record:
            prompt_messages = messages
            response = {"role": "assistant", "content": record["response_text"]}
        else:
            return None

        tools = record.get("tools")
        has_tool_calls = bool(response.get("tool_calls"))

        # OpenClaw: use next_state.score for success signal
        is_successful = 1
        if "next_state" in record and isinstance(record.get("next_state"), dict):
            score = record["next_state"].get("score", 1)
            is_successful = 0 if score < 0 else 1

        return {
            "messages": prompt_messages,
            "response": response,
            "tools": tools,
            "has_tool_calls": int(has_tool_calls),
            "is_successful": is_successful,
        }

    # Format C: ShareGPT {"conversations": [...]}
    if "conversations" in record and isinstance(record["conversations"], list):
        role_map = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}
        messages = []
        for turn in record["conversations"]:
            role = role_map.get(turn.get("from", ""), turn.get("from", "user"))
            messages.append({"role": role, "content": turn.get("value", "")})

        if messages and messages[-1]["role"] == "assistant":
            return {
                "messages": messages[:-1],
                "response": messages[-1],
                "tools": None,
                "has_tool_calls": 0,
                "is_successful": 1,
            }
        return None

    # Format D: Simple {"prompt": "...", "response": "..."}
    if "prompt" in record and "response" in record:
        messages = []
        if record.get("system"):
            messages.append({"role": "system", "content": record["system"]})
        messages.append({"role": "user", "content": record["prompt"]})
        return {
            "messages": messages,
            "response": {"role": "assistant", "content": record["response"]},
            "tools": None,
            "has_tool_calls": 0,
            "is_successful": 1,
        }

    return None


def classify_task_type(messages: list, response: dict) -> str:
    """Simple heuristic task classification."""
    all_text = " ".join(
        m.get("content", "") or "" for m in messages + [response]
        if isinstance(m.get("content"), str)
    ).lower()

    if response.get("tool_calls"):
        return "tool_call"
    for kw in ["translate", "翻译", "translation"]:
        if kw in all_text[:500]:
            return "translation"
    for kw in ["def ", "function ", "class ", "```", "import ", "console.log"]:
        if kw in (response.get("content") or ""):
            return "code"
    return "general"


def ingest(input_dir: str, db_path: str, source: str = "cloud",
           dry_run: bool = False) -> int:
    """Ingest all JSONL files from input_dir into the database."""
    files = sorted(glob.glob(str(Path(input_dir) / "*.jsonl")))
    if not files:
        print(f"No .jsonl files found in {input_dir}", file=sys.stderr)
        return 0

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)

    total = 0
    skipped = 0

    for filepath in files:
        filename = Path(filepath).name
        file_count = 0
        with open(filepath) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  SKIP {filename}:{line_num} — invalid JSON", file=sys.stderr)
                    skipped += 1
                    continue

                converted = detect_and_convert(record)
                if converted is None:
                    skipped += 1
                    continue

                task_type = classify_task_type(
                    converted["messages"], converted["response"]
                )

                if not dry_run:
                    conn.execute(
                        "INSERT OR IGNORE INTO training_samples "
                        "(id, request_messages, request_tools, response_content, "
                        " request_model, actual_model, provider_type, task_type, "
                        " has_tool_calls, is_successful, source) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            str(uuid.uuid4()),
                            json.dumps(converted["messages"], ensure_ascii=False),
                            json.dumps(converted["tools"]) if converted["tools"] else None,
                            json.dumps(converted["response"], ensure_ascii=False),
                            "",  # request_model
                            "",  # actual_model
                            "",  # provider_type
                            task_type,
                            converted["has_tool_calls"],
                            converted["is_successful"],
                            source,
                        ),
                    )

                file_count += 1
                total += 1

        print(f"  {filename}: {file_count} samples ingested", file=sys.stderr)

    if not dry_run:
        conn.commit()
    conn.close()

    print(f"\nTotal: {total} samples ingested, {skipped} skipped", file=sys.stderr)
    if dry_run:
        print("(dry run — nothing written to database)", file=sys.stderr)
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Ingest external conversation logs into Louter's training_samples table"
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing .jsonl conversation log files",
    )
    parser.add_argument(
        "--db", default="louter.db",
        help="Path to louter.db (will be created if missing)",
    )
    parser.add_argument(
        "--source", default="cloud", choices=["cloud", "local"],
        help="Label all ingested samples as this source (default: cloud)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and validate without writing to database",
    )

    args = parser.parse_args()

    if not Path(args.input_dir).is_dir():
        print(f"Error: {args.input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Ingesting from: {args.input_dir}", file=sys.stderr)
    print(f"Database: {args.db}", file=sys.stderr)
    print(f"Source label: {args.source}", file=sys.stderr)
    print(file=sys.stderr)

    ingest(args.input_dir, args.db, args.source, args.dry_run)


if __name__ == "__main__":
    main()
