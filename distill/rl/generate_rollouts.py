#!/usr/bin/env python3
"""
Generate multiple completions (rollouts) per prompt for GRPO training.

For each prompt in the RL episode buffer, generates G completions from the
current local model. These are later scored and used to compute group-relative
advantages for GRPO training.

Supports three inference backends:
  - transformers: Direct HuggingFace model loading (default, works everywhere)
  - ollama:       REST API generation (simplest setup)
  - vllm:         High-throughput batched generation (best for GPU)

Usage:
    python generate_rollouts.py --episodes episodes.jsonl --output rollouts.jsonl
    python generate_rollouts.py --episodes episodes.jsonl --backend ollama --model qwen2.5:1.5b
    python generate_rollouts.py --episodes episodes.jsonl --backend vllm --model ./merged_model
"""

import argparse
import json
import sys
import uuid
from pathlib import Path


# ── Transformers backend ──

def load_transformers_model(model_path: str, adapter_path: str | None = None):
    """Load a HuggingFace model with optional LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_path}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        dtype = torch.bfloat16
        device_map = "auto"
    elif torch.backends.mps.is_available():
        dtype = torch.float16
        device_map = "mps"
    else:
        dtype = torch.float32
        device_map = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True,
    )

    if adapter_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter: {adapter_path}", file=sys.stderr)
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def generate_transformers(model, tokenizer, messages: list[dict], temperature: float, max_tokens: int) -> str:
    """Generate a single completion using HuggingFace Transformers."""
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens (not the prompt)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ── Ollama backend ──

def generate_ollama(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    base_url: str = "http://localhost:11434",
) -> str:
    """Generate a single completion using Ollama REST API."""
    import requests

    response = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


# ── vLLM backend ──

def generate_vllm(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    base_url: str = "http://localhost:8000",
) -> str:
    """Generate a single completion using vLLM OpenAI-compatible API."""
    import requests

    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ── Rollout generation ──

def generate_rollouts_for_episode(
    episode: dict,
    G: int,
    backend: str,
    model,
    tokenizer,
    model_name: str,
    temperature: float,
    max_tokens: int,
    ollama_url: str,
    vllm_url: str,
) -> dict:
    """Generate G completions for a single episode's prompt."""
    messages = episode["prompt_messages"]
    if isinstance(messages, str):
        messages = json.loads(messages)

    completions = []
    for g in range(G):
        try:
            if backend == "transformers":
                text = generate_transformers(model, tokenizer, messages, temperature, max_tokens)
            elif backend == "ollama":
                text = generate_ollama(messages, model_name, temperature, max_tokens, ollama_url)
            elif backend == "vllm":
                text = generate_vllm(messages, model_name, temperature, max_tokens, vllm_url)
            else:
                raise ValueError(f"Unknown backend: {backend}")

            completions.append({
                "index": g,
                "content": text,
                "finish_reason": "stop",
            })
        except Exception as e:
            print(f"    Warning: generation {g} failed: {e}", file=sys.stderr)
            completions.append({
                "index": g,
                "content": "",
                "finish_reason": "error",
                "error": str(e),
            })

    return {
        "id": str(uuid.uuid4()),
        "episode_id": episode.get("id", ""),
        "sample_id": episode.get("sample_id", ""),
        "prompt_messages": messages,
        "completions": completions,
        "source": episode.get("source", ""),
        "reference_completion": episode.get("completion"),
        "reference_reward": episode.get("reward"),
        "backend": backend,
        "model": model_name,
        "temperature": temperature,
        "G": G,
    }


def load_episodes(path: str) -> list[dict]:
    """Load episodes from JSONL file."""
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def load_episodes_from_db(db_path: str, source: str | None = None, limit: int = 0) -> list[dict]:
    """Load scored, unused episodes directly from SQLite."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT * FROM rl_episodes
        WHERE reward IS NOT NULL AND is_used_for_training = 0
    """
    params: list = []

    if source:
        query += " AND source = ?"
        params.append(source)

    query += " ORDER BY created_at ASC"

    if limit > 0:
        query += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    episodes = []
    for row in rows:
        row = dict(row)
        row["prompt_messages"] = json.loads(row["prompt_messages"])
        row["completion"] = json.loads(row["completion"])
        if row.get("reward_details"):
            row["reward_details"] = json.loads(row["reward_details"])
        episodes.append(row)

    return episodes


def main():
    parser = argparse.ArgumentParser(description="Generate rollouts for GRPO training")
    parser.add_argument("--episodes", help="Path to episodes JSONL file")
    parser.add_argument("--db", help="Path to louter.db (alternative to --episodes)")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file for rollouts")

    # Backend
    parser.add_argument(
        "--backend", default="transformers",
        choices=["transformers", "ollama", "vllm"],
        help="Inference backend (default: transformers)",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Model name or path")
    parser.add_argument("--adapter", help="LoRA adapter path (transformers backend only)")

    # Generation
    parser.add_argument("--G", type=int, default=4, help="Number of completions per prompt (default: 4)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per completion (default: 2048)")

    # Filtering
    parser.add_argument("--source", choices=["local", "cloud"], help="Filter episodes by source")
    parser.add_argument("--limit", type=int, default=0, help="Max episodes to process (0 = all)")

    # Backend URLs
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--vllm-url", default="http://localhost:8000", help="vLLM base URL")

    args = parser.parse_args()

    if not args.episodes and not args.db:
        print("Error: provide --episodes (JSONL) or --db (SQLite path)", file=sys.stderr)
        sys.exit(1)

    # Load episodes
    if args.episodes:
        episodes = load_episodes(args.episodes)
    else:
        episodes = load_episodes_from_db(args.db, source=args.source, limit=args.limit)

    if args.limit > 0:
        episodes = episodes[:args.limit]

    if not episodes:
        print("No episodes to process.", file=sys.stderr)
        return

    print(f"Generating {args.G} rollouts each for {len(episodes)} episodes", file=sys.stderr)
    print(f"Backend: {args.backend}, Model: {args.model}", file=sys.stderr)

    # Load model for transformers backend
    hf_model, hf_tokenizer = None, None
    if args.backend == "transformers":
        hf_model, hf_tokenizer = load_transformers_model(args.model, args.adapter)

    # Generate rollouts
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_completions = 0
    with open(output_path, "w") as out:
        for i, episode in enumerate(episodes):
            print(
                f"  [{i+1}/{len(episodes)}] Episode {str(episode.get('id', ''))[:8]}...",
                end=" ",
                file=sys.stderr,
            )

            rollout = generate_rollouts_for_episode(
                episode=episode,
                G=args.G,
                backend=args.backend,
                model=hf_model,
                tokenizer=hf_tokenizer,
                model_name=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                ollama_url=args.ollama_url,
                vllm_url=args.vllm_url,
            )

            successful = sum(1 for c in rollout["completions"] if c["finish_reason"] != "error")
            total_completions += successful
            print(f"{successful}/{args.G} ok", file=sys.stderr)

            out.write(json.dumps(rollout, ensure_ascii=False) + "\n")

    print(
        f"\nDone: {total_completions} completions across {len(episodes)} episodes → {output_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
