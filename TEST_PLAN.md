# End-to-End Testing Plan: SFT + RL Pipeline

A step-by-step plan for testing Louter's distillation and RL pipelines using external conversation logs. Designed to be followed by any coding agent or human operator.

## Prerequisites

- Python 3.10+
- ~10 GB disk space (model weights + training artifacts)
- GPU recommended (CUDA or Apple Silicon MPS), CPU works but is slow
- Internet access (to download the base model on first run)
- An API key for the RL judge model: set `ANTHROPIC_API_KEY` (recommended) or `OPENAI_API_KEY`

## Overview

```
Input:  folder of JSONL conversation logs
                ↓
        ingest_conversations.py  (new script)
                ↓
        louter.db  (SQLite with training_samples table)
               ╱            ╲
          SFT path          RL path
         export.py       export_episodes.py
         compress.py     generate_rollouts.py
         train.py        reward_rollouts.py
              ↓           train_grpo.py
        merged model          ↓
              ↓          merged model
        serve locally    serve locally
```

---

## Step 0: Setup Environment

```bash
cd /path/to/louter

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (SFT + RL)
pip install -r distill/requirements.txt
pip install -r distill/rl/requirements.txt
```

## Step 1: Prepare Conversation Logs

Place JSONL conversation log files in a single folder. Each `.jsonl` file contains one JSON object per line.

**Supported input formats** (the ingestion script auto-detects):

### Format A: OpenAI chat format (most common)

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."}
  ]
}
```

### Format B: OpenClaw session format

```json
{
  "session_id": "abc123",
  "turn": 1,
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "response_text": "The assistant's response...",
  "tool_calls": null,
  "next_state": {"score": 1}
}
```

### Format C: ShareGPT format

```json
{
  "conversations": [
    {"from": "system", "value": "You are a helpful assistant."},
    {"from": "human", "value": "What is 2+2?"},
    {"from": "gpt", "value": "2+2 equals 4."}
  ]
}
```

### Format D: Simple prompt/response

```json
{
  "prompt": "What is 2+2?",
  "response": "2+2 equals 4.",
  "system": "You are a helpful assistant."
}
```

**Example folder structure:**

```
test_data/
├── conversations_01.jsonl
├── conversations_02.jsonl
└── conversations_03.jsonl
```

## Step 2: Ingestion Script

The ingestion script `distill/ingest_conversations.py` converts external JSONL logs into Louter's `training_samples` SQLite table. It is already included in the repository.

```bash
python3 distill/ingest_conversations.py --help
```

## Step 3: Download Qwen3 1.5B (if not cached)

The model downloads automatically on first use via HuggingFace. To pre-download:

```bash
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'Qwen/Qwen3-1.5B'
print(f'Downloading {model_name}...')
AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
print('Done. Model cached at ~/.cache/huggingface/')
"
```

Verify download:

```bash
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
t = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.5B', trust_remote_code=True)
print(f'Vocab size: {t.vocab_size}')
print(f'Model loaded successfully')
"
```

> **Note**: The existing pipeline defaults to `Qwen/Qwen2.5-1.5B-Instruct`. To use Qwen3 instead, set `BASE_MODEL=Qwen/Qwen3-1.5B` in all commands below. The scripts accept any HuggingFace model name.

## Step 4: Ingest Conversation Logs

```bash
# Ingest external JSONL files into the SQLite database
python3 distill/ingest_conversations.py \
    --input-dir ./test_data \
    --db ./louter.db

# Verify
python3 distill/export.py --db ./louter.db --stats
```

Expected output:

```
Ingesting from: ./test_data
Database: ./louter.db

  conversations_01.jsonl: 150 samples ingested
  conversations_02.jsonl: 200 samples ingested

Total: 350 samples ingested, 5 skipped

=== Training Sample Statistics ===

Task Type        Total  Success    Tools Unexport       Tokens
-----------------------------------------------------------------
general            280      280        0      280            0
code                50       50        0       50            0
tool_call           20       20       20       20            0
-----------------------------------------------------------------
TOTAL              350
```

## Step 5: Choose a Pipeline Mode

### Option A: SFT Only

Supervised fine-tuning on successful conversations. Best for initial training when you have good reference data.

```bash
BASE_MODEL="Qwen/Qwen3-1.5B" \
LOUTER_DB="./louter.db" \
./distill/run_distill.sh
```

This runs: export → compress → train (LoRA) → merge → deploy.

Output: `distill/merged_model/` (full merged model with Modelfile)

**Verify SFT output:**

```bash
# Check that the merged model exists
ls -la distill/merged_model/
# Should contain: config.json, model*.safetensors, tokenizer*, Modelfile

# Quick sanity test
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained('./distill/merged_model', trust_remote_code=True)
t = AutoTokenizer.from_pretrained('./distill/merged_model', trust_remote_code=True)
inputs = t.apply_chat_template(
    [{'role': 'user', 'content': 'Hello!'}],
    tokenize=True, add_generation_prompt=True, return_tensors='pt'
)
out = m.generate(inputs, max_new_tokens=50, do_sample=False)
print(t.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
"
```

### Option B: RL Only

Reinforcement learning using GRPO. Requires data already in the database. Works best when you have a mix of successful and failed samples.

```bash
BASE_MODEL="Qwen/Qwen3-1.5B" \
LOUTER_DB="./louter.db" \
JUDGE_PROVIDER="anthropic" \
MIN_EPISODES=10 \
./distill/rl/run_rl.sh
```

> Set `MIN_EPISODES=10` (or lower) for testing with small datasets. Default is 100.
>
> **RL Judge**: The judge model scores rollout quality during RL training. Use a strong cloud model for best results:
> - `JUDGE_PROVIDER=anthropic` (default) — uses `claude-sonnet-4-20250514`. Requires `ANTHROPIC_API_KEY`.
> - `JUDGE_PROVIDER=openai` — uses `gpt-4o-mini`. Requires `OPENAI_API_KEY`.

Output: `distill/rl/rl_merged/` (full merged model with Modelfile)

**Verify RL output:**

```bash
ls -la distill/rl/rl_merged/
# Should contain: config.json, model*.safetensors, tokenizer*, Modelfile

# Check training artifacts
cat distill/rl/output/rl_adapter/training_meta.json | python3 -m json.tool
# Should show hyperparameters, steps completed, final KL divergence
```

### Option C: SFT + RL (Recommended)

Run SFT first to get a baseline, then RL to refine it. The RL pipeline automatically uses the SFT adapter as its starting point.

```bash
export BASE_MODEL="Qwen/Qwen3-1.5B"
export LOUTER_DB="./louter.db"

# Step C.1: SFT
./distill/run_distill.sh

# Step C.2: RL (starts from SFT adapter at distill/output)
ADAPTER_PATH="./distill/output" \
JUDGE_PROVIDER="anthropic" \
MIN_EPISODES=10 \
./distill/rl/run_rl.sh
```

Output: `distill/rl/rl_merged/` (RL-refined model, built on top of SFT)

## Step 6: Serve the Merged Model

Pick one serving method. Both expose an OpenAI-compatible API at `http://localhost:8000/v1`.

### Option 1: HuggingFace Transformers (recommended — works everywhere)

```bash
# Serve the SFT model
python3 distill/rl/serve_hf.py --model ./distill/merged_model --port 8000

# Or serve the RL model
python3 distill/rl/serve_hf.py --model ./distill/rl/rl_merged --port 8000

# Or serve base model + LoRA adapter (no merge step needed)
python3 distill/rl/serve_hf.py \
    --model Qwen/Qwen3-1.5B \
    --adapter ./distill/output \
    --port 8000
```

Works on CUDA, Apple Silicon (MPS), and CPU. No extra dependencies beyond the training requirements.

### Option 2: vLLM (GPU only, highest throughput)

```bash
pip install vllm  # requires CUDA
MODEL_PATH=./distill/rl/rl_merged PORT=8000 ./distill/rl/serve_vllm.sh
```

3-5x higher throughput than HuggingFace via continuous batching and PagedAttention. Requires a CUDA GPU.

## Step 7: Verify the Served Model

Test that the served model responds correctly:

```bash
# Health check (serve_hf.py only)
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "louter-rl",
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "louter-rl",
    "messages": [
      {"role": "user", "content": "Write a haiku about coding."}
    ],
    "stream": true
  }'
```

Expected: a valid JSON response with `choices[0].message.content` containing the model's answer.

---

## Quick Reference: Complete Commands

### Minimal test (SFT only, ~30 min on GPU)

```bash
source .venv/bin/activate
python3 distill/ingest_conversations.py --input-dir ./test_data --db ./louter.db
BASE_MODEL="Qwen/Qwen3-1.5B" LOUTER_DB="./louter.db" ./distill/run_distill.sh
python3 distill/rl/serve_hf.py --model ./distill/merged_model --port 8000
```

### Full test (SFT + RL, ~1-2 hours on GPU)

```bash
source .venv/bin/activate
python3 distill/ingest_conversations.py --input-dir ./test_data --db ./louter.db
export BASE_MODEL="Qwen/Qwen3-1.5B" LOUTER_DB="./louter.db"
./distill/run_distill.sh
ADAPTER_PATH="./distill/output" MIN_EPISODES=10 \
  JUDGE_PROVIDER="anthropic" \
  ./distill/rl/run_rl.sh
python3 distill/rl/serve_hf.py --model ./distill/rl/rl_merged --port 8000
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `No .jsonl files found` | Check `--input-dir` points to a directory with `.jsonl` files |
| `Not enough episodes (N < 100)` | Set `MIN_EPISODES=10` for testing, or add more data |
| `CUDA out of memory` | Reduce batch size: edit `run_distill.sh` to use `--batch-size 2 --gradient-accumulation 8` |
| `MPS not available` | Update to macOS 12.3+ and PyTorch 2.0+ |
| `Model not found at ./rl_merged` | Run training first — the merged model is created by the train+merge steps |
| `score_with_judge fails` | Check that `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` is set |
| `vLLM import error` | vLLM requires CUDA GPU — use `serve_hf.py` on Mac/CPU instead |
