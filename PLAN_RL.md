# Reinforcement Learning Plan for Louter

> **Status: Implemented.** All phases (1-5) have been implemented. See `distill/rl/` for the complete pipeline.

Inspired by [OpenClaw RL](https://github.com/Gen-Verse/OpenClaw-RL), this plan extends Louter's existing LoRA distillation pipeline with reinforcement learning to continuously improve the local model from natural conversation feedback.

## Background

### What we have today

Louter already collects training data from the hybrid routing pipeline:

1. **Data collection**: Every chat completion is saved to `training_samples` in SQLite (messages, tools, response, task type, success status)
2. **Implicit feedback**: The `FeedbackTracker` detects agent retries within 60 seconds and marks previous samples as failed — free success/failure labels with zero manual annotation
3. **LoRA distillation**: `distill/run_distill.sh` runs Export → Compress → Train (SFT) → Merge → Deploy to Ollama
4. **Routing history**: Per-task-type success rates tracked in `routing_history` table

This gives us supervised fine-tuning (SFT) on successful cloud responses. But SFT only imitates — it cannot learn *from mistakes* or optimize for specific reward signals.

### What OpenClaw RL adds

OpenClaw RL introduces three key ideas we can adapt:

| OpenClaw Concept | Louter Adaptation |
|---|---|
| **GRPO (Group Relative Policy Optimization)** | Train the local model with RL using grouped completions and relative advantages — no separate critic network needed |
| **Next-state reward** | Use Louter's existing retry detection + cloud model as judge to score local responses without manual labels |
| **On-Policy Distillation (OPD)** | When local fails and cloud succeeds, use the cloud response as a teacher signal at the token level |
| **Async training loop** | Model keeps serving while RL training runs in background; hot-swap weights on completion |

### Why RL on top of SFT

SFT teaches the model to imitate good responses. RL teaches it to *avoid bad responses* and *maximize reward*. Concretely:

- SFT cannot use failed samples (it only trains on good outputs)
- RL uses both positive and negative signal — a retry-marked sample becomes a negative reward
- GRPO compares multiple completions for the same prompt, learning which response *style* works better
- OPD provides token-level gradient signal from cloud→local, stronger than sequence-level SFT

## Architecture Overview

```
                         Louter Proxy (:6188)
                              │
                  ┌───────────┼───────────┐
                  │           │           │
              Serve local   Collect    Implicit
              model via     samples    feedback
              OpenAI API               (retry detection)
                  │           │           │
                  │           ▼           ▼
                  │     ┌─────────────────────┐
                  │     │   SQLite Database    │
                  │     │  training_samples    │
                  │     │  routing_history     │
                  │     │  rl_episodes (new)   │
                  │     └─────────────────────┘
                  │               │
                  │               ▼
                  │     ┌─────────────────────┐
                  │     │  RL Pipeline (new)   │
                  │     │                     │
                  │     │  1. Export episodes  │
                  │     │  2. Score (reward)   │
                  │     │  3. Generate rollouts│
                  │     │  4. GRPO training    │
                  │     │  5. Merge + deploy   │
                  │     └─────────────────────┘
                  │               │
                  ▼               ▼
              ┌───────────────────────┐
              │  Updated local model  │
              │  (vLLM / Ollama /     │
              │   HF Transformers)    │
              └───────────────────────┘
```

## Implementation Plan

### Phase 1: Reward Infrastructure [DONE]

**Goal**: Enrich the existing data collection with reward signals usable for RL.

#### 1.1 New database table: `rl_episodes`

**Migration**: `migrations/005_rl_episodes.sql` — auto-applied on Louter startup.

**Rust layer**: `src/db/schema.rs` (`RlEpisodeRow`) + `src/db/mod.rs` (insert, update reward, mark used, stats).

```sql
CREATE TABLE IF NOT EXISTS rl_episodes (
    id TEXT PRIMARY KEY,
    sample_id TEXT NOT NULL,               -- FK to training_samples.id
    prompt_messages TEXT NOT NULL,          -- JSON: input messages (without response)
    completion TEXT NOT NULL,               -- JSON: model response
    source TEXT NOT NULL,                   -- 'local' or 'cloud'
    reward REAL,                            -- scalar reward [-1.0, 1.0]
    reward_source TEXT,                     -- 'implicit', 'judge', 'environment', 'manual'
    reward_details TEXT,                    -- JSON: breakdown of reward components
    is_used_for_training INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

**Script**: `distill/rl/export_episodes.py`

```bash
# Export to JSONL file
python export_episodes.py --db ../../louter.db --output episodes.jsonl

# Export and write to DB simultaneously
python export_episodes.py --db ../../louter.db --output episodes.jsonl --write-db

# Filter by source
python export_episodes.py --db ../../louter.db --source local --output local_episodes.jsonl

# View stats
python export_episodes.py --db ../../louter.db --stats
```

Reward assignment from implicit feedback:
- `is_successful=1` and no retry → reward = +1.0
- `is_successful=0` or retry detected → reward = -1.0
- Cloud fallback after local failure → local gets -0.5, cloud gets +1.0

Fallback detection: joins `training_samples` to find local failures followed by cloud successes within 60 seconds on the same task type. Deduplication: tracks `sample_id` to avoid re-exporting.

#### 1.2 Judge-based reward scoring

**Script**: `distill/rl/score_with_judge.py`

Scores on three dimensions (correctness, helpfulness, completeness) rated 1-5, normalized to [-1.0, 1.0] via `(avg - 3.0) / 2.0`.

```bash
# Score unscored episodes using Anthropic (default)
python score_with_judge.py --db ../../louter.db

# Use local judge via Ollama
python score_with_judge.py --db ../../louter.db --provider ollama --model qwen2.5:7b

# Score only local episodes, preview without writing
python score_with_judge.py --db ../../louter.db --source local --dry-run --limit 10

# Adjust rate limiting (default: 1.0s between calls)
python score_with_judge.py --db ../../louter.db --rate-limit 0.5
```

Default judge models per provider:
- `anthropic`: `claude-sonnet-4-20250514`
- `openai`: `gpt-4o-mini`
- `ollama`: `qwen2.5:7b`

The judge prompt requests JSON output. Parsing handles markdown code fences and extracts JSON from mixed text. Failed parses are logged and skipped.

#### 1.3 Environment-based rewards (for tool-calling tasks)

**Script**: `distill/rl/score_tool_calls.py`

Deterministic structural scoring — no API calls needed. Runs in milliseconds.

```bash
# Score all unscored tool-call episodes
python score_tool_calls.py --db ../../louter.db

# Re-score previously scored episodes
python score_tool_calls.py --db ../../louter.db --rescore

# Preview
python score_tool_calls.py --db ../../louter.db --dry-run
```

Four structural checks per tool call:
1. **Valid JSON arguments**: Can `function.arguments` be parsed as JSON? (+0.25 / -0.5)
2. **Known function name**: Does the function name exist in the provided tools? (+0.25 / -0.5)
3. **Required parameters**: Are all required params present? (+0.25 / -0.25)
4. **No error patterns**: Do subsequent tool-result messages contain error keywords? (+0 / -0.25)

Scores are normalized per tool-call count and clamped to [-1.0, 1.0]. Stored as `reward_source='environment'`.

### Phase 2: GRPO Training Pipeline [DONE]

**Goal**: Implement Group Relative Policy Optimization adapted for small models on consumer hardware.

#### 2.1 Rollout generation

**Script**: `distill/rl/generate_rollouts.py`

```bash
# From episodes file (Transformers backend)
python generate_rollouts.py --episodes episodes.jsonl --output rollouts.jsonl \
    --model Qwen/Qwen2.5-1.5B-Instruct --adapter ../output

# From database directly
python generate_rollouts.py --db ../../louter.db --output rollouts.jsonl

# Using Ollama backend
python generate_rollouts.py --episodes episodes.jsonl --output rollouts.jsonl \
    --backend ollama --model qwen2.5:1.5b

# Using vLLM backend
python generate_rollouts.py --episodes episodes.jsonl --output rollouts.jsonl \
    --backend vllm --model ./rl_merged --vllm-url http://localhost:8000

# Customize generation
python generate_rollouts.py --episodes episodes.jsonl --output rollouts.jsonl \
    --G 8 --temperature 0.9 --max-tokens 1024 --limit 100
```

Three inference backends:
- **transformers** (default): Loads model + optional LoRA adapter directly. Auto-detects CUDA/MPS/CPU.
- **ollama**: REST API at `--ollama-url` (default `localhost:11434`). Simplest setup.
- **vllm**: OpenAI-compatible API at `--vllm-url` (default `localhost:8000`). Highest throughput.

Each rollout group contains: prompt messages, G completions (with index and finish_reason), reference completion from the original episode, and metadata.

#### 2.2 Reward assignment for rollouts

**Script**: `distill/rl/reward_rollouts.py`

```bash
# Score with judge + reference comparison (default)
python reward_rollouts.py --rollouts rollouts.jsonl --output scored_rollouts.jsonl

# Skip judge, use reference comparison only (fast, no API calls)
python reward_rollouts.py --rollouts rollouts.jsonl --output scored.jsonl --no-judge

# Use local judge
python reward_rollouts.py --rollouts rollouts.jsonl --output scored.jsonl \
    --judge-provider ollama --judge-model qwen2.5:7b

# Adjust scoring weights
python reward_rollouts.py --rollouts rollouts.jsonl --output scored.jsonl \
    --reference-weight 0.3 --judge-weight 0.7
```

Scoring combines two signals per completion:
1. **Reference comparison**: ROUGE-L similarity to the cloud model's response (if available). Uses `rouge_score` library with fallback to a built-in LCS implementation.
2. **Judge scoring**: 1-5 score from judge model, normalized to [-1, 1].

Combined reward: weighted average of signals (default: 40% reference, 60% judge).

**GRPO advantage computation** — the core algorithm:
```python
advantage[i] = (reward[i] - group_mean) / group_std
```
Completions above the group mean get positive advantage (reinforced), below get negative (penalized). No critic network needed.

#### 2.3 GRPO training step

**Script**: `distill/rl/train_grpo.py`

```bash
# Train from scored rollouts
python train_grpo.py --data scored_rollouts.jsonl --base-model Qwen/Qwen2.5-1.5B-Instruct

# Start from existing SFT adapter (recommended)
python train_grpo.py --data scored_rollouts.jsonl --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter ../output

# Custom hyperparameters
python train_grpo.py --data scored_rollouts.jsonl --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --clip-epsilon 0.15 --kl-beta 0.05 --lr 5e-6 --max-steps 500

# Merge after training
python train_grpo.py --merge --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter ./rl_output --output-dir ./rl_merged
```

GRPO loss: clipped surrogate objective with KL penalty:
```
L = -E[ min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A) ] + β * KL(π_θ || π_ref)
```

Implementation details:
- Loads a frozen reference model for KL computation (same base model, no adapter)
- Per-sample: tokenizes prompt+completion, computes log-probs for completion tokens only, calculates probability ratio vs reference, applies clipped surrogate with advantage
- KL penalty: mean per-token KL divergence between policy and reference
- **KL cap safety**: If average KL exceeds `--kl-cap` (default 0.1), training stops early to prevent mode collapse
- Gradient clipping at norm 1.0
- Cosine LR schedule with 10% warmup
- Saves `training_meta.json` with all hyperparameters for reproducibility

Key parameters (all configurable via CLI flags):

| Parameter | Default | Flag | Rationale |
|---|---|---|---|
| Clip epsilon | 0.2 | `--clip-epsilon` | Standard PPO/GRPO clip range |
| KL penalty (beta) | 0.02 | `--kl-beta` | Prevent divergence from base model |
| KL cap | 0.1 | `--kl-cap` | Emergency stop if model diverges |
| Learning rate | 1e-5 | `--lr` | Lower than SFT to avoid catastrophic forgetting |
| LoRA rank | 8 | `--lora-r` | Match existing SFT configuration |
| Batch size | 2 | `--batch-size` | Fit in consumer GPU memory |
| Gradient accumulation | 8 | `--gradient-accumulation` | Effective batch = 16 |
| Max steps | 200 | `--max-steps` | Short RL rounds, frequent evaluation |
| Max sequence length | 2048 | `--max-seq-len` | Truncate long sequences |

**LoRA reuse**: Pass `--adapter ../output` to start from the SFT adapter. If the path doesn't exist, a new LoRA adapter is created.

#### 2.4 On-Policy Distillation (OPD) — optional hybrid mode

**Script**: `distill/rl/train_opd.py`

```bash
# Combined GRPO + OPD (default weights: both 1.0)
python train_opd.py --data scored_rollouts.jsonl --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter ../output

# OPD only (disable GRPO loss)
python train_opd.py --data scored_rollouts.jsonl --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --w-rl 0.0 --w-opd 1.0

# Adjust weights
python train_opd.py --data scored_rollouts.jsonl --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --w-rl 0.5 --w-opd 1.5

# Or via the pipeline orchestrator
./run_rl.sh --opd
```

OPD extracts pairs where `reference_completion` exists in the rollout data (i.e., the original episode had a cloud response). For each pair:
1. **Student completion** (from rollout): used for GRPO loss
2. **Teacher completion** (from cloud): used for OPD loss — negative log-likelihood of teacher tokens under the student model

The OPD loss gives per-token gradient: tokens where the student assigns low probability to the correct teacher token get stronger updates. This is strictly more informative than sequence-level SFT.

Combined loss: `L_total = W_RL * L_grpo + W_OPD * L_opd`

### Phase 3: Training Orchestration [DONE]

**Goal**: End-to-end pipeline script, similar to the existing `run_distill.sh`.

#### 3.1 Pipeline orchestrator

**Script**: `distill/rl/run_rl.sh`

```bash
# Full pipeline
./run_rl.sh

# Step-by-step execution
./run_rl.sh --score-only       # Export + score episodes
./run_rl.sh --rollout-only     # Generate rollouts
./run_rl.sh --train-only       # GRPO training (scored rollouts must exist)
./run_rl.sh --eval-only        # Evaluate RL vs SFT
./run_rl.sh --deploy-only      # Merge + deploy to Ollama
./run_rl.sh --opd              # Use combined GRPO + OPD training

# Custom configuration
LOUTER_DB=./my.db BASE_MODEL=Qwen/Qwen2.5-3B-Instruct ./run_rl.sh
JUDGE_PROVIDER=ollama JUDGE_MODEL=qwen2.5:7b ./run_rl.sh
INFERENCE_BACKEND=vllm ./run_rl.sh
MIN_EPISODES=500 ./run_rl.sh
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `LOUTER_DB` | `../../louter.db` | Path to SQLite database |
| `BASE_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model for RL |
| `ADAPTER_PATH` | `../output` | Starting LoRA adapter (reuse SFT weights) |
| `JUDGE_PROVIDER` | `anthropic` | Judge provider: `anthropic`, `openai`, `ollama` |
| `JUDGE_MODEL` | auto per provider | Judge model name |
| `INFERENCE_BACKEND` | `transformers` | Rollout backend: `transformers`, `ollama`, `vllm` |
| `OLLAMA_MODEL` | `louter-rl` | Ollama model name for deployment |
| `MIN_EPISODES` | `100` | Skip training if fewer episodes available |
| `LOUTER_RL_STRICT_EVAL` | `1` | Block deployment if eval fails (`0` to override) |

Pipeline steps (full run):

```
1. Export episodes        →  training_samples → rl_episodes + episodes.jsonl
2. Score tool calls       →  Structural rewards (fast, no API calls)
3. Score with judge       →  Judge model scores unscored local episodes
4. Generate rollouts      →  G=4 completions per prompt
5. Score rollouts         →  Judge + reference comparison → GRPO advantages
6. GRPO training          →  LoRA update with clipped surrogate + KL penalty
7. Evaluate               →  Compare RL vs SFT on held-out prompts
8. Merge + deploy         →  Merge LoRA → safetensors → Ollama
```

Output files:
- `distill/rl/output/episodes.jsonl` — exported episodes
- `distill/rl/output/rollouts.jsonl` — generated rollouts
- `distill/rl/output/scored_rollouts.jsonl` — scored rollouts with advantages
- `distill/rl/output/rl_adapter/` — trained LoRA adapter
- `distill/rl/output/eval_report.json` — evaluation results
- `distill/rl/rl_merged/` — final merged model + Ollama Modelfile

**Rollback**: Before deploying, the script copies the existing Ollama model to `louter-rl-prev` for easy rollback.

#### 3.2 Evaluation checkpoint

**Script**: `distill/rl/evaluate.py`

```bash
# Compare RL vs SFT (transformers backend)
python evaluate.py --episodes episodes.jsonl --rl-model ./rl_merged --sft-model ../merged_model

# Using Ollama
python evaluate.py --episodes episodes.jsonl --rl-model louter-rl --sft-model louter-distilled \
    --backend ollama

# View previous report
python evaluate.py --report output/eval_report.json
```

Before deploying, validates the RL model is actually better:
- Holds out 10% of episodes (max 50) as eval set with deterministic seed (42) for reproducibility
- Generates completions from both RL model and SFT baseline
- Scores both with the judge model
- Reports: win/loss/tie counts, mean scores, improvement delta
- **Exit code 0** if RL wins (deploy), **exit code 1** if not (skip deployment)
- Saves detailed report to `output/eval_report.json`

### Phase 4: Flexible Model Serving [DONE]

**Goal**: Support serving the RL-trained model through multiple backends, not just Ollama.

#### 4.1 vLLM serving (recommended for GPU)

**Script**: `distill/rl/serve_vllm.sh`

```bash
# Default: serve ./rl_merged on port 8000
./serve_vllm.sh

# Custom model and port
MODEL_PATH=./custom_model PORT=8080 ./serve_vllm.sh

# Adjust GPU memory and context length
GPU_UTIL=0.8 MAX_MODEL_LEN=8192 ./serve_vllm.sh
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `./rl_merged` | Path to merged model directory |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Listen port |
| `MAX_MODEL_LEN` | `4096` | Maximum context length |
| `GPU_UTIL` | `0.9` | GPU memory utilization fraction |

vLLM provides:
- 3-5x higher throughput than HuggingFace for batched inference
- OpenAI-compatible API (drop-in replacement for Ollama in Louter config)
- Continuous batching, PagedAttention
- Requires `pip install vllm` (GPU only, not available on Apple Silicon)

#### 4.2 Ollama serving (existing)

Already supported. The `run_rl.sh` deploy step automatically creates the Ollama model from the merged safetensors:

```bash
# Automatic (done by run_rl.sh step 5)
ollama create louter-rl -f ./rl_merged/Modelfile

# Manual rollback
ollama cp louter-rl-prev louter-rl
```

Update `louter.toml`:
```toml
[hybrid]
local_model = "louter-rl"
```

#### 4.3 HuggingFace Transformers serving

**Script**: `distill/rl/serve_hf.py`

Lightweight OpenAI-compatible server using FastAPI + Transformers. Works everywhere including Apple Silicon (MPS) and CPU.

```bash
# Serve merged model
python serve_hf.py --model ./rl_merged --port 8000

# Serve base model with LoRA adapter (no merge needed)
python serve_hf.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter ./output/rl_adapter --port 8000

# Custom model name in API responses
python serve_hf.py --model ./rl_merged --model-name my-model --port 8000
```

Endpoints:
- `POST /v1/chat/completions` — streaming and non-streaming
- `GET /v1/models` — list available models
- `GET /health` — health check

Auto-detects hardware: CUDA (bfloat16) → MPS (float16) → CPU (float32). Supports LoRA adapter loading via `--adapter` flag without merging.

#### 4.4 Louter provider configuration

All three serving options expose an OpenAI-compatible API, so Louter can use any of them:

```toml
[hybrid]
local_provider = "ollama"      # or any OpenAI-compatible provider
local_endpoint = "http://localhost:8000/v1"
local_model = "louter-rl"
```

### Phase 5: Continuous RL Loop [DONE]

**Goal**: Automate the RL improvement cycle so the model keeps getting better over time.

#### 5.1 Scheduled RL rounds

The pipeline supports a `MIN_EPISODES` threshold to skip training when insufficient new data exists:

```bash
# Default: require at least 100 new episodes
./run_rl.sh

# Higher threshold for production
MIN_EPISODES=500 ./run_rl.sh

# Run via cron (e.g., weekly)
# 0 3 * * 0  cd /path/to/louter && ./distill/rl/run_rl.sh >> /var/log/louter-rl.log 2>&1
```

The `export_episodes.py` script tracks which samples have already been exported (via `is_used_for_training` flag), so each run only processes new data. When episode count is below `MIN_EPISODES`, the pipeline exits cleanly without error.

#### 5.2 Progressive task expansion

As RL improves the model, gradually route more task types locally:

```
Round 1: general only          (lowest complexity)
Round 2: general + translation (pattern-heavy tasks)
Round 3: + code                (structured output)
Round 4: + tool_call           (most complex)
```

Track success rates per task type after each RL round. Only expand when `min_local_success_rate` is met for current types. Use `export_episodes.py --stats` to review per-source reward distributions.

#### 5.3 Safety rails

All three safety mechanisms are implemented:

- **KL divergence cap**: `train_grpo.py` and `train_opd.py` monitor per-step KL divergence. If average KL exceeds `--kl-cap` (default 0.1), training stops early with a warning. Prevents mode collapse.
- **Evaluation gate**: `evaluate.py` compares RL vs SFT on held-out prompts before deployment. Exit code 1 blocks deployment if RL doesn't improve. `run_rl.sh` enforces this by default (`LOUTER_RL_STRICT_EVAL=1`). Set `LOUTER_RL_STRICT_EVAL=0` to override during development.
- **Rollback mechanism**: Before deploying to Ollama, `run_rl.sh` copies the existing model to `louter-rl-prev`:
  ```bash
  ollama cp louter-rl louter-rl-prev    # automatic backup
  ollama cp louter-rl-prev louter-rl    # manual rollback
  ```

## File Structure

```
distill/
├── export.py                    # (existing) Export training samples
├── compress.py                  # (existing) Compress training data
├── train.py                     # (existing) SFT LoRA training
├── run_distill.sh               # (existing) SFT pipeline
├── requirements.txt             # (existing) SFT dependencies
└── rl/
    ├── requirements.txt         # RL-specific dependencies
    ├── run_rl.sh                # Main RL pipeline orchestrator
    ├── export_episodes.py       # Convert training_samples → rl_episodes
    ├── score_with_judge.py      # Judge-based reward scoring
    ├── score_tool_calls.py      # Structural rewards for tool-call tasks
    ├── generate_rollouts.py     # Generate G completions per prompt
    ├── reward_rollouts.py       # Score rollouts + compute GRPO advantages
    ├── train_grpo.py            # GRPO training with LoRA
    ├── train_opd.py             # On-Policy Distillation (optional)
    ├── evaluate.py              # Compare RL model vs SFT baseline
    ├── serve_vllm.sh            # vLLM serving script
    ├── serve_hf.py              # HuggingFace Transformers server
    └── output/                  # Training outputs, checkpoints, eval reports
```

## Dependencies

**`distill/rl/requirements.txt`**:

```
# Core RL training
torch>=2.1.0
transformers>=4.40.0
peft>=0.10.0
datasets>=2.18.0
accelerate>=0.28.0
bitsandbytes>=0.43.0
trl>=0.12.0

# Reward scoring
rouge-score>=0.1.2              # ROUGE-L for reference comparison
anthropic>=0.40.0               # Judge model API (optional)
openai>=1.50.0                  # Alternative judge (optional)

# Serving
fastapi>=0.115.0                # HF Transformers server (serve_hf.py)
uvicorn>=0.32.0

# General
numpy>=1.26.0
requests>=2.31.0                # Ollama/vLLM REST API calls

# Optional (install separately)
# vllm>=0.6.0                   # GPU inference — requires CUDA, install manually
# ollama via REST API — no pip dependency
```

## Hardware Requirements

| Setup | Hardware | Notes |
|---|---|---|
| **Minimal (Apple Silicon)** | M1/M2/M3 with 16GB+ | MPS for training, HF Transformers for serving. G=2, batch=1. Slow but works. |
| **Recommended (single GPU)** | 1x RTX 3090/4090 (24GB) | vLLM for rollouts, GRPO training with LoRA. G=4, batch=2. ~1 hour per RL round (500 episodes). |
| **Fast (multi-GPU)** | 2-4x GPUs | vLLM on GPU 0 for rollouts, training on remaining GPUs. G=8, batch=4. |
| **Cloud** | Any cloud GPU (A100, H100) | Same scripts, just faster. |

For the 1.5B base model, a single 24GB GPU comfortably handles the full pipeline. The 3B model fits with gradient checkpointing.

## Implementation Order

All phases are implemented. The order below reflects how they were built:

1. ~~**Phase 1.1**: `rl_episodes` schema + `export_episodes.py`~~ ✓
2. ~~**Phase 1.2**: `score_with_judge.py` + `score_tool_calls.py`~~ ✓
3. ~~**Phase 2.1-2.2**: `generate_rollouts.py` + `reward_rollouts.py`~~ ✓
4. ~~**Phase 2.3**: `train_grpo.py`~~ ✓
5. ~~**Phase 3.1**: `run_rl.sh`~~ ✓
6. ~~**Phase 3.2**: `evaluate.py`~~ ✓
7. ~~**Phase 4**: `serve_vllm.sh` + `serve_hf.py`~~ ✓
8. ~~**Phase 2.4**: `train_opd.py`~~ ✓
9. ~~**Phase 5**: Continuous loop (MIN_EPISODES, rollback, KL cap)~~ ✓

## Key Differences from OpenClaw RL

| Aspect | OpenClaw RL | Louter RL |
|---|---|---|
| **Scale** | 8+ GPUs, Megatron+SGLang | Single GPU / Apple Silicon, HF Transformers / vLLM |
| **Async loop** | 4 fully async components | Sequential pipeline with optional background rollouts |
| **Reward model** | Dedicated PRM with majority voting | Cloud model as judge + implicit retry feedback |
| **Data source** | Live conversation interception | Existing Louter proxy data collection |
| **Training framework** | Slime (Megatron+SGLang) | trl GRPOTrainer + LoRA (PEFT) |
| **Model size** | 4B-32B | 1.5B-8B (consumer hardware focus) |
| **Serving** | SGLang only | Ollama, vLLM, HF Transformers |

The core RL algorithm (GRPO) is the same. We adapt the infrastructure for single-machine, consumer-hardware deployment while reusing Louter's existing data pipeline.
