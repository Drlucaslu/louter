<p align="center">
  <h1 align="center">Louter</h1>
  <p align="center">Local LLM gateway with hybrid inference and distillation.</p>
</p>

<p align="center">
  <a href="#install">Install</a> &bull;
  <a href="#agent-setup">Agent Setup</a> &bull;
  <a href="#features">Features</a> &bull;
  <a href="#hybrid-inference">Hybrid Inference</a> &bull;
  <a href="#distillation">Distillation</a> &bull;
  <a href="#reinforcement-learning">Reinforcement Learning</a> &bull;
  <a href="#sft-vs-rl">SFT vs RL</a> &bull;
  <a href="#routing">Routing</a> &bull;
  <a href="#%E4%B8%AD%E6%96%87%E8%AF%B4%E6%98%8E">中文说明</a>
</p>

---

Run one local gateway, route every agent to the right model — cloud or local.

```
  Claude Code ──┐                         ┌── OpenAI
    OpenClaw ───┤                         ├── Anthropic
       Cline ───┼──→  Louter (:6188)  ───┼── DeepSeek
       aider ───┤     localhost           ├── Ollama (local, distilled)
    your app ───┘                         └── Qwen / Groq / Azure / ...
                        │
                  ┌─────▼─────┐
                  │  Hybrid   │  Session-aware routing
                  │  Router   │  Local-first + cloud fallback
                  │           │  Auto-escalation on failure
                  └─────┬─────┘
                        │
                  ┌─────▼─────┐
                  │ Distill   │  SFT: Compress → Train → Deploy
                  │ + RL      │  RL:  Score → Rollout → GRPO → Deploy
                  └───────────┘  Data flywheel
```

Louter is a single-binary LLM API gateway that sits between your AI agents and LLM providers. Beyond routing, it implements **hybrid inference** — automatically routing simple requests to a fast local model and complex ones to the cloud — and a **distillation pipeline** that continuously improves the local model from cloud responses.

**No Docker. No Redis. No Postgres.** Just one binary with embedded SQLite and a Web UI.

## Install

**One-line install** (requires Rust and Node.js):

```bash
curl -fsSL https://raw.githubusercontent.com/Drlucaslu/louter/main/install.sh | bash
```

**Or build manually:**

```bash
git clone https://github.com/Drlucaslu/louter.git && cd louter
cd web && npm install && npm run build && cd ..
cargo build --release
./target/release/louter
```

Then open **http://localhost:6188** — add your providers and create an API key (`lot_xxxx`).

## Agent Setup

Once Louter is running and you've added providers via the Web UI, configure your agents:

### Claude Code

```bash
export OPENAI_API_BASE=http://localhost:6188/v1
export OPENAI_API_KEY=lot_your_key_here

claude --model gpt-4o          # → routes to OpenAI
claude --model deepseek-chat   # → routes to DeepSeek
```

### OpenClaw

```yaml
api_base: http://localhost:6188/v1
api_key: lot_your_key_here
```

### Cline (VS Code)

1. Open Cline settings → **API Provider** → "OpenAI Compatible"
2. **Base URL**: `http://localhost:6188/v1`
3. **API Key**: your `lot_xxxx` key

### aider

```bash
export OPENAI_API_BASE=http://localhost:6188/v1
export OPENAI_API_KEY=lot_your_key_here
aider --model deepseek-chat
```

### Any OpenAI-compatible agent

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:6188/v1",
    api_key="lot_your_key_here",
)

response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",  # routes to Anthropic
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Features

- **One endpoint for all providers** — OpenAI, Anthropic, Azure, DeepSeek, Ollama, and any OpenAI-compatible API
- **Hybrid inference** — Route simple requests to a fast local model, complex ones to the cloud. Session-aware: same model throughout a conversation, with auto-escalation on failure
- **Distillation pipeline** — Automatically collect cloud responses as training data, compress, fine-tune a local model via LoRA, and serve via HuggingFace Transformers or vLLM
- **Reinforcement learning** — GRPO-based RL training (inspired by [OpenClaw RL](https://github.com/Gen-Verse/OpenClaw-RL)) learns from both successes and failures, with judge-based rewards and on-policy distillation
- **Smart routing** — `claude-*` → Anthropic, `gpt-*` → OpenAI, `deepseek-*` → DeepSeek. Custom glob rules with priorities. `model: "auto"` for content-based routing
- **Tool call normalizer** — Parse Hermes/Qwen/ReAct/JSON tool call formats from local models and convert to standard OpenAI format
- **Implicit feedback** — Detect agent retries to automatically label training samples as success/failure
- **Runtime tuning** — Adjust routing thresholds via API without restarting. Relax limits as the distilled model improves
- **Full API coverage** — Chat, streaming, models, images, embeddings, audio — all OpenAI-compatible
- **Native format conversion** — Anthropic tool use, Azure deployment URLs, provider-specific auth — handled transparently
- **Built-in Web UI** — Manage providers, keys, routing rules, distillation stats, and usage analytics
- **Single binary** — Rust (Axum + Tokio) / SQLite / React + Tailwind / embedded via `rust-embed`

## Hybrid Inference

Louter can route requests between a local model (via Ollama) and a cloud model, based on task type, context size, and historical success rates.

```toml
# louter.toml
[hybrid]
enabled = true
local_provider = "ollama"
local_model = "qwen2.5:1.5b"           # Fast local model (~90 tok/s)
cloud_provider = "anthropic"
cloud_model = "claude-sonnet-4-20250514" # Powerful cloud fallback
min_local_success_rate = 0.7             # Route to cloud if local success < 70%
min_samples = 20                         # Min data before trusting success rate
fallback_enabled = true                  # Try local first, fall back to cloud
local_task_types = ["general"]           # Only route simple conversations locally
max_local_context_tokens = 2000          # Large contexts go to cloud
max_local_latency_ms = 30000             # Slow responses go to cloud
```

### How it works

1. **New conversation** — Hybrid router decides local vs cloud based on task type, context size, and historical success rate
2. **Continuation** — Same model is used for all turns in the conversation (session-aware routing)
3. **Local failure** — If the local model returns a bad response, the session is escalated to cloud for all remaining turns
4. **Data collection** — Cloud responses are automatically saved as training samples for distillation

### Runtime adjustment

As your distilled model improves, relax the limits without restarting:

```bash
# After first distillation round — try tool_call locally
curl -X PUT http://localhost:6188/api/admin/distill/config \
  -H "Content-Type: application/json" \
  -d '{"local_task_types": ["general", "tool_call"], "max_local_context_tokens": 4000}'

# After more training — route everything locally
curl -X PUT http://localhost:6188/api/admin/distill/config \
  -d '{"local_task_types": ["general", "tool_call", "code"], "max_local_context_tokens": 8000}'
```

## Distillation

Louter automatically collects cloud model responses as training data, then provides tools to compress, fine-tune, and deploy a local model — creating a **data flywheel** where the local model gets better over time.

```
Use Louter normally
    ↓
Cloud responses saved to training_samples table
    ↓
Export & compress: ./distill/run_distill.sh --export-only
    ↓
Fine-tune local model (LoRA on Qwen2.5-1.5B)
    ↓
Serve via HF / vLLM → local model handles more requests
    ↓
Fewer cloud calls → lower cost → repeat
```

### Prerequisites

**Python 3.10+** and a virtual environment:

```bash
cd distill
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

This installs PyTorch, Transformers, PEFT, TRL, and other dependencies. The base model (**Qwen2.5-1.5B-Instruct**) is downloaded automatically from [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) the first time you run training — no manual download needed. It caches to `~/.cache/huggingface/hub/` (~3 GB).

> To use a different base model (e.g. `Qwen/Qwen2.5-3B-Instruct` or `meta-llama/Llama-3.2-3B-Instruct`), set the `BASE_MODEL` environment variable or pass `--base-model` to `train.py`.

**Hardware requirements:**

| Setup | What to expect |
|-------|----------------|
| NVIDIA GPU (8 GB+ VRAM) | Fastest. Uses bfloat16 automatically |
| Apple Silicon (M1/M2/M3/M4) | Good. Uses MPS with float16. 1.5B model trains fine on 16 GB RAM |
| CPU only | Slow but works. Uses float32. Expect hours for 1000+ samples |

### Step 1: Collect training data

Enable data collection in `louter.toml` (enabled by default):

```toml
[hybrid]
enabled = true
local_provider = "ollama"
local_model = "qwen2.5:1.5b"
cloud_provider = "anthropic"
cloud_model = "claude-sonnet-4-20250514"
fallback_enabled = true

[distillation]
collect_training_data = true        # default: true
max_samples = 100000                # prune oldest when exceeded
only_successful = true              # only collect status-200 responses
```

Use Louter normally. Every cloud response is automatically saved to the `training_samples` table in SQLite. Check progress:

```bash
# Via API
curl http://localhost:6188/api/admin/distill/stats

# Or via the Web UI at http://localhost:6188 → Distillation dashboard
```

Aim for **1000+ samples** before your first training run (more is better).

### Step 2: Export and compress

```bash
cd distill
source venv/bin/activate

# Export from SQLite + compress in one step
./run_distill.sh --export-only

# Or manually:
python export.py --db ../louter.db --output output/training_data.jsonl --mark-exported --format openai
python compress.py output/training_data.jsonl -o output/training_data_compressed.jsonl --stats
```

Output files:
- `distill/output/training_data.jsonl` — raw exported samples
- `distill/output/training_data_compressed.jsonl` — deduplicated and truncated (30-50% smaller)

### Step 3: Fine-tune with LoRA

```bash
# Full pipeline (export + compress + train + merge + deploy)
./run_distill.sh

# Or train only (data must already be exported)
./run_distill.sh --train-only

# Or run train.py directly for more control
python train.py \
    --data output/training_data_compressed.jsonl \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --output-dir ./output \
    --epochs 5 --batch-size 8 --lr 5e-4

# Resume from a checkpoint
python train.py --data output/training_data_compressed.jsonl \
    --resume-from ./output/checkpoint-500
```

**Choosing a different base model:** You can fine-tune any HuggingFace causal LM — pass `--base-model` to `train.py` or set `BASE_MODEL` for `run_distill.sh`:

```bash
# Larger Qwen for better quality (needs more VRAM/RAM)
python train.py --data output/training_data_compressed.jsonl --base-model Qwen/Qwen2.5-3B-Instruct

# Llama-based model
BASE_MODEL=meta-llama/Llama-3.2-3B-Instruct ./run_distill.sh
```

Any model with standard attention layers works (Qwen, Llama, Mistral, Gemma, Phi, etc.). The model is downloaded automatically from Hugging Face on first use.

**Output files after training:**
- `distill/output/` — LoRA adapter weights (`adapter_model.safetensors`, `adapter_config.json`)
- Checkpoints at `distill/output/checkpoint-*/`

### Step 4: Merge and serve the model

```bash
# Merge adapter (as part of full pipeline)
./run_distill.sh --deploy-only

# Or manually:
python train.py --merge \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter-path ./output \
    --output-dir ./merged_model
```

**Output files after merge:**
- `distill/merged_model/` — full merged model (safetensors + tokenizer + config)

**Serve the merged model** (pick one):

```bash
# HuggingFace Transformers — works on CUDA, Apple Silicon, CPU
python rl/serve_hf.py --model ./merged_model --port 8000

# vLLM — highest throughput, requires CUDA GPU
MODEL_PATH=./merged_model ./rl/serve_vllm.sh

# Ollama — alternative if already installed
ollama create louter-distilled -f merged_model/Modelfile
```

**Verify it works:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "louter-distilled", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### Step 5: Configure Louter to use the distilled model

Update `louter.toml` with the distilled model:

```toml
[hybrid]
local_endpoint = "http://localhost:8000/v1"  # HF or vLLM
local_model = "louter-distilled"             # your distilled model name
cloud_provider = "anthropic"
cloud_model = "claude-sonnet-4-20250514"
min_local_success_rate = 0.7
min_samples = 20
fallback_enabled = true                  # try local first, fall back to cloud on failure
local_task_types = ["general"]           # start conservative, expand later
max_local_context_tokens = 2000
max_local_latency_ms = 30000
```

Then set the API key's routing mode to **hybrid** in the Web UI (Keys page) to enable local/cloud routing for that key.

#### Gradually expand local routing

As the distilled model proves itself, relax limits at runtime without restarting:

```bash
# After first distillation — also route tool_call tasks locally
curl -X PUT http://localhost:6188/api/admin/distill/config \
  -H "Content-Type: application/json" \
  -d '{"local_task_types": ["general", "tool_call"], "max_local_context_tokens": 4000}'

# After more training rounds — route everything locally
curl -X PUT http://localhost:6188/api/admin/distill/config \
  -d '{"local_task_types": ["general", "tool_call", "code"], "max_local_context_tokens": 8000}'
```

### Pipeline components

| Tool | Purpose |
|------|---------|
| `distill/export.py` | Export training samples from SQLite as JSONL (OpenAI / ShareGPT format) |
| `distill/compress.py` | Compress training data: dedup system prompts, truncate tool results, strip verbose sections |
| `distill/train.py` | LoRA fine-tuning on Qwen2.5-1.5B (supports CUDA / MPS / CPU) |
| `distill/run_distill.sh` | End-to-end: export → compress → train → merge → serve |

### Training parameters (optimized for 1.5B model)

| Parameter | Value | Reason |
|-----------|-------|--------|
| Base model | Qwen2.5-1.5B-Instruct | Fast inference (~90 tok/s on Apple Silicon) |
| LoRA rank | 8 | Small model, low rank sufficient |
| LoRA alpha | 16 | 2x rank |
| Batch size | 8 | Small model fits larger batches |
| Learning rate | 5e-4 | Small models learn faster |
| Epochs | 5 | More passes for limited data |

### Monitoring

The Web UI dashboard (`http://localhost:6188/distill`) shows:

- Training sample counts by task type
- Local vs cloud routing ratio
- Local success rate per task type
- Session statistics (active, escalated)
- Current hybrid config + dynamic overrides

### API endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/admin/distill/stats` | Training sample statistics |
| `GET /api/admin/distill/routing` | Routing history and success rates |
| `POST /api/admin/distill/export` | Export training samples as JSON |
| `GET /api/admin/distill/config` | Current hybrid + distillation config |
| `PUT /api/admin/distill/config` | Update routing thresholds at runtime |

## Reinforcement Learning

While distillation (SFT) teaches the local model to *imitate* good cloud responses, reinforcement learning teaches it to *avoid bad responses* and *maximize reward*. Louter's RL pipeline is inspired by [OpenClaw RL](https://github.com/Gen-Verse/OpenClaw-RL) and adapted for single-machine, consumer-hardware deployment.

See [PLAN_RL.md](PLAN_RL.md) for the full implementation plan.

### How it works

```
Louter collects training samples (same as SFT)
    ↓
Export episodes with reward signals:
  +1.0  successful response, no retry
  -1.0  retry detected (agent re-sent similar request)
  -0.5  local failed, cloud took over
    ↓
Score unscored episodes with a judge model (cloud or local)
    ↓
Generate G=4 completions per prompt from current local model
    ↓
Score each completion → compute group-relative advantages (GRPO)
    ↓
Train with clipped surrogate loss + KL penalty (LoRA)
    ↓
Evaluate: only deploy if RL model beats SFT baseline
    ↓
Merge + deploy via Ollama / vLLM / HF Transformers
```

**GRPO (Group Relative Policy Optimization)** is the core algorithm. For each training prompt, the model generates multiple completions. These are scored and compared *within the group* — completions that score above the group mean get positive advantage, below get negative. This relative comparison is what lets the model learn which response *style* works better, without needing a separate critic network.

### Prerequisites

Same Python environment as SFT, plus RL-specific dependencies. You can reuse the existing SFT venv or create a separate one:

```bash
cd distill
source venv/bin/activate         # reuse existing SFT venv
pip install -r rl/requirements.txt
```

### Pipeline commands

```bash
# Full RL pipeline (export → score → rollout → train → evaluate → deploy)
cd distill/rl
./run_rl.sh

# Or step by step:
./run_rl.sh --score-only       # Score episodes with judge model
./run_rl.sh --rollout-only     # Generate rollouts from current model
./run_rl.sh --train-only       # Run GRPO training
./run_rl.sh --eval-only        # Run evaluation only
./run_rl.sh --deploy-only      # Merge and deploy
./run_rl.sh --opd              # Use combined GRPO + OPD training
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `LOUTER_DB` | `../../louter.db` | Path to SQLite database |
| `BASE_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model for RL |
| `ADAPTER_PATH` | `../output` | Starting LoRA adapter (reuse SFT weights) |
| `JUDGE_PROVIDER` | `anthropic` | Judge model provider: `anthropic`, `openai`, `ollama` |
| `JUDGE_MODEL` | auto per provider | Model used to score responses |
| `INFERENCE_BACKEND` | `transformers` | Rollout backend: `vllm`, `ollama`, `transformers` |
| `OLLAMA_MODEL` | `louter-rl` | Name for the deployed model |
| `MIN_EPISODES` | `100` | Minimum episodes required before training starts |

### RL training parameters

| Parameter | Value | Rationale |
|---|---|---|
| Group size (G) | 4 | Balance between signal quality and compute |
| Clip epsilon | 0.2 | Standard PPO/GRPO clip range |
| KL penalty (beta) | 0.02 | Prevent divergence from base model |
| Learning rate | 1e-5 | Lower than SFT to avoid catastrophic forgetting |
| LoRA rank | 8 | Match existing SFT configuration |
| Batch size | 2 | Each sample has G completions, so effective size is larger |
| Gradient accumulation | 8 | Effective batch = 16 |
| Max steps per round | 200 | Short RL rounds, frequent evaluation |

### Serving the RL model

The RL pipeline supports multiple serving backends. All expose an OpenAI-compatible API.

**HuggingFace Transformers** (recommended — works on CUDA, Apple Silicon, CPU):

```bash
python serve_hf.py --model ./rl_merged --port 8000
```

**vLLM** (highest throughput on GPU):

```bash
./serve_vllm.sh    # starts OpenAI-compatible API on :8000
```

**Ollama** (alternative, if already installed):

```bash
ollama create louter-rl -f rl_merged/Modelfile
```

Point Louter at whichever backend you choose:

```toml
[hybrid]
local_endpoint = "http://localhost:8000/v1"  # for HF or vLLM
local_model = "louter-rl"
```

### On-Policy Distillation (OPD)

When the local model fails and the cloud model succeeds on the same prompt, OPD provides token-level gradient signal — stronger than both SFT (sequence-level) and GRPO (scalar reward). The cloud response acts as a teacher: tokens where the local model assigns low probability to the correct cloud token get stronger gradient updates.

Enable with `--opd` flag: `./run_rl.sh --opd`

OPD is combined with GRPO: `L_total = W_RL * L_grpo + W_OPD * L_opd`. Weights are configurable via `--w-rl` and `--w-opd` when calling `train_opd.py` directly.

### Safety rails

- **KL divergence cap** — Training stops early if the model drifts too far from the reference (threshold: 0.1)
- **Reward hacking detection** — If judge scores improve but actual retry rates don't, the reward model may be exploited
- **Automatic rollback** — Previous model kept as `louter-rl-prev` for fallback
- **Evaluation gate** — Model is only deployed if it beats the SFT baseline on held-out prompts (`LOUTER_RL_STRICT_EVAL=1` by default; set to `0` to override)

### RL pipeline components

| Tool | Purpose |
|------|---------|
| `distill/rl/export_episodes.py` | Convert training_samples → RL episodes with reward signals |
| `distill/rl/score_with_judge.py` | Score local model responses using a judge model |
| `distill/rl/score_tool_calls.py` | Structural rewards for tool-call tasks (valid JSON, correct schema) |
| `distill/rl/generate_rollouts.py` | Generate G completions per prompt via vLLM/Ollama/Transformers |
| `distill/rl/reward_rollouts.py` | Score rollouts + compute GRPO group-relative advantages |
| `distill/rl/train_grpo.py` | GRPO training with LoRA (clipped surrogate + KL penalty) |
| `distill/rl/train_opd.py` | On-Policy Distillation from cloud→local (optional) |
| `distill/rl/evaluate.py` | Compare RL model vs SFT baseline before deployment |
| `distill/rl/serve_vllm.sh` | Serve merged model via vLLM |
| `distill/rl/serve_hf.py` | Serve merged model via HuggingFace Transformers |
| `distill/rl/run_rl.sh` | End-to-end RL pipeline orchestrator |

## SFT vs RL

Louter offers two approaches to improve the local model. They are complementary — SFT provides the foundation, RL refines it.

### What each approach does

**SFT (Supervised Fine-Tuning)** — the existing `distill/` pipeline:

- Filters to `is_successful=1` samples only — failed responses are thrown away
- Trains the model to imitate cloud responses token-by-token
- One response per prompt — no comparison signal
- The model learns "what a good answer looks like" but not "what makes one answer better than another"

**GRPO RL (Group Relative Policy Optimization)** — the `distill/rl/` pipeline:

- Uses *both* successful and failed samples as reward signal
- Generates **multiple completions** per prompt, scores them, and trains on the *relative differences*
- A retry-detected response becomes a -1.0 reward — the model actively learns to avoid that behavior
- Judge model scores responses on a spectrum, not just pass/fail
- KL penalty prevents the model from diverging too far from the base

### Concrete example

User asks for a tool call. With **SFT**, Louter exports the cloud model's correct response and trains the local model to copy it.

With **GRPO RL**, the local model generates 4 different responses for that prompt:

| Completion | Score | Advantage |
|---|---|---|
| Correct JSON, right function, valid args | +0.8 | +1.2 (above group mean) |
| Right function, minor arg typo | +0.1 | -0.2 |
| Right function, malformed JSON | -0.2 | -0.5 |
| Hallucinated nonexistent function | -1.0 | -1.8 (far below group mean) |

GRPO pushes the model toward completion 1 and away from completion 4. SFT would never see those 3 failures — it only trains on the cloud's correct response.

### When to use which

| Scenario | Recommendation |
|---|---|
| First training round, < 1000 samples | SFT only — need a solid baseline |
| 1000+ samples, model already SFT-trained | Add RL on top of SFT adapter |
| Local model keeps making the same mistakes | RL — it explicitly penalizes failure patterns |
| Plenty of cloud responses, few local failures | SFT — most signal is positive anyway |
| Tool-calling quality needs improvement | RL — structural rewards catch format errors SFT misses |

### Recommended workflow

```
1. Collect 1000+ samples via normal Louter usage
2. Run SFT:  ./distill/run_distill.sh
3. Deploy SFT model, use it for a while, collect more data
4. Run RL:   ./distill/rl/run_rl.sh    (starts from SFT adapter)
5. Deploy RL model → fewer failures → better data → repeat
```

The RL pipeline reuses the SFT adapter as its starting point (`ADAPTER_PATH`), so you never lose the SFT foundation. Each RL round further refines what SFT learned.

## Routing

Three-tier routing — no configuration needed for common providers:

| Model name | Routes to | How |
|---|---|---|
| `claude-*` | Anthropic | Built-in prefix match |
| `gpt-*`, `o1-*`, `o3-*`, `o4-*` | OpenAI | Built-in prefix match |
| `deepseek-*` | DeepSeek | Built-in prefix match |
| `llama*`, `mistral*`, `gemma*` | Ollama | Built-in prefix match |
| `qwen-*` | Qwen (custom) | Provider name match |

Override with per-key rules in the Web UI:

| Pattern | Provider | Priority |
|---|---|---|
| `gpt-4o*` | Azure OpenAI | 10 |
| `gpt-*` | OpenAI | 1 |
| `*` | DeepSeek | 0 |

### Smart Routing (`model: "auto"`)

Send `model: "auto"` and Louter classifies your message and routes to the best provider. Zero latency overhead.

```toml
[smart_routing]
code = "anthropic/claude-sonnet-4-20250514"
math = "deepseek/deepseek-chat"
translation = "openai/gpt-4o"
general = "openai/gpt-4o"
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Louter (single binary)              │
│                                                     │
│  ┌──────────┐  ┌──────────────────────────────┐     │
│  │  Web UI  │  │  /v1/* API Gateway           │     │
│  │  React   │  │  (OpenAI-compatible)         │     │
│  └──────────┘  └──────────────┬───────────────┘     │
│                               │                     │
│                    ┌──────────▼──────────┐           │
│                    │   Session Router    │           │
│                    │   (per-conversation │           │
│                    │    model affinity)  │           │
│                    └──────────┬──────────┘           │
│                               │                     │
│                    ┌──────────▼──────────┐           │
│                    │   Hybrid Router     │           │
│                    │   task type / ctx   │           │
│                    │   size / success    │           │
│                    │   rate filtering    │           │
│                    └────┬──────────┬─────┘           │
│                         │          │                │
│              ┌──────────▼┐   ┌────▼──────────┐      │
│              │   Local   │   │    Cloud      │      │
│              │  (Ollama) │   │  (Anthropic,  │      │
│              │  + Tool   │   │   OpenAI...)  │      │
│              │  Call     │   │              │      │
│              │  Normal.  │   │  → training  │      │
│              └───────────┘   │    samples   │      │
│                              └──────────────┘      │
│                                                     │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐     │
│  │  SQLite  │  │ Feedback  │  │  Dynamic     │     │
│  │  (keys,  │  │ Tracker   │  │  Config      │     │
│  │  rules,  │  │ (retry    │  │  (runtime    │     │
│  │  usage,  │  │  detect)  │  │   tunable)   │     │
│  │  samples)│  └───────────┘  └──────────────┘     │
│  └──────────┘                                       │
└─────────────────────────────────────────────────────┘
```

## Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | Chat (streaming + non-streaming) |
| `GET /v1/models` | List all models from all providers |
| `POST /v1/images/generations` | Text-to-image |
| `POST /v1/embeddings` | Text embeddings |
| `POST /v1/audio/speech` | Text-to-speech |
| `POST /v1/audio/transcriptions` | Speech-to-text |

## License

MIT

---

# 中文说明

## Louter — 本地 Agent 的 LLM 统一网关 + 混合推理 + 蒸馏

在本地运行一个网关，让你所有的 AI Agent 智能路由到最合适的模型 — 本地或云端。

```
  Claude Code ──┐                         ┌── OpenAI
    OpenClaw ───┤                         ├── Anthropic
       Cline ───┼──→  Louter (:6188)  ───┼── DeepSeek
       aider ───┤     localhost           ├── Ollama（本地蒸馏模型）
      你的应用 ──┘                         └── 通义千问 / Groq / Azure / ...
```

Louter 不仅是一个 API 网关，更是一个**混合推理 + 蒸馏**系统：

- **混合推理 (Hybrid Inference)** — 简单请求走快速本地模型，复杂请求走云端大模型，同一会话内保持同一模型不切换
- **蒸馏飞轮 (Distillation)** — 自动收集云端响应作为训练数据，压缩后微调本地模型，本地模型越来越强，云端调用越来越少

**无需 Docker。无需 Redis。无需 Postgres。** 单一二进制文件，内嵌 SQLite 和 Web 管理界面。

## 安装

```bash
git clone https://github.com/Drlucaslu/louter.git && cd louter
cd web && npm install && npm run build && cd ..
cargo build --release
./target/release/louter
```

打开 **http://localhost:6188**，添加供应商并创建 API Key。

## 混合推理

```toml
# louter.toml
[hybrid]
enabled = true
local_provider = "ollama"
local_model = "qwen2.5:1.5b"            # 快速本地模型（~90 tok/s）
cloud_provider = "anthropic"
cloud_model = "claude-sonnet-4-20250514"  # 强大的云端后备
min_local_success_rate = 0.7              # 本地成功率 < 70% 就走云端
fallback_enabled = true                   # 先试本地，失败转云端
local_task_types = ["general"]            # 只让简单对话走本地
max_local_context_tokens = 2000           # 大上下文走云端
```

### 工作原理

1. **新会话** — 根据任务类型、上下文大小、历史成功率决定走本地还是云端
2. **会话延续** — 同一会话始终使用同一模型（Session 级路由）
3. **本地失败** — 自动升级到云端，该会话后续全部走云端
4. **数据收集** — 云端响应自动保存为蒸馏训练数据

### 运行时调整

随着蒸馏模型改善，逐步放宽限制，无需重启：

```bash
# 蒸馏后 → 放开 tool_call
curl -X PUT http://localhost:6188/api/admin/distill/config \
  -d '{"local_task_types": ["general", "tool_call"], "max_local_context_tokens": 4000}'
```

## 蒸馏流水线

### 环境准备

```bash
cd distill
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

基础模型（**Qwen2.5-1.5B-Instruct**）在首次训练时自动从 [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 下载，缓存在 `~/.cache/huggingface/hub/`（约 3 GB），无需手动下载。

### 完整流程

```bash
# 1. 正常使用 Louter，云端响应自动收集到 SQLite

# 2. 积累 1000+ 样本后，一键蒸馏
cd distill && source venv/bin/activate
./run_distill.sh   # 导出 → 压缩 → 训练 → 合并 → 部署

# 3. 或分步执行：
./run_distill.sh --export-only   # 导出 + 压缩训练数据
./run_distill.sh --train-only    # 训练 LoRA 适配器
./run_distill.sh --deploy-only   # 合并 + 部署
```

### 输出文件

| 路径 | 内容 |
|------|------|
| `distill/output/` | LoRA 适配器权重（`adapter_model.safetensors`） |
| `distill/merged_model/` | 合并后的完整模型（safetensors + tokenizer） |

### 服务模型

```bash
# HuggingFace Transformers（推荐 — 支持 CUDA、Apple Silicon、CPU）
python rl/serve_hf.py --model ./merged_model --port 8000

# vLLM（GPU 上吞吐最高）
MODEL_PATH=./merged_model ./rl/serve_vllm.sh

# Ollama（备选，如已安装）
ollama create louter-distilled -f merged_model/Modelfile
```

### 配置 Louter 使用蒸馏模型

更新 `louter.toml`：

```toml
[hybrid]
local_endpoint = "http://localhost:8000/v1"  # HF 或 vLLM
local_model = "louter-distilled"             # ← 蒸馏模型名称
cloud_provider = "anthropic"
cloud_model = "claude-sonnet-4-20250514"
min_local_success_rate = 0.7
fallback_enabled = true
local_task_types = ["general"]           # 先保守，后续逐步放开
```

然后在 Web UI 的 Keys 页面将 API Key 的路由模式设为 **hybrid** 即可启用本地/云端混合路由。

| 工具 | 用途 |
|------|------|
| `distill/export.py` | 从 SQLite 导出训练数据（OpenAI / ShareGPT 格式） |
| `distill/compress.py` | 压缩训练数据：去重 system prompt、截断工具结果 |
| `distill/train.py` | LoRA 微调（基于 Qwen2.5-1.5B，支持 CUDA / MPS / CPU） |
| `distill/run_distill.sh` | 端到端：导出 → 压缩 → 训练 → 合并 → 部署服务 |

### 数据飞轮

```
正常使用 Louter → 云端响应自动收集 → 压缩 + 微调 → 部署服务
    ↑                                                    │
    └──── 本地模型更强 → 更多请求走本地 → 成本更低 ←───────┘
```

## 核心特性

- **一个端点接入所有供应商** — OpenAI、Anthropic、Azure、DeepSeek、Ollama 及任意 OpenAI 兼容 API
- **混合推理** — 本地优先 + 云端回退，Session 级路由，失败自动升级
- **蒸馏飞轮** — 自动收集、压缩、训练、部署，本地模型持续进化
- **强化学习** — 基于 GRPO 的 RL 训练（受 [OpenClaw RL](https://github.com/Gen-Verse/OpenClaw-RL) 启发），从成功和失败中学习，支持评判模型打分和在线蒸馏
- **工具调用标准化** — 解析 Hermes/Qwen/ReAct/JSON 格式，统一转换为 OpenAI 格式
- **隐式反馈** — 检测 Agent 重试行为，自动标注训练样本的成功/失败
- **运行时可调** — 通过 API 动态调整路由阈值，无需重启
- **智能路由** — 按模型名前缀自动匹配，或用 `model: "auto"` 按内容分类路由
- **内置 Web UI** — 管理供应商、Key、路由规则、蒸馏状态和用量统计

## 强化学习

蒸馏（SFT）教本地模型*模仿*好的云端响应，强化学习则教它*避免坏的响应*并*最大化奖励*。Louter 的 RL 流水线受 [OpenClaw RL](https://github.com/Gen-Verse/OpenClaw-RL) 启发，适配单机消费级硬件。

详见 [PLAN_RL.md](PLAN_RL.md)。

### 工作原理

```
Louter 收集训练样本（与 SFT 相同）
    ↓
导出带奖励信号的 episode：
  +1.0  成功响应，无重试
  -1.0  检测到重试（Agent 重新发送了类似请求）
  -0.5  本地失败，云端接管
    ↓
用评判模型对未打分的 episode 评分
    ↓
每个 prompt 用当前本地模型生成 G=4 个回复
    ↓
对每个回复评分 → 计算组内相对优势（GRPO）
    ↓
用裁剪代理损失 + KL 惩罚训练（LoRA）
    ↓
评估：只有 RL 模型超过 SFT 基线才部署
    ↓
合并 + 部署（Ollama / vLLM / HF Transformers）
```

### 命令

```bash
# 完整 RL 流水线
cd distill/rl
./run_rl.sh

# 分步执行
./run_rl.sh --score-only       # 用评判模型打分
./run_rl.sh --rollout-only     # 生成多个回复
./run_rl.sh --train-only       # GRPO 训练
./run_rl.sh --eval-only        # 仅运行评估
./run_rl.sh --deploy-only      # 合并部署
./run_rl.sh --opd              # 使用 GRPO + OPD 联合训练
```

### 模型服务

RL 训练后的模型支持多种服务方式，均提供 OpenAI 兼容 API：

```bash
# HuggingFace Transformers（推荐 — 支持 CUDA、Apple Silicon、CPU）
python serve_hf.py --model ./rl_merged --port 8000

# vLLM（GPU 上吞吐最高）
./serve_vllm.sh

# Ollama（备选，如已安装）
ollama create louter-rl -f rl_merged/Modelfile
```

## SFT vs RL 对比

| | SFT（蒸馏） | GRPO RL（强化学习） |
|---|---|---|
| **训练数据** | 只用成功样本 | 成功和失败样本都用 |
| **学习方式** | 逐 token 模仿云端响应 | 生成多个回复，对比组内相对优劣 |
| **失败处理** | 丢弃失败样本 | 失败样本获得 -1.0 奖励，模型学会避免 |
| **评分** | 二元（成功/失败） | 评判模型连续打分 |
| **适用场景** | 首次训练，< 1000 样本 | SFT 之后进一步优化 |

### 具体例子

用户请求一个工具调用。**SFT** 导出云端的正确响应，训练本地模型模仿它。

**GRPO RL** 让本地模型对同一个 prompt 生成 4 个不同回复：

| 回复 | 得分 | 优势 |
|---|---|---|
| JSON 正确，函数正确，参数有效 | +0.8 | +1.2（高于组均值） |
| 函数正确，参数小错误 | +0.1 | -0.2 |
| 函数正确，JSON 格式错误 | -0.2 | -0.5 |
| 幻觉了一个不存在的函数 | -1.0 | -1.8（远低于组均值） |

GRPO 推动模型趋向回复 1，远离回复 4。SFT 永远不会看到那 3 个失败 — 它只训练云端的正确响应。

### 推荐工作流

```
1. 正常使用 Louter，积累 1000+ 样本
2. 运行 SFT：./distill/run_distill.sh
3. 部署 SFT 模型，继续使用，收集更多数据
4. 运行 RL：./distill/rl/run_rl.sh（从 SFT 适配器开始）
5. 部署 RL 模型 → 更少失败 → 更好数据 → 重复
```

## 许可证

MIT
