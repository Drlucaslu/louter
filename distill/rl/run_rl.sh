#!/bin/bash
# Louter RL Pipeline — End-to-end: export → score → rollout → train → evaluate → deploy
#
# Usage:
#   ./run_rl.sh                    # Full pipeline with defaults
#   ./run_rl.sh --score-only       # Just score episodes with judge
#   ./run_rl.sh --rollout-only     # Just generate rollouts
#   ./run_rl.sh --train-only       # Just run GRPO training
#   ./run_rl.sh --deploy-only      # Just merge and deploy
#   ./run_rl.sh --eval-only        # Just run evaluation
#   ./run_rl.sh --opd              # Use combined GRPO + OPD training
#
# Environment:
#   LOUTER_DB          Path to louter.db (default: ../../louter.db)
#   BASE_MODEL         Base model (default: Qwen/Qwen2.5-1.5B-Instruct)
#   ADAPTER_PATH       Starting adapter (default: ../output — reuse SFT adapter)
#   JUDGE_PROVIDER     Judge provider: anthropic|openai|ollama (default: anthropic)
#   JUDGE_MODEL        Judge model name (default: auto per provider)
#   INFERENCE_BACKEND  Rollout backend: transformers|ollama|vllm (default: transformers)
#   OLLAMA_MODEL       Name for deployed model (default: louter-rl)
#   MIN_EPISODES       Min episodes required to start training (default: 100)
#   LOUTER_RL_STRICT_EVAL  Block deployment if eval fails: 1=strict (default), 0=override

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOUTER_DB="${LOUTER_DB:-${SCRIPT_DIR}/../../louter.db}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-${SCRIPT_DIR}/../output}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-anthropic}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-transformers}"
OLLAMA_MODEL="${OLLAMA_MODEL:-louter-rl}"
MIN_EPISODES="${MIN_EPISODES:-100}"

OUTPUT_DIR="${SCRIPT_DIR}/output"
EPISODES_FILE="${OUTPUT_DIR}/episodes.jsonl"
ROLLOUTS_FILE="${OUTPUT_DIR}/rollouts.jsonl"
SCORED_FILE="${OUTPUT_DIR}/scored_rollouts.jsonl"
RL_ADAPTER_DIR="${OUTPUT_DIR}/rl_adapter"
MERGED_DIR="${SCRIPT_DIR}/rl_merged"

# Parse flags
score_only=false
rollout_only=false
train_only=false
deploy_only=false
eval_only=false
use_opd=false

for arg in "$@"; do
    case $arg in
        --score-only)   score_only=true ;;
        --rollout-only) rollout_only=true ;;
        --train-only)   train_only=true ;;
        --deploy-only)  deploy_only=true ;;
        --eval-only)    eval_only=true ;;
        --opd)          use_opd=true ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

# ── Step 1: Export episodes ──
if [ "$rollout_only" = false ] && [ "$train_only" = false ] && [ "$deploy_only" = false ] && [ "$eval_only" = false ]; then
    echo "=== Step 1: Exporting RL episodes ==="
    echo "Database: $LOUTER_DB"

    python3 "$SCRIPT_DIR/export_episodes.py" \
        --db "$LOUTER_DB" \
        --output "$EPISODES_FILE" \
        --write-db

    episode_count=$(wc -l < "$EPISODES_FILE" 2>/dev/null || echo "0")
    echo "Exported $episode_count episodes"

    if [ "$episode_count" -lt "$MIN_EPISODES" ]; then
        echo "Not enough episodes ($episode_count < $MIN_EPISODES). Collect more data first."
        if [ "$score_only" = false ]; then
            exit 0
        fi
    fi

    # Score tool-call episodes (fast, no API calls)
    echo ""
    echo "=== Step 1b: Scoring tool-call episodes ==="
    python3 "$SCRIPT_DIR/score_tool_calls.py" --db "$LOUTER_DB"

    # Score with judge
    echo ""
    echo "=== Step 1c: Scoring episodes with judge ==="
    judge_args="--db $LOUTER_DB --provider $JUDGE_PROVIDER --source local"
    if [ -n "$JUDGE_MODEL" ]; then
        judge_args="$judge_args --model $JUDGE_MODEL"
    fi
    python3 "$SCRIPT_DIR/score_with_judge.py" $judge_args

    python3 "$SCRIPT_DIR/export_episodes.py" --db "$LOUTER_DB" --stats

    if [ "$score_only" = true ]; then
        echo "Scoring complete."
        exit 0
    fi
fi

# ── Step 2: Generate rollouts ──
if [ "$train_only" = false ] && [ "$deploy_only" = false ] && [ "$eval_only" = false ]; then
    echo ""
    echo "=== Step 2: Generating rollouts ==="
    echo "Backend: $INFERENCE_BACKEND, Model: $BASE_MODEL"

    rollout_args="--output $ROLLOUTS_FILE --backend $INFERENCE_BACKEND --model $BASE_MODEL"

    # Use episodes from file or DB
    if [ -f "$EPISODES_FILE" ]; then
        rollout_args="$rollout_args --episodes $EPISODES_FILE"
    else
        rollout_args="$rollout_args --db $LOUTER_DB"
    fi

    # Add adapter if using transformers and adapter exists
    if [ "$INFERENCE_BACKEND" = "transformers" ] && [ -d "$ADAPTER_PATH" ]; then
        rollout_args="$rollout_args --adapter $ADAPTER_PATH"
    fi

    python3 "$SCRIPT_DIR/generate_rollouts.py" $rollout_args

    if [ "$rollout_only" = true ]; then
        echo "Rollout generation complete: $ROLLOUTS_FILE"
        exit 0
    fi

    # Score rollouts
    echo ""
    echo "=== Step 2b: Scoring rollouts ==="
    score_args="--rollouts $ROLLOUTS_FILE --output $SCORED_FILE --judge-provider $JUDGE_PROVIDER"
    if [ -n "$JUDGE_MODEL" ]; then
        score_args="$score_args --judge-model $JUDGE_MODEL"
    fi
    python3 "$SCRIPT_DIR/reward_rollouts.py" $score_args
fi

# ── Step 3: GRPO training ──
if [ "$deploy_only" = false ] && [ "$eval_only" = false ]; then
    echo ""
    echo "=== Step 3: GRPO training ==="
    echo "Base model: $BASE_MODEL"

    if [ ! -f "$SCORED_FILE" ]; then
        echo "Error: Scored rollouts not found at $SCORED_FILE"
        echo "Run without --train-only first, or generate rollouts separately."
        exit 1
    fi

    train_args="--data $SCORED_FILE --base-model $BASE_MODEL --output-dir $RL_ADAPTER_DIR"

    # Reuse SFT adapter as starting point
    if [ -d "$ADAPTER_PATH" ]; then
        echo "Starting from SFT adapter: $ADAPTER_PATH"
        train_args="$train_args --adapter $ADAPTER_PATH"
    fi

    if [ "$use_opd" = true ]; then
        echo "Using combined GRPO + OPD training"
        python3 "$SCRIPT_DIR/train_opd.py" $train_args
    else
        python3 "$SCRIPT_DIR/train_grpo.py" $train_args
    fi

    echo "Training complete. Adapter saved to: $RL_ADAPTER_DIR"

    if [ "$train_only" = true ]; then
        exit 0
    fi
fi

# ── Step 4: Evaluate ──
if [ "$deploy_only" = false ]; then
    echo ""
    echo "=== Step 4: Evaluation ==="

    # Merge RL model for evaluation
    echo "Merging RL adapter for evaluation..."
    python3 "$SCRIPT_DIR/train_grpo.py" --merge \
        --base-model "$BASE_MODEL" \
        --adapter "$RL_ADAPTER_DIR" \
        --output-dir "$MERGED_DIR"

    if [ -f "$EPISODES_FILE" ]; then
        eval_args="--episodes $EPISODES_FILE --rl-model $MERGED_DIR --backend $INFERENCE_BACKEND"
        eval_args="$eval_args --judge-provider $JUDGE_PROVIDER --report $OUTPUT_DIR/eval_report.json"

        # SFT baseline: use existing merged model or adapter
        sft_merged="${SCRIPT_DIR}/../merged_model"
        if [ -d "$sft_merged" ]; then
            eval_args="$eval_args --sft-model $sft_merged"
        else
            eval_args="$eval_args --sft-model $BASE_MODEL"
        fi

        if [ -n "$JUDGE_MODEL" ]; then
            eval_args="$eval_args --judge-model $JUDGE_MODEL"
        fi

        echo "Comparing RL model vs SFT baseline..."
        if python3 "$SCRIPT_DIR/evaluate.py" $eval_args; then
            echo "RL model is better — proceeding to deploy."
        else
            echo "RL model did NOT improve over SFT baseline."
            echo "Review: $OUTPUT_DIR/eval_report.json"
            if [ "$eval_only" = true ]; then exit 0; fi
            if [ "${LOUTER_RL_STRICT_EVAL:-1}" = "1" ]; then
                echo "Evaluation gate failed. Aborting deployment (set LOUTER_RL_STRICT_EVAL=0 to override)."
                exit 1
            else
                echo "WARNING: Deploying despite evaluation gate failure (LOUTER_RL_STRICT_EVAL=0)."
            fi
        fi
    else
        echo "No episodes file for evaluation — skipping comparison."
    fi

    if [ "$eval_only" = true ]; then
        exit 0
    fi
fi

# ── Step 5: Deploy ──
echo ""
echo "=== Step 5: Deploying RL model ==="

# Merge if not already done
if [ ! -d "$MERGED_DIR" ]; then
    python3 "$SCRIPT_DIR/train_grpo.py" --merge \
        --base-model "$BASE_MODEL" \
        --adapter "$RL_ADAPTER_DIR" \
        --output-dir "$MERGED_DIR"
fi

if command -v ollama &> /dev/null; then
    # Keep previous model as rollback
    if ollama list 2>/dev/null | grep -q "$OLLAMA_MODEL"; then
        echo "Keeping previous model as ${OLLAMA_MODEL}-prev"
        ollama cp "$OLLAMA_MODEL" "${OLLAMA_MODEL}-prev" 2>/dev/null || true
    fi

    echo "Importing '$OLLAMA_MODEL' from safetensors..."
    ollama create "$OLLAMA_MODEL" -f "$MERGED_DIR/Modelfile"

    echo ""
    echo "=== Deployment complete! ==="
    echo "Model: $OLLAMA_MODEL"
    echo "Previous: ${OLLAMA_MODEL}-prev (rollback if needed)"
    echo ""
    echo "Verify: ollama run $OLLAMA_MODEL \"Hello\""
    echo ""
    echo "To use with Louter, update louter.toml:"
    echo "  [hybrid]"
    echo "  local_model = \"$OLLAMA_MODEL\""
else
    echo "Ollama not found. Merged model available at: $MERGED_DIR"
    echo ""
    echo "Serve with vLLM:"
    echo "  ./serve_vllm.sh"
    echo ""
    echo "Or with HuggingFace Transformers:"
    echo "  python serve_hf.py --model $MERGED_DIR --port 8000"
    echo ""
    echo "Or install Ollama and import:"
    echo "  ollama create $OLLAMA_MODEL -f $MERGED_DIR/Modelfile"
fi
