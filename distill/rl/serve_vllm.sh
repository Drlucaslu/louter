#!/bin/bash
# Serve the RL-trained model with vLLM (OpenAI-compatible API).
#
# vLLM provides 3-5x higher throughput than HuggingFace Transformers
# with continuous batching and PagedAttention.
#
# Usage:
#   ./serve_vllm.sh                              # Defaults
#   MODEL_PATH=./custom_model ./serve_vllm.sh    # Custom model
#   PORT=8080 ./serve_vllm.sh                    # Custom port
#
# Requirements:
#   pip install vllm
#
# Then configure Louter:
#   [hybrid]
#   local_provider = "ollama"      # or any OpenAI-compatible provider
#   local_endpoint = "http://localhost:8000/v1"
#   local_model = "louter-rl"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/rl_merged}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.9}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Run ./run_rl.sh first to train and merge, or set MODEL_PATH."
    exit 1
fi

echo "Starting vLLM server..."
echo "  Model: $MODEL_PATH"
echo "  Endpoint: http://${HOST}:${PORT}/v1"
echo "  Max context: ${MAX_MODEL_LEN} tokens"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype auto \
    --gpu-memory-utilization "$GPU_UTIL" \
    --served-model-name "louter-rl"
