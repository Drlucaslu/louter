#!/usr/bin/env python3
"""
Lightweight OpenAI-compatible server using HuggingFace Transformers.

Serves the RL-trained model as a drop-in replacement for Ollama or vLLM.
Works on Apple Silicon (MPS), CPU, and CUDA — useful for environments
where vLLM isn't available.

Supports:
  - POST /v1/chat/completions (streaming + non-streaming)
  - GET  /v1/models

Usage:
    python serve_hf.py --model ./rl_merged --port 8000
    python serve_hf.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter ./rl_output --port 8000
"""

import argparse
import json
import sys
import time
import uuid

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str, adapter_path: str | None = None):
    """Load model with optional LoRA adapter."""
    print(f"Loading model: {model_path}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        dtype = torch.bfloat16
        device_map = "auto"
        print("Using CUDA with bfloat16", file=sys.stderr)
    elif torch.backends.mps.is_available():
        dtype = torch.float16
        device_map = "mps"
        print("Using MPS with float16", file=sys.stderr)
    else:
        dtype = torch.float32
        device_map = "cpu"
        print("Using CPU with float32", file=sys.stderr)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True,
    )

    if adapter_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter: {adapter_path}", file=sys.stderr)
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, messages: list[dict], temperature: float = 0.7,
                      max_tokens: int = 2048, stream: bool = False):
    """Generate a chat completion."""
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if stream:
        return generate_stream(model, tokenizer, inputs, temperature, max_tokens)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=0.95 if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    prompt_tokens = inputs["input_ids"].shape[1]
    completion_tokens = len(generated)

    return text, prompt_tokens, completion_tokens


def generate_stream(model, tokenizer, inputs, temperature: float, max_tokens: int):
    """Generate tokens one at a time for streaming."""
    from transformers import TextIteratorStreamer
    import threading

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "temperature": temperature if temperature > 0 else None,
        "do_sample": temperature > 0,
        "top_p": 0.95 if temperature > 0 else None,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer,
    }

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for text_chunk in streamer:
        yield text_chunk

    thread.join()


def create_app(model, tokenizer, model_name: str):
    """Create FastAPI app with OpenAI-compatible endpoints."""
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field

    app = FastAPI(title="Louter RL Model Server")

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str = model_name
        messages: list[ChatMessage]
        temperature: float = 0.7
        max_tokens: int = 2048
        stream: bool = False

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if request.stream:
            async def stream_response():
                for chunk in generate_stream(
                    model, tokenizer,
                    tokenizer(
                        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                        return_tensors="pt",
                    ).to(model.device),
                    request.temperature, request.max_tokens,
                ):
                    data = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(data)}\n\n"

                # Final chunk
                data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_response(), media_type="text/event-stream")

        text, prompt_tokens, completion_tokens = generate_response(
            model, tokenizer, messages, request.temperature, request.max_tokens,
        )

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{
                "id": model_name,
                "object": "model",
                "owned_by": "louter",
            }],
        }

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": model_name}

    return app


def main():
    parser = argparse.ArgumentParser(description="Louter RL Model Server (HuggingFace Transformers)")
    parser.add_argument("--model", required=True, help="Model path or HuggingFace name")
    parser.add_argument("--adapter", help="LoRA adapter path (optional)")
    parser.add_argument("--model-name", default="louter-rl", help="Model name in API responses")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    hf_model, hf_tokenizer = load_model(args.model, args.adapter)
    app = create_app(hf_model, hf_tokenizer, args.model_name)

    print(f"\nServer ready at http://{args.host}:{args.port}", file=sys.stderr)
    print(f"  POST /v1/chat/completions", file=sys.stderr)
    print(f"  GET  /v1/models", file=sys.stderr)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
