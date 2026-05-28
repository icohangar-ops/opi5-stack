"""OpenAI-compatible API in front of RKLLM (rknn-llm runtime).

Endpoints:
  GET  /v1/models
  POST /v1/chat/completions       (supports stream=true via SSE)

Why a shim? Home Assistant's *Extended OpenAI Conversation* integration
talks the OpenAI REST shape. By matching that shape, switching between
Ollama and RKLLM is just a base-URL change.

Production hardening left as exercises: auth, concurrency lock,
prompt template per-model, function-calling translation.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from threading import Lock
from typing import AsyncIterator, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# rkllm bindings ship with the runtime repo (PYTHONPATH set in Dockerfile)
try:
    from rkllm_binding import RKLLM   # type: ignore
except ImportError:  # fallback name used by some releases
    from rkllm import RKLLM           # type: ignore


MODEL_PATH   = os.environ["RKLLM_MODEL"]
MODEL_NAME   = os.environ.get("RKLLM_MODEL_NAME", "rkllm")
CORE_MASK    = int(os.environ.get("RKLLM_CORE_MASK", "7"))     # all 3 cores
MAX_NEW_TOK  = int(os.environ.get("RKLLM_MAX_NEW_TOKENS", "512"))
MAX_CTX      = int(os.environ.get("RKLLM_MAX_CONTEXT",   "4096"))

# RKLLM is single-session; serialize calls.
_lock = Lock()
_llm: Optional[RKLLM] = None


def _load() -> RKLLM:
    global _llm
    if _llm is None:
        _llm = RKLLM(
            model_path=MODEL_PATH,
            core_mask=CORE_MASK,
            max_new_tokens=MAX_NEW_TOK,
            max_context_len=MAX_CTX,
        )
    return _llm


# ---------------- OpenAI request/response models ----------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = None
    stream: bool = False


def _format_prompt(msgs: List[ChatMessage]) -> str:
    """Minimal ChatML-style template; works for Qwen2.5, Llama 3.x, Phi-3.5."""
    parts: List[str] = []
    for m in msgs:
        parts.append(f"<|im_start|>{m.role}\n{m.content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ---------------- API ----------------
app = FastAPI(title="rkllm-openai-shim")


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    if req.model != MODEL_NAME:
        raise HTTPException(404, f"Unknown model {req.model!r}")
    llm = _load()
    prompt = _format_prompt(req.messages)
    max_new = req.max_tokens or MAX_NEW_TOK
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if not req.stream:
        with _lock:
            text = llm.generate(prompt, max_new_tokens=max_new,
                                temperature=req.temperature, top_p=req.top_p)
        return {
            "id": cid, "object": "chat.completion", "created": created,
            "model": MODEL_NAME,
            "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }],
        }

    async def event_stream() -> AsyncIterator[dict]:
        with _lock:
            for token in llm.generate_stream(
                prompt, max_new_tokens=max_new,
                temperature=req.temperature, top_p=req.top_p,
            ):
                chunk = {
                    "id": cid, "object": "chat.completion.chunk",
                    "created": created, "model": MODEL_NAME,
                    "choices": [{"index": 0, "delta": {"content": token},
                                 "finish_reason": None}],
                }
                yield {"data": json.dumps(chunk)}
        final = {
            "id": cid, "object": "chat.completion.chunk",
            "created": created, "model": MODEL_NAME,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield {"data": json.dumps(final)}
        yield {"data": "[DONE]"}

    return EventSourceResponse(event_stream())


@app.get("/healthz")
def health():
    return {"ok": True, "model": MODEL_NAME}
