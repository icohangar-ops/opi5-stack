"""OpenAI-compatible API in front of RKLLM (rknn-llm runtime).

Endpoints:
  GET  /v1/models
  POST /v1/chat/completions       (supports stream=true via SSE)

Why a shim? Home Assistant's *Extended OpenAI Conversation* integration
talks the OpenAI REST shape. By matching that shape, switching between
Ollama and RKLLM is just a base-URL change.

Hardening in place: fail-closed bearer auth (RKLLM_API_KEY), a single-session
concurrency lock, a per-request inference deadline, and inference error
handling. Remaining exercises: prompt template per-model, function-calling
translation.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from threading import Lock
from typing import AsyncIterator, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from cubiczan_resilience import resilient
from cubiczan_resilience.fastapi_helpers import require_auth

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
# Per-request deadline (seconds) for a single NPU inference call. A hung NPU
# call would otherwise hold _lock forever and wedge the single-session model.
GEN_TIMEOUT  = float(os.environ.get("RKLLM_GEN_TIMEOUT", "120"))

# Fail-closed bearer auth. The expected token is read from RKLLM_API_KEY at
# request time; if it is unset/empty every request is rejected (503).
_auth = require_auth(env_var="RKLLM_API_KEY")

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


@resilient(timeout=GEN_TIMEOUT, max_attempts=3, retryable_exceptions=[RuntimeError, OSError])
async def _generate(prompt: str, max_new: int, temperature: float, top_p: float) -> str:
    """Run a single blocking NPU generate off the event loop, under the lock.

    Wrapped with @resilient: a hung call is hard-bounded by GEN_TIMEOUT
    (asyncio.wait_for) and transient NPU/runtime errors are retried. The lock
    is taken inside the worker so a timeout/cancellation always releases it.
    """
    llm = _load()

    def _call() -> str:
        with _lock:
            return llm.generate(
                prompt, max_new_tokens=max_new,
                temperature=temperature, top_p=top_p,
            )

    return await asyncio.get_event_loop().run_in_executor(None, _call)


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


@app.get("/v1/models", dependencies=[Depends(_auth)])
def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(_auth)])
async def chat(req: ChatRequest):
    if req.model != MODEL_NAME:
        raise HTTPException(404, f"Unknown model {req.model!r}")
    prompt = _format_prompt(req.messages)
    max_new = req.max_tokens or MAX_NEW_TOK
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if not req.stream:
        try:
            text = await _generate(prompt, max_new, req.temperature, req.top_p)
        except asyncio.TimeoutError:
            raise HTTPException(504, "inference timed out")
        except Exception as exc:  # noqa: BLE001 - surface NPU errors as 502
            raise HTTPException(502, f"inference failed: {exc}")
        return {
            "id": cid, "object": "chat.completion", "created": created,
            "model": MODEL_NAME,
            "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }],
        }

    async def event_stream() -> AsyncIterator[dict]:
        # Drain the blocking generator off the event loop into a queue so a
        # hung NPU call is bounded and the lock is always released.
        loop = asyncio.get_event_loop()
        queue: "asyncio.Queue[object]" = asyncio.Queue()
        _SENTINEL = object()
        llm = _load()

        def _produce() -> None:
            try:
                with _lock:
                    for token in llm.generate_stream(
                        prompt, max_new_tokens=max_new,
                        temperature=req.temperature, top_p=req.top_p,
                    ):
                        loop.call_soon_threadsafe(queue.put_nowait, token)
            except Exception as exc:  # noqa: BLE001 - forward to consumer
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        producer = loop.run_in_executor(None, _produce)
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=GEN_TIMEOUT)
                except asyncio.TimeoutError:
                    err = {"error": {"message": "inference timed out",
                                     "type": "timeout"}}
                    yield {"data": json.dumps(err)}
                    break
                if item is _SENTINEL:
                    break
                if isinstance(item, BaseException):
                    err = {"error": {"message": f"inference failed: {item}",
                                     "type": "inference_error"}}
                    yield {"data": json.dumps(err)}
                    break
                chunk = {
                    "id": cid, "object": "chat.completion.chunk",
                    "created": created, "model": MODEL_NAME,
                    "choices": [{"index": 0, "delta": {"content": item},
                                 "finish_reason": None}],
                }
                yield {"data": json.dumps(chunk)}
        finally:
            producer.cancel()
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
