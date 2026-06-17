# LLM Serving on Orange Pi 5 (RK3588)

Two paths, both wired into Home Assistant as a **conversation agent** via
the OpenAI-compatible API.

| Path | Hardware | Speed (Qwen2.5-3B, q4) | When to use |
|------|----------|------------------------|-------------|
| **RKLLM** (`rkllm/`) | NPU (all 3 cores) | ~15–25 tok/s | Best perf, frees CPU for other work. Requires `.rkllm` converted models. |
| **Ollama** (`ollama/`) | CPU (4×A76) | ~6–10 tok/s | Easiest. Huge model library. No conversion. |

Both expose an **OpenAI-compatible** `/v1/chat/completions` endpoint on
port `8080`, so Home Assistant's *OpenAI Conversation* (or *Extended OpenAI
Conversation*) integration points at either one with the same config.

```
opi5-stack/llm/
├── rkllm/              # NPU server (rkllm-server in Docker)
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── server.py       # FastAPI OpenAI-compatible shim over RKLLM
│   └── models/         # drop Qwen2.5-3B.rkllm here
└── ollama/             # CPU fallback
    └── docker-compose.yml
```

## Quick start — Ollama (5 minutes)

```bash
cd ollama && docker compose up -d
docker exec -it ollama ollama pull qwen2.5:3b-instruct-q4_K_M
curl http://localhost:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5:3b-instruct-q4_K_M","messages":[{"role":"user","content":"hi"}]}'
```

## Quick start — RKLLM (NPU)

1. On a Linux x86 box, install `rkllm-toolkit` and convert a model:
   ```bash
   # See https://github.com/airockchip/rknn-llm
   python convert.py --hf Qwen/Qwen2.5-3B-Instruct \
                     --quantized-dtype w8a8 \
                     --target-platform rk3588 \
                     --output qwen2.5-3b.rkllm
   ```
2. Copy `qwen2.5-3b.rkllm` to `rkllm/models/` on the Pi.
3. `cd rkllm && docker compose up -d --build`
4. Test: `curl http://localhost:8080/v1/chat/completions -d '{...}'`

## Wire into Home Assistant

1. Install **Extended OpenAI Conversation** via HACS (preferred — supports tool calling).
2. Settings → Devices → Add Integration → *Extended OpenAI Conversation*.
3. Config:
   - **API key**:
     - Ollama: `not-needed` (Ollama does not authenticate by default)
     - RKLLM: the value of `RKLLM_API_KEY` set on the `rkllm` service. The shim
       is fail-closed — if `RKLLM_API_KEY` is unset/empty every request is
       rejected, so set it in the environment and use the same value here.
   - **Base URL**:
     - Ollama: `http://<pi-ip>:11434/v1`
     - RKLLM: `http://<pi-ip>:8080/v1`
   - **Model**: `qwen2.5:3b-instruct-q4_K_M` (ollama) or `qwen2.5-3b` (rkllm)
4. Settings → Voice assistants → create new assistant → set **Conversation
   agent** to the new OpenAI entry. (Speech-to-text: `whisper`; TTS: `piper`.)

A ready-to-paste system prompt and a sample voice automation are in
`../homeassistant/config/packages/llm_assistant.yaml` (added below).
