# RKLLM server image for RK3588 (aarch64).
# Builds on Ubuntu 22.04 with the rkllm runtime + a small FastAPI shim
# that exposes an OpenAI-compatible /v1/chat/completions endpoint.
FROM arm64v8/ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git curl ca-certificates \
        libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# --- rkllm runtime ---------------------------------------------------------
# Pull librkllmrt.so + the Python bindings from airockchip's rknn-llm repo.
# Pin a release tag in production builds.
RUN git clone --depth 1 https://github.com/airockchip/rknn-llm.git /opt/rknn-llm \
    && install -m 0755 /opt/rknn-llm/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so \
        /usr/local/lib/librkllmrt.so \
    && ldconfig

ENV PYTHONPATH=/opt/rknn-llm/rkllm-toolkit/examples/rkllm_server

# --- API shim --------------------------------------------------------------
WORKDIR /app
RUN pip3 install --no-cache-dir fastapi==0.115.* uvicorn[standard]==0.30.* \
                                pydantic==2.* sse-starlette==2.*

COPY server.py /app/server.py
EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
