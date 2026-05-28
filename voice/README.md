# Wyoming Voice Stack

One-file Docker Compose for **Home Assistant voice** on the Orange Pi 5.

Services exposed on Wyoming protocol (HA Settings → Devices → Wyoming):

| Service | Port | Hardware | Description |
|---------|------|----------|-------------|
| Wyoming-Whisper | `10300` | CPU (A76) | Speech-to-text (local Whisper) |
| Wyoming-Piper | `10200` | CPU (any) | Text-to-speech (Piper) |
| Wyoming-openWakeWord | `10400` | CPU (any) | Wake-word detection |

## Quick start

```bash
cd voice && docker compose up -d
```

Then in Home Assistant:
1. **Settings → Devices & Services → Add Integration → Wyoming**
2. Add each service by IP + port (`<pi-ip>:10300`, `:10200`, `:10400`).
3. **Settings → Voice assistants → New assistant**
   - Name: "Casa"
   - Conversation agent: Extended OpenAI Conversation (local LLM)
   - Speech-to-text: Wyoming Whisper
   - Text-to-speech: Wyoming Piper
   - Wake word: Wyoming openWakeWord → "hey_jarvis_v0.1"

## Model downloads

### Whisper

```bash
# Download a model inside the container on first start, or pre-cache:
mkdir -p /mnt/nvme/models/whisper /mnt/nvme/cache/whisper
cd /mnt/nvme/models/whisper
# tiny = 39 MB, base = 74 MB, small = 466 MB (base is the sweet spot for RK3588)
curl -LO https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
# If you change the model name, update docker-compose.yml
```

### Piper

```bash
mkdir -p /mnt/nvme/models/piper
cd /mnt/nvme/models/piper
pip install piper-tts
python3 -m piper.download --voice en_US-lessac-medium
# or manually:
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

### openWakeWord

```bash
mkdir -p /mnt/nvme/models/openwakeword
cd /mnt/nvme/models/openwakeword
# Default model (included in image), but you can add custom .tflite wake words:
curl -LO https://github.com/dscripka/openWakeWord/raw/main/openwakeword/resources/models/hey_jarvis_v0.1.tflite
```

## Custom wake word

1. Record 20–50 samples of your phrase (~1–2 seconds each).
2. Train with [openWakeWord's notebooks](https://github.com/dscripka/openWakeWord) or the web UI.
3. Copy the `.tflite` to `/mnt/nvme/models/openwakeword/`.
4. In HA, add a second openWakeWord instance (or edit compose) pointing at the new model.

## Full voice loop architecture

```
Microphone  →  openWakeWord ("hey jarvis")
      ↓
Wyoming-Whisper  →  HA Conversation Agent  →  Extended OpenAI Conversation
      ↑                                    (local LLM: Ollama or RKLLM)
Wyoming-Piper  ←  (service calls / spoken reply)
      ↓
Speaker
```

Everything runs on the Pi. No cloud for STT, TTS, wake word, or LLM.
