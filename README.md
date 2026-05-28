# Orange Pi 5 — Edge AI Stack

Production-ready scaffold for running **Frigate NVR + Home Assistant** with
NPU-accelerated detection on the Rockchip RK3588(S), plus a standalone
**Python RKNN inference template** you can fork for custom models
(YOLOv8, RT-DETR, classifiers, etc.).

```
opi5-stack/
├── frigate/                 # Frigate NVR with rknn detector
│   ├── docker-compose.yml
│   ├── config.yml
│   └── .env.example
├── homeassistant/           # HA + MQTT + Frigate integration
│   ├── docker-compose.yml
│   └── config/
│       ├── configuration.yaml
│       └── automations.yaml
└── rknn-inference/          # Python RKNN template (YOLOv8 on NPU)
    ├── requirements.txt
    ├── src/
    │   ├── rknn_engine.py   # Reusable RKNN runtime wrapper
    │   ├── yolo_postprocess.py
    │   ├── infer_image.py   # Single image demo
    │   └── infer_stream.py  # RTSP / V4L2 live demo
    └── models/              # drop .rknn files here
```

## Prerequisites (host)

1. **OS**: Joshua Riek's Ubuntu 24.04 for Rockchip (kernel 6.1+ with `rknpu` driver).
2. Verify NPU: `sudo cat /sys/kernel/debug/rknpu/load` should print 3 cores.
3. Install Docker + Compose v2: `curl -fsSL https://get.docker.com | sh`.
4. Add user to `video`, `render`, `dialout` groups, re-login.
5. NVMe mounted at `/mnt/nvme` for clips/recordings (recommended).

## Quick start

```bash
# 1. Frigate
cd frigate && cp .env.example .env   # edit secrets
docker compose up -d

# 2. Home Assistant
cd ../homeassistant && docker compose up -d
# open http://<pi-ip>:8123, complete onboarding, then add Frigate integration

# 3. RKNN Python template
cd ../rknn-inference
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place yolov8n.rknn in models/ then:
python src/infer_image.py --model models/yolov8n.rknn --image test.jpg
```

See each subfolder's comments for the full configuration surface.
