"""Live RTSP / V4L2 YOLOv8 demo. Prints FPS and (optionally) shows a window.

    python src/infer_stream.py --model models/yolov8n.rknn \
        --src rtsp://user:pass@192.168.1.50:554/Streaming/Channels/102

Tip: spread two streams across two NPU cores by launching twice with
--core 0 and --core 1.
"""
from __future__ import annotations
import argparse
import sys
import time
import cv2

from rknn_engine import RKNNEngine
from yolo_postprocess import letterbox, postprocess, COCO_LABELS

CORE_MAP = {
    0: RKNNEngine.NPU_CORE_0,
    1: RKNNEngine.NPU_CORE_1,
    2: RKNNEngine.NPU_CORE_2,
    -1: RKNNEngine.NPU_CORE_0_1_2,
}

# Reconnect backoff: 1s, 2s, 4s ... capped at this many seconds.
RECONNECT_BASE = 1.0
RECONNECT_CAP = 30.0


def _log(event: str, msg: str) -> None:
    """Structured stderr log so operators can detect outages without attaching."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f"{ts} [{event}] {msg}", file=sys.stderr, flush=True)


def _open_stream(src: str) -> "cv2.VideoCapture":
    """Open a capture with exponential backoff until it succeeds."""
    delay = RECONNECT_BASE
    attempt = 0
    while True:
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            if attempt:
                _log("stream_reconnect", f"reconnected to {src} after {attempt} attempt(s)")
            return cap
        cap.release()
        attempt += 1
        _log("stream_error", f"could not open {src}; retry in {delay:.0f}s (attempt {attempt})")
        time.sleep(delay)
        delay = min(delay * 2, RECONNECT_CAP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--src",   required=True, help="RTSP URL, file, or /dev/video0")
    ap.add_argument("--size",  type=int, default=640)
    ap.add_argument("--core",  type=int, default=-1, choices=[-1, 0, 1, 2])
    ap.add_argument("--show",  action="store_true")
    args = ap.parse_args()

    cap = _open_stream(args.src)

    with RKNNEngine(args.model, core_mask=CORE_MAP[args.core]) as eng:
        t_log = time.perf_counter()
        frames = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                # Stream dropped: release, back off, and reopen. Keeps a
                # long-running camera feed self-healing across RTSP hiccups.
                _log("stream_error", f"read() failed on {args.src}; reconnecting")
                cap.release()
                cap = _open_stream(args.src)
                continue
            lb, r, pad = letterbox(frame, (args.size, args.size))
            nhwc = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)[None, ...]
            outs = eng.infer(nhwc)
            dets = postprocess(outs, r, pad, frame.shape[:2])

            frames += 1
            if time.perf_counter() - t_log > 2.0:
                fps = frames / (time.perf_counter() - t_log)
                print(f"{fps:5.1f} FPS | infer {eng.stats.avg_ms:5.1f} ms | "
                      f"{len(dets)} dets")
                frames = 0
                t_log = time.perf_counter()

            if args.show:
                for x1, y1, x2, y2, score, cid in dets:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{COCO_LABELS[cid]} {score:.2f}",
                                (x1, max(y1 - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow("rknn", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
