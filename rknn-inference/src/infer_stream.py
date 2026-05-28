"""Live RTSP / V4L2 YOLOv8 demo. Prints FPS and (optionally) shows a window.

    python src/infer_stream.py --model models/yolov8n.rknn \
        --src rtsp://user:pass@192.168.1.50:554/Streaming/Channels/102

Tip: spread two streams across two NPU cores by launching twice with
--core 0 and --core 1.
"""
from __future__ import annotations
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--src",   required=True, help="RTSP URL, file, or /dev/video0")
    ap.add_argument("--size",  type=int, default=640)
    ap.add_argument("--core",  type=int, default=-1, choices=[-1, 0, 1, 2])
    ap.add_argument("--show",  action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.src}")

    with RKNNEngine(args.model, core_mask=CORE_MAP[args.core]) as eng:
        t_log = time.perf_counter()
        frames = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
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
