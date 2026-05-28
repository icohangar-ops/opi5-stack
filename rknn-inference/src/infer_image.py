"""Single-image YOLOv8 demo on the RK3588 NPU.

    python src/infer_image.py --model models/yolov8n.rknn --image bus.jpg
"""
from __future__ import annotations
import argparse
import cv2

from rknn_engine import RKNNEngine
from yolo_postprocess import letterbox, postprocess, COCO_LABELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out",   default="out.jpg")
    ap.add_argument("--size",  type=int, default=640)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read {args.image}")
    lb, r, pad = letterbox(img, (args.size, args.size))
    nhwc = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)[None, ...]  # 1xHxWx3 uint8

    with RKNNEngine(args.model, core_mask=RKNNEngine.NPU_CORE_0_1_2) as eng:
        outs = eng.infer(nhwc)
        dets = postprocess(outs, r, pad, img.shape[:2])
        print(f"Inference: {eng.stats.last_ms:.1f} ms | {len(dets)} detections")

    for x1, y1, x2, y2, score, cid in dets:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{COCO_LABELS[cid]} {score:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(args.out, img)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
