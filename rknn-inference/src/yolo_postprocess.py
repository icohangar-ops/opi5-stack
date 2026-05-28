"""YOLOv8 post-processing (letterbox + NMS) for RKNN outputs.

Assumes a model exported with rknn-model-zoo's YOLOv8 recipe, which outputs
three feature maps + scores. For quick experimentation we use the simpler
"single output" head: [1, 84, N] where 84 = 4 box + 80 class.
"""
from __future__ import annotations
from typing import List, Tuple

import cv2
import numpy as np

CONF_THRES = 0.35
IOU_THRES  = 0.45


def letterbox(img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top  = (new_shape[0] - nh) // 2
    left = (new_shape[1] - nw) // 2
    out = np.full((new_shape[0], new_shape[1], 3), color, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, r, (left, top)


def _nms(boxes, scores, iou=IOU_THRES):
    idxs = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONF_THRES, iou)
    return [int(i) for i in np.array(idxs).flatten()] if len(idxs) else []


def postprocess(
    outputs: List[np.ndarray],
    ratio: float,
    pad: Tuple[int, int],
    orig_shape: Tuple[int, int],
) -> List[Tuple[int, int, int, int, float, int]]:
    """Return list of (x1, y1, x2, y2, score, class_id) in original-image coords."""
    pred = outputs[0]
    if pred.ndim == 3:
        pred = pred[0]
    # pred shape (84, N) → transpose to (N, 84)
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T
    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:]
    cls_ids = cls_scores.argmax(axis=1)
    scores  = cls_scores.max(axis=1)
    keep = scores > CONF_THRES
    boxes_xywh = boxes_xywh[keep]
    scores     = scores[keep]
    cls_ids    = cls_ids[keep]
    if boxes_xywh.shape[0] == 0:
        return []

    # xywh → xyxy
    xy = boxes_xywh[:, :2]
    wh = boxes_xywh[:, 2:4]
    xyxy = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)

    # undo letterbox
    xyxy[:, [0, 2]] -= pad[0]
    xyxy[:, [1, 3]] -= pad[1]
    xyxy /= ratio
    h, w = orig_shape
    xyxy[:, 0::2] = xyxy[:, 0::2].clip(0, w - 1)
    xyxy[:, 1::2] = xyxy[:, 1::2].clip(0, h - 1)

    # NMS in xywh form for cv2
    nms_boxes = np.column_stack([xyxy[:, 0], xyxy[:, 1],
                                 xyxy[:, 2] - xyxy[:, 0],
                                 xyxy[:, 3] - xyxy[:, 1]])
    keep_idx = _nms(nms_boxes, scores)

    results = []
    for i in keep_idx:
        x1, y1, x2, y2 = xyxy[i].astype(int).tolist()
        results.append((x1, y1, x2, y2, float(scores[i]), int(cls_ids[i])))
    return results


COCO_LABELS = (
    "person bicycle car motorcycle airplane bus train truck boat traffic_light "
    "fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow "
    "elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
    "skis snowboard sports_ball kite baseball_bat baseball_glove skateboard "
    "surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana "
    "apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch "
    "potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
    "cell_phone microwave oven toaster sink refrigerator book clock vase "
    "scissors teddy_bear hair_drier toothbrush"
).split()
