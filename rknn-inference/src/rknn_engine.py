"""Thin wrapper around rknn-toolkit-lite2 for on-device inference.

Usage:
    eng = RKNNEngine("models/yolov8n.rknn", core_mask=RKNNEngine.NPU_CORE_0)
    outputs = eng.infer(nhwc_uint8_image)
    eng.close()
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import numpy as np

try:
    from rknnlite.api import RKNNLite
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "rknn-toolkit-lite2 is not installed. Download the matching wheel from "
        "https://github.com/airockchip/rknn-toolkit2 and `pip install` it."
    ) from e


@dataclass
class InferStats:
    last_ms: float = 0.0
    avg_ms: float = 0.0
    count: int = 0


class RKNNEngine:
    NPU_CORE_0     = RKNNLite.NPU_CORE_0
    NPU_CORE_1     = RKNNLite.NPU_CORE_1
    NPU_CORE_2     = RKNNLite.NPU_CORE_2
    NPU_CORE_0_1   = RKNNLite.NPU_CORE_0_1
    NPU_CORE_0_1_2 = RKNNLite.NPU_CORE_0_1_2

    def __init__(self, model_path: str, core_mask: int = NPU_CORE_0):
        self._rknn = RKNNLite(verbose=False)
        if self._rknn.load_rknn(model_path) != 0:
            raise RuntimeError(f"Failed to load {model_path}")
        if self._rknn.init_runtime(core_mask=core_mask) != 0:
            raise RuntimeError("Failed to init RKNN runtime (check NPU driver)")
        self.stats = InferStats()

    def infer(self, inputs) -> List[np.ndarray]:
        """Run a single inference. `inputs` is a numpy array or list of arrays."""
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
        t0 = time.perf_counter()
        outs = self._rknn.inference(inputs=inputs)
        dt = (time.perf_counter() - t0) * 1000.0
        s = self.stats
        s.last_ms = dt
        s.count += 1
        s.avg_ms = s.avg_ms + (dt - s.avg_ms) / s.count
        return outs

    def close(self) -> None:
        self._rknn.release()

    def __enter__(self):  return self
    def __exit__(self, *exc):  self.close()
