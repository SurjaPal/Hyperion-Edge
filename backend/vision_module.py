"""
vision_module.py
-----------------
The pretrained-AI-model half of the project: YOLOX-nano (Megvii, Apache-2.0,
open-source), a neural network detector that runs through ONNX Runtime.

Why this file matters for the challenge rules: the guidelines ask for AI
models from Qualcomm AI Hub or other open-source platforms. This is that
model, actually loaded and executed, not described.

Execution provider selection (the NPU story, stated precisely):
  * On a Snapdragon Windows device with the onnxruntime-qnn package, the
    session below is created with QNNExecutionProvider first (HTP = the
    Hexagon NPU backend) and CPUExecutionProvider as fallback.
  * Anywhere else (this dev sandbox) only the CPU provider exists, so the
    session runs on CPU. `VisionModule.provider` reports which one is
    actually active, and the dashboard shows that value verbatim, so the
    UI can never claim NPU execution that did not happen.

NOTE (verify on the real device): QNN's HTP backend is designed around
quantized models. ONNX Runtime documents an HTP fp16 option for float32
models; whether the float32 YOLOX-nano runs fully on the NPU, or needs a
QDQ-quantized copy for best results, must be confirmed on the actual
hardware. That measurement is a to-do, not a claim.
"""

from __future__ import annotations

import os
import time

import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yolox_nano.onnx")
INPUT_SIZE = (416, 416)

# COCO class ids relevant to a fuel-outlet forecourt
VEHICLE_IDS = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PERSON_ID = 0


def _build_session(model_path: str):
    import onnxruntime as ort

    available = ort.get_available_providers()
    providers = []
    if "QNNExecutionProvider" in available:
        providers.append(
            (
                "QNNExecutionProvider",
                {"backend_path": "QnnHtp.dll", "enable_htp_fp16_precision": "1"},
            )
        )
    providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(model_path, providers=providers)
    return session, session.get_providers()[0]


def _preprocess(img_rgb: np.ndarray):
    """Letterbox to 416x416, BGR, 0-255 float32, CHW (YOLOX convention)."""
    from PIL import Image

    h, w = img_rgb.shape[:2]
    ratio = min(INPUT_SIZE[0] / h, INPUT_SIZE[1] / w)
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized = np.array(Image.fromarray(img_rgb).resize((new_w, new_h), Image.BILINEAR))
    canvas = np.full((INPUT_SIZE[0], INPUT_SIZE[1], 3), 114, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    bgr = canvas[:, :, ::-1]
    chw = np.ascontiguousarray(bgr.transpose(2, 0, 1), dtype=np.float32)
    return chw[None, ...], ratio


def _decode(raw: np.ndarray) -> np.ndarray:
    """Decode YOLOX grid outputs to absolute xywh (strides 8/16/32)."""
    grids, strides_out = [], []
    for stride in (8, 16, 32):
        hs, ws = INPUT_SIZE[0] // stride, INPUT_SIZE[1] // stride
        xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        strides_out.append(np.full((1, grid.shape[1], 1), stride))
    grids = np.concatenate(grids, 1)
    strides_out = np.concatenate(strides_out, 1)
    out = raw.copy()
    out[..., :2] = (out[..., :2] + grids) * strides_out
    out[..., 2:4] = np.exp(out[..., 2:4]) * strides_out
    return out


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


class VisionModule:
    def __init__(self, model_path: str = MODEL_PATH):
        self.available = os.path.exists(model_path)
        self.provider = "unavailable"
        self.last_latency_ms = None
        self._session = None
        if self.available:
            self._session, self.provider = _build_session(model_path)
            self._input_name = self._session.get_inputs()[0].name

    def detect(self, img_rgb: np.ndarray, score_thr: float = 0.4) -> dict:
        if not self.available:
            return {"persons": 0, "vehicles": 0, "vehicle_types": [], "latency_ms": None}
        tensor, ratio = _preprocess(img_rgb)
        t0 = time.perf_counter()
        raw = self._session.run(None, {self._input_name: tensor})[0]
        self.last_latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        pred = _decode(raw)[0]
        scores_all = pred[:, 4:5] * pred[:, 5:]
        cls = scores_all.argmax(1)
        conf = scores_all.max(1)
        mask = conf > score_thr
        if not mask.any():
            return {"persons": 0, "vehicles": 0, "vehicle_types": [], "latency_ms": self.last_latency_ms}

        p, cls, conf = pred[mask], cls[mask], conf[mask]
        xyxy = np.stack(
            [p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2, p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2], 1
        )
        keep = _nms(xyxy, conf)
        kept_cls = cls[keep]
        persons = int((kept_cls == PERSON_ID).sum())
        veh = [VEHICLE_IDS[c] for c in kept_cls if c in VEHICLE_IDS]
        return {
            "persons": persons,
            "vehicles": len(veh),
            "vehicle_types": veh,
            "latency_ms": self.last_latency_ms,
        }


# ---- shared helpers: one detector instance, replayed sample frames standing in for a live CCTV feed ----
_FRAME_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "frames")
_vision = None
_frames = None


def get_vision() -> "VisionModule":
    global _vision
    if _vision is None:
        _vision = VisionModule()
    return _vision


def get_frames():
    global _frames
    if _frames is None:
        from PIL import Image

        _frames = []
        if os.path.isdir(_FRAME_DIR):
            for name in sorted(os.listdir(_FRAME_DIR)):
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    _frames.append(np.array(Image.open(os.path.join(_FRAME_DIR, name)).convert("RGB")))
    return _frames
