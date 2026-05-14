"""Server-side YOLO26 wheat-head detector — same ONNX as the mobile app.

The Flutter pipeline in
``mobile-app/lib/services/wheat_head_pipeline.dart`` runs the same
``yolo26_wheat_head.onnx`` export. We load it via onnxruntime here so
detections on the server match the on-device pipeline byte-for-byte
(same letterbox, same conf/iou thresholds, same NMS).

The ONNX export is **post-NMS, end-to-end**: ``output0`` has shape
``[1, 300, 38]`` where each row is ``[x1, y1, x2, y2, conf, cls,
...32 mask coeffs]`` in 640-letterbox space. The mask coefficients are
dropped — SAM produces per-instance masks downstream.

Why a custom runner and not Ultralytics' ``YOLO('.onnx')``? The
Ultralytics post-processing layer expects raw ``[1, 4+nc, N]`` heads
and re-runs NMS itself; feeding it the post-NMS export silently
mis-interprets the rows. Mirroring the Dart parser line-by-line is
cheaper than wrestling the Ultralytics layer back into agreement.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except ImportError:  # pragma: no cover - import error surfaced at construction
    ort = None


# Matches the on-device pipeline constants in ``wheat_head_pipeline.dart``.
YOLO_INPUT_SIZE = 640
DEFAULT_SCORE_THRESHOLD = 0.25
DEFAULT_NMS_IOU = 0.5
DEFAULT_MAX_DETECTIONS = 100
# Columns per row in the post-NMS export: xyxy + conf + cls + 32 mask
# coefficients. The mobile parser uses the same constant.
YOLO_FEATURES = 38


@dataclass
class _Letterbox:
    """Even-padded 640 letterbox state — used to remap boxes back."""

    scale: float
    pad_x: float
    pad_y: float


class WheatHeadDetector:
    """Mobile-parity YOLO26 wheat-head detector.

    `detect()` accepts an HxWx3 RGB uint8 array (the working image at
    the same scale the mobile pipeline uses) and returns image-space
    ``(x1, y1, x2, y2, score)`` tuples. NMS and the conf threshold
    match the on-device defaults so the two paths can be diffed without
    parameter tweaks.
    """

    def __init__(
        self,
        onnx_path: str,
        providers: Optional[Sequence[str]] = None,
    ):
        if ort is None:
            raise RuntimeError(
                "onnxruntime is not installed; add 'onnxruntime-gpu' to requirements.txt"
            )
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(onnx_path)
        if providers is None:
            # Prefer GPU when the installed wheel exposes it. The default
            # `onnxruntime` wheel is CPU-only — to actually get CUDA here
            # the deploy must install `onnxruntime-gpu` against a matching
            # CUDA runtime. We always keep CPU as a fallback so a missing
            # CUDA runtime degrades gracefully instead of failing boot.
            available = set(ort.get_available_providers())
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
                print(
                    "[wheat-head] WARNING: CUDAExecutionProvider not "
                    "available; running YOLO26 on CPU. Install "
                    "'onnxruntime-gpu' with a matching CUDA runtime to "
                    "enable GPU inference."
                )
        self.session = ort.InferenceSession(onnx_path, providers=list(providers))
        print(
            f"[wheat-head] onnxruntime providers active: "
            f"{self.session.get_providers()}"
        )
        self.input_name = self.session.get_inputs()[0].name

    def detect(
        self,
        image_rgb: np.ndarray,
        *,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        nms_iou: float = DEFAULT_NMS_IOU,
        max_detections: int = DEFAULT_MAX_DETECTIONS,
    ) -> List[tuple]:
        canvas, lb = self._letterbox(image_rgb, YOLO_INPUT_SIZE)
        tensor = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))
        tensor = tensor[None, ...].astype(np.float32)
        outputs = self.session.run(None, {self.input_name: tensor})
        # Output0 is the post-NMS table. We ignore output1 (mask
        # prototypes) — SAM handles segmentation.
        return self._parse_output(
            outputs[0],
            lb,
            image_w=image_rgb.shape[1],
            image_h=image_rgb.shape[0],
            score_threshold=score_threshold,
            nms_iou=nms_iou,
            max_detections=max_detections,
        )

    @staticmethod
    def _letterbox(image: np.ndarray, target: int):
        """Aspect-preserving resize into a centred 640x640 zero canvas.

        Even (not top-left) padding — the YOLO encoder was trained on
        centred letterboxes, so we match that here. SAM's encoder uses
        top-left padding instead; the two are independent because each
        detector owns its own letterbox state.
        """
        h, w = image.shape[:2]
        scale = min(target / float(w), target / float(h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target, target, 3), dtype=np.uint8)
        pad_x = (target - new_w) // 2
        pad_y = (target - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return canvas, _Letterbox(scale=scale, pad_x=float(pad_x), pad_y=float(pad_y))

    @staticmethod
    def _parse_output(
        raw: np.ndarray,
        lb: _Letterbox,
        *,
        image_w: int,
        image_h: int,
        score_threshold: float,
        nms_iou: float,
        max_detections: int,
    ) -> List[tuple]:
        flat = np.asarray(raw, dtype=np.float32).reshape(-1, YOLO_FEATURES)
        # Score filter first — the export pads the table to 300 rows
        # with low-conf entries we need to drop.
        scores = flat[:, 4]
        keep = scores >= score_threshold
        if not np.any(keep):
            return []
        xyxy = flat[keep, :4]
        scores = scores[keep]

        # Undo letterbox into working-image space.
        xyxy = xyxy.copy()
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - lb.pad_x) / lb.scale
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - lb.pad_y) / lb.scale
        xyxy[:, 0] = np.clip(xyxy[:, 0], 0.0, image_w)
        xyxy[:, 1] = np.clip(xyxy[:, 1], 0.0, image_h)
        xyxy[:, 2] = np.clip(xyxy[:, 2], 0.0, image_w)
        xyxy[:, 3] = np.clip(xyxy[:, 3], 0.0, image_h)

        valid = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
        if not np.any(valid):
            return []
        xyxy = xyxy[valid]
        scores = scores[valid]

        # Greedy NMS — the export already does class-agnostic NMS at
        # train-time defaults, but we re-run it with the mobile
        # pipeline's iou_thresh so cross-class duplicates can't slip
        # past on multi-class fine-tunes.
        order = np.argsort(-scores)
        xyxy = xyxy[order]
        scores = scores[order]
        suppressed = np.zeros(len(xyxy), dtype=bool)
        kept: List[int] = []
        for i in range(len(xyxy)):
            if suppressed[i]:
                continue
            kept.append(i)
            if len(kept) >= max_detections:
                break
            ious = _iou_one_to_many(xyxy[i], xyxy)
            suppressed |= ious > nms_iou
            suppressed[i] = True

        return [
            (float(xyxy[i, 0]), float(xyxy[i, 1]),
             float(xyxy[i, 2]), float(xyxy[i, 3]),
             float(scores[i]))
            for i in kept
        ]


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    iw = np.clip(x2 - x1, 0.0, None)
    ih = np.clip(y2 - y1, 0.0, None)
    inter = iw * ih
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)
