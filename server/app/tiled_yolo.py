"""Tiled YOLO detection.

The original pipeline ran YOLO on the full image at whatever input size
Ultralytics chose. Wheat-spike / leaf images come from phones at 4032×3024
or larger, so the model saw heavy downsampling and lost small targets near
image edges.

This module slides a fixed-size window across the image, runs YOLO on each
tile at its native resolution, and merges the per-tile detections in
image-space via NMS. The window/stride defaults (1280, 640) give 50%
overlap so a target straddling a tile boundary is still seen whole by the
neighbouring window.

Public surface:
    tiled_detect(image, model, ...) -> List[Detection]

`Detection.xyxy` is in *image* pixels (not normalised, not tile-local).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


TILE_SIZE = 1280
TILE_STRIDE = 640


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def _tile_origins(extent: int, tile: int, stride: int) -> List[int]:
    """Origins along one axis so the last tile ends exactly at `extent`.

    If `extent <= tile` there is a single tile at 0. Otherwise we step by
    `stride` and clamp the final origin to `extent - tile` so we never
    feed YOLO a tile that runs off the image (it would have to be padded
    and produce phantom detections in the padded region).
    """
    if extent <= tile:
        return [0]
    origins = list(range(0, extent - tile, stride))
    last = extent - tile
    if origins[-1] != last:
        origins.append(last)
    return origins


def _iou_matrix(boxes: np.ndarray) -> np.ndarray:
    """Pairwise IoU for a [N, 4] xyxy array — used only inside NMS."""
    if boxes.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    ix1 = np.maximum(x1[:, None], x1[None, :])
    iy1 = np.maximum(y1[:, None], y1[None, :])
    ix2 = np.minimum(x2[:, None], x2[None, :])
    iy2 = np.minimum(y2[:, None], y2[None, :])
    iw = np.clip(ix2 - ix1, 0.0, None)
    ih = np.clip(iy2 - iy1, 0.0, None)
    inter = iw * ih
    union = areas[:, None] + areas[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def nms(detections: Sequence[Detection], iou_thresh: float = 0.5) -> List[Detection]:
    """Greedy NMS: keep the highest-scoring box, drop anything that
    overlaps it past `iou_thresh`, repeat. Sufficient for "merge the
    strongest matches when boxes overlap heavily."""
    if not detections:
        return []
    boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32)
    scores = np.array([d.score for d in detections], dtype=np.float32)
    order = np.argsort(-scores)
    iou = _iou_matrix(boxes)

    kept: List[int] = []
    suppressed = np.zeros(len(detections), dtype=bool)
    for idx in order:
        if suppressed[idx]:
            continue
        kept.append(int(idx))
        suppressed |= iou[idx] > iou_thresh
        suppressed[idx] = True
    return [detections[i] for i in kept]


def tiled_detect(
    image: np.ndarray,
    model,
    tile: int = TILE_SIZE,
    stride: int = TILE_STRIDE,
    conf: float = 0.25,
    iou: float = 0.5,
) -> List[Detection]:
    """Run YOLO over a grid of overlapping tiles, return image-space boxes.

    Args:
        image: HxWx3 RGB uint8 (cv2-style; OK if BGR — YOLO doesn't care).
        model: An Ultralytics YOLO instance. We call it as `model(tile, ...)`
            and only consume `result.boxes.xyxy` — masks are intentionally
            ignored; SAM does the segmentation downstream.
        tile, stride: Window size and step. Default 1280/640 → 50% overlap.
        conf: Per-tile confidence threshold passed to YOLO.
        iou: IoU threshold for the final cross-tile NMS merge.
    """
    h, w = image.shape[:2]
    x_origins = _tile_origins(w, tile, stride)
    y_origins = _tile_origins(h, tile, stride)

    raw: List[Detection] = []
    for y0 in y_origins:
        for x0 in x_origins:
            crop = image[y0 : y0 + tile, x0 : x0 + tile]
            # Ultralytics accepts a numpy array directly; verbose=False
            # keeps the per-tile spam out of the server logs.
            result = model(crop, conf=conf, verbose=False)[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue
            xyxy = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            for (bx1, by1, bx2, by2), s in zip(xyxy, scores):
                raw.append(
                    Detection(
                        x1=float(bx1) + x0,
                        y1=float(by1) + y0,
                        x2=float(bx2) + x0,
                        y2=float(by2) + y0,
                        score=float(s),
                    )
                )
    return nms(raw, iou_thresh=iou)
