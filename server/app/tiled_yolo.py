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


CROP_BBOX_PADDING = 32


def pack_into_crops(
    detections: Sequence[Detection],
    image_w: int,
    image_h: int,
    crop_size: int = 1024,
    bbox_padding: int = CROP_BBOX_PADDING,
) -> List[tuple[tuple[int, int, int, int], List[Detection]]]:
    """Group detections into ``crop_size × crop_size`` windows so SAM
    sees each instance at native resolution instead of a global
    downscale.

    Each crop is guaranteed to fully contain the padded bbox of every
    detection assigned to it — `bbox_padding` extra pixels on every side
    of the YOLO box. That margin is the whole point of this step:
    YOLO's spike-head boxes are often tight on the kernel cluster and
    don't include the awns, so a tight crop would clip the structure
    SAM needs to see to draw a correct mask.

    Greedy clustering: pick the highest-scoring uncovered detection as
    an anchor, iteratively absorb other detections whose padded bbox
    can co-exist with the anchor's inside a single `crop_size` window,
    place the window centred on that union (clamped to image bounds
    while still containing the union), then drop the included
    detections and repeat.

    Returns a list of `((x0, y0, x1, y1), detections_in_window)` tuples.
    Each detection appears in exactly one window. Windows may overlap —
    that's fine, SAM runs independently on each.
    """
    if not detections:
        return []

    def padded(d: Detection) -> tuple[float, float, float, float]:
        return (
            d.x1 - bbox_padding,
            d.y1 - bbox_padding,
            d.x2 + bbox_padding,
            d.y2 + bbox_padding,
        )

    # If the image already fits inside one crop, one window covers it all.
    if image_w <= crop_size and image_h <= crop_size:
        return [((0, 0, image_w, image_h), list(detections))]

    uncovered = sorted(detections, key=lambda d: -d.score)
    crops: List[tuple[tuple[int, int, int, int], List[Detection]]] = []

    while uncovered:
        anchor = uncovered[0]
        ax1, ay1, ax2, ay2 = padded(anchor)
        # Grow the cluster greedily: a candidate joins iff the union of
        # padded bboxes still fits inside one crop_size window in both
        # axes. Iterating uncovered in score order means we prefer to
        # group with the next-best detections.
        ux1, uy1, ux2, uy2 = ax1, ay1, ax2, ay2
        cluster = [anchor]
        for d in uncovered[1:]:
            dx1, dy1, dx2, dy2 = padded(d)
            new_ux1 = min(ux1, dx1)
            new_uy1 = min(uy1, dy1)
            new_ux2 = max(ux2, dx2)
            new_uy2 = max(uy2, dy2)
            if (new_ux2 - new_ux1) <= crop_size and (new_uy2 - new_uy1) <= crop_size:
                ux1, uy1, ux2, uy2 = new_ux1, new_uy1, new_ux2, new_uy2
                cluster.append(d)

        # Centre the window on the union, then clamp so it (a) still
        # contains the union and (b) doesn't run off the image. The
        # union/window arithmetic uses image-space pixel coords.
        center_x = (ux1 + ux2) / 2.0
        center_y = (uy1 + uy2) / 2.0
        x0 = int(round(center_x - crop_size / 2.0))
        y0 = int(round(center_y - crop_size / 2.0))

        if image_w <= crop_size:
            x0, x1 = 0, image_w
        else:
            # Keep the union inside the window: x0 must satisfy
            #   x0 <= ux1   and   x0 + crop >= ux2
            x0 = max(int(np.ceil(ux2 - crop_size)), x0)
            x0 = min(int(np.floor(ux1)), x0)
            x0 = max(0, min(image_w - crop_size, x0))
            x1 = x0 + crop_size

        if image_h <= crop_size:
            y0, y1 = 0, image_h
        else:
            y0 = max(int(np.ceil(uy2 - crop_size)), y0)
            y0 = min(int(np.floor(uy1)), y0)
            y0 = max(0, min(image_h - crop_size, y0))
            y1 = y0 + crop_size

        # Final membership check uses the actual window — clamping above
        # is conservative but we still want to be precise about who is in.
        inside = []
        for d in uncovered:
            dx1, dy1, dx2, dy2 = padded(d)
            if dx1 >= x0 and dy1 >= y0 and dx2 <= x1 and dy2 <= y1:
                inside.append(d)
        if not inside:
            # Anchor's padded bbox is bigger than `crop_size` (rare — only
            # when a single detection is huge). Take it alone; SAM will
            # still see the clipped padded region.
            inside = [anchor]

        crops.append(((x0, y0, x1, y1), inside))
        covered = {id(d) for d in inside}
        uncovered = [d for d in uncovered if id(d) not in covered]
    return crops


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
