"""SAM (efficient) segmenter prompted by detection centroids.

Wraps the encoder + decoder ONNX pair the mobile app already ships
(``sam3_efficient_encoder.onnx`` / ``sam3_efficient_decoder.onnx``).
Filenames are a label — any compatible SAM ONNX export works.

The expected I/O contract is the standard SAM one:

    encoder:
        input:  image           float32[1, 3, 1024, 1024]
        output: image_embeddings

    decoder:
        inputs: image_embeddings, point_coords, point_labels,
                mask_input, has_mask_input, orig_im_size
        output: masks (logits at original image size)

Encoder runs once per image (expensive); decoder runs cheaply per
prompt so we can batch over all the detection centroids without
re-encoding.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except ImportError:  # pragma: no cover - import error surfaced at construction
    ort = None


SAM_INPUT_SIZE = 1024
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class _Letterbox:
    """Scale + paste used to produce the encoder input. Decoder prompts
    must be transformed by the same `scale` so they land on the encoded
    region of the 1024×1024 canvas."""

    scale: float
    pad_x: float
    pad_y: float
    orig_h: int
    orig_w: int


class SamSegmenter:
    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        providers: Optional[Sequence[str]] = None,
    ):
        if ort is None:
            raise RuntimeError(
                "onnxruntime is not installed; add 'onnxruntime' to requirements.txt"
            )
        if not os.path.isfile(encoder_path):
            raise FileNotFoundError(encoder_path)
        if not os.path.isfile(decoder_path):
            raise FileNotFoundError(decoder_path)

        providers = list(providers) if providers else ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(encoder_path, providers=providers)
        self.decoder = ort.InferenceSession(decoder_path, providers=providers)
        self._encoder_input = self.encoder.get_inputs()[0].name

    # ----- encode -----

    def encode(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, _Letterbox]:
        """Run the encoder once. Returns (embedding, letterbox-state).

        The letterbox-state captures the resize that produced the 1024×1024
        encoder input — needed to transform prompt coords later.
        """
        tensor, lb = self._prepare_encoder_input(image_rgb)
        outputs = self.encoder.run(None, {self._encoder_input: tensor})
        return outputs[0], lb

    def _prepare_encoder_input(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, _Letterbox]:
        h, w = image_rgb.shape[:2]
        long_edge = max(h, w)
        scale = SAM_INPUT_SIZE / float(long_edge)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((SAM_INPUT_SIZE, SAM_INPUT_SIZE, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        norm = canvas.astype(np.float32) / 255.0
        norm = (norm - _IMAGENET_MEAN) / _IMAGENET_STD
        # HWC -> CHW -> NCHW
        tensor = np.transpose(norm, (2, 0, 1))[None, ...].astype(np.float32)
        return tensor, _Letterbox(scale=scale, pad_x=0.0, pad_y=0.0, orig_h=h, orig_w=w)

    # ----- decode -----

    def predict(
        self,
        embedding: np.ndarray,
        letterbox: _Letterbox,
        points: Sequence[Tuple[float, float]],
        point_labels: Sequence[int],
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> np.ndarray:
        """Decode a binary mask for one set of prompts.

        `points` and `bbox` are in the *original* image's pixel space. We
        apply the encoder's letterbox transform here so callers can stay
        in image coordinates throughout.
        """
        if not points and bbox is None:
            raise ValueError("SAM decoder requires at least one prompt")

        coords: List[Tuple[float, float]] = []
        labels: List[float] = []
        for (px, py), lbl in zip(points, point_labels):
            coords.append((px * letterbox.scale + letterbox.pad_x,
                           py * letterbox.scale + letterbox.pad_y))
            labels.append(float(lbl))
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            coords.append((x1 * letterbox.scale + letterbox.pad_x,
                           y1 * letterbox.scale + letterbox.pad_y))
            coords.append((x2 * letterbox.scale + letterbox.pad_x,
                           y2 * letterbox.scale + letterbox.pad_y))
            # SAM box convention: label 2 = top-left, label 3 = bottom-right.
            labels.extend([2.0, 3.0])

        point_coords = np.array(coords, dtype=np.float32)[None, ...]
        point_labels_arr = np.array(labels, dtype=np.float32)[None, ...]
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.zeros((1,), dtype=np.float32)
        orig_im_size = np.array(
            [letterbox.orig_h, letterbox.orig_w], dtype=np.float32
        )

        outputs = self.decoder.run(
            None,
            {
                "image_embeddings": embedding,
                "point_coords": point_coords,
                "point_labels": point_labels_arr,
                "mask_input": mask_input,
                "has_mask_input": has_mask_input,
                "orig_im_size": orig_im_size,
            },
        )
        # Standard SAM ONNX returns masks first; logits, threshold at 0.
        mask_logits = outputs[0]
        # Mask shape is typically [1, 1, H, W]; squeeze defensively.
        mask = np.squeeze(mask_logits)
        if mask.ndim != 2:
            # Some exports include the multi-mask dim — take the top one.
            mask = mask.reshape(-1, mask.shape[-2], mask.shape[-1])[0]
        binary = (mask > 0).astype(np.uint8)
        if binary.shape != (letterbox.orig_h, letterbox.orig_w):
            binary = cv2.resize(
                binary,
                (letterbox.orig_w, letterbox.orig_h),
                interpolation=cv2.INTER_NEAREST,
            )
        return binary
