"""SAM 3 / SAM 3.1 segmenters for wheat-head + leaf masks.

Two classes share the same checkpoint:

  * ``Sam3Segmenter`` — Promptable Visual Segmentation (PVS) via the
    ``Sam3Tracker`` head. Accepts point + box prompts and returns one
    mask per prompted object.
  * ``Sam3TextSegmenter`` — Promptable Concept Segmentation (PCS) via
    the base ``Sam3Model`` head. Accepts a free-text concept (e.g.
    ``"wheat head"``) and returns every instance the model finds.

The wheat pipeline uses the tracker head to mask YOLO detections, then
the text head twice — once at default confidence to reject YOLO
false positives, once at high confidence to recover YOLO false
negatives. See ``application._reconcile_with_text``.

Encode-once + decode-many is handled implicitly: each segmenter batches
every detection's prompts into a single ``model(**inputs)`` call. SAM
runs the image encoder once and the prompt encoder / mask decoder per
object inside that one forward pass, so this is equivalent to running
the encoder once and the decoder N times the way the mobile pipeline
does — without us needing to drive the two halves separately.

Why HuggingFace transformers and not Meta's ``sam3`` package? SAM 3.1
was released as a video-multiplex enhancement (``Object Multiplex``)
on top of SAM 3. Meta's release notes report image-side metrics
unchanged between SAM 3 and SAM 3.1, and the only published 3.1
checkpoint (``sam3.1_multiplex.pt``) is laid out for the multiplex
video stack — the image model does not load it. The HF integration
exposes the same image weights through ``Sam3Model`` /
``Sam3TrackerModel`` from ``facebook/sam3`` without the extra git
install or CUDA pinning the ``sam3`` package wants. So for image-only
inference (which is what this server does) we stay on transformers.

Image size: every image we hand to the model is resized to fit
inside 1024×1024 (longest-edge preserving aspect ratio) before being
passed to SAM. SAM's processor does its own internal resize to the
training resolution, but capping our working image at 1024 keeps the
encoder receptive field aligned with the resolution everything else
in the pipeline (mobile parity, YOLO letterbox math, normalised
polygon export) is calibrated against, and bounds the wall-time
cost on very large captures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


# Official Meta release on HuggingFace. Gated — the host needs an HF
# token with the licence accepted (``huggingface-cli login`` and visit
# https://huggingface.co/facebook/sam3 to agree to the terms).
#
# This is the "SAM 3" image checkpoint; SAM 3.1's published checkpoint
# is video-multiplex only and is not loadable by the HF image model.
# See module docstring.
SAM3_MODEL_ID = "facebook/sam3"

# Working image size: every input is resized so its longest edge is
# at most this many pixels (aspect ratio preserved). 1024 keeps us
# aligned with the mobile SAM pipeline's working frame.
WORKING_IMAGE_SIZE = 1024


def _select_device(requested: Optional[str], *, role: str) -> str:
    """Pick a torch device for a SAM 3 head.

    If the caller passes an explicit device we honour it — and raise if
    they asked for CUDA but the runtime can't see it, so a misconfigured
    deploy fails loudly at boot instead of silently running ~450× slower
    on CPU. With ``device=None`` we auto-detect: CUDA when available,
    CPU otherwise. The CPU fallback prints a loud WARNING because in
    every production setup we've seen, ending up on CPU is a bug — see
    ``application._reconcile_with_text`` perf comments — not a choice.
    """
    import torch  # noqa: WPS433 - lazy import, matches caller modules

    if requested is not None:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Sam3 {role} requested device={requested!r} but "
                f"torch.cuda.is_available() is False. Install a CUDA "
                f"torch build or unset CUDA_VISIBLE_DEVICES."
            )
        return requested
    if torch.cuda.is_available():
        return "cuda"
    print(
        f"[sam3] WARNING: CUDA not available; SAM 3 {role} will run on "
        f"CPU (~30-90s per image vs ~0.2s on GPU). Install a CUDA "
        f"torch build or check CUDA_VISIBLE_DEVICES."
    )
    return "cpu"


def _resize_to_1024(image_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """Resize ``image_rgb`` so its longest edge is at most
    ``WORKING_IMAGE_SIZE`` (aspect ratio preserved).

    Returns the resized image and the ``scale`` applied
    (output_px / input_px), so callers can re-map prompt coords from
    the caller's pixel space into the resized space, and the
    resulting masks / bboxes back out.
    """
    h, w = image_rgb.shape[:2]
    long_edge = max(h, w)
    if long_edge <= WORKING_IMAGE_SIZE:
        return image_rgb, 1.0
    import cv2  # noqa: WPS433 - kept local to avoid hard dep at import

    scale = WORKING_IMAGE_SIZE / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale


@dataclass
class Sam3Prompt:
    """One object's prompt. Either ``point``, ``bbox``, or both must be
    set — SAM accepts a foreground point + a box for the same instance
    and they reinforce each other, which is the prompt pair the wheat
    pipeline uses."""

    point: Optional[Tuple[float, float]] = None
    bbox: Optional[Tuple[float, float, float, float]] = None

    def __post_init__(self) -> None:
        if self.point is None and self.bbox is None:
            raise ValueError("Sam3Prompt needs at least a point or a bbox")


class Sam3Segmenter:
    """Wrapper around ``Sam3TrackerModel`` + ``Sam3TrackerProcessor``.

    Exposes a single ``segment(image, prompts)`` method so callers stay
    out of the HF prompt-tensor packing convention. Coordinates in
    ``prompts`` are in the same pixel space as ``image`` (working
    image for the wheat pipeline).
    """

    def __init__(
        self,
        model_id: str = SAM3_MODEL_ID,
        device: Optional[str] = None,
    ):
        # Import lazily so the module stays importable in environments
        # without torch/transformers (e.g. running unit tests on the
        # storage layer alone).
        import torch  # noqa: WPS433
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor

        self.device = _select_device(device, role="tracker")
        self.model = (
            Sam3TrackerModel.from_pretrained(model_id).eval().to(self.device)
        )
        self.processor = Sam3TrackerProcessor.from_pretrained(model_id)
        self._torch = torch
        print(f"[sam3] tracker loaded on device={self.device}")

    def segment(
        self,
        image_rgb: np.ndarray,
        prompts: Sequence[Sam3Prompt],
    ) -> List[np.ndarray]:
        """Return one binary mask per prompt at ``image_rgb`` resolution.

        Prompts that pass both ``point`` and ``bbox`` get the standard
        SAM box-and-point treatment (label=1 foreground point +
        x1y1x2y2 box). Empty list short-circuits.
        """
        return [m for m, _ in self.segment_with_scores(image_rgb, prompts)]

    def segment_with_scores(
        self,
        image_rgb: np.ndarray,
        prompts: Sequence[Sam3Prompt],
    ) -> List[Tuple[np.ndarray, float]]:
        """Like ``segment`` but also returns SAM 3's predicted IoU score
        per mask — the model's own confidence in its segmentation.
        """
        if not prompts:
            return []
        import cv2  # noqa: WPS433
        torch = self._torch

        orig_h, orig_w = image_rgb.shape[:2]
        # Cap input at 1024×1024 (longest edge) before SAM ever sees
        # it; downscale prompt coords by the same factor.
        resized, scale = _resize_to_1024(image_rgb)
        pil = Image.fromarray(resized)

        has_points = any(p.point is not None for p in prompts)
        has_boxes = any(p.bbox is not None for p in prompts)

        # Per the processor signature, shapes are
        #   input_points: [image, object, points_per_object, 2]
        #   input_labels: [image, object, points_per_object]
        #   input_boxes:  [image, object, 4]
        # With one image and ``n`` objects (one point + one box each).
        input_points = None
        input_labels = None
        input_boxes = None

        if has_points:
            input_points = [[
                ([[float(p.point[0]) * scale, float(p.point[1]) * scale]]
                 if p.point is not None else [[0.0, 0.0]])
                for p in prompts
            ]]
            # SAM convention: 1 = foreground, 0 = background,
            # -1 = padding (so the slot doesn't influence the mask).
            input_labels = [[
                [1 if p.point is not None else -1]
                for p in prompts
            ]]
        if has_boxes:
            input_boxes = [[
                ([float(p.bbox[0]) * scale, float(p.bbox[1]) * scale,
                  float(p.bbox[2]) * scale, float(p.bbox[3]) * scale]
                 if p.bbox is not None else [0.0, 0.0, 0.0, 0.0])
                for p in prompts
            ]]

        inputs = self.processor(
            images=pil,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            # multimask_output=False — we already have a strong box
            # prompt per object, so we want a single committed mask
            # rather than three ambiguity candidates.
            outputs = self.model(**inputs, multimask_output=False)

        # ``pred_masks`` is logits at the SAM internal resolution.
        # ``post_process_masks`` resizes back to ``original_sizes`` and
        # thresholds. Returns a list-per-image; we passed one image.
        per_image_masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]
        # ``iou_scores`` shape: [batch, num_objects, num_masks_per_object].
        # One image, ``multimask_output=False`` => index [0, i, 0].
        iou_scores = outputs.iou_scores.detach().cpu().numpy()
        # Shape after post_process: [num_objects, num_masks_per_object, H, W]
        # at the resized image resolution. Resize back to the caller's
        # original pixel grid so polygon export aligns with the rest
        # of the pipeline.
        out: List[Tuple[np.ndarray, float]] = []
        for i in range(per_image_masks.shape[0]):
            mask = per_image_masks[i, 0].numpy().astype(np.uint8)
            if mask.shape != (orig_h, orig_w):
                mask = cv2.resize(
                    mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                )
            out.append((mask, float(iou_scores[0, i, 0])))
        return out


@dataclass
class Sam3TextDetection:
    """One instance returned by ``Sam3TextSegmenter.detect``.

    ``mask`` is a uint8 binary mask at the input image's resolution;
    ``bbox`` is xyxy in image-pixel coords; ``score`` is the
    instance-level confidence the model assigned to the concept match.
    """

    mask: np.ndarray
    bbox: Tuple[float, float, float, float]
    score: float


class Sam3TextSegmenter:
    """Promptable Concept Segmentation via ``Sam3Model``.

    Given an image and a text concept (e.g. ``"wheat head"``), returns
    every instance the model thinks matches the concept above the
    confidence threshold. Used by the wheat pipeline to cross-check
    YOLO detections against a language-grounded second opinion.
    """

    def __init__(
        self,
        model_id: str = SAM3_MODEL_ID,
        device: Optional[str] = None,
    ):
        # See ``Sam3Segmenter`` for the same lazy-import rationale.
        import torch  # noqa: WPS433
        from transformers import Sam3Model, Sam3Processor

        self.device = _select_device(device, role="text")
        self.model = (
            Sam3Model.from_pretrained(model_id).eval().to(self.device)
        )
        self.processor = Sam3Processor.from_pretrained(model_id)
        self._torch = torch
        print(f"[sam3] text head loaded on device={self.device}")

    def detect(
        self,
        image_rgb: np.ndarray,
        text: str,
        *,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> List[Sam3TextDetection]:
        """Run text-prompted instance segmentation.

        ``threshold`` filters the per-instance confidence; only
        detections above this survive. ``mask_threshold`` is the cutoff
        applied to the predicted mask logits during binarisation.
        """
        import cv2  # noqa: WPS433
        torch = self._torch

        orig_h, orig_w = image_rgb.shape[:2]
        resized, scale = _resize_to_1024(image_rgb)
        pil = Image.fromarray(resized)

        inputs = self.processor(
            images=pil, text=text, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        raw_masks = results["masks"]
        raw_boxes = results["boxes"]
        raw_scores = results["scores"]
        # The processor can return either tensors or lists depending on
        # how many instances survived the threshold. Normalise to a
        # Python list of (mask, bbox, score). Boxes are returned at
        # the resized resolution; lift them and the masks back to the
        # caller's original coords.
        out: List[Sam3TextDetection] = []
        inv_scale = 1.0 / scale if scale != 0 else 1.0
        for mask_t, box_t, score_t in zip(raw_masks, raw_boxes, raw_scores):
            mask = mask_t.cpu().numpy() if hasattr(mask_t, "cpu") else np.asarray(mask_t)
            mask = (mask > 0).astype(np.uint8)
            if mask.shape != (orig_h, orig_w):
                mask = cv2.resize(
                    mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                )
            box = (box_t.cpu().tolist() if hasattr(box_t, "cpu") else list(box_t))
            score = float(score_t.cpu().item() if hasattr(score_t, "cpu") else score_t)
            out.append(Sam3TextDetection(
                mask=mask,
                bbox=(
                    float(box[0]) * inv_scale,
                    float(box[1]) * inv_scale,
                    float(box[2]) * inv_scale,
                    float(box[3]) * inv_scale,
                ),
                score=score,
            ))
        return out
