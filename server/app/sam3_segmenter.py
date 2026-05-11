"""SAM 3 segmenter (``facebook/sam3``) for wheat-head + leaf masks.

Drives Meta's regular SAM 3 model via HuggingFace transformers — no
ONNX export. The ``Sam3Tracker`` head accepts visual prompts (points,
boxes) and returns per-object masks at the image's original
resolution, which matches the on-device pipeline's contract.

Encode-once + decode-many is handled implicitly: we batch every
detection's prompts into a single ``model(**inputs)`` call. SAM 3 runs
the image encoder once and the prompt encoder / mask decoder per
object inside that one forward pass, so this is equivalent to running
the encoder once and the decoder N times the way the mobile pipeline
does — without us needing to drive the two halves separately.

Why HuggingFace and not ONNX? SAM 3 ships with three subgraphs (image
encoder, language encoder, mask decoder) and the only pre-built ONNX
exports come from third-party mirrors. Loading the official Meta
weights through the ``transformers`` runtime keeps us on the trusted
release at the cost of a torch dependency — already pulled in by
Ultralytics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


# Official Meta release on HuggingFace. Gated — the host needs an HF
# token with the licence accepted (``huggingface-cli login`` and visit
# https://huggingface.co/facebook/sam3 to agree to the terms).
SAM3_MODEL_ID = "facebook/sam3"


@dataclass
class Sam3Prompt:
    """One object's prompt. Either ``point``, ``bbox``, or both must be
    set — SAM 3 accepts a foreground point + a box for the same
    instance and they reinforce each other, which is the prompt pair
    the wheat pipeline uses."""

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

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            Sam3TrackerModel.from_pretrained(model_id).eval().to(self.device)
        )
        self.processor = Sam3TrackerProcessor.from_pretrained(model_id)
        self._torch = torch

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
        if not prompts:
            return []
        torch = self._torch
        pil = Image.fromarray(image_rgb)

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
                ([[float(p.point[0]), float(p.point[1])]]
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
                ([float(p.bbox[0]), float(p.bbox[1]),
                  float(p.bbox[2]), float(p.bbox[3])]
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
        # Shape after post_process: [num_objects, num_masks_per_object, H, W].
        masks: List[np.ndarray] = []
        for i in range(per_image_masks.shape[0]):
            mask = per_image_masks[i, 0]
            masks.append(mask.numpy().astype(np.uint8))
        return masks
