# On-device wheat-head pipeline

The Flutter app runs a two-stage on-device pipeline in
[`lib/services/wheat_head_pipeline.dart`](../../lib/services/wheat_head_pipeline.dart):

1. **YOLO26** detects wheat-head bounding boxes on the captured image.
2. **SAM-efficient** turns each box into a tight per-instance mask, using
   the box as a prompt (SAM's two-corner prompt encoding, labels `2.0` /
   `3.0`).

All ONNX files below are gitignored and must be dropped in this folder
before building.

```
assets/models/yolo26_wheat_head.onnx              # wheat detector
assets/models/sam3_efficient_encoder.onnx         # SAM image encoder
assets/models/sam3_efficient_decoder.onnx         # SAM prompt decoder
assets/models/yolo11_grape_leaf_seg.onnx          # grape leaf segmenter
assets/models/swinv2_grape_leaf_classifier.onnx   # grape leaf classifier
```

## YOLO26 detector

**Input**

- `images` — `float32[1, 3, 640, 640]`, RGB, scaled to `[0, 1]` (no
  ImageNet mean/std). The pipeline letterboxes the working image into 640
  with zero padding and undoes the letterbox when reading boxes.

**Output** — the parser accepts any of the standard Ultralytics shapes for
a single-class export:

- `[1, 5, N]` raw head: `cx, cy, w, h, conf` per anchor.
- `[1, N, 5]` transposed.
- `[1, N, 6]` post-NMS: `x1, y1, x2, y2, conf, cls`.

A single-class fine-tune is expected (class = wheat head). Multi-class
exports work too, but only the conf channel is consumed.

## SAM-efficient encoder

- Input: `image` — `float32[1, 3, 1024, 1024]`, ImageNet mean/std
  normalized, longest-edge resize + zero pad to 1024.
- Output: `image_embeddings` — `float32[1, 256, 64, 64]`, cached for the
  decoder calls below.

## SAM-efficient decoder (box prompt)

The pipeline drives the decoder with **box prompts**, not point prompts:

- `image_embeddings` — embedding from the encoder.
- `point_coords` — `float32[1, 2, 2]` carrying the box's top-left and
  bottom-right corner in the encoder's 1024-letterbox coordinate space.
- `point_labels` — `float32[1, 2]` = `[2.0, 3.0]` (SAM's box-corner
  labels).
- `mask_input` — `float32[1, 1, 256, 256]`, all zeros.
- `has_mask_input` — `float32[1]` = `0.0`.
- `orig_im_size` — `float32[2]` = `[H, W]` of the working image.
- Output (first): mask logits at the working image resolution; positive
  values are foreground.

If your decoder export emits low-resolution masks instead of orig-size
masks, the pipeline nearest-neighbour upsamples to `(H, W)` — no schema
changes needed.

## Why two SAM files?

SAM-family models split the work so the expensive encoder runs once per
photo, while every box prompt only re-runs the small decoder. The
pipeline encodes once per image and reuses the embedding for every
detection.

## Grape-leaf pipeline

Driven from [`lib/services/wheat_head_pipeline.dart`](../../lib/services/wheat_head_pipeline.dart)
(`runGrapeLeaf` + `analyzeDisease(grapeLeaf)`). Mirrors the server-side
`leaf` task in [`server/app/application.py`](../../../server/app/application.py):

1. **YOLO11-seg** locates leaf bounding boxes (its mask prototypes are
   discarded; SAM produces tighter masks).
2. **SAM3** refines each bbox into a per-leaf mask using the same
   per-crop point + box prompt setup as the wheat pipeline.
3. **Filters**: leaves whose mask covers less than 1/20 of the working
   image area, or whose Laplacian-variance focus score is below the
   floor in `wheat_head_pipeline.dart`, are dropped before classification.
4. **SwinV2** (`Healthy-Leaf` / `Downy-Leaf` / `Powdery-Leaf`) classifies
   the masked crop. The label flows into `FhbReport.severity` so the
   existing per-instance disease overlay + distribution chart render
   unchanged.

Generated from the checkpoints in `server/models/` via:

```bash
tools/.venv/bin/python tools/export_yolo_onnx.py \
  --checkpoint ../server/models/leaf-yolo11m-seg.pt \
  --output assets/models/yolo11_grape_leaf_seg.onnx

tools/.venv/bin/python tools/export_swinv2_onnx.py \
  --checkpoint ../server/models/swinv2-tiny-patch4-window8-256 \
  --output assets/models/swinv2_grape_leaf_classifier.onnx
```

### YOLO11 grape-leaf segmenter

- Input: `images` — `float32[1, 3, 640, 640]`, RGB scaled to `[0, 1]`
  (no ImageNet normalization), letterboxed.
- Output `output0` — `float32[1, 37, 8400]`: per-anchor `cx, cy, w, h,
  conf` (single class) followed by 32 mask coefficients.
- Output `output1` — `float32[1, 32, 160, 160]`: mask prototypes. Final
  per-instance mask = `sigmoid(coeffs · prototypes)` clipped by the
  bbox.

### SwinV2 grape-leaf classifier

- Input: `pixel_values` — `float32[1, 3, 224, 224]`, RGB scaled to
  `[0, 1]` (matches the server's `transforms.functional.to_tensor` +
  `Resize((224, 224))` — no ImageNet mean/std).
- Output: `logits` — `float32[1, 3]`. `argmax` indexes into:

  ```
  0 → Healthy-Leaf
  1 → Downy-Leaf
  2 → Powdery-Leaf
  ```
