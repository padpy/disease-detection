import os
import time
import uuid
import threading
from application_interface import ApplicationInterface
from ultralytics import YOLO
from classification import Classification
from PIL import Image
import numpy as np
import cv2
from scipy import ndimage

from services import StorageService, SqliteStorageService
from tiled_yolo import (
    Detection,
    tiled_detect,
    pack_into_crops,
    TILE_SIZE,
    TILE_STRIDE,
    CROP_BBOX_PADDING,
)
from sam_segmenter import SAM_INPUT_SIZE, SamSegmenter
from sam3_segmenter import (
    SAM3_MODEL_ID,
    Sam3Prompt,
    Sam3Segmenter,
    Sam3TextSegmenter,
)
from wheat_head_detector import WheatHeadDetector

LEAF_MODEL = "models/leaf-yolo11m-seg.pt"
# Same YOLO26 ONNX export the Flutter app loads via onnxruntime — see
# `mobile-app/lib/services/wheat_head_pipeline.dart`. Running the same
# weights with the same letterbox + parser keeps the server and on-device
# wheat detections bit-for-bit comparable.
WHEAT_HEAD_ONNX = "models/yolo26-wheat-head.onnx"
# Legacy SAM ONNX (mobile-style encoder + decoder). Kept for the leaf
# pipeline's per-crop encode/decode pattern; if the files aren't
# present we fall back to rect-over-bbox masks like before.
SAM_ENCODER = "models/sam3_efficient_encoder.onnx"
SAM_DECODER = "models/sam3_efficient_decoder.onnx"

# Mobile-parity wheat-head pipeline constants (see
# `wheat_head_pipeline.dart`). The working image is the longest-edge-1024
# resize that every prompt, mask, and stored normalised coord references.
# The detector itself owns the 640-letterbox; the YOLO input size lives
# inside ``WheatHeadDetector``.
WHEAT_WORKING_LONG_EDGE = SAM_INPUT_SIZE
WHEAT_CONF_THRESHOLD = 0.25
WHEAT_NMS_IOU = 0.5
WHEAT_MAX_DETECTIONS = 100

# Text-coverage dedupe (runs inside ``_reconcile_with_text`` between
# pass-1 and pass-2). For each high-conf text hit, any YOLO mask where
# the text mask covers more than this fraction of the YOLO mask's
# area (``intersection / yolo_mask_area``) is dropped and replaced
# with the text mask. The relation is asymmetric on purpose: we trust
# the text head's whole-object segmentation over a YOLO box that only
# captured a sliver of the same wheat head. Several YOLO masks may
# collapse into one text mask — e.g. two boxes carving the spike body
# and the awns separately, both ≥75% inside the text mask, fold to
# one detection.
WHEAT_TEXT_COVERAGE_MIN = 0.75

# Two-pass SAM 3.1 text reconciliation (see ``_reconcile_with_text``).
# Pass 1 — default conf — runs ``"wheat spike"`` PCS over the working
# image and treats any YOLO mask without overlap with a default-conf
# text mask as a false positive (rejected).
# Pass 2 — high conf — re-runs PCS at a stricter threshold and adds
# any text-derived mask that doesn't overlap a surviving YOLO mask,
# recovering wheat spikes the YOLO detector missed.
# ``WHEAT_TEXT_OVERLAP_MIN`` is the IoU floor used for "the masks
# overlap". 0.1 is permissive on purpose — the goal is to filter
# masks that are completely unrelated to the language opinion, not to
# enforce tight agreement.
# These values are kept in lockstep with
# ``server/notebooks/reconcile_pipeline.ipynb`` — the notebook is the
# tuning surface for the reconcile pass, and detections-per-image
# agreement between the two paths depends on the constants matching.
WHEAT_TEXT_PROMPT = "wheat spike"
WHEAT_TEXT_DEFAULT_CONF = 0.35
WHEAT_TEXT_HIGH_CONF = 0.35
WHEAT_TEXT_OVERLAP_MIN = 0.1
# The text head's image encoder scales ~quadratically with input
# area; running it at 768-longest-edge instead of the 1024 working
# image cuts reconcile wall time by ~1.8x. The PCS overlap floor
# (``WHEAT_TEXT_OVERLAP_MIN``) is loose enough that the noisier
# masks still resolve the overlap test.
WHEAT_TEXT_WORKING_LONG_EDGE = 768

class Application(ApplicationInterface):
    def __init__(self,
                 image_folder="images",
                 storage_service: StorageService = None,
                 db_path: str = "data/gopher_eye.sqlite3",
                 load_models: bool = True):

        self.image_folder = image_folder
        os.makedirs(self.image_folder, exist_ok=True)

        self.storage: StorageService = storage_service or SqliteStorageService(db_path=db_path)
        self._plants = self.storage.load_plants()
        self._trials = self.storage.load_trials()

        if load_models:
            self.segmentation = YOLO(LEAF_MODEL)
            # Wheat-head detector is the same ONNX export the mobile
            # pipeline runs on-device. See ``wheat_head_detector.py`` for
            # the parser; the runtime defaults to CPUExecutionProvider so
            # the server stays portable.
            self.wheat_head = WheatHeadDetector(WHEAT_HEAD_ONNX)
            label2id = {'Healthy-Leaf': 0, 'Downy-Leaf': 1, 'Powdery-Leaf': 2}
            id2label = {0: 'Healthy-Leaf', 1: 'Downy-Leaf', 2: 'Powdery-Leaf'}
            self.classification = Classification("models/swinv2-tiny-patch4-window8-256", label2id=label2id, id2label=id2label)
            # Legacy SAM ONNX (only used by the leaf path). Optional —
            # if the files aren't present, the leaf pipeline falls back
            # to rect-over-bbox masks.
            self.sam = self._try_load_sam()
            # SAM 3.1 box-prompted segmenter — the wheat-head
            # pipeline's per-detection mask source. Pulled from
            # ``facebook/sam3.1`` via Meta's ``sam3`` package. The gated
            # repos require an accepted licence on the HF token the
            # runtime has access to; loading failures degrade the
            # wheat path to rect masks (same fallback as ``self.sam``).
            self.sam3 = self._try_load_sam3()
            # SAM 3.1 text head — same checkpoint, exposed via
            # ``Sam3Processor.set_text_prompt``. Drives the two-pass
            # ``"wheat head"`` PCS reconciliation. Optional: if it
            # fails to load, the pipeline runs without text
            # reconciliation and the YOLO output is taken at face
            # value.
            self.sam3_text = self._try_load_sam3_text()
        else:
            self.segmentation = None
            self.wheat_head = None
            self.classification = None
            self.sam = None
            self.sam3 = None
            self.sam3_text = None

    @staticmethod
    def _try_load_sam():
        if not (os.path.isfile(SAM_ENCODER) and os.path.isfile(SAM_DECODER)):
            return None
        try:
            return SamSegmenter(SAM_ENCODER, SAM_DECODER)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[application] SAM load failed, falling back to YOLO-seg: {exc}")
            return None

    @staticmethod
    def _try_load_sam3():
        try:
            return Sam3Segmenter(SAM3_MODEL_ID)
        except Exception as exc:  # pragma: no cover - defensive
            # Common reasons this fails: HF cache not mounted, token
            # not accepted on the gated repo, ``sam3`` package not
            # installed, torch/CUDA mismatch. Log the full traceback
            # so operators can diagnose; fall back to rect masks
            # rather than crashing boot.
            import traceback  # noqa: WPS433 - one-shot import at fail
            print(f"[application] SAM 3.1 load failed, falling back to rect masks: {exc}")
            traceback.print_exc()
            return None

    @staticmethod
    def _try_load_sam3_text():
        try:
            return Sam3TextSegmenter(SAM3_MODEL_ID)
        except Exception as exc:  # pragma: no cover - defensive
            import traceback  # noqa: WPS433
            print(
                f"[application] SAM 3.1 text head load failed, "
                f"reconciliation disabled: {exc}"
            )
            traceback.print_exc()
            return None

    def segment_plant(self, file, task='leaf', data=None):
        guid = str(uuid.uuid4())
        # TODO: Check if the image is valid

        try:
            with open(os.path.join(self.image_folder, f'{guid}.jpeg'), 'wb') as fs:
                fs.write(file)
        except:
            return None

        self._plants[guid] = {
            "plant_id": guid,
            "status": "pending",
            "image": f"{guid}.jpeg",
            "bounding_boxes": [],
            "masks": [],
            "labels": [],
            "trial_id": data.get("trial_id", "") if data else "",
            "datetime": data.get("datetime", "") if data else "",
            "plot_label_name": data.get("plot_label_name", "") if data else "",
            "plot_id": data.get("plot_id", "") if data else "",
            "plot_location": data.get("plot_location", "") if data else "",
            "user": data.get("user", "") if data else ""
        }

        # Start segmentation in a separate thread
        thread = threading.Thread(target=self._process_segmentation, args=(guid, task))
        thread.start()

        return guid

    def _process_segmentation(self, guid, task):
        """Dispatch to the per-task pipeline.

        - ``leaf``: full-resolution tiled YOLO + per-crop SAM. Phones
          shoot at 4032×3024+, so the leaf detector needs to see tiles
          at native scale to find small targets near the edges.
        - ``spike``: mobile-parity wheat-head pipeline (1024-longest-
          edge working image → YOLO26 at 640 → single SAM encode →
          per-box SAM decode). See ``_process_wheat_head``.
        """
        if task == 'spike':
            return self._process_wheat_head(guid)
        return self._process_leaf(guid)

    def _process_leaf(self, guid):
        """Leaf pipeline:

        1. Slide a 1280×1280 window across the image with stride 640 and
           run YOLO detection on each tile (boxes only — we drop masks).
        2. NMS-merge the cross-tile detections so a target straddling a
           tile boundary collapses back to one box.
        3. For each surviving box, take the centroid and prompt SAM to
           produce the actual segmentation mask. Falls back to YOLO-seg's
           own masks if SAM isn't loaded (no ONNX files present).
        """
        image_path = os.path.join(self.image_folder, f'{guid}.jpeg')
        image = self.read_image(image_path)
        h, w = image.shape[:2]

        detections = tiled_detect(
            image,
            self.segmentation,
            tile=TILE_SIZE,
            stride=TILE_STRIDE,
        )

        # Bounding boxes are stored normalised (matching the old contract
        # that the mobile app reads from `bounding_boxes`).
        self._plants[guid]["bounding_boxes"] = [
            [d.x1 / w, d.y1 / h, d.x2 / w, d.y2 / h] for d in detections
        ]

        masks = self._segment_from_detections(image, detections)
        # Masks are persisted as normalised polygon points for parity with
        # the previous YOLO-seg output (`results.masks.xyn`). Empty masks
        # — e.g. SAM produced nothing for a centroid — are stored as `[]`
        # so the index lines up with `bounding_boxes`.
        self._plants[guid]["masks"] = [
            self._mask_to_normalised_polygon(m, w, h) for m in masks
        ]

        for mask in masks:
            if mask is None or mask.sum() == 0:
                self._plants[guid]["labels"].append("Unknown")
                continue
            mask_chw = mask[:, :, np.newaxis]
            subimage = self._crop_image(image, mask_chw)
            cropped_mask = self._crop_image(mask_chw, mask_chw)
            label = self.classification.classify(cropped_mask * subimage)
            self._plants[guid]["labels"].append(label)

        self._plants[guid]["status"] = "complete"
        self.storage.save_plant(self._plants[guid])

    def _process_wheat_head(self, guid):
        """Mobile-parity wheat-head pipeline.

        Mirrors ``mobile-app/lib/services/wheat_head_pipeline.dart``:

          1. Decode the capture and resize so its longest edge equals
             1024 — the working image. This is the canonical reference
             frame for every prompt, mask, and stored normalised coord.
             SAM's decoder math assumes the encoder's input is exactly
             1024 on the long side; the on-device pipeline produces a
             working image at that scale and we do the same here so the
             two paths can be diffed.
          2. Run YOLO26 (wheat-head detector) on the working image at
             ``imgsz=640``. Ultralytics handles letterboxing into the
             640 canvas and reports xyxy in working-image coords,
             exactly matching the Dart pipeline's ``_parseYoloOutput``
             remap.
          3. Encode the working image **once** with SAM. The encoder's
             longest-edge-to-1024 paste uses the same scale + zero-pad
             as the mobile encoder, so the embedding is interchangeable.
          4. For each detection prompt the SAM decoder with the box's
             centroid (label 1 = foreground) and the box itself (corner
             labels 2/3). SAM reconstructs the mask at working-image
             resolution.

        Bounding boxes / masks are stored as normalised coords against
        the working image; uniform-scale resize means
        ``x / working_w == x / original_w`` so the existing contract
        the rest of the server consumes is unchanged.
        """
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        image_path = os.path.join(self.image_folder, f'{guid}.jpeg')
        full_image = self.read_image(image_path)
        fh, fw = full_image.shape[:2]
        print(f"[wheat][{guid[:8]}] decode {fw}x{fh} dt={time.perf_counter() - t0:.3f}s")

        t0 = time.perf_counter()
        working = self._resize_longest_edge(
            full_image, WHEAT_WORKING_LONG_EDGE
        )
        h, w = working.shape[:2]
        print(f"[wheat][{guid[:8]}] resize {w}x{h} dt={time.perf_counter() - t0:.3f}s")

        t0 = time.perf_counter()
        detections = self._wheat_head_detect(working)
        print(
            f"[wheat][{guid[:8]}] yolo n={len(detections)} "
            f"dt={time.perf_counter() - t0:.3f}s"
        )

        t0 = time.perf_counter()
        masks = self._wheat_head_masks(working, detections)
        print(
            f"[wheat][{guid[:8]}] sam3.1-masks n={len(masks)} "
            f"dt={time.perf_counter() - t0:.3f}s"
        )

        # Two-pass SAM 3 text reconciliation: drop YOLO false
        # positives that don't agree with a "wheat head" PCS at
        # default conf, then add high-conf PCS instances YOLO missed.
        # Done before the normalised-bbox/polygon dump so both lists
        # stay aligned with the final detection set.
        t0 = time.perf_counter()
        pre_n = len(detections)
        detections, masks = self._reconcile_with_text(
            working, detections, masks
        )
        print(
            f"[wheat][{guid[:8]}] reconcile {pre_n}->{len(detections)} "
            f"dt={time.perf_counter() - t0:.3f}s"
        )

        t0 = time.perf_counter()
        self._plants[guid]["bounding_boxes"] = [
            [d.x1 / w, d.y1 / h, d.x2 / w, d.y2 / h] for d in detections
        ]
        self._plants[guid]["masks"] = [
            self._mask_to_normalised_polygon(m, w, h) for m in masks
        ]
        print(
            f"[wheat][{guid[:8]}] polygon-dump n={len(masks)} "
            f"dt={time.perf_counter() - t0:.3f}s"
        )

        t0 = time.perf_counter()
        scored = 0
        for mask in masks:
            if mask is None or mask.sum() == 0:
                self._plants[guid]["labels"].append("Unknown")
                continue
            mask_chw = mask[:, :, np.newaxis]
            subimage = self._crop_image(working, mask_chw)
            cropped_mask = self._crop_image(mask_chw, mask_chw)
            self._plants[guid]["labels"].append(
                self._score_spike(subimage, cropped_mask)
            )
            scored += 1
        print(
            f"[wheat][{guid[:8]}] score-spike n={scored}/{len(masks)} "
            f"dt={time.perf_counter() - t0:.3f}s"
        )

        t0 = time.perf_counter()
        self._plants[guid]["status"] = "complete"
        self.storage.save_plant(self._plants[guid])
        print(f"[wheat][{guid[:8]}] save dt={time.perf_counter() - t0:.3f}s")

        print(
            f"[wheat][{guid[:8]}] TOTAL n={len(detections)} "
            f"dt={time.perf_counter() - t_total:.3f}s"
        )

    def _wheat_head_detect(self, working):
        """Single-pass YOLO26 ONNX on the working image.

        Returns image-space ``Detection`` objects in working-image
        coordinates. Delegates to the same ONNX runner the mobile app
        uses (``WheatHeadDetector``), so the conf/IoU thresholds, the
        letterbox math, and the parsed boxes match the on-device path.
        """
        raw = self.wheat_head.detect(
            working,
            score_threshold=WHEAT_CONF_THRESHOLD,
            nms_iou=WHEAT_NMS_IOU,
            max_detections=WHEAT_MAX_DETECTIONS,
        )
        return [
            Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=s)
            for (x1, y1, x2, y2, s) in raw
        ]

    def _wheat_head_masks(self, working, detections):
        """SAM 3.1 masks for each wheat-head detection.

        Each detection contributes one ``Sam3Prompt`` carrying the
        YOLO bbox; SAM 3.1's batched ``predict_inst`` takes one box
        per object and runs the image encoder once for the whole
        batch, so this is functionally the "encode-once, decode-many"
        pattern the mobile pipeline uses without us having to drive
        the two halves separately.

        Falls back to rect-over-bbox masks if SAM 3.1 isn't loaded so
        the rest of the pipeline (scoring, polygon export) still has
        something to operate on.
        """
        if not detections:
            return []
        h, w = working.shape[:2]
        if self.sam3 is None:
            fallback = []
            for d in detections:
                m = np.zeros((h, w), dtype=np.uint8)
                x1 = int(max(0, d.x1))
                y1 = int(max(0, d.y1))
                x2 = int(min(w, d.x2))
                y2 = int(min(h, d.y2))
                m[y1:y2, x1:x2] = 1
                fallback.append(m)
            return fallback

        prompts = [
            Sam3Prompt(
                point=(d.centroid[0], d.centroid[1]),
                bbox=(d.x1, d.y1, d.x2, d.y2),
            )
            for d in detections
        ]
        masks = self.sam3.segment(working, prompts)
        # Defensive: ``Sam3Segmenter.segment`` already resizes masks
        # back to the input resolution, but a future processor-config
        # change could regress this — nearest-resize so polygon
        # export stays well-defined.
        out = []
        for m in masks:
            if m.shape == (h, w):
                out.append(m)
            else:
                out.append(cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST))
        return out

    def _reconcile_with_text(self, working, detections, masks):
        """Cross-check YOLO detections against SAM 3 "wheat head" PCS.

        One PCS forward over a downscaled copy of the working image,
        then two filters on the same hits:

          1. Default-conf set (``WHEAT_TEXT_DEFAULT_CONF``). Any YOLO
             mask that does NOT overlap (mask IoU >
             ``WHEAT_TEXT_OVERLAP_MIN``) at least one PCS instance is
             dropped — the language model is unwilling to agree it's
             a wheat head, so we treat it as a YOLO false positive.
          2. High-conf subset (``WHEAT_TEXT_HIGH_CONF``). Any PCS
             instance that does NOT overlap a surviving YOLO mask is
             added as a new detection — these are wheat heads YOLO
             missed but SAM is confident about.

        Two perf tricks vs. the naive implementation:

          * The text head is run **once** at the lower threshold; the
            high-conf set is just a score-filter of the same hits.
            Halves encoder forwards.
          * The text image is downscaled to
            ``WHEAT_TEXT_WORKING_LONG_EDGE`` (768) before being fed
            to SAM. The encoder dominates wall time and scales with
            input area, so this is another ~1.8x. Output masks +
            bboxes are upscaled back into ``working`` coords so the
            IoU compare and pass-2 additions stay in one space.

        Falls through unmodified if the text head failed to load
        (``self.sam3_text is None``) or no detections came in.
        """
        if self.sam3_text is None or not detections:
            return detections, masks

        wh, ww = working.shape[:2]
        text_working = self._resize_longest_edge(
            working, WHEAT_TEXT_WORKING_LONG_EDGE
        )
        th, tw = text_working.shape[:2]
        sx = ww / float(tw)
        sy = wh / float(th)

        t0 = time.perf_counter()
        hits = self.sam3_text.detect(
            text_working,
            WHEAT_TEXT_PROMPT,
            threshold=WHEAT_TEXT_DEFAULT_CONF,
        )
        # Lift hit masks + bboxes from the 768-text frame back into
        # the 1024-working frame so ``_mask_iou`` and the
        # ``Detection`` we may emit in pass 2 line up with the YOLO
        # side. Mutating in place is fine — these dataclass instances
        # are local to this call.
        for hit in hits:
            if hit.mask.shape != (wh, ww):
                hit.mask = cv2.resize(
                    hit.mask, (ww, wh), interpolation=cv2.INTER_NEAREST
                )
            x1, y1, x2, y2 = hit.bbox
            hit.bbox = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        high_hits = [h for h in hits if h.score >= WHEAT_TEXT_HIGH_CONF]
        print(
            f"[wheat] reconcile.text hits={len(hits)} "
            f"high={len(high_hits)} "
            f"dt={time.perf_counter() - t0:.3f}s"
        )

        # Pass 1 — reject YOLO false positives.
        if hits:
            default_masks = [hit.mask for hit in hits]
            keep_idx = [
                i for i, m in enumerate(masks)
                if any(
                    self._mask_iou(m, dm) > WHEAT_TEXT_OVERLAP_MIN
                    for dm in default_masks
                )
            ]
            if len(keep_idx) != len(detections):
                print(
                    f"[wheat] text-pass-1 dropped "
                    f"{len(detections) - len(keep_idx)}/{len(detections)} "
                    f"YOLO detections without text overlap"
                )
            detections = [detections[i] for i in keep_idx]
            masks = [masks[i] for i in keep_idx]

        # Pass 1.5 — text-coverage replacement. For each high-conf
        # text hit, find YOLO masks where the text mask covers
        # ``> WHEAT_TEXT_COVERAGE_MIN`` of the YOLO mask's area
        # (asymmetric: intersection / yolo_mask_area). Drop those
        # YOLO masks and emit the text mask once. Multiple YOLO masks
        # can collapse into a single text mask — e.g. two boxes
        # carving the spike body and the awns of one wheat head, both
        # ≥75% inside the same text mask. Done before pass-2 so the
        # high-conf-add step naturally skips any text hit already
        # emitted here (its mask is now in ``masks`` and the IoU
        # check below trips at 1.0).
        if high_hits:
            drop_idx: set[int] = set()
            replacement_pairs: list[tuple[Detection, np.ndarray]] = []
            # Higher-scored hits get first claim on overlapping YOLO
            # masks. Score-sort here is independent of the input order
            # we got the hits in.
            for hit in sorted(high_hits, key=lambda h: -h.score):
                h_bool = hit.mask.astype(bool)
                h_count = int(h_bool.sum())
                if h_count == 0:
                    continue
                covered: list[int] = []
                for i, m in enumerate(masks):
                    if i in drop_idx or m is None:
                        continue
                    m_bool = m.astype(bool)
                    yolo_area = int(m_bool.sum())
                    if yolo_area == 0:
                        continue
                    inter = int(np.logical_and(h_bool, m_bool).sum())
                    if inter / float(yolo_area) > WHEAT_TEXT_COVERAGE_MIN:
                        covered.append(i)
                if covered:
                    drop_idx.update(covered)
                    x1, y1, x2, y2 = hit.bbox
                    replacement_pairs.append((
                        Detection(
                            x1=float(x1), y1=float(y1),
                            x2=float(x2), y2=float(y2),
                            score=float(hit.score),
                        ),
                        hit.mask,
                    ))
            if drop_idx or replacement_pairs:
                survivors = [
                    (d, m) for i, (d, m) in enumerate(zip(detections, masks))
                    if i not in drop_idx
                ]
                detections = (
                    [d for d, _ in survivors]
                    + [d for d, _ in replacement_pairs]
                )
                masks = (
                    [m for _, m in survivors]
                    + [m for _, m in replacement_pairs]
                )
                print(
                    f"[wheat] text-coverage replaced {len(drop_idx)} YOLO "
                    f"masks with {len(replacement_pairs)} text masks"
                )

        # Pass 2 — add missed wheat heads (high-conf only).
        added = 0
        for hit in high_hits:
            if any(self._mask_iou(hit.mask, m) > WHEAT_TEXT_OVERLAP_MIN
                   for m in masks):
                continue
            x1, y1, x2, y2 = hit.bbox
            detections.append(Detection(
                x1=float(x1), y1=float(y1),
                x2=float(x2), y2=float(y2),
                score=float(hit.score),
            ))
            masks.append(hit.mask)
            added += 1
        if added:
            print(f"[wheat] text-pass-2 added {added} missed wheat heads")
        return detections, masks

    @staticmethod
    def _mask_iou(a, b):
        """Binary mask IoU. Returns 0 for differing shapes."""
        if a is None or b is None:
            return 0.0
        if a.shape != b.shape:
            return 0.0
        a_bool = a.astype(bool)
        b_bool = b.astype(bool)
        inter = int(np.logical_and(a_bool, b_bool).sum())
        if inter == 0:
            return 0.0
        union = int(np.logical_or(a_bool, b_bool).sum())
        return inter / float(union) if union > 0 else 0.0

    @staticmethod
    def _resize_longest_edge(image, target):
        """Uniform-scale resize so ``max(h, w) == target``. Matches
        ``WheatHeadPipeline.resizeLongestEdge`` in the mobile code:
        SAM's decoder reconstructs masks against a 1024-letterbox, so
        the working image must already be at 1024 on the long side
        before it ever reaches the encoder."""
        h, w = image.shape[:2]
        long_edge = max(h, w)
        if long_edge == target:
            return image
        scale = target / float(long_edge)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _segment_from_detections(self, image, detections):
        """Produce one binary mask per detection.

        Detections are packed into ~1024×1024 image crops (matching the
        SAM encoder's native input size) so we encode at the model's
        actual receptive field instead of a downscaled global view.
        Each crop is sized so it fully contains every assigned
        detection's padded bbox — that margin prevents clipping the
        spike's full extent (e.g. awns past the YOLO box) before SAM
        ever sees it. The encoder runs once per crop; the decoder runs
        once per detection within that crop.

        Falls back to rectangular masks over the bbox if SAM isn't
        loaded, so the rest of the pipeline still has something to
        crop against.
        """
        if not detections:
            return []
        h, w = image.shape[:2]
        if self.sam is None:
            fallback = []
            for d in detections:
                m = np.zeros((h, w), dtype=np.uint8)
                x1, y1, x2, y2 = (int(max(0, d.x1)), int(max(0, d.y1)),
                                  int(min(w, d.x2)), int(min(h, d.y2)))
                m[y1:y2, x1:x2] = 1
                fallback.append(m)
            return fallback

        # Pack the post-NMS detections into crops of the SAM input size.
        # `mask_by_detection` keys are `id(detection)` so the original
        # order of `detections` is preserved on the way out.
        crops = pack_into_crops(
            detections,
            image_w=w,
            image_h=h,
            crop_size=SAM_INPUT_SIZE,
            bbox_padding=CROP_BBOX_PADDING,
        )
        mask_by_detection: dict[int, np.ndarray] = {}

        for (x0, y0, x1, y1), members in crops:
            crop_img = image[y0:y1, x0:x1]
            ch, cw = crop_img.shape[:2]
            embedding, letterbox = self.sam.encode(crop_img)
            for d in members:
                # Translate prompts into crop-local coords; SAM emits a
                # mask at `crop_img` resolution, which we then paste
                # back into a full-image-sized zero buffer.
                cx, cy = d.centroid
                local_centroid = (cx - x0, cy - y0)
                local_bbox = (d.x1 - x0, d.y1 - y0,
                              d.x2 - x0, d.y2 - y0)
                local_mask = self.sam.predict(
                    embedding=embedding,
                    letterbox=letterbox,
                    points=[local_centroid],
                    point_labels=[1],
                    bbox=local_bbox,
                )
                full = np.zeros((h, w), dtype=np.uint8)
                full[y0:y0 + ch, x0:x0 + cw] = local_mask
                mask_by_detection[id(d)] = full

        return [mask_by_detection[id(d)] for d in detections]

    @staticmethod
    def _mask_to_normalised_polygon(mask, width, height):
        """Convert a binary mask to the normalised polygon format the
        mobile app already consumes (`results.masks.xyn` from Ultralytics:
        a list of [x, y] pairs in 0..1). We use the largest external
        contour — multi-component masks are rare for instance segmentation
        and choosing the largest matches Ultralytics' own behaviour."""
        if mask is None:
            return []
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return []
        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 3:
            return []
        pts = largest.reshape(-1, 2).astype(np.float32)
        pts[:, 0] /= width
        pts[:, 1] /= height
        return pts.tolist()

    def read_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _score_spike(self, image, mask, alpha=0.81):
        masked_image = image * mask

        b = masked_image[:,:,0]
        g = masked_image[:,:,1]

        bg = alpha * b - g
        bg[bg < 0] = 0
        bg[bg > 0] = 1

        labeled_array, num_features = ndimage.label(bg)
        min_area = 0.0005 * masked_image.shape[0] * masked_image.shape[1]

        filtered_bg = np.zeros_like(bg)

        large_regions = 0

        for label in range(1, num_features+1):
            region = labeled_array == label
            if np.sum(region) > min_area:
                filtered_bg[region] = 1
                large_regions += 1


        return f'FHB: {np.sum(filtered_bg) / np.sum(mask):.2f}'

    def _points_to_mask(self, points, img_shape):
        mask = np.zeros(img_shape[:-1], dtype=np.uint8)
        points = points * np.flip(img_shape[:-1])
        points = points.reshape(-1, 1, 2).astype(int)

        cv2.fillPoly(mask, [points], 1)
        return mask[:,:, np.newaxis]

    def _get_mask_bounding_box(self, mask):
        mask = mask.squeeze()
        y, x = np.where(mask)
        return np.min(x), np.min(y), np.max(x), np.max(y)

    def _crop_image(self, image, mask):
        x1, y1, x2, y2 = self._get_mask_bounding_box(mask)
        return image[y1:y2, x1:x2]

    def record_plant(self, data):
        """Persist a plant record. Kept for backward compatibility with callers
        that wrote to the JSONL file directly; new code should prefer
        ``self.storage.save_plant``."""
        self.storage.save_plant(data)

    def plant_status(self, plant_id):
        if plant_id in self._plants:
            return self._plants[plant_id]["status"]
        return "plant not found"

    def plant_data(self, plant_id):
        response = {}
        if plant_id in self._plants:
            response = self._plants[plant_id]

        return response

    def get_image(self, plant_id, image_name):
        # TODO: This needs a custom error message
        try:
            image_file_path = os.path.join(self.image_folder, self._plants[plant_id][image_name].rstrip())
            return open(image_file_path, 'rb'), "image/png" if ("png" in image_file_path) else "image/jpeg"
        except:
            return None, None

    def get_plant_ids(self):
        return list(self._plants.keys())

    def get_trials(self):
        return self._trials

    def create_trial(self, trial_data):
        trial_id = str(uuid.uuid4())
        trial_data["trial_id"] = trial_id

        self._trials[trial_id] = trial_data
        self.storage.save_trial(trial_data)

        return trial_id
