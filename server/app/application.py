import os
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
from sam3_segmenter import SAM3_MODEL_ID, Sam3Prompt, Sam3Segmenter
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
            # Regular SAM 3 — the wheat-head pipeline's segmenter. Pulled
            # from ``facebook/sam3`` via HF transformers. The gated repo
            # requires an accepted licence on the HF token the runtime
            # has access to; loading failures degrade the wheat path to
            # rect masks (same fallback as ``self.sam``).
            self.sam3 = self._try_load_sam3()
        else:
            self.segmentation = None
            self.wheat_head = None
            self.classification = None
            self.sam = None
            self.sam3 = None

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
            # Common reasons this fails: HF cache not mounted, token not
            # accepted on the gated repo, network blocked at startup.
            # Log and fall back to rect masks rather than crashing boot.
            print(f"[application] SAM3 load failed, falling back to rect masks: {exc}")
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
        image_path = os.path.join(self.image_folder, f'{guid}.jpeg')
        full_image = self.read_image(image_path)
        working = self._resize_longest_edge(
            full_image, WHEAT_WORKING_LONG_EDGE
        )
        h, w = working.shape[:2]

        detections = self._wheat_head_detect(working)

        self._plants[guid]["bounding_boxes"] = [
            [d.x1 / w, d.y1 / h, d.x2 / w, d.y2 / h] for d in detections
        ]

        masks = self._wheat_head_masks(working, detections)
        self._plants[guid]["masks"] = [
            self._mask_to_normalised_polygon(m, w, h) for m in masks
        ]

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

        self._plants[guid]["status"] = "complete"
        self.storage.save_plant(self._plants[guid])

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
        """SAM 3 masks for each wheat-head detection.

        Each detection contributes one ``Sam3Prompt`` with the YOLO
        centroid (foreground point) and the YOLO bbox. We hand every
        prompt to the segmenter in a single batched call: SAM 3 runs
        the image encoder once and the prompt/decoder stack per object
        inside that forward pass, so this is functionally the
        "encode-once, decode-many" pattern the mobile pipeline uses,
        without us having to drive the two halves separately.

        Falls back to rect-over-bbox masks if SAM 3 isn't loaded so
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
        # Defensive: if SAM 3 returned a mask at a different resolution
        # than the working image (shouldn't happen — post_process_masks
        # resizes to original_sizes — but a malformed processor config
        # could regress this), nearest-resize back so polygon export
        # works.
        out = []
        for m in masks:
            if m.shape == (h, w):
                out.append(m)
            else:
                out.append(cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST))
        return out

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
