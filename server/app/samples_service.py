"""Server-side counterpart to ``mobile-app/lib/services/sample_repository.dart``.

The mobile app captures images, runs detection on-device, and stores the
results in a local sqflite DB (samples / sample_instances / chat_messages).
This service exposes the same data shapes over HTTP so the mobile app — or any
other client — can sync samples up to the server, fetch them back, and append
chat turns. Storage is delegated to ``StorageService.sqlite``; original
captures (large JPEGs) are kept on disk under ``source_dir``, while the
working image / overlays / instance masks stay as BLOB columns to mirror the
mobile schema 1:1.
"""

from __future__ import annotations

import base64
import binascii
import os
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

from services import StorageService
from services.storage_service import (
    INSTANCE_BLOB_FIELDS,
    INSTANCE_REQUIRED_BLOBS,
    SAMPLE_BLOB_FIELDS,
    normalize_sample,
)


SUPPORTED_DETECTION_MODES = {"wheat_fhb", "grape_leaf"}

SAMPLE_BLOB_NAMES = SAMPLE_BLOB_FIELDS  # exposed as URL-friendly aliases below
SAMPLE_BLOB_ALIASES = {
    "working_image": "working_image_png",
    "working_image_png": "working_image_png",
    "disease_overlay": "disease_overlay_png",
    "disease_overlay_png": "disease_overlay_png",
    "segmentation_overlay": "segmentation_overlay_png",
    "segmentation_overlay_png": "segmentation_overlay_png",
}

INSTANCE_BLOB_ALIASES = {
    "mask": "mask_png",
    "mask_png": "mask_png",
    "preview": "preview_png",
    "preview_png": "preview_png",
    "disease_preview": "disease_preview_png",
    "disease_preview_png": "disease_preview_png",
}


class SamplesError(Exception):
    """Maps to a structured JSON error in the API layer."""

    def __init__(self, message: str, status_code: int = 400, code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code

    def to_payload(self) -> Dict[str, Any]:
        return {"error": {"message": self.message, "code": self.code}}


class SamplesService:
    def __init__(self, storage: StorageService, source_dir: str = "data/sample_sources"):
        self.storage = storage
        self.source_dir = source_dir
        os.makedirs(self.source_dir, exist_ok=True)

    # ---------------------- samples ----------------------

    def list_samples(
        self,
        user: Optional[str] = None,
        detection_mode: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None,
        include_blobs: Iterable[str] = (),
    ) -> List[Dict[str, Any]]:
        if detection_mode is not None and detection_mode not in SUPPORTED_DETECTION_MODES:
            raise SamplesError(f"unsupported detection_mode: {detection_mode}", code="invalid_filter")
        rows = self.storage.list_samples(
            user=user, detection_mode=detection_mode, since=since, limit=limit
        )
        wanted = self._resolve_sample_blob_kinds(include_blobs)
        return [self._serialize_sample(row, wanted) for row in rows]

    def get_sample(self, sample_id: int, include_blobs: Iterable[str] = ()) -> Optional[Dict[str, Any]]:
        row = self.storage.get_sample(int(sample_id))
        if row is None:
            return None
        wanted = self._resolve_sample_blob_kinds(include_blobs)
        return self._serialize_sample(row, wanted)

    def create_sample(self, image_bytes: bytes, metadata: Dict[str, Any], filename: Optional[str] = None) -> Dict[str, Any]:
        if not image_bytes:
            raise SamplesError("image is required", code="missing_image")

        detection_mode = metadata.get("detection_mode", "wheat_fhb")
        if detection_mode not in SUPPORTED_DETECTION_MODES:
            raise SamplesError(
                f"unsupported detection_mode: {detection_mode}",
                code="invalid_detection_mode",
            )

        suffix = self._safe_suffix(filename)
        guid = uuid.uuid4().hex
        rel_path = f"{guid}{suffix}"
        abs_path = os.path.join(self.source_dir, rel_path)
        with open(abs_path, "wb") as fp:
            fp.write(image_bytes)

        record = normalize_sample({
            "file_path": rel_path,
            "taken_at": metadata.get("taken_at") or int(time.time() * 1000),
            "latitude": metadata.get("latitude"),
            "longitude": metadata.get("longitude"),
            "accuracy": metadata.get("accuracy"),
            "detection_mode": detection_mode,
            "user": metadata.get("user", ""),
            "trial_id": metadata.get("trial_id", ""),
            "collection_id": metadata.get("collection_id"),
            "qr_id": metadata.get("qr_id"),
            "qr_line": metadata.get("qr_line"),
            "qr_rep": metadata.get("qr_rep"),
            "qr_location": metadata.get("qr_location"),
            "qr_note": metadata.get("qr_note"),
        })
        for blob_field in SAMPLE_BLOB_FIELDS:
            if blob_field in metadata and metadata[blob_field] is not None:
                record[blob_field] = self._coerce_blob(metadata[blob_field], blob_field)
        if "working_image_w" in metadata:
            record["working_image_w"] = metadata.get("working_image_w")
        if "working_image_h" in metadata:
            record["working_image_h"] = metadata.get("working_image_h")

        sample_id = self.storage.create_sample(record)
        return self.get_sample(sample_id) or {}

    def update_sample(self, sample_id: int, partial: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        clean: Dict[str, Any] = {}
        for key, value in partial.items():
            if key in {"id", "inserted_at", "updated_at"}:
                continue
            if key == "detection_mode" and value not in SUPPORTED_DETECTION_MODES:
                raise SamplesError(
                    f"unsupported detection_mode: {value}",
                    code="invalid_detection_mode",
                )
            if key in SAMPLE_BLOB_FIELDS and value is not None:
                clean[key] = self._coerce_blob(value, key)
            else:
                clean[key] = value
        updated = self.storage.update_sample(int(sample_id), clean)
        if updated is None:
            return None
        return self._serialize_sample(updated, set())

    def delete_sample(self, sample_id: int) -> bool:
        existing = self.storage.get_sample(int(sample_id))
        if existing is None:
            return False
        deleted = self.storage.delete_sample(int(sample_id))
        if deleted:
            file_path = existing.get("file_path") or ""
            if file_path:
                full = os.path.join(self.source_dir, file_path)
                try:
                    os.remove(full)
                except OSError:
                    # The file may already be gone — match the mobile repo's
                    # tolerance for missing source files.
                    pass
        return deleted

    def get_sample_source(self, sample_id: int) -> Optional[Tuple[str, str]]:
        sample = self.storage.get_sample(int(sample_id))
        if sample is None:
            return None
        rel = sample.get("file_path") or ""
        if not rel:
            return None
        full = os.path.join(self.source_dir, rel)
        if not os.path.isfile(full):
            return None
        return full, self._guess_mimetype(full)

    def get_sample_blob(self, sample_id: int, kind: str) -> Optional[bytes]:
        column = self._sample_blob_column(kind)
        sample = self.storage.get_sample(int(sample_id))
        if sample is None:
            return None
        blob = sample.get(column)
        if blob is None:
            return None
        return bytes(blob)

    def set_sample_blob(self, sample_id: int, kind: str, blob: bytes,
                        width: Optional[int] = None, height: Optional[int] = None) -> Optional[Dict[str, Any]]:
        column = self._sample_blob_column(kind)
        partial: Dict[str, Any] = {column: blob}
        if column == "working_image_png":
            if width is not None:
                partial["working_image_w"] = width
            if height is not None:
                partial["working_image_h"] = height
        return self.update_sample(sample_id, partial)

    def clear_sample_blob(self, sample_id: int, kind: str) -> Optional[Dict[str, Any]]:
        column = self._sample_blob_column(kind)
        partial: Dict[str, Any] = {column: None}
        if column == "working_image_png":
            partial["working_image_w"] = None
            partial["working_image_h"] = None
        return self.update_sample(sample_id, partial)

    # ---------------------- instances ----------------------

    def list_instances(self, sample_id: int, include_blobs: Iterable[str] = ()) -> List[Dict[str, Any]]:
        rows = self.storage.list_instances(int(sample_id))
        wanted = self._resolve_instance_blob_kinds(include_blobs)
        return [self._serialize_instance(row, wanted) for row in rows]

    def get_instance(self, instance_id: int, include_blobs: Iterable[str] = ()) -> Optional[Dict[str, Any]]:
        row = self.storage.get_instance(int(instance_id))
        if row is None:
            return None
        wanted = self._resolve_instance_blob_kinds(include_blobs)
        return self._serialize_instance(row, wanted)

    def create_instance(self, sample_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        sample = self.storage.get_sample(int(sample_id))
        if sample is None:
            raise SamplesError("sample not found", status_code=404, code="sample_not_found")

        record = self._instance_payload_to_record(int(sample_id), payload, allow_idx=True)
        if "idx" not in record:
            existing = self.storage.list_instances(int(sample_id))
            record["idx"] = max((row["idx"] for row in existing), default=-1) + 1

        instance_id = self.storage.create_instance(record)
        return self.get_instance(instance_id) or {}

    def replace_instances(self, sample_id: int, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sample = self.storage.get_sample(int(sample_id))
        if sample is None:
            raise SamplesError("sample not found", status_code=404, code="sample_not_found")
        records = [
            self._instance_payload_to_record(int(sample_id), p, allow_idx=True, fallback_idx=i)
            for i, p in enumerate(payloads)
        ]
        rows = self.storage.replace_instances(int(sample_id), records)
        return [self._serialize_instance(row, set()) for row in rows]

    def update_instance(self, instance_id: int, partial: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        clean: Dict[str, Any] = {}
        for key, value in partial.items():
            if key in {"id", "sample_id", "created_at", "updated_at"}:
                continue
            if key in INSTANCE_BLOB_FIELDS and value is not None:
                clean[key] = self._coerce_blob(value, key)
            else:
                clean[key] = value
        updated = self.storage.update_instance(int(instance_id), clean)
        if updated is None:
            return None
        return self._serialize_instance(updated, set())

    def delete_instance(self, instance_id: int) -> bool:
        return self.storage.delete_instance(int(instance_id))

    def get_instance_blob(self, instance_id: int, kind: str) -> Optional[bytes]:
        column = self._instance_blob_column(kind)
        row = self.storage.get_instance(int(instance_id))
        if row is None:
            return None
        blob = row.get(column)
        if blob is None:
            return None
        return bytes(blob)

    # ---------------------- chat ----------------------

    def list_chat(self, instance_id: int) -> List[Dict[str, Any]]:
        rows = self.storage.list_chat_messages(int(instance_id))
        return [self._serialize_chat(row) for row in rows]

    def append_chat(self, instance_id: int, role: str, content: str) -> Dict[str, Any]:
        if self.storage.get_instance(int(instance_id)) is None:
            raise SamplesError("instance not found", status_code=404, code="instance_not_found")
        if not content:
            raise SamplesError("content is required", code="missing_content")
        try:
            new_id = self.storage.append_chat_message({
                "instance_id": int(instance_id),
                "role": role,
                "content": content,
                "created_at": int(time.time() * 1000),
            })
        except ValueError as exc:
            raise SamplesError(str(exc), code="invalid_role") from exc
        rows = self.storage.list_chat_messages(int(instance_id))
        for row in rows:
            if row["id"] == new_id:
                return self._serialize_chat(row)
        # Should be unreachable — the row we just inserted must exist.
        raise SamplesError("chat append succeeded but row missing", status_code=500, code="server_error")

    def clear_chat(self, instance_id: int) -> int:
        return self.storage.clear_chat_messages(int(instance_id))

    # ---------------------- helpers ----------------------

    @staticmethod
    def _safe_suffix(filename: Optional[str]) -> str:
        if not filename:
            return ".jpg"
        ext = os.path.splitext(filename)[1].lower()
        if ext in {".jpg", ".jpeg", ".png", ".webp", ".heic"}:
            return ext
        return ".jpg"

    @staticmethod
    def _guess_mimetype(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".heic": "image/heic",
        }.get(ext, "application/octet-stream")

    @staticmethod
    def _sample_blob_column(kind: str) -> str:
        column = SAMPLE_BLOB_ALIASES.get(kind)
        if column is None:
            raise SamplesError(
                f"unknown sample blob '{kind}'; expected one of "
                f"{sorted(set(SAMPLE_BLOB_ALIASES.values()))}",
                code="unknown_blob",
            )
        return column

    @staticmethod
    def _instance_blob_column(kind: str) -> str:
        column = INSTANCE_BLOB_ALIASES.get(kind)
        if column is None:
            raise SamplesError(
                f"unknown instance blob '{kind}'; expected one of "
                f"{sorted(set(INSTANCE_BLOB_ALIASES.values()))}",
                code="unknown_blob",
            )
        return column

    @staticmethod
    def _resolve_sample_blob_kinds(values: Iterable[str]) -> set:
        out = set()
        for v in values:
            if not v:
                continue
            if v == "*":
                return set(SAMPLE_BLOB_FIELDS)
            out.add(SamplesService._sample_blob_column(v))
        return out

    @staticmethod
    def _resolve_instance_blob_kinds(values: Iterable[str]) -> set:
        out = set()
        for v in values:
            if not v:
                continue
            if v == "*":
                return set(INSTANCE_BLOB_FIELDS)
            out.add(SamplesService._instance_blob_column(v))
        return out

    @staticmethod
    def _coerce_blob(value: Any, field: str) -> bytes:
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        if isinstance(value, memoryview):
            return value.tobytes()
        if isinstance(value, str):
            try:
                return base64.b64decode(value, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise SamplesError(
                    f"'{field}' base64 decode failed: {exc}", code="invalid_blob",
                ) from exc
        raise SamplesError(
            f"'{field}' must be bytes or a base64-encoded string", code="invalid_blob",
        )

    def _instance_payload_to_record(
        self,
        sample_id: int,
        payload: Dict[str, Any],
        allow_idx: bool = True,
        fallback_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise SamplesError("instance payload must be an object", code="invalid_payload")
        record: Dict[str, Any] = {"sample_id": sample_id}

        # Bbox can be passed flat (bbox_left/top/right/bottom) or as
        # {bbox: {left, top, right, bottom}} for ergonomic JSON.
        bbox = payload.get("bbox")
        if isinstance(bbox, dict):
            record["bbox_left"] = bbox.get("left", 0.0)
            record["bbox_top"] = bbox.get("top", 0.0)
            record["bbox_right"] = bbox.get("right", 0.0)
            record["bbox_bottom"] = bbox.get("bottom", 0.0)
        for key in ("bbox_left", "bbox_top", "bbox_right", "bbox_bottom"):
            if key in payload:
                record[key] = payload[key]

        centroid = payload.get("centroid")
        if isinstance(centroid, dict):
            record["centroid_x"] = centroid.get("x", 0.0)
            record["centroid_y"] = centroid.get("y", 0.0)
        for key in ("centroid_x", "centroid_y"):
            if key in payload:
                record[key] = payload[key]

        for key in (
            "score", "image_w", "image_h",
            "fhb_green", "fhb_necrotic", "fhb_other", "fhb_total", "fhb_ratio", "fhb_severity",
            "created_at", "updated_at",
        ):
            if key in payload:
                record[key] = payload[key]

        if allow_idx and "idx" in payload:
            record["idx"] = int(payload["idx"])
        elif fallback_idx is not None:
            record["idx"] = fallback_idx

        for blob_field in INSTANCE_BLOB_FIELDS:
            if blob_field in payload and payload[blob_field] is not None:
                record[blob_field] = self._coerce_blob(payload[blob_field], blob_field)

        for blob in INSTANCE_REQUIRED_BLOBS:
            if not record.get(blob):
                raise SamplesError(
                    f"instance.{blob} is required (provide bytes or base64 string)",
                    code="missing_blob",
                )
        return record

    @staticmethod
    def _serialize_sample(row: Dict[str, Any], include_blobs: set) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in row.items():
            if key in SAMPLE_BLOB_FIELDS:
                if key in include_blobs and value is not None:
                    out[key] = base64.b64encode(value).decode("ascii")
                else:
                    out[key] = None
                out[f"has_{key}"] = value is not None
                continue
            out[key] = value
        return out

    @staticmethod
    def _serialize_instance(row: Dict[str, Any], include_blobs: set) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in row.items():
            if key in INSTANCE_BLOB_FIELDS:
                if key in include_blobs and value is not None:
                    out[key] = base64.b64encode(value).decode("ascii")
                else:
                    out[key] = None
                out[f"has_{key}"] = value is not None
                continue
            out[key] = value
        # Convenience nested shapes that match how mobile uses Rect/Offset.
        out["bbox"] = {
            "left": row.get("bbox_left"),
            "top": row.get("bbox_top"),
            "right": row.get("bbox_right"),
            "bottom": row.get("bbox_bottom"),
        }
        out["centroid"] = {"x": row.get("centroid_x"), "y": row.get("centroid_y")}
        return out

    @staticmethod
    def _serialize_chat(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": row.get("id"),
            "instance_id": row.get("instance_id"),
            "role": row.get("role"),
            "content": row.get("content"),
            "created_at": row.get("created_at"),
        }
