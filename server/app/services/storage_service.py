from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

PLANT_FIELDS = (
    "plant_id",
    "status",
    "image",
    "bounding_boxes",
    "masks",
    "labels",
    "trial_id",
    "datetime",
    "plot_label_name",
    "plot_id",
    "plot_location",
    "user",
)

PLANT_LIST_FIELDS = ("bounding_boxes", "masks", "labels")
PLANT_STRING_DEFAULTS = (
    "trial_id",
    "datetime",
    "plot_label_name",
    "plot_id",
    "plot_location",
    "user",
)

TRIAL_FIELDS = (
    "trial_id",
    "trial_name",
    "datetime",
    "description",
    "user",
)

TRIAL_STRING_DEFAULTS = ("datetime", "description", "user")


def normalize_plant(data: dict) -> dict:
    return {
        "plant_id": data["plant_id"],
        "status": data["status"],
        "image": data["image"],
        "bounding_boxes": data.get("bounding_boxes", []),
        "masks": data.get("masks", []),
        "labels": data.get("labels", []),
        "trial_id": data.get("trial_id", ""),
        "datetime": data.get("datetime", ""),
        "plot_label_name": data.get("plot_label_name", ""),
        "plot_id": data.get("plot_id", ""),
        "plot_location": data.get("plot_location", ""),
        "user": data.get("user", ""),
    }


def normalize_trial(data: dict) -> dict:
    return {
        "trial_id": data["trial_id"],
        "trial_name": data["trial_name"],
        "datetime": data.get("datetime", ""),
        "description": data.get("description", ""),
        "user": data.get("user", ""),
    }


# ---------- Sample / instance / chat schema (mirror of the mobile sqflite ----------
# The mobile app's local DB is the source of truth for the field names; see
# ``mobile-app/lib/services/sample_repository.dart`` and the model classes
# under ``mobile-app/lib/model/`` for the canonical column list. The server
# stores them with identical names so a sync layer can map 1:1.

SAMPLE_DEFAULTS: Dict[str, Any] = {
    "file_path": "",
    "taken_at": 0,
    "latitude": None,
    "longitude": None,
    "accuracy": None,
    "detection_mode": "wheat_fhb",
    "working_image_w": None,
    "working_image_h": None,
    "working_image_png": None,
    "disease_overlay_png": None,
    "segmentation_overlay_png": None,
    "user": "",
    "trial_id": "",
    # Mobile-side grouping. Collections themselves are not synced; we just
    # round-trip the id so a device pulling its own samples back can re-link
    # them to the local collection.
    "collection_id": None,
    "qr_id": None,
    "qr_line": None,
    "qr_rep": None,
    "qr_location": None,
    "qr_note": None,
}

SAMPLE_BLOB_FIELDS = (
    "working_image_png",
    "disease_overlay_png",
    "segmentation_overlay_png",
)

INSTANCE_DEFAULTS: Dict[str, Any] = {
    "sample_id": 0,
    "idx": 0,
    "bbox_left": 0.0,
    "bbox_top": 0.0,
    "bbox_right": 0.0,
    "bbox_bottom": 0.0,
    "centroid_x": 0.0,
    "centroid_y": 0.0,
    "score": 0.0,
    "image_w": 0,
    "image_h": 0,
    "mask_png": None,
    "preview_png": None,
    "fhb_green": None,
    "fhb_necrotic": None,
    "fhb_other": None,
    "fhb_total": None,
    "fhb_ratio": None,
    "fhb_severity": None,
    "disease_preview_png": None,
}

INSTANCE_BLOB_FIELDS = ("mask_png", "preview_png", "disease_preview_png")
INSTANCE_REQUIRED_BLOBS = ("mask_png", "preview_png")

CHAT_ROLES = ("user", "assistant", "system")


def normalize_sample(data: dict) -> dict:
    record: Dict[str, Any] = {key: data.get(key, default) for key, default in SAMPLE_DEFAULTS.items()}
    if "id" in data and data["id"] is not None:
        record["id"] = int(data["id"])
    if record["taken_at"] is None:
        record["taken_at"] = 0
    record["taken_at"] = int(record["taken_at"])
    if record["detection_mode"] not in {"wheat_fhb", "grape_leaf"}:
        record["detection_mode"] = "wheat_fhb"
    if record["collection_id"] is not None:
        record["collection_id"] = int(record["collection_id"])
    return record


def normalize_instance(data: dict) -> dict:
    record: Dict[str, Any] = {key: data.get(key, default) for key, default in INSTANCE_DEFAULTS.items()}
    if "id" in data and data["id"] is not None:
        record["id"] = int(data["id"])
    return record


def normalize_chat(data: dict) -> dict:
    role = data.get("role", "user")
    if role not in CHAT_ROLES:
        raise ValueError(f"invalid chat role: {role!r}")
    return {
        "instance_id": int(data["instance_id"]),
        "role": role,
        "content": data.get("content", ""),
        "created_at": int(data.get("created_at") or 0),
    }


class StorageService(ABC):
    """Persistence boundary for plant/trial records.

    Implementations are responsible for durable storage; the application layer
    keeps a working in-memory view and delegates reads/writes here. Swap
    implementations by passing a different subclass into Application.
    """

    @abstractmethod
    def load_plants(self) -> Dict[str, dict]:
        """Return all stored plants keyed by plant_id."""

    @abstractmethod
    def save_plant(self, plant: dict) -> None:
        """Insert or replace a plant record."""

    @abstractmethod
    def get_plant(self, plant_id: str) -> Optional[dict]:
        """Return a single plant by id, or None if absent."""

    @abstractmethod
    def list_plant_ids(self) -> List[str]:
        """Return all plant_ids in insertion order."""

    @abstractmethod
    def load_trials(self) -> Dict[str, dict]:
        """Return all stored trials keyed by trial_id."""

    @abstractmethod
    def save_trial(self, trial: dict) -> None:
        """Insert or replace a trial record."""

    def close(self) -> None:
        """Release any held resources. Default is a no-op."""
        return None

    # ---------- Samples / instances / chat ----------
    # These are the mobile-app-aligned tables. The legacy JSONL store does not
    # implement them; storage backends that don't support samples raise
    # ``NotImplementedError`` so the API layer can return a 501.

    def list_samples(
        self,
        user: Optional[str] = None,
        detection_mode: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        raise NotImplementedError

    def get_sample(self, sample_id: int) -> Optional[dict]:
        raise NotImplementedError

    def create_sample(self, record: dict) -> int:
        raise NotImplementedError

    def update_sample(self, sample_id: int, partial: dict) -> Optional[dict]:
        raise NotImplementedError

    def delete_sample(self, sample_id: int) -> bool:
        raise NotImplementedError

    def list_instances(self, sample_id: int) -> List[dict]:
        raise NotImplementedError

    def get_instance(self, instance_id: int) -> Optional[dict]:
        raise NotImplementedError

    def create_instance(self, record: dict) -> int:
        raise NotImplementedError

    def replace_instances(self, sample_id: int, records: List[dict]) -> List[dict]:
        raise NotImplementedError

    def update_instance(self, instance_id: int, partial: dict) -> Optional[dict]:
        raise NotImplementedError

    def delete_instance(self, instance_id: int) -> bool:
        raise NotImplementedError

    def list_chat_messages(self, instance_id: int) -> List[dict]:
        raise NotImplementedError

    def append_chat_message(self, record: dict) -> int:
        raise NotImplementedError

    def clear_chat_messages(self, instance_id: int) -> int:
        raise NotImplementedError
