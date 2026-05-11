import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

from .storage_service import (
    INSTANCE_DEFAULTS,
    INSTANCE_REQUIRED_BLOBS,
    PLANT_LIST_FIELDS,
    SAMPLE_DEFAULTS,
    StorageService,
    normalize_chat,
    normalize_instance,
    normalize_plant,
    normalize_sample,
    normalize_trial,
)

_PLANT_COLUMNS = (
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

_TRIAL_COLUMNS = (
    "trial_id",
    "trial_name",
    "datetime",
    "description",
    "user",
)

_PLANTS_DDL = """
CREATE TABLE IF NOT EXISTS plants (
    plant_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    image TEXT NOT NULL,
    bounding_boxes TEXT NOT NULL DEFAULT '[]',
    masks TEXT NOT NULL DEFAULT '[]',
    labels TEXT NOT NULL DEFAULT '[]',
    trial_id TEXT NOT NULL DEFAULT '',
    datetime TEXT NOT NULL DEFAULT '',
    plot_label_name TEXT NOT NULL DEFAULT '',
    plot_id TEXT NOT NULL DEFAULT '',
    plot_location TEXT NOT NULL DEFAULT '',
    user TEXT NOT NULL DEFAULT '',
    inserted_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
)
"""

_TRIALS_DDL = """
CREATE TABLE IF NOT EXISTS trials (
    trial_id TEXT PRIMARY KEY,
    trial_name TEXT NOT NULL,
    datetime TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    user TEXT NOT NULL DEFAULT '',
    inserted_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
)
"""

# Mirror of the mobile sqflite schema in ``sample_repository.dart``. Field
# names match mobile column names so a sync layer maps 1:1. ``user`` and
# ``trial_id`` are extras that only exist server-side (the mobile app is
# single-tenant on-device). ``updated_at`` lets ``?since=`` queries support
# incremental sync without a separate journal table.
_SAMPLES_DDL = """
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL DEFAULT '',
    taken_at INTEGER NOT NULL,
    latitude REAL,
    longitude REAL,
    accuracy REAL,
    detection_mode TEXT NOT NULL DEFAULT 'wheat_fhb',
    working_image_w INTEGER,
    working_image_h INTEGER,
    working_image_png BLOB,
    disease_overlay_png BLOB,
    segmentation_overlay_png BLOB,
    user TEXT NOT NULL DEFAULT '',
    trial_id TEXT NOT NULL DEFAULT '',
    collection_id INTEGER,
    qr_id TEXT,
    qr_line TEXT,
    qr_rep TEXT,
    qr_location TEXT,
    qr_note TEXT,
    inserted_at INTEGER NOT NULL DEFAULT (strftime('%s','now')),
    updated_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
)
"""

# Columns added after the initial samples table existed in the wild. On fresh
# DBs ``_SAMPLES_DDL`` creates them; on existing DBs they get ``ALTER TABLE``d
# in ``_migrate_samples_columns``. Keep the SQL type aligned with the DDL above.
_SAMPLES_ADDED_COLUMNS = (
    ("collection_id", "INTEGER"),
    ("qr_id", "TEXT"),
    ("qr_line", "TEXT"),
    ("qr_rep", "TEXT"),
    ("qr_location", "TEXT"),
    ("qr_note", "TEXT"),
)

_SAMPLES_TAKEN_AT_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_samples_taken_at ON samples(taken_at DESC)"
)

_SAMPLES_UPDATED_AT_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_samples_updated_at ON samples(updated_at DESC)"
)

_INSTANCES_DDL = """
CREATE TABLE IF NOT EXISTS sample_instances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id INTEGER NOT NULL,
    idx INTEGER NOT NULL,
    bbox_left REAL NOT NULL,
    bbox_top REAL NOT NULL,
    bbox_right REAL NOT NULL,
    bbox_bottom REAL NOT NULL,
    centroid_x REAL NOT NULL,
    centroid_y REAL NOT NULL,
    score REAL NOT NULL,
    image_w INTEGER NOT NULL,
    image_h INTEGER NOT NULL,
    mask_png BLOB NOT NULL,
    preview_png BLOB NOT NULL,
    fhb_green INTEGER,
    fhb_necrotic INTEGER,
    fhb_other INTEGER,
    fhb_total INTEGER,
    fhb_ratio REAL,
    fhb_severity TEXT,
    disease_preview_png BLOB,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    FOREIGN KEY (sample_id) REFERENCES samples(id) ON DELETE CASCADE
)
"""

_INSTANCES_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_instances_sample ON sample_instances(sample_id, idx)"
)

_CHAT_DDL = """
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instance_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (instance_id) REFERENCES sample_instances(id) ON DELETE CASCADE
)
"""

_CHAT_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_chat_instance "
    "ON chat_messages(instance_id, created_at ASC)"
)

_SAMPLE_COLUMNS = (
    "id",
    "file_path",
    "taken_at",
    "latitude",
    "longitude",
    "accuracy",
    "detection_mode",
    "working_image_w",
    "working_image_h",
    "working_image_png",
    "disease_overlay_png",
    "segmentation_overlay_png",
    "user",
    "trial_id",
    "collection_id",
    "qr_id",
    "qr_line",
    "qr_rep",
    "qr_location",
    "qr_note",
    "inserted_at",
    "updated_at",
)

_INSTANCE_COLUMNS = (
    "id",
    "sample_id",
    "idx",
    "bbox_left",
    "bbox_top",
    "bbox_right",
    "bbox_bottom",
    "centroid_x",
    "centroid_y",
    "score",
    "image_w",
    "image_h",
    "mask_png",
    "preview_png",
    "fhb_green",
    "fhb_necrotic",
    "fhb_other",
    "fhb_total",
    "fhb_ratio",
    "fhb_severity",
    "disease_preview_png",
    "created_at",
    "updated_at",
)

_CHAT_COLUMNS = ("id", "instance_id", "role", "content", "created_at")

# Columns the API may write through ``PATCH``. ``id``, ``inserted_at`` etc.
# are server-managed.
_SAMPLE_WRITABLE = tuple(SAMPLE_DEFAULTS.keys())
_INSTANCE_WRITABLE = tuple(k for k in INSTANCE_DEFAULTS.keys())


def _encode_plot_location(value):
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _decode_plot_location(value: str):
    if not value:
        return ""
    if value[0] in "{[\"":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


class SqliteStorageService(StorageService):
    """SQLite-backed storage.

    Single-file database; writes serialize through a process-wide lock so the
    background segmentation thread and request threads cannot collide.
    Connection uses ``check_same_thread=False`` because Application processes
    uploads on a worker thread.
    """

    def __init__(self, db_path: str = "data/gopher_eye.sqlite3"):
        self.db_path = db_path
        directory = os.path.dirname(self.db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; we wrap multi-stmt work explicitly
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        with self._lock:
            self._conn.execute(_PLANTS_DDL)
            self._conn.execute(_TRIALS_DDL)
            self._conn.execute(_SAMPLES_DDL)
            self._migrate_samples_columns_locked()
            self._conn.execute(_SAMPLES_TAKEN_AT_INDEX)
            self._conn.execute(_SAMPLES_UPDATED_AT_INDEX)
            self._conn.execute(_INSTANCES_DDL)
            self._conn.execute(_INSTANCES_INDEX)
            self._conn.execute(_CHAT_DDL)
            self._conn.execute(_CHAT_INDEX)

    def _migrate_samples_columns_locked(self) -> None:
        """Add any columns defined in ``_SAMPLES_ADDED_COLUMNS`` that are
        missing on an existing ``samples`` table. SQLite has no ``ADD COLUMN
        IF NOT EXISTS`` so we introspect ``PRAGMA table_info`` first."""
        existing = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(samples)").fetchall()
        }
        for column, sql_type in _SAMPLES_ADDED_COLUMNS:
            if column not in existing:
                self._conn.execute(
                    f"ALTER TABLE samples ADD COLUMN {column} {sql_type}"
                )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _row_to_plant(self, row: sqlite3.Row) -> dict:
        plant = {col: row[col] for col in _PLANT_COLUMNS}
        for field in PLANT_LIST_FIELDS:
            plant[field] = json.loads(plant[field]) if plant[field] else []
        plant["plot_location"] = _decode_plot_location(plant["plot_location"])
        return plant

    def _row_to_trial(self, row: sqlite3.Row) -> dict:
        return {col: row[col] for col in _TRIAL_COLUMNS}

    def load_plants(self) -> Dict[str, dict]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM plants ORDER BY inserted_at ASC, plant_id ASC"
            )
            rows = cursor.fetchall()
        return {row["plant_id"]: self._row_to_plant(row) for row in rows}

    def save_plant(self, plant: dict) -> None:
        record = normalize_plant(plant)
        params = (
            record["plant_id"],
            record["status"],
            record["image"],
            json.dumps(record["bounding_boxes"]),
            json.dumps(record["masks"]),
            json.dumps(record["labels"]),
            record["trial_id"],
            record["datetime"],
            record["plot_label_name"],
            record["plot_id"],
            _encode_plot_location(record["plot_location"]),
            record["user"],
        )
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO plants (
                    plant_id, status, image, bounding_boxes, masks, labels,
                    trial_id, datetime, plot_label_name, plot_id,
                    plot_location, user
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plant_id) DO UPDATE SET
                    status=excluded.status,
                    image=excluded.image,
                    bounding_boxes=excluded.bounding_boxes,
                    masks=excluded.masks,
                    labels=excluded.labels,
                    trial_id=excluded.trial_id,
                    datetime=excluded.datetime,
                    plot_label_name=excluded.plot_label_name,
                    plot_id=excluded.plot_id,
                    plot_location=excluded.plot_location,
                    user=excluded.user
                """,
                params,
            )

    def get_plant(self, plant_id: str) -> Optional[dict]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM plants WHERE plant_id = ?", (plant_id,)
            )
            row = cursor.fetchone()
        return self._row_to_plant(row) if row else None

    def list_plant_ids(self) -> List[str]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT plant_id FROM plants ORDER BY inserted_at ASC, plant_id ASC"
            )
            return [row["plant_id"] for row in cursor.fetchall()]

    def load_trials(self) -> Dict[str, dict]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM trials ORDER BY inserted_at ASC, trial_id ASC"
            )
            rows = cursor.fetchall()
        return {row["trial_id"]: self._row_to_trial(row) for row in rows}

    def save_trial(self, trial: dict) -> None:
        record = normalize_trial(trial)
        params = (
            record["trial_id"],
            record["trial_name"],
            record["datetime"],
            record["description"],
            record["user"],
        )
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO trials (trial_id, trial_name, datetime, description, user)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(trial_id) DO UPDATE SET
                    trial_name=excluded.trial_name,
                    datetime=excluded.datetime,
                    description=excluded.description,
                    user=excluded.user
                """,
                params,
            )

    # ---------- Samples / instances / chat ----------

    @staticmethod
    def _row_to_dict(row: sqlite3.Row, columns) -> dict:
        out: Dict[str, Any] = {}
        for col in columns:
            value = row[col]
            if isinstance(value, (bytes, memoryview)):
                # Convert sqlite blobs into bytes so json serializers can
                # base64-encode them at the API layer.
                out[col] = bytes(value)
            else:
                out[col] = value
        return out

    def list_samples(
        self,
        user: Optional[str] = None,
        detection_mode: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        clauses: List[str] = []
        params: List[Any] = []
        if user:
            clauses.append("user = ?")
            params.append(user)
        if detection_mode:
            clauses.append("detection_mode = ?")
            params.append(detection_mode)
        if since is not None:
            clauses.append("updated_at >= ?")
            params.append(int(since))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM samples {where} ORDER BY taken_at DESC, id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._lock:
            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()
        return [self._row_to_dict(row, _SAMPLE_COLUMNS) for row in rows]

    def get_sample(self, sample_id: int) -> Optional[dict]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM samples WHERE id = ?", (int(sample_id),)
            )
            row = cursor.fetchone()
        return self._row_to_dict(row, _SAMPLE_COLUMNS) if row else None

    def create_sample(self, record: dict) -> int:
        normalized = normalize_sample(record)
        now = int(time.time())
        cols = list(_SAMPLE_WRITABLE) + ["inserted_at", "updated_at"]
        values = [normalized[k] for k in _SAMPLE_WRITABLE] + [now, now]
        placeholders = ", ".join("?" for _ in cols)
        sql = f"INSERT INTO samples ({', '.join(cols)}) VALUES ({placeholders})"
        with self._lock:
            cursor = self._conn.execute(sql, values)
            return int(cursor.lastrowid)

    def update_sample(self, sample_id: int, partial: dict) -> Optional[dict]:
        fields = {k: v for k, v in partial.items() if k in _SAMPLE_WRITABLE}
        if not fields:
            return self.get_sample(sample_id)
        now = int(time.time())
        assignments = ", ".join(f"{k} = ?" for k in fields.keys())
        params = list(fields.values()) + [now, int(sample_id)]
        sql = f"UPDATE samples SET {assignments}, updated_at = ? WHERE id = ?"
        with self._lock:
            cursor = self._conn.execute(sql, params)
            if cursor.rowcount == 0:
                return None
        return self.get_sample(sample_id)

    def delete_sample(self, sample_id: int) -> bool:
        with self._lock:
            # SQLite ``ON DELETE CASCADE`` requires foreign_keys=ON (set in __init__).
            cursor = self._conn.execute(
                "DELETE FROM samples WHERE id = ?", (int(sample_id),)
            )
            return cursor.rowcount > 0

    def list_instances(self, sample_id: int) -> List[dict]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM sample_instances WHERE sample_id = ? ORDER BY idx ASC",
                (int(sample_id),),
            )
            rows = cursor.fetchall()
        return [self._row_to_dict(row, _INSTANCE_COLUMNS) for row in rows]

    def get_instance(self, instance_id: int) -> Optional[dict]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM sample_instances WHERE id = ?", (int(instance_id),)
            )
            row = cursor.fetchone()
        return self._row_to_dict(row, _INSTANCE_COLUMNS) if row else None

    def create_instance(self, record: dict) -> int:
        normalized = normalize_instance(record)
        for blob in INSTANCE_REQUIRED_BLOBS:
            if not normalized.get(blob):
                raise ValueError(f"sample_instance.{blob} is required")
        now_ms = int(time.time() * 1000)
        created_at = int(record.get("created_at") or now_ms)
        updated_at = int(record.get("updated_at") or now_ms)

        cols = list(_INSTANCE_WRITABLE) + ["created_at", "updated_at"]
        values = [normalized[k] for k in _INSTANCE_WRITABLE] + [created_at, updated_at]
        placeholders = ", ".join("?" for _ in cols)
        sql = f"INSERT INTO sample_instances ({', '.join(cols)}) VALUES ({placeholders})"
        with self._lock:
            cursor = self._conn.execute(sql, values)
            self._touch_sample_locked(int(normalized["sample_id"]))
            new_id = int(cursor.lastrowid)
        return new_id

    def replace_instances(self, sample_id: int, records: List[dict]) -> List[dict]:
        sample_id = int(sample_id)
        prepared: List[tuple] = []
        now_ms = int(time.time() * 1000)
        for i, raw in enumerate(records):
            payload = dict(raw)
            payload["sample_id"] = sample_id
            payload.setdefault("idx", i)
            normalized = normalize_instance(payload)
            for blob in INSTANCE_REQUIRED_BLOBS:
                if not normalized.get(blob):
                    raise ValueError(
                        f"sample_instances[{i}].{blob} is required"
                    )
            created_at = int(payload.get("created_at") or now_ms)
            updated_at = now_ms
            prepared.append((normalized, created_at, updated_at))

        cols = list(_INSTANCE_WRITABLE) + ["created_at", "updated_at"]
        placeholders = ", ".join("?" for _ in cols)
        sql = f"INSERT INTO sample_instances ({', '.join(cols)}) VALUES ({placeholders})"

        with self._lock:
            try:
                self._conn.execute("BEGIN")
                self._conn.execute(
                    "DELETE FROM sample_instances WHERE sample_id = ?", (sample_id,)
                )
                for normalized, created_at, updated_at in prepared:
                    values = [normalized[k] for k in _INSTANCE_WRITABLE] + [created_at, updated_at]
                    self._conn.execute(sql, values)
                self._touch_sample_locked(sample_id)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return self.list_instances(sample_id)

    def update_instance(self, instance_id: int, partial: dict) -> Optional[dict]:
        fields = {k: v for k, v in partial.items() if k in _INSTANCE_WRITABLE}
        if not fields:
            return self.get_instance(instance_id)
        now_ms = int(time.time() * 1000)
        assignments = ", ".join(f"{k} = ?" for k in fields.keys())
        params = list(fields.values()) + [now_ms, int(instance_id)]
        sql = f"UPDATE sample_instances SET {assignments}, updated_at = ? WHERE id = ?"
        with self._lock:
            cursor = self._conn.execute(sql, params)
            if cursor.rowcount == 0:
                return None
            inst = self._conn.execute(
                "SELECT sample_id FROM sample_instances WHERE id = ?",
                (int(instance_id),),
            ).fetchone()
            if inst is not None:
                self._touch_sample_locked(int(inst["sample_id"]))
        return self.get_instance(instance_id)

    def delete_instance(self, instance_id: int) -> bool:
        with self._lock:
            inst = self._conn.execute(
                "SELECT sample_id FROM sample_instances WHERE id = ?",
                (int(instance_id),),
            ).fetchone()
            if inst is None:
                return False
            self._conn.execute(
                "DELETE FROM sample_instances WHERE id = ?", (int(instance_id),)
            )
            self._touch_sample_locked(int(inst["sample_id"]))
        return True

    def list_chat_messages(self, instance_id: int) -> List[dict]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM chat_messages WHERE instance_id = ? "
                "ORDER BY created_at ASC, id ASC",
                (int(instance_id),),
            )
            rows = cursor.fetchall()
        return [self._row_to_dict(row, _CHAT_COLUMNS) for row in rows]

    def append_chat_message(self, record: dict) -> int:
        payload = dict(record)
        if not payload.get("created_at"):
            payload["created_at"] = int(time.time() * 1000)
        normalized = normalize_chat(payload)
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO chat_messages (instance_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    normalized["instance_id"],
                    normalized["role"],
                    normalized["content"],
                    normalized["created_at"],
                ),
            )
            return int(cursor.lastrowid)

    def clear_chat_messages(self, instance_id: int) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM chat_messages WHERE instance_id = ?",
                (int(instance_id),),
            )
            return cursor.rowcount

    # ---------- helpers ----------

    def _touch_sample_locked(self, sample_id: int) -> None:
        self._conn.execute(
            "UPDATE samples SET updated_at = ? WHERE id = ?",
            (int(time.time()), int(sample_id)),
        )
