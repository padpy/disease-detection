"""SQLite-backed registry of ZeroTier members managed by the broker.

The store is the single source of truth. The reconciler diffs this table
against the live controller member list and applies authorize/deauthorize
calls to converge.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS peer (
  peer_id        TEXT PRIMARY KEY,
  kind           TEXT NOT NULL CHECK (kind IN ('server', 'user')),
  assigned_ip    TEXT NOT NULL,
  node_id        TEXT NOT NULL,
  created_at     TEXT NOT NULL,
  revoked_at     TEXT
);
-- Uniqueness only applies to live peers so that a revoked peer's IP / node id
-- can be reused by a new registration without deleting the audit row.
CREATE UNIQUE INDEX IF NOT EXISTS peer_active_ip
    ON peer(assigned_ip) WHERE revoked_at IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS peer_active_node_id
    ON peer(node_id) WHERE revoked_at IS NULL;
CREATE INDEX IF NOT EXISTS peer_active_idx
    ON peer(revoked_at) WHERE revoked_at IS NULL;
"""


@dataclass(frozen=True)
class Peer:
    peer_id: str
    kind: str            # 'server' | 'user'
    assigned_ip: str
    node_id: str
    created_at: datetime
    revoked_at: Optional[datetime]


class PeerStore:
    def __init__(self, db_path: Path | str):
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def upsert(self, *, peer_id: str, kind: str, assigned_ip: str, node_id: str) -> Peer:
        now = _utcnow()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO peer (peer_id, kind, assigned_ip, node_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(peer_id) DO UPDATE SET
                    kind = excluded.kind,
                    assigned_ip = excluded.assigned_ip,
                    node_id = excluded.node_id,
                    revoked_at = NULL
                """,
                (peer_id, kind, assigned_ip, node_id, _iso(now)),
            )
        return self.get(peer_id)  # type: ignore[return-value]

    def get(self, peer_id: str) -> Optional[Peer]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM peer WHERE peer_id = ?", (peer_id,)).fetchone()
        return _row_to_peer(row) if row else None

    def revoke(self, peer_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE peer SET revoked_at = ? WHERE peer_id = ? AND revoked_at IS NULL",
                (_iso(_utcnow()), peer_id),
            )
            return cur.rowcount > 0

    def list_active(self) -> list[Peer]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM peer WHERE revoked_at IS NULL ORDER BY created_at"
            ).fetchall()
        return [_row_to_peer(r) for r in rows]

    def used_ips(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT assigned_ip FROM peer WHERE revoked_at IS NULL"
            ).fetchall()
        return {r["assigned_ip"] for r in rows}


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _row_to_peer(row: sqlite3.Row) -> Peer:
    return Peer(
        peer_id=row["peer_id"],
        kind=row["kind"],
        assigned_ip=row["assigned_ip"],
        node_id=row["node_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        revoked_at=datetime.fromisoformat(row["revoked_at"]) if row["revoked_at"] else None,
    )
