"""Reconcile the peer store with the live ZeroTier controller member list.

Single-pass diff: anything active in the store but not authorized on the
controller (or assigned a different IP) is re-authorized, anything authorized
on the controller but not active in the store is deauthorized + removed.
Idempotent and crash-safe — a restart resumes from the next pass.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from . import zerotier as zt
from .peer_store import PeerStore

log = logging.getLogger(__name__)


def reconcile_once(store: PeerStore, zt_cmd: zt.ZtCommand, network_id: str) -> tuple[int, int]:
    """Run a single reconciliation pass.

    Returns (authorized, deauthorized) counts.
    """
    desired = {p.node_id: p.assigned_ip for p in store.list_active()}
    live = {m.node_id: m for m in zt.show_members(zt_cmd, network_id)}

    authorized = 0
    deauthorized = 0

    for node_id, ip in desired.items():
        current = live.get(node_id)
        if (
            current is None
            or not current.authorized
            or ip not in current.ip_assignments
        ):
            zt.authorize_member(zt_cmd, network_id, node_id=node_id, assigned_ip=ip)
            authorized += 1

    for node_id, member in live.items():
        if node_id in desired:
            continue
        if member.authorized or member.ip_assignments:
            zt.deauthorize_member(zt_cmd, network_id, node_id=node_id)
            deauthorized += 1

    if authorized or deauthorized:
        log.info("reconcile: +%d -%d members", authorized, deauthorized)
    return authorized, deauthorized


class Reconciler:
    """Background thread that calls `reconcile_once` periodically.

    Calls to `kick()` wake the loop early so writes show up on the wire in
    ~milliseconds instead of waiting out the full interval.
    """

    def __init__(
        self,
        store: PeerStore,
        zt_cmd: zt.ZtCommand,
        network_id: str,
        *,
        interval_seconds: float = 10.0,
    ):
        self._store = store
        self._zt = zt_cmd
        self._network_id = network_id
        self._interval = interval_seconds
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="zt-reconciler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def kick(self) -> None:
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                reconcile_once(self._store, self._zt, self._network_id)
            except Exception:  # noqa: BLE001
                log.exception("reconcile pass failed; will retry")
            self._wake.wait(timeout=self._interval)
            self._wake.clear()
