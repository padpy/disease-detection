"""Broker entrypoint.

`uvicorn app.broker:app` loads `app`, which is built once at import time by
`build_app()`. The ZT backend is selected via env: `BROKER_ZT_BACKEND=stub`
skips the controller HTTP calls (useful for smoke-testing the API without a
running zerotier-one; never use in production).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import zerotier as zt
from .api import create_app
from .config import BrokerConfig
from .peer_store import PeerStore
from .reconciler import Reconciler


def build_app() -> FastAPI:
    logging.basicConfig(level=os.environ.get("BROKER_LOG_LEVEL", "INFO"))

    config = BrokerConfig.from_env()
    store = PeerStore(config.db_path)

    zt_backend = os.environ.get("BROKER_ZT_BACKEND", "system")
    if zt_backend == "stub":
        zt_cmd: zt.ZtCommand = _StubZt()
    else:
        zt_cmd = zt.HttpZt(config.zt_controller_url, config.zt_auth_token)

    reconciler = Reconciler(store, zt_cmd, config.zt_network_id)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        reconciler.start()
        try:
            yield
        finally:
            reconciler.stop()

    return create_app(config, store, reconciler, lifespan=lifespan)


class _StubZt:
    """No-op ZT backend for smoke tests; logs invocations only."""
    def __init__(self):
        self._log = logging.getLogger(f"{__name__}.stub_zt")
        self._members: dict[str, dict] = {}

    def get_members(self, network_id: str) -> dict:
        self._log.info("zt get_members %s", network_id)
        return {node_id: 1 for node_id in self._members}

    def get_member(self, network_id: str, node_id: str) -> dict:
        self._log.info("zt get_member %s/%s", network_id, node_id)
        return dict(self._members.get(node_id, {"authorized": False, "ipAssignments": []}))

    def set_member(self, network_id: str, node_id: str, payload: dict) -> dict:
        self._log.info("zt set_member %s/%s %s", network_id, node_id, payload)
        current = self._members.setdefault(node_id, {"authorized": False, "ipAssignments": []})
        if payload:
            current.update(payload)
        return current

    def delete_member(self, network_id: str, node_id: str) -> None:
        self._log.info("zt delete_member %s/%s", network_id, node_id)
        self._members.pop(node_id, None)


app = build_app()
