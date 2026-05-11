"""Shared fixtures: stub ZeroTier controller, in-memory store, FastAPI TestClient."""

from __future__ import annotations

import ipaddress
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import BrokerConfig
from app.peer_store import PeerStore
from app.reconciler import Reconciler


class StubZt:
    """In-memory ZT controller backend. Records `set_member`/`delete_member`
    calls and serves the recorded state back to the reconciler.
    """

    def __init__(self):
        self.members: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str, str, dict[str, Any]]] = []

    def get_members(self, network_id: str) -> dict[str, Any]:
        self.calls.append(("get_members", network_id, "", {}))
        return {node_id: 1 for node_id in self.members}

    def get_member(self, network_id: str, node_id: str) -> dict[str, Any]:
        self.calls.append(("get_member", network_id, node_id, {}))
        return dict(self.members[node_id])

    def set_member(self, network_id: str, node_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("set_member", network_id, node_id, dict(payload)))
        current = self.members.setdefault(node_id, {"authorized": False, "ipAssignments": []})
        if payload:
            if "authorized" in payload:
                current["authorized"] = bool(payload["authorized"])
            if "ipAssignments" in payload:
                current["ipAssignments"] = list(payload["ipAssignments"])
        return current

    def delete_member(self, network_id: str, node_id: str) -> None:
        self.calls.append(("delete_member", network_id, node_id, {}))
        self.members.pop(node_id, None)


@pytest.fixture
def stub_zt() -> StubZt:
    return StubZt()


@pytest.fixture
def config(tmp_path: Path) -> BrokerConfig:
    return BrokerConfig(
        admin_token="test-token",
        db_path=tmp_path / "broker.db",
        zt_network_id="0123456789abcdef",
        zt_controller_url="http://127.0.0.1:9993",
        zt_auth_token="test-zt-token",
        server_ip="10.66.66.2",
        subnet=ipaddress.IPv4Network("10.66.66.0/24"),
    )


@pytest.fixture
def store(config: BrokerConfig) -> PeerStore:
    return PeerStore(config.db_path)


@pytest.fixture
def reconciler(store: PeerStore, stub_zt: StubZt, config: BrokerConfig) -> Reconciler:
    # We do not call .start() — tests drive reconciliation explicitly when needed.
    return Reconciler(store, stub_zt, config.zt_network_id)


@pytest.fixture
def client(config, store, reconciler) -> TestClient:
    app = create_app(config, store, reconciler)
    return TestClient(app)


@pytest.fixture
def auth_headers(config) -> dict[str, str]:
    return {"X-Broker-Token": config.admin_token}
