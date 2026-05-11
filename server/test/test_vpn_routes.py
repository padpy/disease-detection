"""Unit tests for `/vpn/config` routes.

The broker is faked via `broker_factory`; Firebase Auth is faked via
`auth_dependency`. Both injection points are wired into `build_router(...)`
specifically so these tests can run without firebase_admin or a live broker.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from vpn_routes import build_router  # noqa: E402


NODE_ID = "abcdef0123"
ALT_NODE_ID = "1234567890"


class FakeBroker:
    def __init__(self):
        self.peers: dict[str, dict] = {}

    def register_peer(self, *, peer_id: str, kind: str, node_id: str) -> dict:
        entry = {
            "peer_id": peer_id,
            "node_id": node_id,
            "network_id": "0123456789abcdef",
            "assigned_ip": "10.66.66.7",
            "server_ip": "10.66.66.2",
        }
        self.peers[peer_id] = entry
        return entry

    def get_peer(self, peer_id: str):
        return self.peers.get(peer_id)

    def revoke_peer(self, peer_id: str) -> bool:
        return self.peers.pop(peer_id, None) is not None


@pytest.fixture
def client():
    fake = FakeBroker()

    def fake_auth():
        return {"uid": "u1", "email": "u1@example.com"}

    app = FastAPI()
    app.include_router(build_router(broker_factory=lambda: fake, auth_dependency=fake_auth))
    return TestClient(app), fake


def test_register_returns_broker_payload(client):
    c, _ = client
    r = c.post("/vpn/config", json={"client_node_id": NODE_ID})
    assert r.status_code == 200
    body = r.json()
    assert body["peer_id"] == "user:u1"
    assert body["assigned_ip"] == "10.66.66.7"
    assert body["network_id"] == "0123456789abcdef"
    assert body["node_id"] == NODE_ID


def test_register_rejects_malformed_node_id(client):
    c, _ = client
    r = c.post("/vpn/config", json={"client_node_id": "NOPE"})
    # Pydantic catches it as 422 before our explicit regex check runs.
    assert r.status_code in (400, 422)


def test_get_returns_existing(client):
    c, fake = client
    fake.peers["user:u1"] = {"peer_id": "user:u1", "assigned_ip": "10.66.66.9", "node_id": ALT_NODE_ID}
    r = c.get("/vpn/config")
    assert r.status_code == 200
    assert r.json()["assigned_ip"] == "10.66.66.9"


def test_get_404_when_missing(client):
    c, _ = client
    r = c.get("/vpn/config")
    assert r.status_code == 404


def test_revoke_204(client):
    c, fake = client
    fake.peers["user:u1"] = {"peer_id": "user:u1"}
    r = c.delete("/vpn/config")
    assert r.status_code == 204
    assert "user:u1" not in fake.peers


def test_missing_token_when_real_auth_dep():
    """The real auth dependency rejects missing Authorization headers."""
    from vpn_routes import _firebase_auth_dependency
    dep = _firebase_auth_dependency()
    with pytest.raises(HTTPException) as ei:
        dep(authorization=None)
    assert ei.value.status_code == 401
