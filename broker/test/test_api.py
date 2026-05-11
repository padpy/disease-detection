from __future__ import annotations

from app.reconciler import reconcile_once


# ZeroTier node IDs are 10 lowercase hex characters.
SERVER_NODE = "aaaaaaaaaa"
USER_A_NODE = "bbbbbbbbbb"
USER_B_NODE = "cccccccccc"
USER_A_NODE_ROT = "dddddddddd"


def test_healthz_no_auth(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_admin_routes_require_token(client):
    r = client.get("/admin/peers/server")
    assert r.status_code == 401


def test_register_server_peer_uses_reserved_ip(client, auth_headers):
    r = client.post(
        "/admin/peers",
        headers=auth_headers,
        json={"peer_id": "server", "kind": "server", "node_id": SERVER_NODE},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["peer_id"] == "server"
    assert body["node_id"] == SERVER_NODE
    assert body["network_id"] == "0123456789abcdef"
    assert body["assigned_ip"] == "10.66.66.2"
    assert body["server_ip"] == "10.66.66.2"


def test_register_user_skips_reserved_addresses(client, auth_headers):
    r = client.post(
        "/admin/peers",
        headers=auth_headers,
        json={"peer_id": "user:a", "kind": "user", "node_id": USER_A_NODE},
    )
    assert r.status_code == 200
    # 10.66.66.0 is the network address and 10.66.66.2 is reserved for server,
    # so the first user gets .1.
    assert r.json()["assigned_ip"] == "10.66.66.1"


def test_register_is_idempotent_for_same_node_id(client, auth_headers):
    body = {"peer_id": "user:a", "kind": "user", "node_id": USER_A_NODE}
    first = client.post("/admin/peers", headers=auth_headers, json=body).json()
    second = client.post("/admin/peers", headers=auth_headers, json=body).json()
    assert first == second


def test_register_rotates_node_id_keeps_ip(client, auth_headers):
    base = {"peer_id": "user:a", "kind": "user", "node_id": USER_A_NODE}
    first = client.post("/admin/peers", headers=auth_headers, json=base).json()

    rotated = dict(base, node_id=USER_A_NODE_ROT)
    second = client.post("/admin/peers", headers=auth_headers, json=rotated).json()
    assert second["assigned_ip"] == first["assigned_ip"]
    assert second["node_id"] == USER_A_NODE_ROT


def test_get_peer_returns_existing(client, auth_headers):
    client.post(
        "/admin/peers",
        headers=auth_headers,
        json={"peer_id": "user:a", "kind": "user", "node_id": USER_A_NODE},
    )
    r = client.get("/admin/peers/user:a", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["assigned_ip"] == "10.66.66.1"
    assert r.json()["node_id"] == USER_A_NODE


def test_get_peer_404_when_missing(client, auth_headers):
    r = client.get("/admin/peers/user:nope", headers=auth_headers)
    assert r.status_code == 404


def test_revoke_peer(client, auth_headers, store):
    client.post(
        "/admin/peers",
        headers=auth_headers,
        json={"peer_id": "user:a", "kind": "user", "node_id": USER_A_NODE},
    )
    r = client.delete("/admin/peers/user:a", headers=auth_headers)
    assert r.status_code == 204
    assert store.get("user:a").revoked_at is not None
    # Second revoke is a no-op 404 (already revoked)
    assert client.delete("/admin/peers/user:a", headers=auth_headers).status_code == 404


def test_revoked_peer_frees_ip_for_next_allocation(client, auth_headers):
    client.post("/admin/peers", headers=auth_headers,
                json={"peer_id": "user:a", "kind": "user", "node_id": USER_A_NODE})
    client.delete("/admin/peers/user:a", headers=auth_headers)
    r = client.post("/admin/peers", headers=auth_headers,
                    json={"peer_id": "user:b", "kind": "user", "node_id": USER_B_NODE})
    assert r.json()["assigned_ip"] == "10.66.66.1"  # reuses freed slot


def test_reconciler_applies_authorizes_and_deauthorizes(store, stub_zt, config):
    store.upsert(peer_id="user:a", kind="user", assigned_ip="10.66.66.3",
                 node_id=USER_A_NODE)
    store.upsert(peer_id="user:b", kind="user", assigned_ip="10.66.66.4",
                 node_id=USER_B_NODE)
    authorized, deauthorized = reconcile_once(store, stub_zt, config.zt_network_id)
    assert (authorized, deauthorized) == (2, 0)
    assert stub_zt.members == {
        USER_A_NODE: {"authorized": True, "ipAssignments": ["10.66.66.3"]},
        USER_B_NODE: {"authorized": True, "ipAssignments": ["10.66.66.4"]},
    }

    store.revoke("user:a")
    authorized, deauthorized = reconcile_once(store, stub_zt, config.zt_network_id)
    assert (authorized, deauthorized) == (0, 1)
    assert list(stub_zt.members.keys()) == [USER_B_NODE]


def test_reconciler_is_idempotent(store, stub_zt, config):
    store.upsert(peer_id="user:a", kind="user", assigned_ip="10.66.66.3",
                 node_id=USER_A_NODE)
    reconcile_once(store, stub_zt, config.zt_network_id)
    authorized, deauthorized = reconcile_once(store, stub_zt, config.zt_network_id)
    assert (authorized, deauthorized) == (0, 0)
