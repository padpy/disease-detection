"""FastAPI surface for the broker's admin API.

All `/admin/*` routes require `X-Broker-Token: <BROKER_ADMIN_TOKEN>`. The
application server is the sole legitimate caller; mobile clients never
touch the broker's HTTP API directly.
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import Depends, FastAPI, Header, HTTPException, status

from .config import BrokerConfig
from .ip_pool import PoolExhausted, allocate
from .models import PeerConfig, RegisterPeerRequest
from .peer_store import PeerStore
from .reconciler import Reconciler


def require_admin_token(config: BrokerConfig):
    def _dep(x_broker_token: str = Header(default="")):
        if not _consteq(x_broker_token, config.admin_token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")
    return _dep


def _consteq(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    result = 0
    for x, y in zip(a.encode(), b.encode()):
        result |= x ^ y
    return result == 0


def create_app(
    config: BrokerConfig,
    store: PeerStore,
    reconciler: Reconciler,
    *,
    lifespan: Callable[[FastAPI], Any] | None = None,
) -> FastAPI:
    app = FastAPI(title="gopher-eye-broker", lifespan=lifespan)
    admin = require_admin_token(config)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.post("/admin/peers", response_model=PeerConfig, dependencies=[Depends(admin)])
    def register_peer(req: RegisterPeerRequest):
        existing = store.get(req.peer_id)
        if existing and existing.revoked_at is None:
            # Idempotent upsert: same node id returns the existing assignment;
            # a different node id rotates it.
            if existing.node_id == req.node_id:
                return _response(req.peer_id, existing.assigned_ip, req.node_id, config)
            store.upsert(
                peer_id=req.peer_id,
                kind=req.kind,
                assigned_ip=existing.assigned_ip,
                node_id=req.node_id,
            )
            reconciler.kick()
            return _response(req.peer_id, existing.assigned_ip, req.node_id, config)

        if req.kind == "server" and req.peer_id == "server":
            assigned = config.server_ip
            if assigned in store.used_ips():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="server IP is already assigned to a different peer",
                )
        else:
            try:
                assigned = allocate(
                    config.subnet,
                    used=store.used_ips(),
                    server_ip=config.server_ip,
                    reserve_server_ip=True,
                )
            except PoolExhausted as exc:
                raise HTTPException(status_code=status.HTTP_507_INSUFFICIENT_STORAGE, detail=str(exc))

        store.upsert(
            peer_id=req.peer_id,
            kind=req.kind,
            assigned_ip=assigned,
            node_id=req.node_id,
        )
        reconciler.kick()
        return _response(req.peer_id, assigned, req.node_id, config)

    @app.get("/admin/peers/{peer_id}", response_model=PeerConfig, dependencies=[Depends(admin)])
    def get_peer(peer_id: str):
        peer = store.get(peer_id)
        if peer is None or peer.revoked_at is not None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="peer not found")
        return _response(peer.peer_id, peer.assigned_ip, peer.node_id, config)

    @app.delete("/admin/peers/{peer_id}", status_code=204, dependencies=[Depends(admin)])
    def revoke_peer(peer_id: str):
        if not store.revoke(peer_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="peer not found")
        reconciler.kick()
        return None

    return app


def _response(peer_id: str, assigned_ip: str, node_id: str, config: BrokerConfig) -> PeerConfig:
    return PeerConfig(
        peer_id=peer_id,
        node_id=node_id,
        network_id=config.zt_network_id,
        assigned_ip=assigned_ip,
        server_ip=config.server_ip,
    )
