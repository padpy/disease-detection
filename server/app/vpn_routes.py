"""`/vpn/config` routes — Firebase-auth gated, proxy to the ZeroTier broker.

Flow:
  client -> POST /vpn/config { client_node_id }
       server identifies user from Firebase ID token
       server calls broker /admin/peers (peer_id = "user:<uid>")
       server returns broker's response shape to the client

The broker is the source of truth for assigned IPs and live network state;
this module owns nothing persistent.
"""

from __future__ import annotations

import re
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

from broker_client import BrokerClient, BrokerError


# ZeroTier node IDs are exactly 10 lowercase hex characters.
_NODE_ID_RE = re.compile(r"^[0-9a-f]{10}$")


class RegisterVpnRequest(BaseModel):
    client_node_id: str = Field(min_length=10, max_length=10)


def _firebase_auth_dependency():
    """Returns a dependency that resolves to the Firebase user record (dict).

    Imports the Firebase Admin SDK lazily so this module is still importable
    in environments that haven't configured Firebase (tests, CI).
    """
    def _dep(authorization: Optional[str] = Header(default=None)) -> dict:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing bearer token",
            )
        token = authorization.split(" ", 1)[1].strip()
        try:
            from firebase_admin import auth as fb_auth  # type: ignore
        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"firebase_admin not installed: {exc}",
            )
        try:
            return fb_auth.verify_id_token(token, check_revoked=True)
        except fb_auth.RevokedIdTokenError:
            raise HTTPException(status_code=401, detail="token revoked")
        except fb_auth.ExpiredIdTokenError:
            raise HTTPException(status_code=401, detail="token expired")
        except fb_auth.InvalidIdTokenError:
            raise HTTPException(status_code=401, detail="invalid token")
    return _dep


def _peer_id_for(user: dict) -> str:
    uid = user.get("uid") or user.get("user_id")
    if not uid:
        raise HTTPException(status_code=401, detail="token missing uid")
    return f"user:{uid}"


def build_router(broker_factory=None, auth_dependency=None) -> APIRouter:
    """Build the router. `broker_factory` and `auth_dependency` are injected
    for tests; production calls `build_router()` with no args.
    """
    broker_factory = broker_factory or BrokerClient.from_env
    auth_dependency = auth_dependency or _firebase_auth_dependency()

    router = APIRouter(prefix="/vpn", tags=["vpn"])

    @router.post("/config")
    def register(req: RegisterVpnRequest, user: dict = Depends(auth_dependency)):
        if not _NODE_ID_RE.match(req.client_node_id):
            raise HTTPException(
                status_code=400,
                detail="client_node_id must be 10 lowercase hex characters",
            )
        peer_id = _peer_id_for(user)
        broker = broker_factory()
        try:
            return broker.register_peer(
                peer_id=peer_id,
                kind="user",
                node_id=req.client_node_id,
            )
        except BrokerError as exc:
            raise HTTPException(status_code=502, detail=f"broker error: {exc.body}")

    @router.get("/config")
    def fetch(user: dict = Depends(auth_dependency)):
        peer_id = _peer_id_for(user)
        broker = broker_factory()
        try:
            existing = broker.get_peer(peer_id)
        except BrokerError as exc:
            raise HTTPException(status_code=502, detail=f"broker error: {exc.body}")
        if existing is None:
            raise HTTPException(status_code=404, detail="no VPN config issued for this user")
        return existing

    @router.delete("/config", status_code=204)
    def revoke(user: dict = Depends(auth_dependency)):
        peer_id = _peer_id_for(user)
        broker = broker_factory()
        try:
            broker.revoke_peer(peer_id)
        except BrokerError as exc:
            raise HTTPException(status_code=502, detail=f"broker error: {exc.body}")
        return None

    return router
