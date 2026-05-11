"""Pydantic request/response shapes for the admin API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RegisterPeerRequest(BaseModel):
    peer_id: str = Field(min_length=1, max_length=128)
    kind: Literal["server", "user"]
    # ZeroTier node IDs are exactly 10 lowercase hex characters.
    node_id: str = Field(min_length=10, max_length=10, pattern=r"^[0-9a-f]{10}$")


class PeerConfig(BaseModel):
    """What gets returned to the application server, which forwards it (minus
    nothing — the shape is what the mobile client ultimately consumes too).
    """
    peer_id: str
    node_id: str                # 10-hex ZeroTier identity
    network_id: str             # 16-hex ZeroTier network to join
    assigned_ip: str            # bare IPv4 (no /mask), e.g. "10.66.66.7"
    server_ip: str              # the application server's address on the network
