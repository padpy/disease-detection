"""Thin wrapper around the ZeroTier local controller API.

The reconciler is the only caller that mutates the live network. Tests inject
a stub via the `ZtCommand` protocol below — keeps `httpx` out of the unit
tests.

Reference: https://docs.zerotier.com/self-hosting/network-controllers
The controller is reachable on the same host that runs `zerotier-one` (default
http://127.0.0.1:9993). The `X-ZT1-Auth` header is the contents of
`/var/lib/zerotier-one/authtoken.secret`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import httpx


class ZtCommand(Protocol):
    def get_members(self, network_id: str) -> dict[str, Any]: ...
    def get_member(self, network_id: str, node_id: str) -> dict[str, Any]: ...
    def set_member(self, network_id: str, node_id: str, payload: dict[str, Any]) -> dict[str, Any]: ...
    def delete_member(self, network_id: str, node_id: str) -> None: ...


@dataclass(frozen=True)
class LiveMember:
    node_id: str
    authorized: bool
    ip_assignments: tuple[str, ...]


class HttpZt:
    """Default ZtCommand that talks to a local ZeroTier controller over HTTP."""

    def __init__(self, base_url: str, auth_token: str, *, timeout_seconds: float = 5.0):
        self._base = base_url.rstrip("/")
        self._headers = {"X-ZT1-Auth": auth_token, "Accept": "application/json"}
        self._timeout = timeout_seconds

    def get_members(self, network_id: str) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as client:
            r = client.get(
                f"{self._base}/controller/network/{network_id}/member",
                headers=self._headers,
            )
        r.raise_for_status()
        return r.json()

    def get_member(self, network_id: str, node_id: str) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as client:
            r = client.get(
                f"{self._base}/controller/network/{network_id}/member/{node_id}",
                headers=self._headers,
            )
        r.raise_for_status()
        return r.json()

    def set_member(self, network_id: str, node_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as client:
            r = client.post(
                f"{self._base}/controller/network/{network_id}/member/{node_id}",
                headers={**self._headers, "Content-Type": "application/json"},
                json=payload,
            )
        r.raise_for_status()
        return r.json()

    def delete_member(self, network_id: str, node_id: str) -> None:
        with httpx.Client(timeout=self._timeout) as client:
            r = client.delete(
                f"{self._base}/controller/network/{network_id}/member/{node_id}",
                headers=self._headers,
            )
        if r.status_code not in (200, 204, 404):
            r.raise_for_status()


def show_members(zt: ZtCommand, network_id: str) -> list[LiveMember]:
    """List currently-known members on the controller for `network_id`.

    The controller's ``/member`` endpoint returns a map of node_id → revision
    counter; we then fetch each member individually to read its authorization
    + IP assignments. For our peer counts (10s, not 1000s) the extra round
    trips are fine.
    """
    raw = zt.get_members(network_id)
    members: list[LiveMember] = []
    for node_id in raw.keys():
        try:
            detail = zt.get_member(network_id, node_id)
        except httpx.HTTPStatusError:
            continue
        members.append(
            LiveMember(
                node_id=node_id,
                authorized=bool(detail.get("authorized", False)),
                ip_assignments=tuple(detail.get("ipAssignments", []) or ()),
            )
        )
    return members


def authorize_member(
    zt: ZtCommand,
    network_id: str,
    *,
    node_id: str,
    assigned_ip: str,
) -> None:
    zt.set_member(
        network_id,
        node_id,
        {"authorized": True, "ipAssignments": [assigned_ip]},
    )


def deauthorize_member(zt: ZtCommand, network_id: str, *, node_id: str) -> None:
    zt.delete_member(network_id, node_id)
