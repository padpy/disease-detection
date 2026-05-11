"""Thin HTTPS client for the ZeroTier broker's admin API.

Used by `/vpn/config` routes to register and revoke peers, and by the deploy
script `server/script/enroll_vpn` to enroll the server itself.

All calls send `X-Broker-Token: <BROKER_ADMIN_TOKEN>` and return the parsed
JSON body (raising `BrokerError` for non-2xx).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


class BrokerError(RuntimeError):
    def __init__(self, status_code: int, body: str):
        super().__init__(f"broker returned {status_code}: {body}")
        self.status_code = status_code
        self.body = body


@dataclass(frozen=True)
class BrokerClient:
    base_url: str           # e.g. "https://broker.example.com:8080"
    admin_token: str
    timeout_seconds: float = 5.0

    @classmethod
    def from_env(cls) -> "BrokerClient":
        return cls(
            base_url=_require_env("BROKER_URL"),
            admin_token=_require_env("BROKER_ADMIN_TOKEN"),
            timeout_seconds=float(os.environ.get("BROKER_TIMEOUT_SECONDS", "5")),
        )

    def _headers(self) -> dict[str, str]:
        return {"X-Broker-Token": self.admin_token, "Accept": "application/json"}

    def register_peer(self, *, peer_id: str, kind: str, node_id: str) -> dict:
        return self._post("/admin/peers", {
            "peer_id": peer_id,
            "kind": kind,
            "node_id": node_id,
        })

    def get_peer(self, peer_id: str) -> dict | None:
        url = f"{self.base_url.rstrip('/')}/admin/peers/{peer_id}"
        with httpx.Client(timeout=self.timeout_seconds) as client:
            r = client.get(url, headers=self._headers())
        if r.status_code == 404:
            return None
        if r.is_error:
            raise BrokerError(r.status_code, r.text)
        return r.json()

    def revoke_peer(self, peer_id: str) -> bool:
        url = f"{self.base_url.rstrip('/')}/admin/peers/{peer_id}"
        with httpx.Client(timeout=self.timeout_seconds) as client:
            r = client.delete(url, headers=self._headers())
        if r.status_code == 404:
            return False
        if r.is_error:
            raise BrokerError(r.status_code, r.text)
        return True

    def _post(self, path: str, body: dict) -> dict:
        url = f"{self.base_url.rstrip('/')}{path}"
        with httpx.Client(timeout=self.timeout_seconds) as client:
            r = client.post(url, headers=self._headers(), json=body)
        if r.is_error:
            raise BrokerError(r.status_code, r.text)
        return r.json()


def _require_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"required environment variable {key} is not set")
    return value
