"""Runtime configuration for the broker.

All values come from environment variables (see `.envrc.example`). One place
to read them so the rest of the code can rely on a typed object.
"""

from __future__ import annotations

import ipaddress
import os
import re
from dataclasses import dataclass
from pathlib import Path


_ZT_NETWORK_ID_RE = re.compile(r"^[0-9a-f]{16}$")


@dataclass(frozen=True)
class BrokerConfig:
    admin_token: str
    db_path: Path
    zt_network_id: str              # 16 hex chars; the ZeroTier network managed by this controller
    zt_controller_url: str          # base URL of the local ZT controller API (typically http://127.0.0.1:9993)
    zt_auth_token: str              # contents of /var/lib/zerotier-one/authtoken.secret
    server_ip: str                  # reserved /32 for the application server peer
    subnet: ipaddress.IPv4Network   # full VPN subnet managed by the controller

    @classmethod
    def from_env(cls, env: dict | None = None) -> "BrokerConfig":
        e = env if env is not None else os.environ
        network_id = _require(e, "BROKER_ZT_NETWORK_ID").lower()
        if not _ZT_NETWORK_ID_RE.match(network_id):
            raise RuntimeError(
                "BROKER_ZT_NETWORK_ID must be exactly 16 hex characters"
            )
        return cls(
            admin_token=_require(e, "BROKER_ADMIN_TOKEN"),
            db_path=Path(e.get("BROKER_DB_PATH", "data/broker.db")),
            zt_network_id=network_id,
            zt_controller_url=e.get("BROKER_ZT_CONTROLLER_URL", "http://127.0.0.1:9993"),
            zt_auth_token=_load_auth_token(e),
            server_ip=e.get("BROKER_SERVER_IP", "10.66.66.2"),
            subnet=ipaddress.IPv4Network(e.get("BROKER_SUBNET", "10.66.66.0/24")),
        )


def _load_auth_token(env: dict) -> str:
    inline = env.get("BROKER_ZT_AUTH_TOKEN")
    if inline:
        return inline.strip()
    path = env.get("BROKER_ZT_AUTH_TOKEN_PATH", "/var/lib/zerotier-one/authtoken.secret")
    try:
        return Path(path).read_text().strip()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"ZeroTier auth token not found at {path}; set BROKER_ZT_AUTH_TOKEN or "
            "BROKER_ZT_AUTH_TOKEN_PATH"
        ) from exc


def _require(env: dict, key: str) -> str:
    value = env.get(key)
    if not value:
        raise RuntimeError(f"required environment variable {key} is not set")
    return value
