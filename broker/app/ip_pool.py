"""Allocate the next free /32 from the broker's ZeroTier network subnet.

Reserved addresses:
  - subnet.network_address       (e.g. 10.66.66.0) — network address
  - subnet.broadcast_address     (e.g. 10.66.66.255) — broadcast
  - server_ip (10.66.66.2)       — the application server peer (reserved
                                   so users can't claim it; assigned only
                                   to the peer with peer_id='server')

ZeroTier does not require a "hub" address inside the network the way
WireGuard's broker did — the controller sits outside the L2/L3 plane, so
.1 stays available for user assignment unless you carve it out yourself.
"""

from __future__ import annotations

import ipaddress
from typing import Iterable


class PoolExhausted(RuntimeError):
    """No free /32 left in the subnet."""


def allocate(
    subnet: ipaddress.IPv4Network,
    *,
    used: Iterable[str],
    server_ip: str,
    reserve_server_ip: bool,
) -> str:
    """Pick the lowest free address in `subnet`, skipping reservations.

    Pass `reserve_server_ip=False` only when allocating for the server peer
    itself — then this returns `server_ip` if it's free.
    """
    used_set = set(used)
    skip = {str(subnet.network_address), str(subnet.broadcast_address)}
    if reserve_server_ip:
        skip.add(server_ip)

    for host in subnet.hosts():
        addr = str(host)
        if addr in skip or addr in used_set:
            continue
        return addr

    raise PoolExhausted(f"no free address in {subnet}")
