# Gopher Eye ZeroTier Broker

The broker is the control plane that issues per-peer ZeroTier network
membership for Gopher Eye. Both the Flask/FastAPI application server and each
mobile client join the same ZeroTier network; the broker is the only thing
that calls the local ZeroTier controller API to authorize members and assign
them stable IPs.

```
mobile (ZT member) ──┐
                     ├──▶ ZeroTier network (managed by this broker)
                     │     └─ app server (ZT member, 10.66.66.2)
mobile (ZT member) ──┘
```

The host running this broker must also run `zerotier-one` configured as a
[network controller](https://docs.zerotier.com/self-hosting/network-controllers).
The application server and mobile clients don't need to reach the broker
host directly — ZeroTier handles peer transport for them. The broker only
needs HTTPS reachability from the application server (for the admin API).

## Components

- `app/api.py` — FastAPI admin routes (`POST/GET/DELETE /admin/peers`)
- `app/peer_store.py` — SQLite-backed peer registry (source of truth)
- `app/ip_pool.py` — allocates the next free `/32` from `BROKER_SUBNET`
- `app/zerotier.py` — thin wrapper around the local ZT controller HTTP API
- `app/reconciler.py` — periodic loop reconciling the peer table with the live
  ZeroTier network members

## Auth

Every admin call must send `X-Broker-Token: <BROKER_ADMIN_TOKEN>`. That secret
is shared with the application server via the matching env var in
`server/.envrc`. The broker does **not** authenticate end users — the
application server does, then proxies to the broker.

## Setup

1. **Provision a host** (small VM is plenty).
2. Install ZeroTier: `curl -s https://install.zerotier.com | sudo bash`, then
   start with `sudo systemctl enable --now zerotier-one`.
3. Create a network controlled by this host:
   `sudo zerotier-cli controller`, then
   `sudo zerotier-cli net-create` (or via the API). Note the 16-hex network ID.
4. Configure the network's managed routes / IP pool to match `BROKER_SUBNET`
   (default `10.66.66.0/24`). The controller will let the broker assign
   addresses inside that pool.
5. `cp .envrc.example .envrc`, edit values (paste the network ID into
   `BROKER_ZT_NETWORK_ID`), `direnv allow` (or `source .envrc`).
6. As root, run `sudo -E ./script/bootstrap`. This sanity-checks the env, the
   controller, and prints the resolved network ID.
7. Start the API: `./script/broker` (local) or `./script/docker`
   (containerized — see the compose file for the required host-network +
   socket mount).

After this, run `server/script/enroll_vpn` on the application server to
register it as peer `server` (assigned IP `10.66.66.2`).

## API

All routes require `X-Broker-Token` and return JSON. See `app/api.py` for
shapes.

- `POST /admin/peers` — register or upsert a peer.
- `GET  /admin/peers/{peer_id}` — fetch a previously registered peer.
- `DELETE /admin/peers/{peer_id}` — revoke; reconciler deauthorizes the member
  from the ZeroTier network on its next pass.
- `GET /healthz` — liveness; no auth.

## Tests

```bash
cd broker
pip install -r requirements.txt
pytest
```

Tests stub `zerotier.py` so they don't require a running `zerotier-one` on the
test host.
