# Private VPS Staging Setup

**Updated:** 2026-06-22

**Scope:** trusted staging only; not a public production deployment

Do not place real hostnames, IP addresses, usernames, tokens, home paths, or private keys in this file. Use a private inventory and environment file on the host.

## Prerequisites

- A dedicated unprivileged service account.
- A private firewall/VPN or strict source allowlist.
- Python 3.12 and a virtual environment.
- TLS termination for the HTTP API.
- Separate high-entropy publisher and API credentials.
- Personal memory and model checkpoints kept off a relay-only host.

## Install

```bash
sudo useradd --system --home /opt/lixyswarm --shell /usr/sbin/nologin lixyswarm
sudo install -d -o lixyswarm -g lixyswarm /opt/lixyswarm
sudo -u lixyswarm git clone https://github.com/OWNER/REPOSITORY.git /opt/lixyswarm/app
sudo -u lixyswarm python3 -m venv /opt/lixyswarm/venv
sudo -u lixyswarm /opt/lixyswarm/venv/bin/pip install -r /opt/lixyswarm/app/api/requirements_api.txt
```

Create a root-readable environment file outside Git, for example `/etc/lixyswarm/api.env`:

```text
LIXYSWARM_PUBLISH_TOKEN=<random-secret>
LIXYSWARM_BOOTSTRAP_SEEDS=<trusted-host>:7338
LIXYSWARM_IDENTITY_PATH=/opt/lixyswarm/state/identity.key
```

Do not enable address publication flags unless the dashboard explicitly requires them and its access is restricted.

## Firewall

For early staging, expose no raw LSP port publicly. Allow UDP 7337 and TCP 7338 only over a VPN or from named trusted peer addresses. Expose the API only through a TLS reverse proxy with rate limits.

## Run a relay node

```bash
sudo -u lixyswarm \
  LIXYSWARM_IDENTITY_PATH=/opt/lixyswarm/state/identity.key \
  /opt/lixyswarm/venv/bin/python /opt/lixyswarm/app/node_daemon.py
```

Configure an explicit upstream only in the private host environment:

```text
LIXYSWARM_PEER_HOST=<trusted-peer-host>
LIXYSWARM_FEROMON_PORT=7337
LIXYSWARM_GOSSIP_PORT=7338
```

## Run the API

The checked-in service file is a development template and must be reviewed before installation. Run as the dedicated account, load secrets from `EnvironmentFile=`, bind Uvicorn to loopback, and let the TLS proxy serve external traffic.

## Validation

```bash
curl --fail http://127.0.0.1:8080/health
pytest -q test_lsp.py test_lsp_v2.py test_network.py
```

Then verify firewall rules, TLS, authentication failures, log redaction, service restart, identity-file permissions, resource limits, and backup/restore.

## Production warning

This setup does not close the protocol, privacy, discovery, reputation, or scaling blockers in `INTERNET_SCALE_READINESS.md`. A reachable VPS is not evidence that the system is Internet-ready.
