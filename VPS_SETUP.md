# Public LSP v3 Seed Deployment

**Updated:** 2026-06-23

The VPS is a replaceable bootstrap seed. It accepts persistent LSP v3 sessions and shares learned peer addresses. It does not relay model traffic, coordinate decisions, or remain necessary after peers connect.

Do not commit real addresses, credentials, private keys, or inventory details. Publish the seed through a stable DNS name.

## Install

```bash
sudo useradd --system --home /opt/lixyswarm --shell /usr/sbin/nologin lixyswarm
sudo install -d -o lixyswarm -g lixyswarm /opt/lixyswarm
sudo -u lixyswarm git clone https://github.com/OWNER/REPOSITORY.git /opt/lixyswarm/app
sudo -u lixyswarm python3 -m venv /opt/lixyswarm/venv
sudo -u lixyswarm /opt/lixyswarm/venv/bin/pip install -e /opt/lixyswarm/app
```

Create `/etc/lixyswarm/seed.env` outside Git:

```text
LIXYSWARM_PUBLIC_HOST=seed.example.net
LIXYSWARM_GOSSIP_PORT=7338
LIXYSWARM_TARGET_OUTBOUND=0
```

The checked-in `lixyswarm-seed.service` stores identity/address state under `/var/lib/lixyswarm`, runs without root, and applies systemd hardening.

```bash
sudo cp /opt/lixyswarm/app/lixyswarm-seed.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now lixyswarm-seed
sudo systemctl status lixyswarm-seed
```

## Web explorer and API

The checked-in API service binds only to loopback on port `8001`. Do not expose that port directly. Serve the static explorer and proxy `/api/` through nginx (or an equivalent TLS reverse proxy):

```nginx
server {
    listen 8080;
    root /opt/lixyswarm/app/frontend;
    index swarm-explorer.html;

    location / {
        try_files $uri $uri/ =404;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8001/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Use a dedicated DNS name plus HTTPS before presenting the explorer as a production endpoint. The API remains a research interface; binding it behind nginx does not replace endpoint authentication.

## Firewall and DNS

- Publish `seed.example.net` A and AAAA records.
- Allow inbound TCP 7338 to the seed.
- No UDP 7337 rule is required for LSP v3.
- Keep the HTTP API on a separate TLS endpoint and policy.
- Add at least one independently operated seed before a public release.

Ordinary participants do not open ports or use a VPN. They initiate outbound TCP sessions to seeds/learned peers.

## Client bootstrap

Until official seed domains are compiled into a release:

```bash
export LIXYSWARM_BOOTSTRAP_SEEDS='seed.example.net:7338'
lixyswarm start
```

The basic command contributes connectivity and explicitly imported artifacts only. A participant can use a separately verified local checkpoint with `--checkpoint`, or configure threshold trust, activate a signed local release, and use `lixyswarm start --release`. The repository does not publish official signer keys, a pinned genesis, or model weights yet, so acquisition must not be automated from an untrusted peer.

After peer exchange, stopping the seed must not interrupt existing direct sessions. `test_lsp_v3.py::test_v3_network_continues_after_seed_shutdown` covers graceful shutdown in one process; `test_lsp_v3_multiprocess.py` runs each node in a separate interpreter and abruptly terminates the seed process.

## Validation

```bash
pytest -q test_lsp_v3.py test_lsp_v3_multiprocess.py
journalctl -u lixyswarm-seed --since today
```

Validate DNS rotation, IPv4/IPv6, restart identity persistence, address-book recovery, connection limits, seed shutdown continuity, and common NAT clients before advertising the seed publicly.
