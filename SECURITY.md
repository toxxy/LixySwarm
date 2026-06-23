# Security and Privacy

**Updated:** 2026-06-22

**Security status:** trusted-development prototype, not approved for public exposure

## Never commit

- `.env` files, tokens, passwords, API keys, or private keys
- `checkpoints/`, `data/`, personal corpora, memory banks, or session histories
- LSP identity PEM/key files or peer caches
- runtime status files, logs, operator IP addresses, or local filesystem paths

If a secret or private datum has ever entered Git history, deleting it from the latest revision is insufficient. Rotate the secret and rewrite/purge history using an approved process.

## Current controls

- Publisher writes require `LIXYSWARM_PUBLISH_TOKEN` and constant-time comparison.
- LSP send paths use Ed25519 signatures and persistent local identities.
- Personal/global Matriarca paths are separate; global export does not read the personal bank.
- Personal memory supports AES-256-GCM when `LIXYSWARM_MATRIARCA_KEY` is configured.
- Checkpoint reads changed in the publisher use `weights_only=True`.
- Publisher IP storage is off by default. Enable only with `LIXYSWARM_STORE_PUBLISHER_IP=true`.
- Publishing peer addresses is off by default. Enable only with `LIXYSWARM_PUBLISH_NETWORK_ADDRESSES=true`.
- Exposing peer addresses through API status is off by default. Enable only with `LIXYSWARM_EXPOSE_PEER_HOSTS=true`.

These controls are incomplete. Encryption is optional, unsigned LSP packets are accepted, no replay protection exists, and most API endpoints are unauthenticated.

## Required production controls

1. TLS at every public HTTP boundary; authenticated chat/history/admin routes; separate credentials by role.
2. Mandatory encrypted personal memory with managed keys, rotation, backups, and deletion.
3. Mandatory LSP signatures, replay prevention, bounded frames/decompression, rate limits, and peer sanctions.
4. Global-memory provenance, per-item validation, quarantine, reputation, revocation, and rollback.
5. Explicit CORS allowlist, request limits, concurrency limits, timeouts, and audit logs that exclude content/secrets.
6. Unprivileged services, minimal filesystem permissions, network firewalling, dependency scanning, SBOM, and signed releases.
7. Automated secret/PII scans on commits and full history plus an independent security review.

See `INTERNET_SCALE_READINESS.md` for the full public-network gate.

## Reporting

Do not post suspected vulnerabilities with exploit details in a public issue. Use the repository owner's private security-reporting channel. The project must add a stable public security contact before an open testnet.
