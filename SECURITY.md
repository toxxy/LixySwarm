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
- LSP v3 requires Ed25519 signatures, persistent identities, bounded frames, freshness checks, message/session IDs, monotonic sequences, replay caching, and per-session rate windows.
- Signed X25519 HELLO negotiation derives an HKDF-SHA256 process-session key; every established payload requires ChaCha20-Poly1305 authentication/encryption.
- Personal/global Matriarca paths are separate; global export does not read the personal bank.
- Personal memory supports AES-256-GCM when `LIXYSWARM_MATRIARCA_KEY` is configured.
- Checkpoint reads changed in the publisher use `weights_only=True`.
- Publisher IP storage is off by default. Enable only with `LIXYSWARM_STORE_PUBLISHER_IP=true`.
- Publishing peer addresses is off by default. Enable only with `LIXYSWARM_PUBLISH_NETWORK_ADDRESSES=true`.
- Exposing peer addresses through API status is off by default. Enable only with `LIXYSWARM_EXPOSE_PEER_HOSTS=true`.
- Compute is disabled until a persisted `balanced` or `maximum` contribution policy records explicit consent.
- Peer work cannot contain executable code and can invoke only fixed local handlers. Recursive payload limits, deadlines, rate windows, and resource leases are enforced.
- Remote inference cannot read/write personal Matriarca memory or session/Dolphin state.
- Artifact manifests expose hashes, sizes, types, and timestamps—not source filenames or paths. Chunk and full-object hashes are verified before commit.
- Training workers never load peer checkpoints and never apply returned gradients; they require the exact locally loaded model hash and safe NumPy token arrays.
- Outbound selection prefers distinct coarse network groups. Protocol violations accrue local decaying scores and persistent temporary bans keyed by hashes rather than raw identifiers.

These controls are incomplete. HELLO/traffic metadata remains visible and long sessions do not rekey; resource claims and compute results remain self-reported/unattested; work runs in-process; local bans do not provide Sybil or global reputation defenses; personal encryption is optional; gradients can leak training data at their endpoints; and most API endpoints are unauthenticated. Legacy v2 has weaker verification and must not be enabled on the public network.

## Required production controls

1. TLS at every public HTTP boundary; authenticated chat/history/admin routes; separate credentials by role.
2. Mandatory encrypted personal memory with managed keys, rotation, backups, and deletion.
3. Independent review of LSP key agreement, in-session key rotation, peer sanctions, eclipse resistance, fuzzing, and load validation.
4. Global-memory provenance, per-item validation, quarantine, reputation, revocation, and rollback.
5. Explicit CORS allowlist, request limits, concurrency limits, timeouts, and audit logs that exclude content/secrets.
6. Unprivileged services, minimal filesystem permissions, network firewalling, dependency scanning, SBOM, and signed releases.
7. Automated secret/PII scans on commits and full history plus an independent security review.
8. Replicated compute verification, poisoning/gradient-inversion defenses, robust aggregation, signed release manifests, and an explicit promotion authority.

See `INTERNET_SCALE_READINESS.md` for the full public-network gate.

## Reporting

Do not post suspected vulnerabilities with exploit details in a public issue. Use the repository owner's private security-reporting channel. The project must add a stable public security contact before an open testnet.
