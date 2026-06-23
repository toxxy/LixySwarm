# Security and Privacy

**Updated:** 2026-06-23

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
- Remote work admission reserves from a fixed queue and enforces per-identity active/queued and minute-window quotas. Every parseable overload rejection is signed; the executor's internal queue cannot grow without the protocol reservation succeeding.
- Work cancellation is bound to the authenticated requester identity and active job ID. Built-in inference/training poll cancellation and deadlines; outputs produced after cancellation are discarded and the final error receipt is signed.
- Automatic single-peer fallback is bounded to five by the local API and three by default, uses distinct already-eligible peers and one total deadline, and never overrides an explicit peer assignment. Authenticated session loss wakes the request immediately; late results from a previous attempt fail the expected-peer check.
- Remote inference cannot read/write personal Matriarca memory or session/Dolphin state.
- Artifact manifests expose hashes, sizes, types, and timestamps—not source filenames or paths. Chunk and full-object hashes are verified before commit.
- Training workers never load peer checkpoints and never apply returned gradients; they require the exact locally loaded model hash and safe NumPy token arrays.
- Verified inference and gradient quorum modes replace only failed workers, up to an explicit bound, while preserving the configured number of accepted distinct identities under one total deadline. Attempted identities cannot re-enter; replacements prefer network groups not retained by successful/active members. Gradient mode validates archive bounds/metadata/tensor shapes/finiteness and produces a coordinate median without applying it.
- Each accepted work result has a portable Ed25519 receipt bound to worker, requester, job, content, and time.
- Verified inference requires an exact deterministic majority from distinct model-matched peers, preferring separate coarse network groups. The prompt is disclosed to every selected worker and cheap Sybil identities can still collude.
- Hashcash-style identity work is disabled by default. Operators may configure a requester-enforced minimum bound to each Ed25519 identity as admission friction, but it does not prove different owners or useful computation.
- Gradient contributors receive credits only after inclusion in a validated quorum aggregate. Each credit contains the worker-signed result receipt plus the requester's signed aggregation claim and is stable across replay. Connected peers present at most 16 credits after encryption; HELLO reputation is discarded, both signatures are verified locally, and the scheduler prefers firsthand accepted work. Third-party issuer trust and Sybil-independent reputation remain unsolved.
- Every fifth local selection may explore the least-selected identity observed for at least 60 seconds. The bounded private history stores no IPs, and quorum exploration is skipped when it would reduce available network-group diversity. This is free anti-starvation scheduling, not economic stake or Sybil resistance.
- Model releases require a locally configured threshold of trusted Ed25519 signers; optional genesis pinning, revocation, monotonic activation, weights-only loading, and explicit rollback prevent silent downgrade.
- Release announcements are signature-checked before download; referenced artifacts are hash-verified. Automatic activation is a persisted explicit opt-in and is off by default.
- Outbound selection prefers distinct coarse network groups. Protocol violations accrue local decaying scores and persistent temporary bans keyed by hashes rather than raw identifiers.

These controls are incomplete. HELLO/traffic metadata remains visible and long sessions do not rekey; resource claims and compute results remain self-reported/unattested; admitted work runs in-process, so cooperative cancellation cannot preempt a stuck handler or CUDA kernel; identity quotas are not Sybil resistance; personal encryption is optional; gradients can leak training data at their endpoints; and most API endpoints are unauthenticated. Legacy v2 has weaker verification and must not be enabled on the public network.

## Required production controls

1. TLS at every public HTTP boundary; authenticated chat/history/admin routes; separate credentials by role.
2. Mandatory encrypted personal memory with managed keys, rotation, backups, and deletion.
3. Independent review of LSP key agreement, in-session key rotation, peer sanctions, eclipse resistance, fuzzing, and load validation.
4. Global-memory provenance, per-item validation, quarantine, reputation, revocation, and rollback.
5. Explicit CORS allowlist, request limits, concurrency limits, timeouts, and audit logs that exclude content/secrets.
6. Unprivileged services, minimal filesystem permissions, network firewalling, dependency scanning, SBOM, and signed releases.
7. Automated secret/PII scans on commits and full history plus an independent security review.
8. Sybil-independent quorum membership, cross-hardware replicated-inference validation, useful-credit issuer reputation, poisoning/gradient-inversion defenses, official separated signing operations/key recovery, and a collective promotion policy. Receipts, credits, threshold manifests, exact majority, and coordinate median alone are insufficient.

See `INTERNET_SCALE_READINESS.md` for the full public-network gate.

## Reporting

Do not post suspected vulnerabilities with exploit details in a public issue. Use the repository owner's private security-reporting channel. The project must add a stable public security contact before an open testnet.
