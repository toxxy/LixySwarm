# Internet-Scale Readiness

**Assessment date:** 2026-06-22

**Verdict:** not ready for untrusted public-Internet or mass deployment

The current network is suitable for local development, loopback tests, trusted LAN experiments, and controlled explicit-peer staging. Setting a public host or relay URL makes a route available; it does not make the system secure, scalable, or production-ready.

## Release blockers

### P0: protocol and API security

- Reject every unsigned LSP packet; define version negotiation and key rotation.
- Add monotonic sequence numbers or nonces, bounded clock skew, replay caches, and duplicate suppression.
- Bound TCP frame size before allocation, cap decompressed size, set connection/read quotas, and limit concurrent handlers.
- Preserve origin, TTL, and decay across relays. The current relay re-signs a fresh packet and resets TTL.
- Validate advertised peer addresses and prevent private-address/loopback SSRF through peer exchange.
- Add node admission policy, rate limits, bans, and abuse telemetry.
- Put the HTTP API behind TLS and a reverse proxy. Authenticate chat, history, administrative status, and publisher endpoints separately.
- Remove wildcard CORS for production and define explicit origins.
- Add input/output size limits, timeouts, cancellation, and per-identity quotas.
- Threat-model model extraction, prompt abuse, memory poisoning, checkpoint attacks, and denial of service.

### P0: privacy and data governance

- Make personal-memory encryption mandatory for production, with managed key rotation and recovery.
- Prove that personal text and embeddings cannot reach global exports across every write/import path.
- Do not publish peer/operator addresses by default; current code now requires explicit opt-in flags.
- Define retention, deletion, consent, export, and incident-response procedures.
- Add automated secret and PII scanning in pre-commit and CI, including Git history.
- Establish dataset licenses, provenance, opt-out handling, and jurisdiction-specific compliance.

### P0: shared-memory integrity

- Sign each global memory item or a verifiable batch manifest and persist origin/provenance.
- Validate schemas, dimensions, numeric ranges, text size, and content policy before import.
- Add reputation/stake/quorum rules, conflict resolution, poisoning detection, revocation, and rollback.
- Separate untrusted candidate memory from trusted active memory.

## Scale architecture gaps

### Discovery and reachability

- Operate multiple public DNS seeds under separate failure domains.
- Implement a Kademlia-style DHT or another measured decentralized discovery layer.
- Add NAT traversal (ICE/STUN/TURN or a documented relay model), IPv6, and relay selection.
- Persist peer quality and use backoff, health checks, circuit breaking, and topology diversity.

### Data plane

- Replace thread-per-connection behavior with a bounded asynchronous or worker architecture.
- Add backpressure, congestion control, batching, real merge windows, and bandwidth budgets.
- Define delivery semantics, message IDs, idempotency, retransmission policy, and protocol compatibility tests.
- Benchmark latency, throughput, loss, churn, partitions, and recovery across real regions and thousands of simulated peers.

### Model and training distribution

- Define how model weights are distributed, verified, versioned, and rolled back. The public Git repository does not ship checkpoints.
- Decide whether nodes perform replicated inference, expert routing, or tensor/pipeline partitioning; the current protocol exchanges signals, not distributed model execution.
- Add scheduler fairness, hardware attestation/capability validation, job isolation, cost limits, and failure recovery.
- Prevent untrusted training data or gradients from corrupting shared releases.

### Operations

- Build reproducible containers/packages, pinned lockfiles, migrations, and release signatures/SBOMs.
- Run as an unprivileged service with read-only filesystem regions and minimal capabilities.
- Add structured logs, metrics, traces, health/readiness probes, SLOs, alerting, capacity planning, backups, and disaster recovery.
- Add rolling upgrades and mixed-version compatibility tests.

## Staged delivery plan

| Gate | Required outcome | Acceptance evidence |
|---|---|---|
| 1. Trusted staging | 3-10 explicitly configured nodes | 7-day run, encrypted personal banks, TLS API, no P0 protocol findings |
| 2. Adversarial testnet | Untrusted nodes allowed with limits | Replay/fuzz/load/poisoning reports, revocation, recovery under churn |
| 3. Regional beta | Redundant seeds and relays | Multi-region SLOs, failover, privacy review, signed releases |
| 4. Public network | Decentralized discovery and governance | Independent security audit, incident process, measured capacity and abuse controls |
| 5. Mass scale | Sustainable operation under high churn | Load evidence at target concurrency, cost model, no single required operator service |

## Definition of Internet-ready

Internet-ready means all P0 items are closed, every public endpoint is authenticated or intentionally public with rate limits, relay loops are bounded, global memory is provenance-checked, at least two independent bootstrap/relay failure domains exist, a node can join behind common NAT, and adversarial multi-node tests demonstrate recovery without private-data leakage.

None of those conditions should be inferred from the dashboard's `internet.ready` field. That field currently reports configuration reachability only and should be treated as an observability hint.
