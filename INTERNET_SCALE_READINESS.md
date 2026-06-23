# Internet-Scale Readiness

**Assessment date:** 2026-06-22

**Verdict:** not ready for untrusted public-Internet or mass deployment

LSP v3 provides the final topology foundation and an initial compute/data plane. Persistent outbound sessions, seed-independent peer exchange, typed jobs, isolated inference, content-addressed artifacts, and gradient-candidate computation now work in code and tests. This is still not a safe public compute network.

## Implemented network foundation

- Mandatory Ed25519 signatures and stable identities.
- Signed X25519 key agreement and mandatory ChaCha20-Poly1305 encryption for every post-HELLO payload.
- Session IDs, message IDs, monotonic sequences, timestamp bounds, and replay cache.
- One-megabyte payload/frame bounds, no compressed frames, and per-session rate windows.
- Asynchronous persistent TCP sessions suitable for outbound-only NAT participants.
- Validated peer advertisements, persistent address book, retry/backoff, and peer exchange.
- Seed-independent three-node communication proven by an automated test.
- Bounded resource profiles registered in runtime `NodeManager`.
- Unprivileged systemd seed service template.
- Persisted opt-in resource policy with CPU/GPU/RAM/disk/bandwidth declarations and leases.
- Coarse network-group-diverse outbound preference plus hashed, decaying local misbehavior scores and exponential temporary bans.
- Declarative content-addressed work; fixed local handlers only, with no peer code execution.
- Isolated remote inference that cannot read/write personal Matriarca or conversation state.
- SHA-256 artifact manifests, resumable chunk transfer, per-chunk validation, atomic commit, and full-file verification.
- Exact-model bounded gradient jobs over safe NumPy token artifacts; gradients are returned but never applied.
- Three-to-31-identity gradient quorums with exact metadata/tensor validation, bounded ZIP inspection, and streaming coordinate-median output.
- Portable worker result receipts and a threshold-signed local model release registry with pinned genesis, revocation, monotonic activation, and explicit rollback.
- Encrypted P2P release announcement, trust-before-download validation, direct artifact acquisition, deduplicated relay, and persisted opt-in auto-activation.
- A complete local run of 167 tests on the assessment date.

## Release blockers

### P0: protocol and API security

- Add in-session encryption-key rotation, persistent-identity rotation/recovery, handshake downgrade tests, and independent cryptographic review.
- Extend local bans into measured peer/result reputation, autonomous-system-aware diversity, feeler connections, eclipse/Sybil resistance, and privacy-safe abuse telemetry.
- Fuzz every decoder and load-test connection, message, and address-book limits.
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

- Operate multiple public DNS seeds under separate operators/failure domains and commit their domains as defaults.
- Implement a Kademlia-style DHT or another measured decentralized discovery layer.
- Validate IPv6 and common consumer NAT networks. Outbound-only nodes must remain full participants without port forwarding.
- Add autonomous-system-aware diversity, health scoring, circuit breaking, feeler connections, and adversarial eclipse tests beyond the current coarse group preference.

### Data plane

- Add backpressure, congestion control, batching, bandwidth budgets, and explicit work queues.
- Define delivery/retransmission semantics and mixed-version compatibility.
- Benchmark latency, throughput, loss, churn, partitions, and recovery across real regions and thousands of simulated peers.

### Model and training distribution

- Threshold signatures now prove authority according to each node's local trust policy, and trusted releases propagate peer-to-peer. Publish independent official signer keys and a pinned genesis, add multi-provider/DHT content lookup, dataset provenance, key-rotation/recovery procedures, and collective promotion rules. The public Git repository does not ship checkpoints.
- Current inference is replicated whole-request execution. Benchmark it, then add redundant candidate verification/expert routing only where measurements justify it.
- Add scheduler fairness, hardware attestation/capability validation, process/container job isolation, quotas, cancellation, cost limits, and failure rescheduling.
- Make quorum membership Sybil-independent, quantify non-determinism, add poisoning/anomaly tests, and connect receipt-backed aggregates to an audited threshold release proposal. Receipts exist, but cheap identities invalidate the median's honest-majority assumption on an open network.
- Address gradient inversion and endpoint training-data leakage; encrypted transport does not make untrusted workers or requesters trustworthy.

### Operations

- Build reproducible containers/packages, pinned lockfiles, migrations, and release signatures/SBOMs.
- Run as an unprivileged service with read-only filesystem regions and minimal capabilities.
- Add structured logs, metrics, traces, health/readiness probes, SLOs, alerting, capacity planning, backups, and disaster recovery.
- Add rolling upgrades and mixed-version compatibility tests.

## Staged delivery plan

| Gate | Required outcome | Acceptance evidence |
|---|---|---|
| 1. Seed integration | Final v3 topology and typed work on 3-10 public nodes | Seed shutdown continuity, NAT clients, artifact/inference/gradient jobs, encrypted personal banks, TLS API |
| 2. Adversarial testnet | Untrusted nodes on the final protocol | Fuzz/load/poisoning reports, revocation, recovery under churn |
| 3. Regional beta | Redundant seeds and direct peers | Multi-region SLOs, seed failover, privacy review, signed releases |
| 4. Public network | Decentralized discovery and governance | Independent security audit, incident process, measured capacity and abuse controls |
| 5. Mass scale | Sustainable operation under high churn | Load evidence at target concurrency, cost model, no single required operator service |

## Definition of Internet-ready

Internet-ready means all P0 items are closed, every public endpoint is authenticated or intentionally public with rate limits, global memory is provenance-checked, at least two independent seed failure domains exist, an outbound-only node can join behind common NAT, and adversarial multi-node tests demonstrate recovery without private-data leakage.

None of those conditions should be inferred from the dashboard's `internet.ready` field. That field currently reports configuration reachability only and should be treated as an observability hint.
