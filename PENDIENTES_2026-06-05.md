# Current Engineering Backlog

**Filename retained for compatibility; backlog re-audited 2026-06-23.**

## P0: block unsafe public deployment

- [x] Reject unsigned packets on the default LSP v3 path; v2 is explicit compatibility only.
- [x] Add session/message IDs, sequences, replay cache, network ID, and freshness rules.
- [x] Bound v3 frames, JSON, vectors, message rates, and byte rates; add global connection quotas and fuzzing next.
- [x] Remove the relay dependency: v3 establishes direct persistent sessions and does not flood packets.
- [x] Validate peer-advertised addresses and reject private/link-local abuse on the public path.
- [x] Encrypt all post-HELLO LSP v3 payloads with signed X25519/HKDF/ChaCha20-Poly1305 sessions.
- [x] Add coarse network-group selection plus hashed, decaying local misbehavior scores and persistent exponential temporary bans.
- [ ] Add in-session key rotation, identity recovery/rotation, result/capability reputation, and adversarial eclipse/Sybil defenses.
- [ ] Put the API behind TLS; authenticate chat/history/admin; add rate and size limits.
- [ ] Replace wildcard production CORS with an explicit allowlist.
- [ ] Make personal-memory encryption mandatory in production with key rotation.
- [ ] Add secret/PII scanning to CI and audit existing Git history.
- [ ] Commission an independent security and privacy review.

## P1: match the paper precisely

- [ ] Reconcile the parameter count and publish a script separating trainable, frozen, and external parameters.
- [ ] Decide whether code or manuscript is authoritative for block size defaults.
- [ ] Implement the paper's exact fitness/confidence/role aggregation or revise the equation.
- [ ] Reconcile the five-factor importance weights and continuation/topic-shift feedback.
- [ ] Unify the six paper roles, three runtime agent weights, and sect role taxonomy.
- [ ] Finish LSP v3 lifecycle integration: `NodeManager` registration and typed-job selection are implemented; verified-capacity-driven Dolphin/sect allocation remains.
- [ ] Make Phase B consolidation a durable background lifecycle and measure its benefit.
- [ ] Make Metabolic Hunger learn from recorded outcomes or describe it as deterministic.

## P1: global-memory integrity

- [ ] Sign and persist provenance for each shared memory item/batch.
- [ ] Validate schemas, dimensions, values, text size, and content policy before import.
- [ ] Quarantine untrusted imports before activation.
- [ ] Add reputation, conflict resolution, poisoning detection, revocation, and rollback.
- [ ] Add end-to-end negative privacy tests across publisher, network, and memory paths.

## P1: reproducibility

- [ ] Publish actual private-data-free artifact hashes and an official threshold-signed genesis manifest; the format/tooling is implemented.
- [ ] Pin dependencies and publish container/build instructions.
- [ ] Provide dataset provenance and exact evaluation commands.
- [ ] Add paper-table regeneration and metric schema validation to CI.
- [ ] Separate historical benchmark claims from current CI results.

## P2: final-topology multi-node validation

- [x] Add in-process and separate-interpreter three-node tests proving discovery and encrypted direct communication after graceful/abrupt seed shutdown.
- [ ] Run the same final v3 topology on 3-10 multi-region public nodes for at least seven days.
- [ ] Measure latency, throughput, packet loss, partitions, churn, and recovery.
- [ ] Implement a real merge window and benchmark its effect.
- [ ] Exercise mixed protocol versions and rolling upgrades.
- [ ] Add fuzzing and property tests for every wire decoder.

## P1: distributed compute and artifacts

- [x] Add persisted opt-in contribution policies and worker-side resource leases.
- [x] Add signed-origin, content-addressed, declarative work with allowlisted handlers and no peer code execution.
- [x] Add isolated distributed inference that cannot access personal runtime state.
- [x] Add SHA-256 artifact manifests, resumable chunks, quotas, atomic commit, and complete verification.
- [x] Add exact-model bounded gradient computation over safe NumPy token artifacts; never apply results automatically.
- [x] Add three-to-31-peer exact-input gradient quorum and streaming coordinate-median candidate artifacts.
- [x] Add threshold-signed model release manifests, local trust thresholds, pinned genesis, revocation, monotonic activation, and explicit rollback.
- [x] Add encrypted P2P release announcement, trust-before-download artifact acquisition, deduplicated relay, and persisted opt-in auto-activation.
- [ ] Add dataset provenance manifests, official key operations/recovery, and multi-provider/DHT content lookup.
- [ ] Add content/provider discovery, replication, availability scoring, and garbage collection.
- [x] Bound the in-memory inbound work queue and add per-identity concurrent/rate quotas with signed overload rejection.
- [x] Add authenticated requester cancellation, deadline-aware handler context, and cooperative polling in built-in inference/training.
- [x] Add bounded same-job fallback for automatically scheduled single-peer work and receipt-derived gradient artifact fetching.
- [x] Add bounded failed-member replacement for exact-cardinality verified inference and gradient quorums under one caller deadline.
- [ ] Add crash-persistent quorum/job recovery, killable process isolation, durable queues where required, and full fair-share accounting.
- [ ] Move work into an OS/container sandbox with CPU/GPU/RAM/disk/network enforcement.
- [x] Persist portable Ed25519 worker result receipts in gradient quorum provenance.
- [x] Add exact-model deterministic replicated inference, coarse network-group selection, strict output majority, and supporting receipts.
- [x] Add optional persistent Ed25519-bound identity work and requester-enforced compute difficulty; keep it disabled by default.
- [x] Issue and persist dual-signed, duplicate-resistant credits for gradient results included in validated quorum aggregates.
- [x] Exchange bounded useful-work proofs with direct peers and prioritize firsthand, then issuer-diverse, verified evidence in scheduling.
- [x] Add persistent identity-aged one-in-five newcomer exploration without reducing available quorum network-group diversity.
- [ ] Add Sybil-independent issuer trust; neither useful-work credits, continuity aging, nor optional Hashcash alone establishes independence.
- [ ] Add poisoning/anomaly tests, gradient privacy defenses, and audited promotion rules before model updates.

## P2: Internet topology and operations

- [ ] Operate redundant DNS seeds under separate failure domains; multi-endpoint resolution exists in code.
- [ ] Implement and test DHT discovery.
- [ ] Validate outbound-only participation across common NATs and IPv6 networks.
- [ ] Build unprivileged packages/containers, migrations, SBOM, and signed releases; the seed systemd unit is hardened and unprivileged.
- [ ] Add structured telemetry, SLOs, alerts, backups, and disaster recovery.

## Long-term research

- [ ] Reputation-weighted decentralized consensus.
- [ ] Safe decentralized model promotion governance; threshold-signed releases and unapplied gradient candidates exist, but collective proposal/acceptance policy does not.
- [ ] GrowthGate with empirical promotion/rollback criteria.
- [ ] Sandboxed self-modification with containment and human release authority.
- [ ] Multimodal nodes after text/network behavior is reproducible and secure.

The complete acceptance gates are in `INTERNET_SCALE_READINESS.md`.
