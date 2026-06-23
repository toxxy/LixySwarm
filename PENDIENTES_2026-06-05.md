# Current Engineering Backlog

**Filename retained for compatibility; backlog re-audited 2026-06-22.**

## P0: block unsafe public deployment

- [ ] Reject unsigned LSP packets.
- [ ] Add message IDs, origin IDs, sequence/nonces, replay cache, and freshness rules.
- [ ] Bound TCP frames, decompression, JSON, vectors, connections, and handler concurrency.
- [ ] Preserve origin TTL/decay across relays and add loop/duplicate suppression.
- [ ] Validate peer-advertised addresses and prevent SSRF/private-address abuse.
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
- [ ] Wire authenticated LSP peer lifecycle into `NodeManager` and Dolphin/sect scaling.
- [ ] Make Phase B consolidation a durable background lifecycle and measure its benefit.
- [ ] Make Metabolic Hunger learn from recorded outcomes or describe it as deterministic.

## P1: global-memory integrity

- [ ] Sign and persist provenance for each shared memory item/batch.
- [ ] Validate schemas, dimensions, values, text size, and content policy before import.
- [ ] Quarantine untrusted imports before activation.
- [ ] Add reputation, conflict resolution, poisoning detection, revocation, and rollback.
- [ ] Add end-to-end negative privacy tests across publisher, network, and memory paths.

## P1: reproducibility

- [ ] Publish artifact hashes and a private-data-free release manifest.
- [ ] Pin dependencies and publish container/build instructions.
- [ ] Provide dataset provenance and exact evaluation commands.
- [ ] Add paper-table regeneration and metric schema validation to CI.
- [ ] Separate historical benchmark claims from current CI results.

## P2: trusted multi-node validation

- [ ] Run a 3-10 node, multi-region staging network for at least seven days.
- [ ] Measure latency, throughput, packet loss, partitions, churn, and recovery.
- [ ] Implement a real merge window and benchmark its effect.
- [ ] Exercise mixed protocol versions and rolling upgrades.
- [ ] Add fuzzing and property tests for every wire decoder.

## P2: Internet topology and operations

- [ ] Operate redundant DNS seeds/relays under separate failure domains.
- [ ] Implement and test DHT discovery.
- [ ] Add NAT traversal and IPv6 support.
- [ ] Build unprivileged packages/containers, migrations, SBOM, and signed releases.
- [ ] Add structured telemetry, SLOs, alerts, backups, and disaster recovery.

## Long-term research

- [ ] Reputation-weighted decentralized consensus.
- [ ] Safe distributed training and model-version governance.
- [ ] GrowthGate with empirical promotion/rollback criteria.
- [ ] Sandboxed self-modification with containment and human release authority.
- [ ] Multimodal nodes after text/network behavior is reproducible and secure.

The complete acceptance gates are in `INTERNET_SCALE_READINESS.md`.
