# Distributed Protocol Overview

**Updated:** 2026-06-23

LSP v3 is the default network path. It uses persistent outbound-friendly TCP sessions for peer exchange, pheromones, Global Matriarca deltas, typed work, and verified artifact chunks.

## Bootstrap without dependency

```text
new node
  -> DNS/bootstrap seed
  -> persistent signed session
  -> receives peer addresses
  -> connects directly to learned peers
  -> seed can disappear
```

Saved addresses and peer exchange are preferred after initial discovery. The VPS runs `node_daemon.py` as a seed, not as a required relay.

Configure multiple endpoints until official DNS seeds are published:

```bash
export LIXYSWARM_BOOTSTRAP_SEEDS='seed-a.example.net:7338,seed-b.example.net:7338'
```

## Current data flows

- Pheromones: float16 binary payload over the persistent signed session.
- Global memory: global-only filtered JSON delta over the same session.
- Peer discovery: validated address lists, persistent address book, retry/backoff.
- Resources: bounded capability declaration registered in `NodeManager`.
- Optional identity cost: when configured, compute peers persist a key-bound SHA-256 work stamp and requesters enforce their own minimum. It is disabled by default and is not the network's useful-work reward.
- Work: canonical signed-origin offers with deadlines, idempotency, allowlisted local handlers, consent/resource leases, a fixed inbound queue, per-identity concurrency/rate quotas, authenticated requester cancellation, and bounded automatic retry across distinct eligible peers. Overload/cancellation results carry the worker's portable signature.
- Results: portable Ed25519 receipts bind worker/requester/job/output and are preserved in gradient quorum provenance.
- Useful-work evidence: connected workers present up to 16 dual-signed credits over the encrypted session. The receiver rejects self-declared HELLO reputation, verifies every credit locally, and ranks firsthand accepted work before third-party evidence and raw hardware capacity.
- Fair entry: requester-local private history persists pseudonymous IDs and counters. Every fifth selection may choose a worker identity known for at least 60 seconds; quorum replacement occurs only when it preserves the retained network-group diversity.
- Releases: trusted threshold manifests gossip across encrypted direct sessions; receivers fetch and verify model/support artifacts from the announcer, then accept or explicitly auto-activate according to persisted local policy.
- Inference: complete prompt jobs on a matching consenting peer; remote requests cannot access or mutate the operator's personal memory/session state.
- Verified inference: deterministic greedy execution on three-to-nine exact-model peers, selected across coarse network groups, requires a strict identical-output majority and returns every supporting signed receipt.
- Artifacts: SHA-256 manifests and resumable verified chunks without source filenames or paths.
- Training: bounded token batches against an exact local model hash; results are NPZ gradient artifacts and are never applied automatically. Quorum mode validates three-to-31 distinct peer results, produces a streaming coordinate median, and returns a dual-signed useful-work credit to each included worker.

Nodes behind NAT need only outbound TCP connectivity. Public nodes and seeds accept inbound sessions. No VPN or port forwarding is required for an outbound-only participant.

## Current limits

- Established-session payloads are signed and ChaCha20-Poly1305 encrypted after a signed X25519 HELLO. HELLO metadata, peer addresses, frame sizes, timing, and volume remain visible; in-session rekeying is not implemented.
- No public built-in DNS seed domain is committed yet.
- No DHT, Sybil defense, decentralized capability reputation, or verified hardware capacity. Local protocol misbehavior scores and temporary bans are implemented.
- Scheduling capacity declarations are self-reported. Verified useful-work evidence and bounded newcomer exploration affect ordering. Inbound admission is memory-bounded and identity-quotad. Cooperative deadline/cancellation checks cover built-in inference, training, and artifact transfer. Automatically selected single-peer jobs retry send failure, disconnect, timeout, overload/capacity rejection, and peer-specific handler failure within one total deadline. Verified inference and gradient quorums replace a bounded number of failed members, retain their exact configured cardinality, prefer replacements outside retained network groups, and expose attempt/replacement counts. Explicit single-peer assignments remain exact. Durable crash recovery, Sybil-independent issuer reputation, hardware attestation, and forced termination remain missing.
- Replication reduces reliance on one worker but does not defeat coordinated Sybil identities. It also exposes the requested prompt to every selected worker.
- Optional identity work can raise the cost of key creation but does not prove distinct ownership. It is a configurable abuse control, not the primary contribution system.
- Useful-work credits bind a worker-signed result receipt to a requester-signed aggregation claim and deduplicate repeated use of one result. Direct peers exchange bounded presentations and the scheduler prioritizes credits it issued itself, then issuer diversity. Malicious or colluding pseudonymous issuers can still make false quality claims.
- Credits are not currency. The protocol defines no payment, fee, stake, mining reward, or mandatory Hashcash cost; all resource contribution remains explicit opt-in under the local policy.
- Coordinate median tolerates a minority of arbitrary candidates only when quorum identities are genuinely independent. Current pseudonymous identities are cheap and therefore do not establish that assumption.
- Threshold-signed release manifests, local trust policies, pinned genesis, encrypted P2P announcement/acquisition, monotonic activation, revocation, and explicit rollback are implemented. Official keys/genesis, multi-provider lookup, and collective promotion policy remain external.

See `LSP_SPEC.md` for the wire format and `INTERNET_SCALE_READINESS.md` for release gates.
