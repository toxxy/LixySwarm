# Distributed Protocol Overview

**Updated:** 2026-06-22

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
- Work: canonical signed-origin offers with deadlines, idempotency, allowlisted local handlers, and consent/resource leases.
- Results: portable Ed25519 receipts bind worker/requester/job/output and are preserved in gradient quorum provenance.
- Inference: complete prompt jobs on a matching consenting peer; remote requests cannot access or mutate the operator's personal memory/session state.
- Artifacts: SHA-256 manifests and resumable verified chunks without source filenames or paths.
- Training: bounded token batches against an exact local model hash; results are NPZ gradient artifacts and are never applied automatically. Quorum mode validates three-to-31 distinct peer results and produces a streaming coordinate median.

Nodes behind NAT need only outbound TCP connectivity. Public nodes and seeds accept inbound sessions. No VPN or port forwarding is required for an outbound-only participant.

## Current limits

- Established-session payloads are signed and ChaCha20-Poly1305 encrypted after a signed X25519 HELLO. HELLO metadata, peer addresses, frame sizes, timing, and volume remain visible; in-session rekeying is not implemented.
- No public built-in DNS seed domain is committed yet.
- No DHT, Sybil defense, decentralized capability reputation, or verified hardware capacity. Local protocol misbehavior scores and temporary bans are implemented.
- Scheduling declarations are self-reported; there is no result reputation, fairness, hardware attestation, redundant execution, or failure rescheduling.
- Coordinate median tolerates a minority of arbitrary candidates only when quorum identities are genuinely independent. Current pseudonymous identities are cheap and therefore do not establish that assumption.
- Threshold-signed release manifests, local trust policies, pinned genesis, monotonic activation, revocation, and explicit rollback are implemented. Official keys/genesis, network announcement/discovery, and collective promotion policy remain external.

See `LSP_SPEC.md` for the wire format and `INTERNET_SCALE_READINESS.md` for release gates.
