# LSP Architecture Decisions

**Updated:** 2026-06-23

## Non-negotiable topology

- The VPS is a replaceable seed, never a required relay or coordinator.
- Ordinary NAT nodes participate through persistent outbound sessions.
- Learned peers replace seed dependency immediately after bootstrap.
- Every message is signed, bounded, freshness-checked, and replay-protected.
- Local resources are declared to the swarm but are not trusted without verification.
- Personal memory remains local; only Global Matriarca deltas use the network.

## Why persistent TCP first

LSP v2 required inbound UDP reachability for pheromones and used short TCP connections for gossip. That model could not support mass participation behind consumer NAT. LSP v3 multiplexes the essential traffic over long-lived connections initiated by the participant. TCP was selected as the first universal transport because it crosses common networks reliably. Signed X25519 negotiation plus ChaCha20-Poly1305 now protects application payloads; QUIC remains an optional future transport, not a requirement for privacy.

## Identity and trust

Ed25519 proves continuity of a pseudonymous node identity and message integrity. It does not prove that resource declarations, memories, gradients, or inference results are correct. Reputation and redundant verification must remain separate from cryptographic identity.

## Discovery layers

1. Persistent address book.
2. Multiple DNS/bootstrap seeds.
3. Peer exchange.
4. DHT after the first three layers are adversarially tested.

A DHT is not a substitute for safe sessions or verified peer data. It is deliberately later in the sequence.

## Compute architecture direction

The compute layer distributes complete inference requests and bounded gradient jobs rather than latency-sensitive tensor operations per token. Training inputs/results use content-addressed dataset/model/gradient identifiers. Gradient quorum mode validates exact inputs from three-to-31 distinct identities and creates a coordinate-median artifact. Verified inference and gradient quorums replace bounded failed members under one deadline without accepting a smaller result set; attempted identities are excluded and available path diversity is retained. No gradient is applied. Included workers receive dual-signed useful-work credits and present bounded evidence over encrypted sessions. The scheduler prefers its own firsthand credits and then issuer diversity, while a persisted continuity-aged exploration slot prevents strict newcomer starvation. Optional identity work is off by default. None of these mechanisms is currency or currently provides Sybil-independent quorum selection or authority to activate a shared release.

## Remaining decisions

- In-session key rotation and independent review of the implemented X25519/HKDF/ChaCha20-Poly1305 construction.
- Seed domains and independent operators.
- Network reputation and eclipse resistance.
- Network announcement/discovery for the implemented threshold-signed local release registry, plus official trust-root/genesis operations.
- Process/container sandboxing beyond the implemented consent governor and allowlisted scheduler.
- Sybil-independent quorum selection beyond the implemented deterministic exact-majority inference and coordinate-median gradient quorums, plus network governance for promotion.
