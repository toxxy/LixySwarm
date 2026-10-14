# LSP Architecture Decisions

**Updated:** 2026-10-13

## Non-negotiable topology

- Seeds introduce peers. A reachable consenting node can additionally offer a replaceable relay service.
- Ordinary NAT nodes use outbound sessions and fall back to an advertised relay when direct TCP fails.
- Direct sessions survive seed loss. A relayed path requires its relay or another reachable relay connected to both endpoints.
- Every message is signed, bounded, freshness-checked, and replay-protected.
- Local resources are declared to the swarm but are not trusted without verification.
- Personal memory remains local; only Global Matriarca deltas use the network.

## Why persistent TCP first

LSP v2 required inbound UDP reachability for pheromones and used short TCP connections for gossip. That model could not support mass participation behind consumer NAT. LSP v3 multiplexes the essential traffic over long-lived connections initiated by the participant. TCP was selected as the first universal transport because it crosses common networks reliably. Signed X25519 negotiation plus ChaCha20-Poly1305 now protects application payloads; QUIC remains an optional future transport, not a requirement for privacy.

## Identity and trust

Ed25519 proves continuity of a pseudonymous node identity and message integrity. It does not prove that resource declarations, memories, gradients, or inference results are correct. Reputation and redundant verification must remain separate from cryptographic identity.

## Discovery layers

1. Persistent address book.
2. Built-in public bootstrap plus operator-configurable DNS/bootstrap seeds.
3. Peer exchange.
4. DHT after the first three layers are adversarially tested.

A DHT is not a substitute for safe sessions or verified peer data. It is deliberately later in the sequence.

## NAT relay fallback

The Internet probe on two physical computers showed that joining a public seed did not make the advertised listen ports reachable. LSP now tries direct TCP first, then opens a bounded virtual byte stream through a connected peer advertising `relay_v1`.

Endpoints perform the existing signed HELLO/X25519 handshake inside that stream and pin the expected endpoint identity. The relay forwards ciphertext; application callbacks, work offers and result receipts retain the endpoint identity. A direct connection supersedes a relayed connection. After relay failure, peer maintenance can use another connected compatible relay.

Serving is explicitly enabled with `LIXYSWARM_RELAY_ENABLED=1` for either `node_daemon.py` or `lixyswarm start`. Defaults bound the relay to 64 streams, four per endpoint, 64 KiB transport chunks and the existing session message/byte quotas. Nested tunnels are rejected. `test_lsp_relay.py` covers NAT fallback, receipt identities, ciphertext integrity, capacity, cleanup, direct upgrade and alternate-relay recovery.

## Compute architecture direction

The compute layer distributes complete inference requests and bounded gradient jobs rather than latency-sensitive tensor operations per token. Training inputs/results use content-addressed dataset/model/gradient identifiers. Gradient quorum mode validates exact inputs from three-to-31 distinct identities and creates a coordinate-median artifact. Verified inference and gradient quorums replace bounded failed members under one deadline without accepting a smaller result set; attempted identities are excluded and available path diversity is retained. No gradient is applied. Included workers receive dual-signed useful-work credits and present bounded evidence over encrypted sessions. The scheduler prefers its own firsthand credits and then issuer diversity, while a persisted continuity-aged exploration slot prevents strict newcomer starvation. Optional identity work is off by default. None of these mechanisms is currency or currently provides Sybil-independent quorum selection or authority to activate a shared release.

## Remaining decisions

- In-session key rotation and independent review of the implemented X25519/HKDF/ChaCha20-Poly1305 construction.
- A second independent seed and stable public seed domains.
- Network reputation and eclipse resistance.
- Network announcement/discovery for the implemented threshold-signed local release registry, plus official trust-root/genesis operations.
- Process/container sandboxing beyond the implemented consent governor and allowlisted scheduler.
- Sybil-independent quorum selection beyond the implemented deterministic exact-majority inference and coordinate-median gradient quorums, plus network governance for promotion.
