# LSP Architecture Decisions

**Updated:** 2026-06-22

**Status:** implemented core plus unimplemented Internet target

## Scope

LSP is a coordination protocol for pheromones, peer metadata, and shareable Matriarca deltas. It is not a general distributed-compute scheduler and does not currently partition model execution across nodes.

## Current decisions

### Identity

Each node generates an Ed25519 key pair and uses the 32-byte public key as its node ID. Private identity files remain local with mode 0600. Public keys provide stable pseudonymous identity, not reputation or authorization.

Known limitation: `LSPPacket.verify()` currently accepts packets that do not set the signed flag. Mandatory signatures are a release blocker.

### Memory sovereignty

Runtime conversation writes go to the Personal Matriarca. Export reads only the Global Matriarca. Personal encryption is enabled only when a key is configured. Shared global items are treated as untrusted input on an open network; current filters are not sufficient for that threat model.

### Transport split

- UDP 7337 for low-latency, lossy pheromone packets.
- TCP 7338 for handshake, peer lists, and global deltas.

TCP is currently thread-per-connection and lacks strict pre-allocation frame bounds. The signed envelope has no sequence number or nonce.

### Discovery

The public source contains no operator IP. Explicit seeds come from `LIXYSWARM_BOOTSTRAP_SEEDS`; saved peers and peer exchange extend the list. Public DNS seeds and DHT discovery are target architecture, not current features.

### Relay semantics

The target design preserves original packet identity, remaining TTL, decay, and message ID across relay hops. The current standalone relay instead emits a new pheromone packet. This must be redesigned before cyclic topologies or untrusted relays.

### Reputation and consensus

Node reputation, weighted consensus, memory conflict resolution, and Byzantine/Sybil resistance are not implemented. Fitness in pheromone payloads is sender-provided and must not be trusted as a security weight.

## Target topology

The intended public network requires multiple interchangeable DNS seeds, a Kademlia-style routing layer, NAT traversal/relay selection, bounded anti-entropy, and independent operators. No single bootstrap or dashboard service may be required for continued operation.

Progress toward that target is gated by `INTERNET_SCALE_READINESS.md` rather than by roadmap dates.
