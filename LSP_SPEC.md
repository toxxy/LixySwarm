# LixySwarm Protocol (LSP) — Specification v2.0

**Status:** Draft + implementation notes (2026-06-05)
**Authors:** LixySwarm Team  
**Repo:** https://github.com/toxxy/LixySwarm  
**Reference implementation:** `src/network/lsp_v2.py`, `src/network/lsp_merge.py`

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Design Goals](#2-design-goals)
3. [Architecture Decisions](#3-architecture-decisions)
4. [Wire Format](#4-wire-format)
   - 4.1 [Base Header (v1 + v2)](#41-base-header)
   - 4.2 [FeromonV2 Binary Payload](#42-feromonv2-binary-payload)
   - 4.3 [Gossip Payload (JSON/TCP)](#43-gossip-payload)
5. [Packet Types](#5-packet-types)
6. [Transport Layer](#6-transport-layer)
7. [Node Identity](#7-node-identity)
8. [TTL and Decay](#8-ttl-and-decay)
9. [Merge-on-Transit](#9-merge-on-transit)
10. [Matriarca Dual Integration](#10-matriarca-dual-integration)
11. [Message Flows](#11-message-flows)
12. [Error Cases](#12-error-cases)
13. [Test Vectors](#13-test-vectors)
14. [Implementation Checklist](#14-implementation-checklist)
15. [Changelog](#15-changelog)

---

## Implementation Status — 2026-06-05

LSP v2 is the canonical path in this repository. The implemented baseline is:

- `SwarmNetwork` defaults to LSP v2 for pheromones and TCP gossip.
- `GOSSIP_DELTA` carries synthetic Global Matriarca deltas; personal memory must not be exported.
- `swarm_publisher.py` publishes dashboard state through `POST /swarm/publish` with token auth.
- Legacy v1 `messages.py` and `transport.py` remain for compatibility only.

Not yet complete: DHT/Kademlia discovery, reputation, nonce/sequence anti-replay, encrypted Personal Matriarca storage, and consensus. Treat those as roadmap items, not available protocol guarantees.

---

## 1. Introduction

LSP (LixySwarm Protocol) is a native binary protocol for distributed pheromone exchange between AI swarm nodes. It is **not** a wrapper over a generic transport — it defines the semantics of pheromone signals, node identity, and swarm topology as first-class protocol concepts.

LSP runs over:
- **UDP** — pheromone signals (fire-and-forget, low latency)
- **TCP** — gossip and handshake (reliable, ordered)

The same protocol runs on a LAN (mDNS/explicit peers) or on the open internet through configured peers/relays. DHT-backed zero-config WAN discovery is planned, not currently complete.

---

## 2. Design Goals

| Goal | Decision |
|------|----------|
| Pheromone packet < 1 KB | float16 tensors, binary encoding (528 bytes for 256d) |
| Zero-copy tensor transfer | Binary struct, not JSON float lists |
| Signal decay across hops | TTL field + decay factor 0.95^hops |
| Prevent duplicate processing | FeromonMergeBuffer: fitness-weighted merge per node |
| Cryptographic node identity | Ed25519 keypair, 32-byte node ID |
| No central server | Peers register each other on handshake/ping |
| Swappable transport | Protocol defines what is sent, not how |
| Any language can implement | This document is the spec; Python is reference only |

---

## 3. Architecture Decisions

### AD-001: Matriarca Dual (Personal + Global)

The Matriarca memory bank is split into two independent banks:

- **PersonalMatriarca**: private to the local node, never shared over the network, optionally encrypted at rest. Contains interaction history, user preferences, session memories.
- **GlobalMatriarca**: shared knowledge accumulated from all nodes in the swarm. Populated via gossip and `merge_global_update()`. Safe to broadcast.

The combined infrasound emission is a weighted sum: `0.7 × personal + 0.3 × global` by current default, configurable per runtime.

### AD-002: Ed25519 Identity Without Central CA

Each node generates its own Ed25519 keypair on first start. The 32-byte public key IS the node ID. There is no registration step. Trust is established by:
1. Checking that the signature in the header verifies against the node_id field.
2. Optionally maintaining a local allowlist of known node IDs.

### AD-003: LSP Is a Native Protocol, Not a Wrapper

LSP defines its own magic bytes (`LYSW`), versioning, packet types, and payload semantics. It is NOT HTTP, MQTT, WebSocket, or gRPC wrapped. This allows:
- Implementations in Rust/Go/C++ without Python dependencies
- Minimal overhead (108-byte fixed header)
- Native understanding of pheromone semantics (TTL, decay, merge hints)

---

## 4. Wire Format

### 4.1 Base Header

All LSP packets share a fixed 108-byte header:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     Magic: 0x4C595357 "LYSW"                  |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Version (1B) |  Type (1B)    |         Flags (2B)            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    Payload Length (4B, LE)                    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|                    Node ID (32 bytes)                         |
|              (first 32 bytes of Ed25519 pubkey)               |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|                   Signature (64 bytes)                        |
|           (Ed25519 over raw/compressed payload bytes)         |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|              Payload (N bytes, see Type)                      |
|              (zlib compressed if Flags.bit0=1)                |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

**Field sizes:** Magic(4) + Version(1) + Type(1) + Flags(2) + PayloadLen(4) + NodeID(32) + Sig(64) = **108 bytes**

**Flags:**
```
bit 0 (0x01): COMPRESSED — payload is zlib-compressed
bit 1 (0x02): SIGNED     — signature field is valid
bit 2 (0x04): URGENT     — high priority, process before non-urgent
```

**Byte order:** Little-endian for all multi-byte integer fields.

---

### 4.2 FeromonV2 Binary Payload

Used when Type = `0x10` (FEROMON_V2). This is the core LSP v2 improvement — no JSON, pure binary.

```
 0       1       2       3       4       5       6       7
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| DimType(1B)   |    N_Dims (2B, LE)    |  TTL (1B)     |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|              Step (4B, uint32 LE)                      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|              Fitness (4B, float32 LE)                  |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|           Timestamp Delta (4B, uint32 LE, ms)          |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|      Tensor Bytes (N_Dims × sizeof(DimType))           |
|              ...                                        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

**DimType values:**
```
0x01 = float16  (2 bytes/element) ← default, recommended
0x02 = float32  (4 bytes/element)
0x03 = bfloat16 (2 bytes/element)
```

**Size examples:**
```
256d float16:  1+2+1+4+4+4 + 256×2 = 16 + 512 = 528 bytes
256d float32:  1+2+1+4+4+4 + 256×4 = 16 + 1024 = 1040 bytes  ← exceeds goal
512d float16:  1+2+1+4+4+4 + 512×2 = 16 + 1024 = 1040 bytes
```

**Recommendation:** Use `feromon_dim=256` with `float16` to stay under 1KB wire size.

---

### 4.3 Gossip Payload (JSON/TCP)

Gossip packets (Type `0x02`, `0x11`) carry JSON payloads over TCP. Wire format is the base header (108B) + length-prefixed JSON bytes.

**Gossip kinds:**
```json
// digest — announce current memory state
{ "kind": "digest", "node_id": "<hex>", "memory_count": 42,
  "newest_ts": 1700000000.0, "bank_hash": "<sha256[:16]>" }

// request — ask for memories newer than timestamp
{ "kind": "request", "node_id": "<hex>", "since_ts": 1699990000.0 }

// global_delta — transfer synthetic Global Matriarca deltas only
{ "kind": "global_delta", "node_id": "<hex>",
  "memories": [{"content_hash": "...", "importance": 0.8, "embedding": [0.1, ...]}],
  "privacy": "global-only" }

// ping — discovery heartbeat
{ "kind": "ping", "node_id": "<hex>", "feromon_port": 7337, "gossip_port": 7338 }
```

`GOSSIP_DELTA` must not carry raw personal conversation text. If a field contains text, it must be synthetic/global-safe content generated for swarm sharing.

---

## 5. Packet Types

| Hex  | Name           | Transport | Description                          |
|------|----------------|-----------|--------------------------------------|
| 0x01 | FEROMON        | UDP       | v1 pheromone (JSON, deprecated)      |
| 0x02 | GOSSIP         | TCP       | Memory sync / anti-entropy           |
| 0x03 | HANDSHAKE      | TCP       | Initial peer exchange                |
| 0x04 | PING           | UDP       | Discovery / keepalive                |
| 0x05 | PONG           | UDP       | Discovery reply                      |
| 0x10 | FEROMON_V2     | UDP       | v2 pheromone (binary, native)        |
| 0x11 | GOSSIP_DELTA   | TCP       | Incremental gossip (delta sync)      |
| 0x12 | MERGE_HINT     | UDP       | Merge suggestion for intermediaries  |

Types `0x01`-`0x05` are v1 (backwards compatible). Types `0x10`+ are v2.

---

## 6. Transport Layer

### UDP — Pheromone Signals (port 7337)

- Fire-and-forget, no acknowledgement
- Max safe payload: 65507 bytes (UDP limit)
- Recommended max: 1400 bytes (MTU-safe, no IP fragmentation)
- LSP pheromone target: < 1KB ✓

### TCP — Gossip and Handshake (port 7338)

- Length-prefixed framing: `[4B uint32 LE: length][N bytes: packet]`
- One packet per connection (simple) or persistent connections (future)
- Timeout: 10 seconds for initial handshake

### Default Ports

```
UDP 7337 — FEROMON / PING / PONG / FEROMON_V2 / MERGE_HINT
TCP 7338 — HANDSHAKE / GOSSIP / GOSSIP_DELTA
```

---

## 7. Node Identity

Each LSP node has a persistent Ed25519 identity stored in a PEM file:

```
~/.lixyswarm/identity.pem   (or configured path)
```

**Node ID** = first 32 bytes of the Ed25519 public key (raw bytes).

**Signature** = Ed25519 signature over the raw payload bytes (compressed if `Flags.COMPRESSED` is set — signature is over the compressed form).

**Verification flow:**
```
1. Parse header → extract node_id (32B) and signature (64B)
2. Extract raw_payload (the bytes as sent, before decompression)
3. Reconstruct Ed25519PublicKey from node_id
4. ed25519_verify(signature, raw_payload, pubkey) → bool
5. If False: discard packet, log warning
```

No central authority. Each node is self-sovereign.

---

## 8. TTL and Decay

Every FeromonV2 packet carries a TTL (Time To Live) counter.

**Initial TTL:** 3 (set by origin node)  
**Decrement:** Each relay decrements TTL by 1 before forwarding  
**Discard:** TTL = 0 → packet is NOT forwarded, NOT delivered to callback  
**Decay factor:** `0.95` per hop (configurable, default `0.95`)

```
hop 0 (origin):  feromon = F,        TTL = 3
hop 1 (relay):   feromon = F × 0.95, TTL = 2
hop 2 (relay):   feromon = F × 0.90, TTL = 1
hop 3 (dest):    feromon = F × 0.86, TTL = 0  ← delivered, not forwarded
```

The decay ensures that signals from distant nodes have less influence than local ones. This mirrors how biological pheromone trails fade with distance.

**API:**
```python
# apply_decay returns a NEW FeromonV2Payload (immutable)
decayed = payload.apply_decay(decay=0.95)
# → decayed.feromon = payload.feromon * 0.95
# → decayed.ttl = payload.ttl - 1

# standalone helper
from src.network.lsp_merge import decay_feromon
p3 = decay_feromon(payload, hops=3, decay=0.95)
# → p3.feromon = payload.feromon * (0.95 ** 3)
```

---

## 9. Merge-on-Transit

`FeromonMergeBuffer` accumulates pheromones from the same source node and merges them before delivering to the application layer.

**Why:** A busy node may emit multiple pheromones per second. Without merging, downstream nodes would process each one independently, wasting compute. The merge ensures each node's contribution arrives as a single, coherent signal.

**Algorithm:**
```
For each node_id with N pending payloads:
  1. Discard payloads older than MAX_AGE_MS (2000ms default)
  2. If N == 1: deliver as-is
  3. If N > 1: fitness-weighted average
     weight_i = fitness_i / sum(all fitnesses)
     merged_feromon = Σ(weight_i × feromon_i)
     merged_ttl = min(all TTLs)
     merged_fitness = mean(all fitnesses)
  4. Apply decay if merged_ttl < 3 (packet has hopped)
```

**Constants:**
```
MAX_AGE_MS   = 2000   # discard stale pheromones
MAX_PER_NODE = 4      # buffer cap per node (oldest evicted)
```

**API:**
```python
buf = FeromonMergeBuffer()
buf.push("node_abc", payload_1)
buf.push("node_abc", payload_2)
buf.push("node_abc", payload_3)
results = buf.flush()
# → [(node_id, merged_payload), ...]
# → one entry per node_id
```

---

## 10. Matriarca Dual Integration

`SwarmNetwork` and `LSPNodeV2` support an optional `MatriarcaDual` instance. When present:

- Incoming `GOSSIP_DELTA` packets update the **GlobalMatriarca** (shared knowledge)
- Incoming pheromones influence the runtime swarm state and may inform later safe/global synthesis
- Local interaction memories go to **PersonalMatriarca** (private)
- `emit_combined()` blends both: `0.7 × personal + 0.3 × global` by default

```
Node A                        Network                        Node B
  │                              │                              │
  │─── FEROMON_V2 (TTL=3) ──────►│                              │
  │                              │── (relay, TTL→2, decay) ────►│
  │                              │                   FeromonMergeBuffer.push()
  │                              │                   flush() → merge
  │                              │                   MatriarcaDual.merge_global_update()
  │                              │                   on_feromon_received(merged_tensor)
```

---

## 11. Message Flows

### Flow A: Pheromone Broadcast

```
Origin node (O)                   Peer nodes (P1, P2, P3)
      │                                     │
      │  send_feromon_v2(tensor, fitness)    │
      │──── FEROMON_V2 (UDP broadcast) ────►│ P1
      │──────────────────────────────────── │ P2
      │──────────────────────────────────── │ P3
      │                                     │
      │                          P1: FeromonMergeBuffer.push()
      │                          P1: flush() → callback(merged)
```

### Flow B: Peer Discovery (LAN)

```
Node A                                  Node B
  │                                        │
  │──── PING (UDP, broadcast :7337) ──────►│
  │                                        │ register_peer(A)
  │◄──── PONG (UDP, unicast) ──────────────│
  │ register_peer(B)                       │
  │                                        │
  │──── HANDSHAKE (TCP :7338) ────────────►│
  │◄──── HANDSHAKE response ───────────────│
  │ peers[B] confirmed                     │ peers[A] confirmed
```

### Flow C: Gossip / Memory Sync

```
Node A                                  Node B
  │                                        │
  │──── GOSSIP_DELTA digest (TCP) ────────►│
  │     {memory_count:42, newest_ts:...}   │
  │                                        │ compare with local
  │◄──── GOSSIP_DELTA request ─────────────│
  │      {since_ts: ...}                   │
  │──── GOSSIP_DELTA global_delta ────────►│
  │     [{content_hash, importance,        │
  │       embedding, privacy}, ...]        │
  │                                        │ MatriarcaDual.merge_global_update()
```

### Flow D: TTL Relay

```
Origin (O)      Relay (R)       Destination (D)
    │               │                  │
    │─ FEROMON_V2 ─►│                  │
    │   TTL=3        │ TTL→2, *0.95    │
    │               │─ FEROMON_V2 ────►│
    │               │   TTL=2           │ TTL→1, *0.95 (if relaying again)
    │               │                   │ TTL=0? → deliver, don't forward
```

---

## 12. Error Cases

| Situation | Behavior |
|-----------|----------|
| Bad magic bytes | Discard, log DEBUG |
| Unsupported version | Discard, log WARNING |
| Invalid Ed25519 signature | Discard, log WARNING |
| Truncated payload | Discard, log DEBUG |
| TTL = 0 on receive | Deliver locally, do NOT forward |
| TTL = 0 on create | Log WARNING, do not send |
| Packet > MAX_UDP_SIZE | Log WARNING, do not send via UDP |
| Unknown packet type | Discard gracefully (forward compat) |
| Decompression failure | Discard, log DEBUG |
| Merge buffer overflow (> MAX_PER_NODE) | Evict oldest, push new |
| Callback raises exception | Log DEBUG, continue other callbacks |

---

## 13. Test Vectors

The following test vectors are deterministic and can be used to validate any implementation.

### TV-01: FeromonV2 Pack Size

```
Input:
  feromon = torch.ones(256)   # all 1.0 values
  ttl = 3
  step = 100
  fitness = 0.75
  dim_type = 0x01 (float16)

Expected:
  len(pack()) == 528
  bytes[0] == 0x01           # DimType
  bytes[1:3] == b'\x00\x01'  # N_Dims = 256 (LE) → actually b'\x00\x01' is big-endian
                              # LE: 256 = 0x0100 → b'\x00\x01'
  bytes[3] == 0x03            # TTL
  bytes[4:8] == struct.pack('<I', 100)  # step
  bytes[8:12] == struct.pack('<f', 0.75)  # fitness
```

### TV-02: Round-Trip cosine similarity

```
Input:  feromon = torch.randn(256, generator=torch.manual_seed(42))
Process: payload = FeromonV2Payload(feromon=x, ttl=3)
         recovered = FeromonV2Payload.unpack(payload.pack())
Expected: cos_sim(feromon, recovered.feromon) >= 0.999
```

### TV-03: Decay chain

```
Input:  feromon = torch.ones(256)
Process: p = FeromonV2Payload(feromon=f, ttl=3)
         p3 = decay_feromon(p, hops=3, decay=0.95)
Expected:
  p3.feromon.mean() ≈ 0.95^3 ≈ 0.8574
  p3.ttl == 0
```

### TV-04: Fitness-weighted merge

```
Input:
  p1 = FeromonV2Payload(feromon=torch.ones(4),  fitness=0.8)
  p2 = FeromonV2Payload(feromon=torch.zeros(4), fitness=0.2)
Process:
  merged = merge_feromons([p1, p2], strategy="fitness_weighted")
Expected:
  merged.feromon ≈ [0.8, 0.8, 0.8, 0.8]  # 0.8/(0.8+0.2) * 1.0 + 0.2/1.0 * 0.0
  merged.fitness == (0.8 + 0.2) / 2 == 0.5
```

---

## 14. Implementation Checklist

For a new implementation in any language to be LSP-compliant:

- [ ] Parse 108-byte header (magic, version, type, flags, payload_len, node_id, signature)
- [ ] Validate magic bytes `LYSW` (0x4C 0x59 0x53 0x57)
- [ ] Support zlib decompression (Flags.bit0)
- [ ] Verify Ed25519 signature (Flags.bit1)
- [ ] Handle packet types: FEROMON(0x01), GOSSIP(0x02), HANDSHAKE(0x03), PING(0x04), PONG(0x05)
- [ ] Handle v2 types: FEROMON_V2(0x10), GOSSIP_DELTA(0x11), MERGE_HINT(0x12)
- [ ] Implement FeromonV2 binary payload (dim_type, n_dims, ttl, step, fitness, ts_delta, tensor)
- [ ] Reject/export-filter personal memory in every `GOSSIP_DELTA` path
- [ ] Add nonce/sequence anti-replay for `GOSSIP_DELTA`
- [ ] Implement TTL decrement and discard on TTL=0
- [ ] Implement decay factor application (0.95 per hop)
- [ ] Implement fitness-weighted merge (FeromonMergeBuffer equivalent)
- [ ] UDP listener on port 7337, TCP listener on port 7338
- [ ] Length-prefixed TCP framing (4B uint32 LE)
- [ ] Gracefully ignore unknown packet types (forward compat)

---

## 15. Changelog

| Version | Date       | Changes |
|---------|------------|---------|
| 1.0     | 2025-01    | Initial protocol: JSON pheromones, Ed25519 identity, UDP+TCP |
| 2.0     | 2026-05-29 | Binary FeromonV2 payload, merge-on-transit, TTL+decay, MatriarcaDual integration, GOSSIP_DELTA, MERGE_HINT types |
