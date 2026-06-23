# LixySwarm Protocol v2: Current Wire Specification

**Updated:** 2026-06-22

**Maturity:** experimental; incompatible changes are still possible

All multibyte fields in the LSP envelope and v2 payload are little-endian unless stated otherwise.

## Signed envelope

| Offset | Size | Field | Current value/meaning |
|---:|---:|---|---|
| 0 | 4 | magic | ASCII `LYSW` |
| 4 | 1 | version | `0x01` |
| 5 | 1 | type | packet type |
| 6 | 2 | flags | compressed `0x01`, signed `0x02`, urgent `0x04` |
| 8 | 4 | payload length | number of transmitted payload bytes |
| 12 | 32 | node ID | raw Ed25519 public key |
| 44 | 64 | signature | Ed25519 signature over transmitted payload bytes |
| 108 | N | payload | compressed when flag `0x01` is set |

The current sender signs every normal packet. The current verifier accepts unsigned packets, which is a known security defect. Receivers must not be exposed to untrusted traffic until unsigned packets are rejected.

Packet types:

| Value | Name | Transport |
|---:|---|---|
| `0x01` | legacy FEROMON | UDP |
| `0x02` | legacy GOSSIP | TCP/compatibility |
| `0x03` | HANDSHAKE | TCP |
| `0x04` | PING | UDP |
| `0x05` | PONG | UDP |
| `0x10` | FEROMON_V2 | UDP |
| `0x11` | GOSSIP_DELTA | TCP |
| `0x12` | MERGE_HINT | reserved/not handled |
| `0x13` | PEER_LIST | TCP |

## FEROMON_V2 payload

| Offset | Size | Field | Meaning |
|---:|---:|---|---|
| 0 | 1 | dimension type | 1 float16, 2 float32, 3 bfloat16 |
| 1 | 2 | dimension count | default 256 |
| 3 | 1 | TTL | sender default 3 |
| 4 | 4 | training step | unsigned integer |
| 8 | 4 | fitness | IEEE float32, sender-provided |
| 12 | 4 | timestamp fragment | epoch milliseconds modulo 2^32 |
| 16 | variable | vector | count × encoded element size |

The default payload is 528 bytes and the complete signed packet is 636 bytes.

`timestamp fragment` cannot establish freshness by itself. The protocol has no sequence, nonce, message ID, or replay cache.

## Handshake

The handshake payload is JSON with node ID, advertised UDP/TCP ports, local step, and capability strings. The TCP frame is a four-byte little-endian packet length followed by the complete LSP packet. A response may include up to 50 peers.

The receiver derives the connection host from the socket but accepts advertised ports. Peer lists encode a four-byte count followed by repeated one-byte host length, host bytes, and two-byte port. The codec limits a list to 100 items; address safety and routability validation remain incomplete.

## GOSSIP_DELTA

The payload is UTF-8 JSON, optionally zlib-compressed by the envelope. The current Matriarca delta includes:

- `kind = matriarca_global_delta`
- `version = 1`
- creation/source metadata
- global-only memory metadata
- embeddings represented as float16-compatible JSON lists

The packet signature covers the compressed payload bytes. Imported memory stores source metadata, but there is no durable per-item signature or trust decision.

## Current processing semantics

- Invalid signed packets are rejected.
- TTL zero pheromones are discarded.
- A merge buffer supports fitness-weighted averaging, but normal receive processing flushes it immediately.
- The standalone relay re-emits pheromones with a fresh TTL and its own identity.
- Peer and global-delta callbacks run in listener-created threads.

## Required v3/security changes

1. Require signatures and bind a declared node ID to the envelope key.
2. Add message ID, origin ID, sequence/nonce, absolute timestamp, and replay rules.
3. Add preserved hop count/TTL and a relay signature chain or verifiable origin signature.
4. Define maximum envelope, compressed, decompressed, JSON, vector, and peer-list sizes.
5. Define error codes, version negotiation, capability negotiation, and graceful incompatibility.
6. Define trust/admission/reputation separately from cryptographic identity.
7. Publish language-neutral test vectors and fuzz every decoder.

Until these changes exist, this document is an implementation description, not a safe interoperability promise.
