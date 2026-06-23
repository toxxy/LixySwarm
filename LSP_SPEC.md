# LixySwarm Protocol v3

**Updated:** 2026-06-22

**Maturity:** implemented development protocol; public testnet hardening remains

LSP v3 replaces the default v2 UDP/short-TCP topology with persistent, bidirectional TCP sessions. Nodes behind NAT initiate outbound sessions, receive peer addresses over those sessions, and communicate directly without requiring the seed after bootstrap.

## Transport

- One persistent TCP listener, default port `7338`.
- Four-byte unsigned big-endian frame length.
- Maximum frame: 1 MiB encrypted payload plus the 164-byte signed header; encrypted plaintext is limited to 1 MiB minus the 16-byte AEAD tag.
- Mandatory Ed25519 signature for every frame.
- Signed plaintext HELLO negotiates X25519; all established-session payloads require ChaCha20-Poly1305 encryption with an HKDF-SHA256 key bound to both identities and process-session IDs.
- No compression, preventing decompression bombs.
- Per-session message and byte windows close abusive connections.
- UDP `7337` is legacy LSP v2 compatibility only.

## Signed envelope

All integer fields use network byte order.

| Field | Size | Meaning |
|---|---:|---|
| Magic | 4 | `LYS3` |
| Version | 1 | `3` |
| Type | 1 | message type |
| Flags | 1 | bit 0 must indicate signed |
| TTL | 1 | reserved hop budget; direct sessions currently do not flood |
| Payload length | 4 | maximum 1 MiB |
| Timestamp | 8 | epoch milliseconds; maximum accepted skew is 5 minutes |
| Sequence | 8 | monotonically increasing within a sender session |
| Session ID | 16 | random process-session identifier |
| Message ID | 16 | random duplicate-suppression identifier |
| Sender ID | 32 | raw Ed25519 public key |
| Network ID | 8 | `LIXYMAIN` for the current network |
| Signature | 64 | signature over the preceding header fields and payload |
| Payload | N | type-specific bytes |

The fixed header is 164 bytes. Flag bit 0 means signed and bit 1 means encrypted. HELLO is signed but intentionally plaintext so peers can negotiate; every later frame must set both bits. Packets from another network, stale packets, invalid signatures/AEAD tags, duplicate message IDs, and non-monotonic sequences are rejected.

## Session encryption

Each process creates a non-persistent X25519 key and includes its raw public key in signed HELLO. Both peers derive the same 256-bit key with HKDF-SHA256 over the X25519 shared secret, network ID, ordered Ed25519 identities, and both process-session IDs. ChaCha20-Poly1305 authenticates the complete envelope prefix as associated data. The nonce combines four bytes of the sender session ID with its monotonically increasing 64-bit sequence.

Captured application payloads cannot be decrypted after both process-ephemeral keys are gone. Long-lived processes do not yet rotate X25519 keys in place. HELLO metadata, peer IP addresses, frame sizes, timing, and traffic volume remain observable.

## Message types

| Value | Name | Payload |
|---:|---|---|
| `0x01` | HELLO | bounded JSON capabilities, resources, listen port, user agent |
| `0x02` | PEERS | up to 100 validated addresses |
| `0x03` | PHEROMONE | v2-compatible binary float16 tensor payload |
| `0x04` | GLOBAL_DELTA | bounded Global Matriarca JSON delta |
| `0x05` | PING | small JSON timestamp |
| `0x06` | PONG | echoed ping payload |
| `0x07` | WORK_OFFER | canonical, content-addressed declarative work JSON, maximum 256 KiB |
| `0x08` | WORK_RESULT | bounded status/output JSON tied to a pending peer/job |

The first frame in each direction must be a fresh HELLO. The sender public key in all subsequent frames must match the session peer identity.

## Discovery

1. Load the persistent `peers_v3.json` address book.
2. Add configured DNS/bootstrap endpoints.
3. Resolve every A/AAAA record for redundancy.
4. Maintain a target of eight outbound sessions by default.
5. Exchange up to 100 peer addresses after every connection.
6. Prefer successful learned peers over seeds.
7. Retry failures with exponential backoff.
8. Prefer candidates from distinct IPv4 `/16`, IPv6 `/32`, or DNS-suffix groups before filling from repeated groups.

The seed is not a relay or coordinator. A seed node runs the same protocol with `target_outbound=0`, accepts sessions, and shares its learned address book. Peers form direct sessions and continue after seed shutdown; this invariant has an automated three-node test.

Private, loopback, link-local, multicast, unspecified, invalid hostname, and invalid port advertisements are rejected on the public path. Private addresses require an explicit LAN/test setting.

Malformed protocol behavior adds a decaying score to local hashed IP/identity keys. Repeated violations create persisted exponential temporary bans without storing the raw banned identifier in the reputation file. Successful sessions reduce pending scores. This is local abuse control, not global reputation and not Sybil resistance.

## Resource declaration

HELLO may declare bounded scheduling metadata:

- contribution mode;
- CPU cores;
- RAM;
- GPU presence and VRAM;
- available disk.
- available work kinds and, when applicable, locally loaded model hashes.

`SwarmNetwork` registers these peers in its runtime `NodeManager` and removes them on disconnect. The scheduler filters candidates with these values, while workers enforce their own persisted consent and resource leases. Values remain self-reported and are not proof of capacity.

## Typed work and artifacts

`WorkUnit` hashes canonical JSON containing origin, operation, kind, resource requirements, payload, deadline, and nonce. The signed session identity must equal the claimed origin. Workers execute only locally registered handlers; peer-supplied code, scripts, commands, shells, and executables are forbidden. Completed IDs are cached for idempotency.

Every `WorkResult` contains a second portable Ed25519 receipt over the job ID, worker, requester, output/error digest, and completion time. The requester verifies this receipt against the transport peer before accepting the result. Gradient quorum artifacts retain the receipts as provenance; a receipt proves what a pseudonymous identity asserted, not that the computation was honest or that identities are independent.

Current operations are isolated inference, artifact describe/read-chunk, and gradient computation. Artifacts use full-file SHA-256 identities, bounded manifests, 96 KiB raw chunks, per-chunk SHA-256, atomic commit, and final full-file verification. Gradient results are candidates and are never applied by the protocol.

## Compatibility

LSP v2 remains available only through `SwarmNetwork(..., protocol="v2")`. LSP v3 is the default. v2 and v3 do not share a socket or negotiate an in-place upgrade.

## Remaining protocol work

- In-session key rotation/rekeying and an external cryptographic review of the custom handshake.
- Public DNS seed domains operated in independent failure domains.
- Stronger autonomous-system/network diversity, feeler connections, and adversarial eclipse tests.
- Capability/result reputation, hardware verification, and Sybil resistance beyond local misbehavior bans.
- DHT discovery after persistent peer exchange is stable.
- Official threshold trust roots/genesis artifacts, network release announcement/discovery, replicated inference verification, fair scheduling, cancellation, and job recovery.
- Fuzzing, load tests, mixed-version upgrades, and an external security audit.
