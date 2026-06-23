# LixySwarm Protocol v3

**Updated:** 2026-06-23

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
| `0x09` | RELEASE_ANNOUNCE | threshold-signed release manifest, maximum 64 KiB |
| `0x0A` | USEFUL_WORK_CREDIT | worker receipt plus requester aggregation attestation, maximum 16 KiB |
| `0x0B` | USEFUL_WORK_PROOFS | up to 16 dual-signed credits presented after encryption, maximum 64 KiB |
| `0x0C` | WORK_CANCEL | requester-authenticated job ID, timestamp, and cancellation reason, maximum 4 KiB |

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

The seed is not a relay or coordinator. A seed node runs the same protocol with `target_outbound=0`, accepts sessions, and shares its learned address book. Peers form direct sessions and continue after seed shutdown. This invariant has both an in-process three-node test and a spawn-based acceptance test that runs each node in a separate interpreter, abruptly terminates the seed process, verifies the surviving route remains encrypted, and delivers a signed Global Matriarca delta directly.

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
- an optional Hashcash-style identity-work proof bound to the HELLO Ed25519 identity.

`SwarmNetwork` registers these peers in its runtime `NodeManager` and removes them on disconnect. The scheduler filters candidates with these values, while workers enforce their own persisted consent and resource leases. Values remain self-reported and are not proof of capacity.

Identity work is disabled by default. An operator can set `LIXYSWARM_IDENTITY_WORK_BITS` from 1 to 28 to mine a persistent stamp whose challenge binds the network ID, Ed25519 public key, and nonce. Requesters then reject inference/training workers below their configured minimum; connectivity and artifact exchange remain open. This is optional admission friction, not proof of useful work.

## Typed work and artifacts

`WorkUnit` hashes canonical JSON containing origin, operation, kind, resource requirements, payload, deadline, and nonce. The signed session identity must equal the claimed origin. Workers execute only locally registered handlers; peer-supplied code, scripts, commands, shells, and executables are forbidden. Completed IDs are cached for idempotency.

Admission reserves a slot before submitting to the executor. Defaults are one-to-64 active workers according to local configuration, 16 queued offers, two active-or-queued offers per remote identity, and 12 attempts per identity per minute. The identity-rate table is capped at 1,024 entries. `LIXYSWARM_MAX_QUEUED_OFFERS`, `LIXYSWARM_MAX_OFFERS_PER_PEER`, and `LIXYSWARM_MAX_OFFERS_PER_MINUTE` change local limits. A rejected offer receives a signed `WorkResult` with `work_queue_full`, `peer_queue_limit`, `peer_rate_limit`, `worker_identity_limit`, or `worker_shutting_down`. Sending from an LSP callback schedules onto the current event loop instead of synchronously waiting on itself.

If the requester timeout expires or its local cancellation event is set, it sends `WORK_CANCEL` over the authenticated session. The worker accepts only version 1, an exact active `(requester identity, job ID)` pair, one of the two fixed reasons, and a timestamp within five minutes. A different identity cannot cancel the job. Handlers receive `WorkExecution`, which remains an `isinstance(..., WorkUnit)`-compatible object and adds `cancelled()`, `raise_if_cancelled()`, and `remaining_s()`. The coordinator checks before/after handler execution; remote token generation checks each loop and gradient work checks fetch/forward/backward/parameter/archive boundaries. A CUDA kernel or third-party handler that never returns or checks cannot be forcibly stopped until process isolation exists.

When no peer was explicitly chosen, submission orders up to three distinct eligible peers. Send failure, authenticated-session loss, per-attempt timeout, signed overload/capacity rejection, missing result, or `handler_failed:*` advances to the next peer. All attempts reuse the same content-addressed `WorkUnit` and share its total deadline; the current remaining time is divided among remaining attempts. A timed-out attempt receives `WORK_CANCEL` before retry. Cancellation/deadline and permanent rejections do not retry. Explicit single-peer selection uses exactly the requested identity. Single-gradient orchestration reads the verified worker receipt after fallback and fetches the resulting artifact from that actual worker.

Verified inference and gradient quorum orchestration dispatches to exact-model distinct identities in parallel. A failed or disconnected member may be replaced by a previously unattempted eligible peer, bounded by `max_replacements` (one quorum by default), without reducing the configured accepted-result count. Selection prefers a coarse network group not present among retained results or active requests when one is available. Compute and artifact retrieval share one caller deadline and cancellation signal. Results expose accepted worker identities, total attempts, and replacement count; no partial quorum is aggregated or accepted.

Every `WorkResult` contains a second portable Ed25519 receipt over the job ID, worker, requester, output/error digest, and completion time. The requester verifies this receipt against the transport peer before accepting the result. Gradient quorum artifacts retain the receipts as provenance; a receipt proves what a pseudonymous identity asserted, not that the computation was honest or that identities are independent.

After a gradient candidate enters a validated quorum aggregate, the requester signs a useful-work credit containing the worker's signed result receipt and the exact model, dataset, candidate, aggregate, and token count. The stable credit ID prevents repeated aggregation of one job/result from increasing the worker's local count. Credits are delivered over the encrypted session and stored by the worker.

A worker presents at most 16 credits, preferring recent credits from distinct issuers, after the encrypted handshake. Presentations received before the local ledger/verifier starts are retained only for the authenticated live session and replayed to the verifier, removing startup-order dependence. HELLO-provided `useful_work` values are discarded. The receiver validates both signatures and worker ownership, stores only privacy-safe counters in its peer snapshot, and identifies credits that it issued firsthand. Scheduling prefers firsthand evidence, then bounded issuer diversity and capacity. Multi-hop discovery, issuer trust/aging, and Sybil independence are not implemented.

The requester persists a private bounded scheduler history containing only pseudonymous node IDs, first-seen/last-selected times, and counters. By default, every fifth scheduling event can replace the lowest-ranked selected slot with the least-selected identity known for at least 60 seconds. A quorum replacement is skipped if it would reduce the network-group diversity of the retained members. `LIXYSWARM_EXPLORATION_INTERVAL` and `LIXYSWARM_EXPLORATION_MIN_AGE_S` tune these local values; zero interval disables exploration. This is free admission and anti-starvation policy, not a fee, stake, reward token, or Sybil proof.

Current operations are isolated inference, artifact describe/read-chunk, and gradient computation. Artifacts use full-file SHA-256 identities, bounded manifests, 96 KiB raw chunks, per-chunk SHA-256, atomic commit, and final full-file verification. Gradient results are candidates and are never applied by the protocol.

Trusted release announcements are deduplicated and relayed only after the receiver's local threshold policy validates them. Acquisition runs outside the transport loop, pulls every referenced artifact from the announcing peer, verifies full hashes, and stores the accepted manifest. Activation remains manual unless the persisted trust policy explicitly enables auto-activation.

## Compatibility

LSP v2 remains available only through `SwarmNetwork(..., protocol="v2")`. LSP v3 is the default. v2 and v3 do not share a socket or negotiate an in-place upgrade.

## Remaining protocol work

- In-session key rotation/rekeying and an external cryptographic review of the custom handshake.
- Public DNS seed domains operated in independent failure domains.
- Stronger autonomous-system/network diversity, feeler connections, and adversarial eclipse tests.
- Sybil-independent issuer/result reputation, identity aging, hardware verification, and eclipse resistance beyond local misbehavior bans and connected-peer useful-work evidence.
- DHT discovery after persistent peer exchange is stable.
- Official threshold trust roots/genesis artifacts, multi-provider content lookup beyond the announcing peer, cross-hardware validation of replicated inference, durable fair-share queues, forced termination/process isolation, and crash-persistent job/quorum recovery.
- Fuzzing, load tests, mixed-version upgrades, and an external security audit.
