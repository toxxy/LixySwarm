# Distributed Protocol Overview

**Updated:** 2026-06-22

**Status:** trusted prototype

LSP v2 exchanges coordination signals and global-memory deltas. It does not distribute transformer layers, gradients, or full inference jobs.

## Current modes

| Mode | Current behavior |
|---|---|
| Local | No network listener; inference and training continue normally. |
| Explicit peers | Reads saved peers and `LIXYSWARM_BOOTSTRAP_SEEDS`; performs TCP handshakes and peer exchange. |
| Trusted LAN | Works when peers are reachable and configured; the default v2 facade does not currently start legacy mDNS discovery. |
| Staging relay | `node_daemon.py` listens on UDP/TCP and can connect to an explicit upstream peer. |
| Open Internet | Unsupported and unsafe. |

## Data flows

### Pheromones

The sender serializes a 256-dimensional float32 tensor as float16 in a signed LSP packet and sends it over UDP 7337. The receiver verifies signed packets, decodes the vector, and stores the latest signal in its peer table. `SwarmNetwork.get_combined_feromon()` can blend remote signals with local state.

The protocol has a TTL and decay implementation, but relay forwarding currently creates a fresh packet. Therefore TTL is not an end-to-end loop bound. The merge buffer also flushes immediately in the normal receive path, so production batching/merge-on-transit is incomplete.

### Global Matriarca

Only `MatriarcaDual.global_` is exported. `GOSSIP_DELTA` carries metadata and float16-list embeddings over signed TCP LSP packets. The receiver filters, deduplicates, discounts importance, and writes accepted items to its global bank.

Transport signatures authenticate the packet's self-asserted key. They do not establish trust in content. There is no per-memory signature, reputation, quorum, or poisoning defense.

### Peer discovery

Discovery order is saved peer cache, configured seeds, and peer exchange. Configure seeds without committing operator addresses:

```bash
export LIXYSWARM_BOOTSTRAP_SEEDS='relay-a.example.net:7338,relay-b.example.net:7338'
```

No DHT, public DNS seed fleet, NAT traversal, or Sybil protection is implemented.

## Privacy defaults

Status publishing does not include network addresses unless `LIXYSWARM_PUBLISH_NETWORK_ADDRESSES=true`. The API redacts peer hosts unless `LIXYSWARM_EXPOSE_PEER_HOSTS=true`, and it does not store publisher IPs unless `LIXYSWARM_STORE_PUBLISHER_IP=true`.

These flags affect observability only. LSP peers necessarily see connection source addresses at the transport layer.

## Operational warning

Bind to loopback or a private firewall-controlled network. Do not port-forward 7337/7338 from an untrusted network. See `LSP_SPEC.md`, `SECURITY.md`, and `INTERNET_SCALE_READINESS.md`.
