# Dolphin Roadmap

**Updated:** 2026-06-22

## Current implementation

- Five learned pings: topic, intent, need, context, and emotion.
- Topic-anchored multi-head attention triangulation.
- Acoustic-map projection into the 256-dimensional pheromone space.
- Confidence head and fixed whistle/identity vector.
- Bounded half-sleep context buffer and persistent awake state.
- Idle-triggered PCA/SVD consolidation checked during runtime invocation.
- `DolphinPool` scaling from one to four instances based on supplied node count.
- Routing helpers for local sect records and adaptive sleep modes.

## Remaining work

### P0: make current behavior measurable

- Add an evaluation that isolates the five-ping attention map against a no-Dolphin baseline.
- Measure response quality, calibration, latency, and memory cost per Dolphin count.
- Publish deterministic Phase A/Phase B configurations and checkpoint hashes.
- Add long-conversation tests for consolidation quality and regression.

### P1: complete runtime integration

- Run idle consolidation in a durable background lifecycle with clean shutdown and persistence.
- Drive pool size from authenticated live network capacity rather than a caller-provided count.
- Persist and version sleep state safely across checkpoint upgrades.
- Define how multiple dolphins route or combine sect outputs in the actual generation path.

### P2: research targets

- Learn ping semantics and routing quality with explicit objectives rather than names alone.
- Evaluate multimodal acoustic spaces only after text behavior is reproducible.
- Study privacy leakage from implicit sleep-state representations before any sharing.

## Non-claims

The current Dolphin is a learned routing/context module, not evidence of consciousness or biological equivalence. Multimodal nodes and network-level emergent specialization remain future work.
