# LixySwarm Implemented Architecture

**Updated:** 2026-06-23

**Status:** research prototype; see `PAPER_COMPLIANCE.md` for gaps

## Runtime pipeline

```text
Input token IDs
  |
  +-> DolphinPool
  |     five independent ping encoders
  |     topic-query cross-attention over all pings
  |     acoustic map -> pheromone vector
  |     persistent half-sleep state
  |
  +-> MatriarcaDual
  |     personal retrieval (default weight 0.7)
  |     global retrieval   (default weight 0.3)
  |     infrasound vector
  |
  +-> InfrasoundMixer
  |
  +-> AgentBase x 3, for two default swarm rounds
  |     fixed identity vector
  |     FeromonGate before transformer blocks
  |     language logits + outgoing pheromone
  |
  +-> confidence heads + optional 20% Matriarca bias
  |
  +-> RuntimeSession task-role adjustment and sampling
        repetition penalty, top-k, top-p, temperature
```

## Ant layer

`src/agents/agent_base.py` defines a GPT-style transformer. The default `AgentConfig` uses 12 layers, 12 heads, width 768, vocabulary 50,304, a 256-dimensional pheromone, and a 64-dimensional identity vector. The generic class defaults to a 1,024-token context; reported training configurations use 512.

`FeromonGate` projects the current pheromone into hidden width, computes a sigmoid gate from hidden and pheromone state, and applies a residual gated signal before the transformer blocks.

`SpecializationTracker` records fitness, divergence, and confidence. `DynamicRoleAdapter` classifies the user query using lexical rules and provides three agent weights plus a recommended temperature. Those role weights are applied in `RuntimeSession`, not in the core `LixySwarm.forward()` aggregation.

## Matriarca layer

The default orchestrator derives two independent stores:

- Personal: runtime conversations and session context. Local-only by export policy. Encryption is optional and activated by `LIXYSWARM_MATRIARCA_KEY`.
- Global: shareable legacy and synthetic knowledge. This is the only bank used by `export_global_delta()`.

Retrieval combines cosine similarity with stored importance. Accessed memories receive an importance increase. `RuntimeSession` computes response importance from length, lexical diversity, repetition, input/output overlap, and topic continuity. The bank compresses low-importance memories when it reaches approximately 90% of its configured capacity.

The privacy boundary is implemented by separate paths plus export filters. It has unit coverage but has not received an independent information-flow audit.

## Dolphin layer

Each dolphin produces five pings: topic, intent, need, context, and emotion. Topic acts as the attention query over the complete ping stack. The resulting acoustic map is projected into pheromone space.

`HalfSleepState` maintains a bounded recent-context buffer and an accumulated state. Phase B can consolidate idle context using PCA/SVD. The check runs when the dolphin/runtime is invoked; it is not an independent durable background scheduler.

The pool uses one Dolphin for one node, two for two-to-four nodes, three for five-to-nine nodes, and then grows by approximately one per three additional nodes without an artificial ceiling. LSP peer profiles are registered in `NodeManager`; automatically resizing a live model from unverified declarations remains unsafe and is not enabled.

## Lifecycle layer

- `NodeManager`: local model of physical nodes, heartbeats, capacity, and contribution modes.
- `SectManager`: sect birth, low-fitness death, low-diversity spawn, bifurcation, and legacy transfer.
- `AntLifecycleManager`: compatibility lifecycle for model agents during training.
- `MatriarcaEnriched`: stores node/sect legacy and supports bifurcation suggestions.

These mechanisms are locally tested. Dynamic addition/removal of live model parameters during distributed training is not a solved production capability.

## Network layer

`SwarmNetwork` defaults to `LSPNodeV3`. It uses persistent TCP 7338 sessions for signed HELLO, encrypted peer exchange, binary float16 pheromones, Global Matriarca deltas, and work. Each node maintains a bounded persistent address book and a target number of outbound sessions. Configured DNS seeds only introduce peers; learned direct sessions survive seed shutdown.

The v3 envelope requires Ed25519 signatures and includes network, session, message, sequence, timestamp, and bounded-length fields. Signed X25519 HELLO derives an HKDF-SHA256 key, and all later payloads require ChaCha20-Poly1305. Replay, stale, malformed, wrong-network, oversized, invalid-AEAD, and identity-changing frames are rejected. Public peer advertisements reject private/link-local/multicast/invalid addresses unless LAN/test mode is explicit.

HELLO resource declarations are bounded and registered in runtime `NodeManager`. The typed scheduler uses them to select consenting peers for inference, artifact, and gradient work, but they remain self-reported. Worker-side policy, operation allowlists, fixed schemas, and leases limit execution; hardware attestation and process/container isolation do not yet exist.

Work units are canonical JSON identified by SHA-256 and tied to the signed origin session. They contain declarative inputs only; executable/script/command fields are rejected recursively. Inference runs without personal Matriarca retrieval, history writes, or Dolphin-state mutation. Training workers accept only a matching local model hash and a safe one-dimensional NumPy token artifact, return a verified NPZ gradient artifact, and never apply it. The orchestrator can request three-to-31 distinct peer identities and stream a coordinate median into a new provenance-bearing artifact without loading complete model-sized candidate sets into RAM at once. Each included worker receives a stable useful-work credit containing its signed result receipt and the requester's signed aggregation claim. Connected peers present bounded credits over encryption; the scheduler prioritizes firsthand accepted work and then issuer-diverse evidence, while a persisted one-in-five exploration slot lets aged newcomers earn their first credit without reducing available quorum path diversity. None of this is treated as currency or Sybil-independent trust.

Inbound offers reserve one of a fixed number of active/queued slots before reaching the executor. Per-identity concurrent and minute-window quotas prevent one connected identity from monopolizing that finite queue; overflow produces a signed rejection rather than allocating more pending futures. Requester timeouts/cancellation set a worker-side event only for the matching authenticated origin and job. Built-in inference and training poll this event/deadline at safe boundaries and discard cancelled output. The queue remains in-memory, and non-cooperative code or an executing CUDA kernel cannot be forcibly terminated without process isolation.

For automatically scheduled single-peer work, the requester retains a bounded ordered fallback set. It reuses the exact job ID across at most three distinct peers and advances on transport failure, authenticated-session loss, timeout, overload/capacity rejection, missing response, or handler failure. One total deadline bounds the sequence. Explicit single-peer assignments remain exact. Verified inference and gradient quorums launch bounded replacements for failed identities, keep the requested result cardinality, avoid attempted identities, prefer unused coarse network groups, and expose attempts/replacements in the result. This recovery is in-process; durable crash recovery remains future work.

`ArtifactStore` addresses objects by SHA-256, omits source filenames and paths from manifests, applies storage quotas, transfers 96 KiB chunks, verifies each chunk, and verifies the complete object before commit.

`ReleaseManifest` separates model governance from peer transport. Nodes configure their own Ed25519 signer set and threshold, may pin a genesis release, reject revoked manifests, activate only a monotonic predecessor chain, and require explicit confirmation for rollback. The managed runtime uses weights-only loading for this path. No official trust roots or genesis are embedded yet.

LSP v2, legacy JSON/HMAC, UDP, and mDNS modules remain for explicit compatibility only.

## Training and evaluation

- `train.py`: base agent training.
- `train_matriarca.py`: memory model training.
- `train_swarm.py`: complete swarm training, optional remote pheromone/global delta paths, optional ant lifecycle.
- `auto_train.py`: repeated cycles, checkpoint rotation, plateau LR reduction, graceful signal handling, and optional metabolic-hunger decisions.
- `benchmark.py`: perplexity, generation statistics, checkpoint comparison, and organism-health observations.

Metabolic Hunger is a deterministic controller using available state. It does not yet learn its coefficients from Matriarca outcomes or allocate resources without an external process invoking it.

## API and frontends

`api/main.py` exposes health, status, component metrics, publisher ingestion, chat, and chat history. `swarm_publisher.py` reads local artifacts and sends a status snapshot. Peer addresses and publisher IP storage are disabled by default unless explicitly enabled.

The API remains a development service: broad CORS, unauthenticated read/chat routes, in-memory chat history, no rate limiter, and no multi-publisher state model.

## Repository boundaries

Checkpoints, data, memory banks, node identities, peer caches, logs, and session files are local runtime artifacts and must remain outside Git. The public source tree is not a complete reproducibility bundle for the paper's reported checkpoint.
