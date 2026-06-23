# LixySwarm Implemented Architecture

**Updated:** 2026-06-22

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

The pool target is `min(1 + floor(n_nodes / 3), 4)`. Today, the runtime usually sees only its local `NodeManager` record because remote LSP peers are not automatically mapped to model capacity.

## Lifecycle layer

- `NodeManager`: local model of physical nodes, heartbeats, capacity, and contribution modes.
- `SectManager`: sect birth, low-fitness death, low-diversity spawn, bifurcation, and legacy transfer.
- `AntLifecycleManager`: compatibility lifecycle for model agents during training.
- `MatriarcaEnriched`: stores node/sect legacy and supports bifurcation suggestions.

These mechanisms are locally tested. Dynamic addition/removal of live model parameters during distributed training is not a solved production capability.

## Network layer

`SwarmNetwork` defaults to `LSPNodeV2` unless run in local mode. The data path is:

- UDP 7337: signed LSP envelope containing a binary float16 pheromone payload.
- TCP 7338: signed handshake, peer list, and JSON global-memory delta inside the LSP envelope.
- Saved peer cache plus `LIXYSWARM_BOOTSTRAP_SEEDS` for discovery.

Legacy JSON/HMAC and mDNS modules remain for compatibility but are not the default v2 path.

Current security limitations include acceptance of unsigned packets, no replay field, insufficient TCP/decompression bounds, self-asserted peer trust, and relay TTL reset. Do not expose these listeners to untrusted networks.

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
