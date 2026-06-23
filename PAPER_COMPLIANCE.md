# Paper Compliance Matrix

**Audit date:** 2026-06-22

**Reference:** `paper/LixySwarm_AntElephantDolphin.pdf`

**Scope:** current source tree and collected tests; ignored local checkpoints are not release artifacts

Status meanings:

- **Implemented:** a production path exists and has direct test or source evidence.
- **Partial:** useful code exists, but the paper's full behavior or validation is missing.
- **Not implemented:** design target only.
- **Evidence gap:** a claim may have been observed historically but cannot be reproduced from the public repository alone.

## Architecture and inference

| Paper requirement | Status | Current evidence and gap |
|---|---|---|
| Three 125.7M AntAgents | Implemented | `SwarmConfig.n_agents=3`; `AgentConfig` defines 12 layers, 12 heads, and width 768. The default 1,024-context test configuration reports 126.1M per agent and 453.6M for the `LixySwarm` module excluding external Matriarca; the manuscript's 568M total needs reconciliation. |
| FeromonGate and immutable identity vector | Implemented | `src/agents/agent_base.py`; exercised indirectly by integration/training paths. |
| Five Dolphin pings and cross-attention triangulation | Implemented | `Echolocation` has topic, intent, need, context, and emotion encoders plus multi-head attention. |
| DolphinPool scales as `min(1 + floor(nodes/3), 4)` | Implemented | `src/swarm/dolphin_pool.py` and lifecycle tests. |
| Phase B sleep consolidation after inactivity | Partial | PCA/SVD consolidation and idle detection exist. Consolidation occurs when the runtime is invoked/ticked, not in a durable independent background service. |
| Matriarca infrasound retrieval | Implemented | Top-k retrieval, importance-weighted scores, and 256-dimensional output exist. |
| Exact paper aggregation: fitness × confidence × role weight + Matriarca vote | Partial | The main forward pass aggregates confidence plus a 20% Matriarca bias. `RuntimeSession` separately applies task-role weights. Fitness is measured after aggregation and is not part of the current logit weights. |
| Repetition penalty + top-k + top-p decoding | Implemented | `src/utils/sampling.py` is used by `RuntimeSession`. `AgentBase.generate()` is a simpler compatibility path and does not use every control. |
| Six named dynamic roles from the paper | Partial | Runtime classification uses six task types and three agent weight slots; sect roles use a different taxonomy. There is no single six-role adapter matching the manuscript table. |

## Matriarca and lifecycle

| Paper requirement | Status | Current evidence and gap |
|---|---|---|
| Separate Personal and Global banks | Implemented | `MatriarcaDual`; runtime interaction writes are personal and legacy/synthetic writes are global. |
| Personal memory never exported | Implemented with limits | Export reads only the global bank and filters private/personal metadata; negative tests exist. This is an application invariant, not yet an independently audited information-flow guarantee. |
| Personal AES-256-GCM at rest | Partial | Implemented only when `LIXYSWARM_MATRIARCA_KEY` is configured. Without it, personal memory remains plaintext. |
| Five-factor importance score | Partial | `RuntimeSession` computes five factors, but weights differ from Equation 7 in the paper. Other write paths accept caller-provided importance directly. |
| +15% continuation and -20% topic-shift feedback | Partial | Continuation reinforcement exists with an overlap-scaled delta; the exact +15% rule and topic-shift penalty are not implemented. |
| Compression at 90% capacity | Implemented with algorithm difference | Compression triggers around 90%, but the selection/grouping behavior is not identical to the simplified paper description. |
| Sect birth, death, bifurcation, and legacy transfer | Implemented locally | Managers and tests exist. Long-running distributed validation and optimizer-safe dynamic topology remain missing. |
| Physical nodes drive runtime capacity | Partial | `NodeManager` models this, but LSP peer events are not wired into the model's `NodeManager` by default. |

## Network and shared memory

| Paper requirement | Status | Current evidence and gap |
|---|---|---|
| Ed25519 identity and signed packets | Partial | Sending signs packets and verification exists, but unsigned packets are currently accepted by `LSPPacket.verify()`. No trust policy exists. |
| 636-byte default pheromone packet | Implemented | 108-byte envelope plus 528-byte float16 payload for 256 dimensions. |
| TTL decay and bounded forwarding | Partial | Payload decay exists. The standalone relay creates a new packet and resets TTL, so end-to-end hop bounds are not preserved. |
| Merge-on-transit | Partial | A merge buffer exists, but the receive path flushes immediately, preventing useful accumulation during normal traffic. |
| TCP handshake and peer exchange | Implemented for trusted tests | Signed self-asserted identities and address exchange work. Authentication, address validation, limits, and Sybil resistance are missing. |
| Global Matriarca gossip delta | Partial | Export, filtering, transport signature, deduplication, and import exist. Per-memory provenance/signature validation, reputation, conflict handling, and poisoning defenses do not. |
| LAN zero-configuration discovery | Partial | Legacy mDNS code exists, but the default LSP v2 `SwarmNetwork` path uses saved peers and configured seeds, not mDNS. Physical multi-host release validation is not automated. |
| VPS relay path | Partial | A relay daemon and explicit-peer path exist. It is a trusted staging design, not a hardened public relay service. |
| DHT/Kademlia discovery | Not implemented | Future work. |
| Reputation-weighted consensus | Not implemented | Future work. |
| Replay protection with sequence/nonce | Not implemented | Release blocker. |

## Training, observability, and autonomy

| Paper requirement | Status | Current evidence and gap |
|---|---|---|
| Continuous auto-training with plateau LR changes and checkpoint rotation | Implemented | `auto_train.py`; human invocation and resource provisioning are still required. |
| Metabolic Hunger | Partial | Deterministic scoring and meal/snack/watch/satiated decisions exist behind `--metabolic-hunger`; weights are not learned from Matriarca history as described in the paper. |
| SwarmExplorer/API status bridge | Implemented as prototype | Publisher, API, and dashboard exist. Multi-publisher aggregation, durable storage, authentication for all endpoints, and production observability are absent. |
| GrowthGate and autonomous stage transition | Not implemented | Stage labels are roadmap concepts only. |
| Sandboxed self-modification | Not implemented | Long-term research work. |

## Experimental claims

The paper reports 11 runs, approximately 400M tokens, validation loss 5.3 to 3.44, FineWeb perplexity 35.22, and short bilingual generation metrics. Local ignored artifacts contain related checkpoints and JSON reports, but the public Git tree does not contain the large checkpoints or datasets needed for independent reproduction.

Status: **evidence gap for release reproducibility**, not a finding that the historical results are false.

Before the next paper/release revision:

1. Publish checkpoint hashes, exact configuration, dataset manifests, commands, seeds, hardware/software versions, and raw metric JSON without private data.
2. Reconcile the manuscript's approximately 568M parameter claim with the actual default three-agent model and independently count trainable, frozen, and external Matriarca parameters.
3. Reconcile block size (`AgentConfig` defaults to 1024 while reported training uses 512).
4. Update the aggregation equation and importance-feedback equations, or change the implementation to match them.
5. Report current `pytest --collect-only` and test results rather than historical aggregate counts.

## Compliance verdict

The repository supports the paper's central prototype thesis: ant agents, Dolphin mapping, Matriarca memory, lifecycle logic, and LSP v2 artifacts are real. It does **not** yet satisfy the paper's target Internet-scale ecosystem. The correct current stage remains **Infant / research prototype**.
