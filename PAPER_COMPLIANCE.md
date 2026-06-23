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
| DolphinPool scales with supplied node count | Implemented with formula difference | One Dolphin at one node, two at two-to-four, three at five-to-nine, then approximately one additional Dolphin per three nodes without an artificial ceiling. This differs from the manuscript's capped formula. |
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
| Physical nodes drive runtime capacity | Partial | LSP v3 peer events add/remove bounded resource profiles in runtime `NodeManager`; the scheduler consumes advertised capabilities for inference/artifact/training selection. Claims are self-reported, and live Dolphin/sect/model topology is not changed from unverified capacity. |

## Network and shared memory

| Paper requirement | Status | Current evidence and gap |
|---|---|---|
| Ed25519 identity and signed packets | Implemented in v3 | LSP v3 requires valid Ed25519 signatures for every frame. Legacy v2 remains explicitly selectable and retains its weaker verifier. Decaying local misbehavior bans exist, but no decentralized reputation/trust policy exists yet. |
| 636-byte default pheromone packet | Implemented | 108-byte envelope plus 528-byte float16 payload for 256 dimensions. |
| TTL decay and bounded forwarding | Superseded in v3 | Legacy v2 contains TTL/decay but its relay test resets origin semantics. Default v3 forms direct persistent sessions and does not flood/relay pheromone packets. |
| Merge-on-transit | Partial | A merge buffer exists, but the receive path flushes immediately, preventing useful accumulation during normal traffic. |
| TCP handshake and peer exchange | Implemented in v3 | Persistent signed/encrypted sessions, X25519/HKDF/ChaCha20-Poly1305, bounded frames, validated addresses, saved peers, retry/backoff, and direct peer exchange are tested. Rekeying, independent cryptographic review, and Sybil/eclipsing resistance remain missing. |
| Global Matriarca gossip delta | Partial | Export, filtering, transport signature, deduplication, and import exist. Per-memory provenance/signature validation, reputation, conflict handling, and poisoning defenses do not. |
| Internet bootstrap and seed independence | Implemented in v3 code | Saved peers plus multiple seed endpoints feed `PeerManager`; a three-node test proves direct communication continues after seed shutdown. Official DNS seed infrastructure is not configured yet. |
| VPS seed path | Implemented in v3 code | `node_daemon.py` is a non-privileged seed service template. It introduces peers and is not a required relay. Deployment validation on the actual VPS remains. |
| DHT/Kademlia discovery | Not implemented | Future work. |
| Reputation-weighted consensus | Not implemented | Future work. |
| Replay protection with sequence/message ID | Implemented in v3 | Per-session monotonic sequences, random message IDs, bounded replay cache, timestamps, and session IDs are verified by tests. |
| Distributed inference contribution | Partial | Signed-origin typed work selects consenting peers and executes an allowlisted full-request inference handler. Remote prompts cannot access/write personal Matriarca, history, or Dolphin state. There is no redundant result verification, fairness, cancellation, or process sandbox. |
| Content-addressed model/dataset artifacts | Partial | SHA-256 manifests, quotas, resumable chunks, per-chunk hashes, atomic commit, and full verification are implemented. Hashes prove content, not publisher authority; release signing, provenance, replication, and discovery by content are missing. |
| Distributed training contribution | Partial | A worker fetches a safe NumPy token artifact, requires the exact locally loaded checkpoint hash, computes a bounded real gradient, and returns a verified NPZ artifact without applying it. Three-to-31-peer quorum mode validates candidates and creates a streaming coordinate median. Sybil-independent membership, signed receipts, poisoning/privacy defenses, and promotion governance are missing. |

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

The repository supports the paper's central prototype thesis and now includes a post-paper encrypted LSP v3 topology plus inference, artifact, gradient quorum/median, diversity, and local-ban work. It does **not** yet satisfy the paper's target Internet-scale compute ecosystem because cheap identities undermine quorum independence, inference results are not replicated, work lacks process isolation/attestation, and governance, key rotation/review, and public seed operations remain. The correct current stage remains **Infant / research prototype**.
