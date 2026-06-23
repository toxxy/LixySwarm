# Orchestrator Runtime

**Updated:** 2026-06-23

## Entry points

- `lixy_chat.py`: interactive CLI backed by `RuntimeSession`.
- `lixy_orchestrator.py`: higher-level runtime, status, evolution, and optional network startup.
- `src/swarm/orchestrator.py`: model composition and training forward pass.
- `src/swarm/runtime_session.py`: cross-turn inference behavior.

## Model forward pass

1. `DolphinPool` returns a pheromone and diagnostic information.
2. Matriarca retrieves relevant personal/global state and emits infrasound.
3. `InfrasoundMixer` combines both signals.
4. Every AntAgent runs for each configured swarm round.
5. Outgoing pheromones are pooled and mixed again with infrasound.
6. Agent confidence heads are optionally biased 20% by Matriarca and softmaxed.
7. Aggregated logits, average training loss, and the final pheromone are returned.

Fitness is calculated after the training aggregation for observation/lifecycle use. It is not currently an input to the same aggregation, despite the simplified paper equation.

## Interactive generation

`RuntimeSession.turn()`:

- classifies the query into a task profile;
- selects a default sampling temperature;
- builds bounded cross-turn context;
- warms the swarm and blends a cached previous pheromone;
- refreshes pheromone periodically during token generation;
- calls every agent directly and mixes 65% confidence with 35% task-role weights;
- samples with repetition penalty, top-k, and top-p;
- stores a quality-scored personal memory after the response;
- persists local session state.

This means training/core-forward aggregation and interactive-token aggregation are related but not identical paths. Benchmarks must state which path they exercise.

## Persistence and privacy

Runtime history and personal memory may contain user content and are ignored by Git. Personal Matriarca encryption requires `LIXYSWARM_MATRIARCA_KEY`; otherwise the files are plaintext. Session history has no independent encryption layer.

Do not enable global export for content that has not been explicitly stored in the Global Matriarca. Do not publish session files or checkpoints.

## Network integration

When enabled, `SwarmNetwork` defaults to persistent LSP v3 sessions, supplies blended remote pheromones, exchanges global deltas, and enables typed work. Signed peer HELLO resource profiles are added to and removed from runtime `NodeManager`. The scheduler can assign inference, artifact, and gradient jobs, but declarations are self-reported and Dolphin/sect allocation is not automatically driven by unverified capacity.

Remote inference uses a fresh non-persistent `RuntimeSession`, disables personal Matriarca retrieval and importance changes, records no history, does not read/update Dolphin sleep/acoustic state, and serializes model access with local inference. Distributed training requires an exact checkpoint file hash, a content-addressed `application/x-npy` token dataset, bounded token range, and declared RAM/disk sufficient for the estimated gradient. `compute_gradient_quorum()` requests matching distinct peers in parallel and emits a streaming coordinate-median artifact; every candidate and aggregate remains unapplied. Once aggregation succeeds, each included worker receives a locally persisted, duplicate-resistant useful-work credit signed by both the worker (result receipt) and requester (aggregation attestation). Direct peers present and verify a bounded credit set; scheduling prefers contributions previously accepted by the current requester, then capped issuer-diverse evidence, with one identity-aged newcomer exploration opportunity per five selections.

Requester timeouts and explicit caller cancellation emit authenticated cancellation messages. Remote inference polls at token/agent boundaries; gradient work polls around fetch, forward, backward, parameter extraction, and artifact commit. The worker returns a signed `work_cancelled` or `work_deadline_exceeded` result when it reaches a cooperative boundary.

Non-explicit single-peer inference/training work may attempt three eligible peers under one deadline. Transient failure cancels the current attempt and advances; the job ID remains unchanged. For a fallback gradient, the orchestrator obtains the actual worker identity from the verified receipt before fetching its artifact. Replicated inference and gradient quorum members remain fixed to preserve independence accounting.

`generate_distributed_verified()` sends a deterministic greedy request to an odd quorum of three-to-nine peers advertising the exact checkpoint hash. Candidate selection prefers different coarse network groups; acceptance requires a strict byte-identical text majority, and the returned record includes supporting Ed25519 receipts. This is replication, not proof against Sybil-controlled workers.

`lixyswarm start --release` loads only the locally active manifest after rechecking its threshold signatures, trust policy, content-addressed artifacts, chain state, and `pytorch-weights-only-v1` format. Direct `--checkpoint` remains an explicit operator-trusted path and does not claim release governance.

## Known runtime gaps

- The inbound executor queue is bounded with per-identity concurrency/rate quotas, but it is not durable. Cooperative cancellation and bounded single-job fallback exist; crash recovery, quorum-member replacement, full fair-share accounting, forced termination of non-cooperative kernels/handlers, and process-level multi-tenant isolation remain missing.
- Model loading is process-global and heavyweight.
- API chat history is in memory and unauthenticated.
- Dynamic topology changes are not integrated safely with a live optimizer.
- Dolphin idle consolidation is invocation-driven, not a durable background worker.
- Paper and runtime aggregation formulas require reconciliation.
