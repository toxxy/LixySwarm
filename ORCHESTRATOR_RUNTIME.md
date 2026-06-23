# Orchestrator Runtime

**Updated:** 2026-06-22

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

Remote inference uses a fresh non-persistent `RuntimeSession`, disables personal Matriarca retrieval and importance changes, records no history, does not update Dolphin sleep/acoustic state, and serializes model access with local inference. Distributed training requires an exact checkpoint file hash, a content-addressed `application/x-npy` token dataset, bounded token range, and declared RAM/disk sufficient for the estimated gradient. Returned gradients remain unapplied candidates.

## Known runtime gaps

- No bounded persistent request queue, cancellation, fair scheduling, or process-level multi-tenant isolation.
- Model loading is process-global and heavyweight.
- API chat history is in memory and unauthenticated.
- Dynamic topology changes are not integrated safely with a live optimizer.
- Dolphin idle consolidation is invocation-driven, not a durable background worker.
- Paper and runtime aggregation formulas require reconciliation.
