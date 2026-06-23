# LixySwarm

LixySwarm is a research prototype for bio-inspired distributed language intelligence. It combines transformer agents (ants), persistent memory (Matriarca/elephant), an echolocation router (dolphin), and an experimental peer protocol (LSP v2).

**Current status (2026-06-22):** local research prototype. The model, memory, runtime, LAN/explicit-peer transport, status publisher, and dashboard paths exist. The repository is not ready for untrusted Internet exposure or mass deployment.

## Paper

The design reference is [LixySwarm: AntElephantDolphin](paper/LixySwarm_AntElephantDolphin.pdf), dated June 22, 2026. The paper deliberately distinguishes implemented, partial, and future work. [PAPER_COMPLIANCE.md](PAPER_COMPLIANCE.md) maps those claims to the current code and tests.

The manuscript's experimental numbers are historical results for a specific checkpoint and evaluation setup. They are not guarantees for every checkout or checkpoint.

## Implemented system

The default swarm configuration contains:

- Three GPT-style `AgentBase` instances with `FeromonGate`, fixed identity vectors, pheromone outputs, and confidence heads.
- A `DolphinPool` whose primary dolphin creates five learned pings and an attention-based acoustic map.
- A dual `Matriarca` runtime with separate personal and global banks, retrieval, importance updates, compression, and sect legacy storage.
- Two swarm rounds followed by confidence aggregation with a 20% Matriarca bias.
- `RuntimeSession` for cross-turn state, dynamic task profiles, sampling controls, and response-memory feedback.
- Node, sect, and ant lifecycle managers exercised by local tests.
- LSP v2 signed envelopes, float16 pheromone payloads, explicit-peer bootstrap, peer exchange, and global-memory deltas.
- FastAPI status/chat endpoints, a status publisher, and two static frontends.
- Continuous training and an opt-in metabolic-hunger decision function.

Some paper descriptions are only partially represented. In particular, the main forward pass does not implement the paper's exact `fitness × confidence × role_weight` equation, remote LSP peers are not automatically registered as runtime `NodeManager` capacity, and LSP relay TTL/merge semantics are not yet safe at Internet scale.

## Architecture

```text
tokens
  -> DolphinPool: five pings + attention triangulation + sleep state
  -> Matriarca: personal/global retrieval -> infrasound
  -> Ant agents: parallel transformer passes with pheromone gating
  -> confidence aggregation + Matriarca bias
  -> repetition-penalized top-k/top-p sampling in RuntimeSession
```

LSP v2 uses UDP `7337` for pheromones and TCP `7338` for handshake, peer lists, and global-memory deltas. There are no built-in public seeds in this public repository. Configure explicit seeds with `LIXYSWARM_BOOTSTRAP_SEEDS=host:7338[,host:7338]`.

## Install and verify

Python 3.12 and a recent PyTorch installation are recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
pytest -q
```

The suite currently collects 121 tests. Test totals in old experiment reports refer to earlier scripts or revisions; they are not the current collection count.

Useful entry points:

```bash
python3 lixy_chat.py
python3 lixy_orchestrator.py --status
python3 auto_train.py --status
python3 benchmark.py --health-only --cpu
uvicorn api.main:app --host 127.0.0.1 --port 8080
```

Large checkpoints and training datasets are intentionally excluded from Git. A fresh clone therefore cannot reproduce paper metrics without separately obtained artifacts and data.

## Network safety

Do not bind LSP or the API directly to the public Internet. Current blockers include mandatory-signature enforcement, replay protection, bounded TCP/decompression, peer reputation, authenticated API access, rate limiting, NAT traversal, redundant bootstrap, and adversarial multi-node tests. See [INTERNET_SCALE_READINESS.md](INTERNET_SCALE_READINESS.md).

Publisher authentication uses `LIXYSWARM_PUBLISH_TOKEN`. Personal Matriarca encryption is enabled only when `LIXYSWARM_MATRIARCA_KEY` is set. Network addresses are not published or exposed by default; enabling that requires explicit environment flags documented in [SECURITY.md](SECURITY.md).

Never commit checkpoints, corpora, session histories, identities, peer databases, logs, `.env` files, or operator addresses.

## Documentation

- [ARQUITECTURA.md](ARQUITECTURA.md): implemented architecture
- [PAPER_COMPLIANCE.md](PAPER_COMPLIANCE.md): paper-to-code compliance matrix
- [INTERNET_SCALE_READINESS.md](INTERNET_SCALE_READINESS.md): production and mass-Internet gap analysis
- [LSP_SPEC.md](LSP_SPEC.md): current wire protocol
- [LSP_ARCHITECTURE.md](LSP_ARCHITECTURE.md): protocol decisions and target architecture
- [DISTRIBUTED_PROTOCOL.md](DISTRIBUTED_PROTOCOL.md): operator-facing network overview
- [ORCHESTRATOR_RUNTIME.md](ORCHESTRATOR_RUNTIME.md): runtime behavior
- [SECURITY.md](SECURITY.md): security and privacy policy
- [VPS_SETUP.md](VPS_SETUP.md): private staging deployment, not public production
- [PENDIENTES_2026-06-05.md](PENDIENTES_2026-06-05.md): current backlog retained under its historical filename

Historical experiment notes are labeled as such and must not be read as current release status.
