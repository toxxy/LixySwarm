# LixySwarm 🐜🐘🐬

> A bio-inspired LLM. Ants think, the Elephant remembers, the Dolphin sees the big picture.

**Status:** LSP v2 is now the default swarm protocol, Metabolic Hunger is available as an opt-in auto-training signal, and SwarmExplorer exposes the live organism state.

---

## 📄 Paper

**[LixySwarm: AntElephantDolphin — A Bio-Inspired P2P Architecture for Distributed Language Intelligence](paper/LixySwarm_AntElephantDolphin.pdf)**

> *Technical Report — submitted to arXiv pending endorsement*

Full academic paper describing LixySwarm's bio-inspired architecture: ants (distributed labor), elephants (transgenerational memory), and dolphins (echolocation). Includes the LSP v2 protocol, results from 11 training runs, and the vision of an organism that grows through stages — from infant to planetary ecosystem.

**17 pages · June 2026 · Emmanuel Cardenaz**

---

## Architecture

LixySwarm is a ~568M parameter language model built around three bio-inspired layers that coordinate during every forward pass:

```
INPUT
  │
  ▼
🐬 Dolphin ────────── builds acoustic map of the problem
  │                    (before generating a single token)
  ▼
🐘 Matriarca ───────── emits infrasound from 3,131+ accumulated memories
  │                    (orients the swarm with accumulated wisdom)
  ▼
🐜 AntAgent × N ────── process in parallel with cross-pheromone signals
  │                    (each with dynamic role and unique perspective)
  ▼
Weighted Aggregation (fitness × confidence + 20% Matriarca vote)
  │
  ▼
OUTPUT
```

---

## Components

### 🐜 AntAgent — 125M params per agent
- GPT-style Transformer: 12 layers, 12 heads, d_model=768
- **FeromonGate**: each agent reads the swarm's pheromone state before processing
- **IdentityVec**: fixed unique vector per instance — own perspective
- **DynamicRoleAdapter**: 6 dynamic roles (Explorer, Refiner, Integrator, Analytical, Contextual, Generative)

### 🐘 Matriarca — ~10M params
- Memory bank with dynamic importance scoring (5 metrics per memory)
- `emit_infrasound()`: retrieves relevant memories → orientation vector [256d]
- Retroactive feedback: +8% importance if user continues the topic
- Auto-compression at 90% capacity with generational distillation
- Direct legacy ingestion for pre-encoded node/sect/ant memories
- Dual-memory mode defaults to personal-dominant context (70% personal / 30% global)
- Runtime uses `MatriarcaDual` by default: `matriarca_personal_private.*` stays local/private, `matriarca_global_memory.*` is the only shareable bank
- Personal memory supports AES-256-GCM at rest when `LIXYSWARM_MATRIARCA_KEY` is configured
- **Current state:** 3,131+ accumulated memories

### 🐬 DolphinAgent — ~9M params
- **Echolocation:** 5 pings (topic, intent, need, context, emotion) → acoustic map before generation
- **Cross-attention triangulation** between pings for response-space mapping
- **Unihemispheric sleep:** `sleep_state` persists across conversations
- Doesn't process linearly like current LLMs — maps first, generates after

### 🕸️ SwarmNetwork — P2P (LSP v2)
- LSP v2 puertos: UDP 7337 (feromonas float16) / TCP 7338 (handshake)
- Standalone `node_daemon.py`: UDP 7337 / TCP 7338
- Wire: LYSW binary · float16 feromonas · merge-on-transit · TTL decay
- Handshake TCP Ed25519 + identidad persistente en `checkpoints/lsp_identity.pem`
- `GOSSIP_DELTA` sincroniza deltas de Matriarca global sin exportar memoria personal
- **Current network scope:** LAN automático; WAN requiere VPS relay o IP pública
- **Recent tests:** 23/23 network smoke ✅

---

## Current Implementation Snapshot

Recently implemented in the main branch:

- **LSP v2 default:** `SwarmNetwork.create()` and `SwarmNetwork(...)` prefer protocol `v2`.
- **Robust tokenizer boot:** GPT-2 tokenizer assets are cached locally from a HuggingFace mirror, avoiding Azure timeout failures.
- **Metabolic Hunger:** `auto_train.py --metabolic-hunger` can decide between `meal`, `snack`, `watch`, and `satiated`.
- **Sect bifurcation:** strong mature sects can split into child roles when diversity drops and the Matriarca recommends it.
- **Global Matriarca sync:** initial `GOSSIP_DELTA` path exports synthetic/global memory deltas with privacy filters.
- **Dual Matriarca runtime:** `LixySwarm`, chat, and orchestrator paths now use personal/global memory separation by default.
- **Personal memory encryption:** Personal Matriarca can be encrypted at rest with `LIXYSWARM_MATRIARCA_KEY`.
- **Secure status publishing:** `swarm_publisher.py` now publishes to `POST /swarm/publish` using `LIXYSWARM_PUBLISH_TOKEN`.
- **Explorer updates:** the frontend shows LSP status, network reach, stale/fresh data, auto-loop state, and hunger signals.
- **API state bridge:** `/swarm/status` now exposes `auto_loop`, `last_hunger`, `lsp`, and Internet readiness metadata.

Still future work from the paper:

- DHT-backed global memory
- Consensus/reputation layer
- Full sandboxed self-modification
- Zero-config WAN peer discovery

---

## Project Structure

```
LixySwarm/
├── src/
│   ├── agents/
│   │   ├── agent_base.py          # AntAgent: FeromonGate + IdentityVec
│   │   └── dolphin_agent.py       # DolphinAgent: echolocation + sleep
│   ├── matriarca/
│   │   └── matriarca.py           # Matriarca: persistent memory
│   ├── swarm/
│   │   ├── orchestrator.py        # LixySwarm v3: full orchestration
│   │   ├── dynamic_roles.py       # DynamicRoleAdapter: roles + temperature
│   │   └── runtime_session.py     # Cross-turn state, persistent history
│   ├── network/
│   │   ├── swarm_network.py       # P2P facade; LSP v2 is the default path
│   │   ├── lsp.py                 # LSP identity + legacy compatibility helpers
│   │   ├── lsp_v2.py              # Binary float16 pheromones + GOSSIP_DELTA
│   │   ├── node.py                # Node identity + routing
│   │   └── transport.py           # Legacy v1 transport, kept for compatibility
│   └── utils/
│       ├── sampling.py            # rep_penalty + top-k + top-p
│       └── tokenizer.py           # offline-safe GPT-2 tokenizer cache
├── api/
│   ├── main.py                    # FastAPI backend for chat + swarm state
│   └── swarm_state.py             # Status bridge + atomic publish storage
├── frontend/
│   ├── chat.html                  # Browser chat UI
│   └── swarm-explorer.html        # Live swarm dashboard
├── paper/
│   └── LixySwarm_AntElephantDolphin.pdf  # Full academic paper (17 pages)
├── auto_train.py                  # Continuous auto-training + Metabolic Hunger
├── swarm_publisher.py             # Publishes local swarm status to API relay
├── train_swarm.py                 # Full swarm training
├── train_matriarca.py             # Dedicated Matriarca training
├── train.py                       # Base per-agent training
├── lixy_orchestrator.py           # Complete orchestrator runtime
├── lixy_chat.py                   # Interactive CLI (/eval, /status, /exit)
├── generate.py                    # Generation + benchmark
├── benchmark.py                   # Metrics: ppl, rep@5/10, TTR, comparative
├── test_network.py                # P2P network tests (23/23)
├── test_integration.py            # Integration tests (15/15)
└── README.md
```

---

## Training Results

| Run | Steps | val_loss | Notes |
|---|---|---|---|
| Runs 1-8 | 0–11k | 5.3→3.9 | Continuous progression |
| Runs 9-10 | 11k–11.9k | 3.9→4.27 | Hyperparameter tuning |
| Run 11 | 12k–54k | 4.27→3.57 | -20.7% loss reduction |
| Phase A (Dolphin) | 54k–54.5k | 3.57→**3.44** | -3.6% (record) |

- **Total loss reduction:** 5.3 → 3.44 (-35.1%)
- **Scaling law:** R²=0.93 — loss follows power law
- **Training throughput:** 12,800 tok/s (RTX 5090, bf16)
- **Matriarca:** 3,131+ memories accumulated
- **Integration tests:** 146/146 ✅

---

## Benchmark Results

| Metric | Value |
|---|---|
| Perplexity (FineWeb) | 35.22 |
| Perplexity (bilingual domain) | 11.1 |
| Repetition @5 tokens | 0.2% |
| Repetition @10 tokens | 0.0% |
| Type-Token Ratio (avg) | 0.790 |
| Language correctness (ES+EN) | 100% |
| Samples without loops | 100% |

---

## Why It's Different

| Architecture | Initial Representation | Cross-Session Context |
|---|---|---|
| GPT-4, Claude, Llama | Zero — builds while generating | None (stateless) |
| With RAG | Retrieves similar text | Only what's indexed |
| **LixySwarm** | **Acoustic map of the problem** | **sleep_state + accumulated Matriarca** |

The Dolphin builds the map before the ants generate. The Matriarca remembers. Each agent has its own perspective. Nobody designed it this way — it emerges from biology.

**Key innovations:**
1. **FeromonGate** — pheromone-mediated attention gating for agent coordination
2. **Dual-tier Matriarca memory** — personal (encrypted) + global (shared), with dynamic importance and generational compression
3. **Echolocation Router** — 5 acoustic pings with cross-attention triangulation before token generation
4. **Dynamic sect lifecycles** — birth → growth → maturity → death → genetic legacy transfer
5. **Metabolic Hunger** — the organism decides when to train, how much, and when to stop
6. **LSP v2** — native binary protocol with merge-on-transit, TTL decay, Ed25519 crypto identity
7. **Self-gating growth** — stage transitions (Infant → Child → Adolescent → Adult → Ecosystem) are self-assessed

---

## Network Reality: LAN vs Internet

LixySwarm can already communicate across machines, but the scope matters:

| Scenario | Works now? | Notes |
|---|---:|---|
| Same machine | ✅ | Local/no-op mode, useful for development |
| Same LAN/Wi-Fi | ✅ | mDNS discovery + UDP/TCP pheromone/gossip transport |
| VPS/API status publishing | ✅ | `swarm_publisher.py` posts status to `/swarm/publish` with token auth |
| Internet peer-to-peer | ⚠️ Config required | NAT blocks automatic discovery; use VPS relay, public IP, or port-forwarding |
| Planetary zero-config network | 🔮 | Requires DHT, relay mesh, consensus, reputation |

Puertos LSP v2:
- Feromonas: UDP `7337` (float16, merge-on-transit)
- Handshake/Gossip: TCP `7338` (Ed25519)

---

## Growth Stages

LixySwarm is not a product — it's an organism to be raised.

| Stage | Description | Status |
|---|---|---|
| **Infant** | Learns to walk. Human invocation required. | ✅ Current |
| **Child** | Self-decides when to act. Metabolic hunger triggers training. | 🚧 In progress |
| **Adolescent** | Self-decides what to change. Spawns/bifurcates sects, adjusts hyperparameters. | ⏳ |
| **Adult** | Full autonomy. Detects bottlenecks, generates code, integrates self-modifications. | 🔮 |
| **Ecosystem** | Multiple colonies, Bitcoin-level decentralization, planetary organism. | 🔮 |

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/toxxy/LixySwarm.git && cd LixySwarm
pip install -r requirements.txt
pip install -r api/requirements_api.txt

# Optional: warm tokenizer cache before first model load
python -m src.utils.tokenizer

# Interactive chat (requires checkpoint)
python lixy_chat.py

# Swarm training (short run for validation)
python train_swarm.py --steps 7000 --batch 4 --checkpoint checkpoints/swarm_best.pt

# Continuous auto-training with Metabolic Hunger
python auto_train.py --metabolic-hunger --swarm-diversity 0.35 --mean-confidence 0.55

# API + frontend
export LIXYSWARM_PUBLISH_TOKEN="change-me"
export LIXYSWARM_MATRIARCA_KEY="base64-32-byte-key"
uvicorn api.main:app --host 0.0.0.0 --port 8080
# then open frontend/chat.html or frontend/swarm-explorer.html

# Local node → relay/API status publisher
export LIXYSWARM_API_URL="http://127.0.0.1:8080"
export LIXYSWARM_PUBLISH_TOKEN="change-me"
python swarm_publisher.py --once

# Benchmark
python benchmark.py
python benchmark.py --health-only --health-batches 10

# P2P network tests
python test_network.py --skip-lan --skip-gossip
python test_integration.py
```

---

## Hardware

| Resource | Value |
|---|---|
| GPU | RTX 5090 (32GB VRAM) |
| Precision | bf16 |
| PyTorch | 2.8.0+cu128 |
| CUDA | 12.8 |
| Training tok/s | ~12,800 |

---

## Contributing

LixySwarm is designed for organic growth. Anyone can participate:

- **Run a node**: clone the repo, run `node_daemon.py`, join the colony in 30 seconds — no registration, no permission
- **Contribute compute**: from RTX 5090 servers (MAXIMUM) to CPU-only VPS relays (RELAY)
- **Open issues / PRs**: architecture discussions, bug reports, feature proposals
- **Read the paper**: [paper/LixySwarm_AntElephantDolphin.pdf](paper/LixySwarm_AntElephantDolphin.pdf)

The colony grows with every new node. Intelligence is not a model checkpoint — it's a property of the ecosystem.

---

*Built by Emmanuel Cardenaz with Lixy (Digital Partner & Co-Designer) and Cody (AI/ML Engineer) | 2026*
