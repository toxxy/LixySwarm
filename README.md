# LixySwarm 🐜🐘🐬

> A bio-inspired swarm language model. Ants think, the Elephant remembers, the Dolphin sees the whole picture.

**Status:** Active development — Run 11 training in progress (~50k/52k steps)

---

## Architecture

LixySwarm is a 434M parameter language model built around three bio-inspired components:

### 🐜 AntAgent (AgentBase — 125M params each)
- Each ant has a **FeromonGate**: attention-gated feromon signals that orient generation
- Each ant has an **IdentityVec**: unique learned identity vector per agent
- **DynamicRoleAdapter**: 6 task types (explorador, refinador, integrador...), temperature adapts dynamically
- Multiple ants work in parallel, contributing complementary perspectives

### 🐘 Matriarca (Elephant layer)
- Persistent memory across conversations (currently 3241+ memories)
- Importance scoring: 5 metrics (length, TTR, bigram-rep, semantic coherence, thematic continuity)
- Emits **infrasound** to orient feromon gates before generation
- Memory survives restarts — the swarm remembers

### 🐬 DolphinAgent
- **Echolocation**: 3 ping projections (topic, intent, need) → acoustic map before generation
- **Unihemispheric sleep**: `sleep_state` tensor persists between conversations
- Builds a representation of the *problem space* before delegating to ants
- Unlike all current LLMs: maps first, generates second

### 🕸️ SwarmNetwork (P2P)
- UDP gossip for feromon broadcast (fire-and-forget)
- TCP reliable channel for gossip protocol
- `inject_remote_feromon()` / `merge_remote_feromons()` — feromon sharing between nodes
- mDNS for LAN auto-discovery (no configuration needed)

---

## Project Structure

```
lixy-llm/
├── src/
│   ├── agents/
│   │   ├── agent_base.py      # AntAgent: FeromonGate + IdentityVec
│   │   └── dolphin_agent.py   # DolphinAgent: echolocation + unihemispheric sleep
│   ├── matriarca/
│   │   └── matriarca.py       # Elephant layer: persistent memory
│   ├── swarm/
│   │   ├── orchestrator.py    # LixySwarm v3: full swarm orchestration
│   │   ├── dynamic_roles.py   # DynamicRoleAdapter: task typing + temperature
│   │   └── runtime_session.py # Cross-turn state management
│   ├── network/
│   │   ├── swarm_network.py   # P2P network layer
│   │   ├── node.py            # Node identity + routing
│   │   └── transport.py       # UDP/TCP transport
│   └── utils/
│       └── sampling.py        # rep_penalty + top-p + recent window
├── train_swarm.py             # Main swarm training loop
├── train_matriarca.py         # Matriarca evolutionary training
├── train.py                   # Base model training
├── generate.py                # Generation + benchmark
├── lixy_orchestrator.py       # Full runtime orchestrator
├── lixy_chat.py               # Interactive CLI
├── benchmark.py               # Perplexity + quality metrics
├── test_network.py            # P2P network tests (23/23)
├── test_integration.py        # Integration tests (15/15)
├── ARQUITECTURA.md            # Architecture deep-dive
├── DISTRIBUTED_PROTOCOL.md   # P2P protocol design
├── DOLPHIN_ROADMAP.md         # DolphinAgent vision + roadmap
└── LSP_ARCHITECTURE.md        # LixySwarm Protocol spec
```

---

## Training Results

| Run | Steps | val_loss | Notes |
|-----|-------|----------|-------|
| Run 1-8 | 0–11k | 4.8→4.3 | Progressive improvement |
| Run 10 | 11k | 4.27 | lr=2e-4, grad_accum=8 |
| Run 11 | 50k+ | 3.59 | Current, near completion |

**Scaling law fit:** R²=0.93 — loss follows power law as expected.

---

## Roadmap

- [ ] **DolphinAgent Phase A** — 5 pings + attention triangulation
- [ ] **Dynamic Ants** — ants born/die based on fitness and network topology
- [ ] **Genetic Legacy** — dying ants transfer essence to Matriarca; new ants inherit DNA
- [ ] **Dynamic Dolphins** — 1 dolphin for small net, N dolphins for large net
- [ ] **SwarmExplorer** — read-only real-time dashboard
- [ ] **LixySwarm Protocol (LSP)** — native wire protocol for distributed training

---

## Requirements

```
torch >= 2.0
transformers
numpy
tiktoken
```

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Interactive chat
python lixy_chat.py

# Train swarm
python train_swarm.py --steps 10000 --batch 4

# Benchmark
python benchmark.py
```

---

## Philosophy

> Most LLMs are built like solo performers.  
> LixySwarm is built like an ecosystem.

The ants don't just run in parallel — they specialize, they compete, they share pheromones. The Matriarca accumulates wisdom across all conversations. The Dolphin maps the problem before anyone starts generating.

No central server. No single point of failure. The swarm is the model.

---

*Built by Emmanuel Cardenaz with Cody (AI engineer) | 2026*
