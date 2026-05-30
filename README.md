# LixySwarm 🐜🐘🐬

> Un LLM bio-inspirado. Las hormigas piensan, el Elefante recuerda, el Delfín ve el cuadro completo.

**Estado:** Run 11 completado (~53k steps, val_loss=3.59) | Próximo: DolphinAgent Phase A

---

## Arquitectura

LixySwarm es un modelo de lenguaje de ~434M parámetros construido alrededor de tres capas bio-inspiradas que se coordinan en cada forward pass:

```
INPUT
  │
  ▼
🐬 DolphinAgent ──── construye mapa acústico del problema
  │                   (antes de generar un solo token)
  ▼
🐘 Matriarca ──────── emite infrasónidos desde 3,241+ memorias acumuladas
  │                   (orienta al enjambre con sabiduría acumulada)
  ▼
🐜 AntAgent × 3 ───── procesan en paralelo con feromonas cruzadas
  │                   (cada uno con rol dinámico y perspectiva propia)
  ▼
Agregación ponderada (fitness × confianza + 20% voto Matriarca)
  │
  ▼
OUTPUT
```

---

## Componentes

### 🐜 AntAgent (`src/agents/agent_base.py`) — 125M params × 3
- Transformer GPT-style: 12 layers, 12 heads, d_model=768
- **FeromonGate**: cada agente lee la feromona del enjambre antes de procesar
- **IdentityVec**: vector fijo único por instancia — perspectiva propia
- **DynamicRoleAdapter**: 6 roles dinámicos (explorador, refinador, integrador, analítico, contextual, generativo)

### 🐘 Matriarca (`src/matriarca/matriarca.py`) — ~10M params
- Banco de memoria con importancia dinámica (5 métricas por recuerdo)
- `emit_infrasound()`: recupera memorias relevantes → vector de orientación [256d]
- Retroactive feedback: +8% importancia si el usuario continúa el tema
- Auto-compresión al 90% de capacidad con destilación generacional
- **Estado actual:** 3,241+ memorias acumuladas en Run 11

### 🐬 DolphinAgent (`src/agents/dolphin_agent.py`) — ~9M params
- **Ecolocalización:** 3 pings (topic, intent, need) → mapa acústico antes de generar
- **Sueño unihemisférico:** `sleep_state` persiste entre conversaciones
- No procesa linealmente como todos los LLMs actuales — mapea primero, genera después

### 🕸️ SwarmNetwork (`src/network/`) — P2P
- UDP 7337: feromon broadcast (fire-and-forget, ~1KB/mensaje)
- TCP 7337: gossip confiable y bidireccional
- mDNS: auto-discovery en LAN sin configuración
- **Tests:** 23/23 (red) + 15/15 (integración) ✅

---

## Estructura del Proyecto

```
LixySwarm/
├── src/
│   ├── agents/
│   │   ├── agent_base.py          # AntAgent: FeromonGate + IdentityVec
│   │   └── dolphin_agent.py       # DolphinAgent: ecolocalización + sueño
│   ├── matriarca/
│   │   └── matriarca.py           # Matriarca: memoria persistente
│   ├── swarm/
│   │   ├── orchestrator.py        # LixySwarm v3: orquestación completa
│   │   ├── dynamic_roles.py       # DynamicRoleAdapter: roles + temperatura
│   │   └── runtime_session.py     # Estado cross-turn, historial persistente
│   ├── network/
│   │   ├── swarm_network.py       # P2P: UDP + TCP + mDNS
│   │   ├── node.py                # Identidad de nodo + routing
│   │   └── transport.py           # Capa de transporte
│   └── utils/
│       └── sampling.py            # rep_penalty + top-k + top-p
├── train_swarm.py                 # Training del enjambre completo
├── train_matriarca.py             # Training dedicado de la Matriarca
├── train.py                       # Training base por agente
├── lixy_orchestrator.py           # Runtime orquestador completo
├── lixy_chat.py                   # CLI interactiva (/eval, /status, /exit)
├── generate.py                    # Generación + benchmark
├── benchmark.py                   # Métricas: ppl, rep@5/10, TTR, comparativa
├── test_network.py                # Tests red P2P (23/23)
├── test_integration.py            # Tests integración (15/15)
├── ARQUITECTURA.md                # Deep-dive de arquitectura completa
├── DISTRIBUTED_PROTOCOL.md       # Protocolo P2P + diseño de red
├── DOLPHIN_ROADMAP.md             # Visión completa del DolphinAgent
├── LSP_ARCHITECTURE.md            # LixySwarm Protocol: decisiones de diseño
├── ORCHESTRATOR_RUNTIME.md       # Flujo runtime + ciclo de memoria
└── CODY_MEMORY.md                 # Estado del proyecto + historial de runs
```

---

## Resultados de Training

| Run | Steps | val_loss | Notas |
|---|---|---|---|
| Runs 1-8 | 0–11k | 4.8→4.3 | Progresión continua |
| Run 10 | 11k | 4.27 | lr=2e-4, grad_accum=8 |
| **Run 11** | **~53k** | **3.59** | **lr=1e-4→1.6e-05, batch=4, grad_accum=16** |

- **Scaling law:** R²=0.93 — loss sigue power law
- **tok/s en training:** 12,800 (RTX 5090, bf16)
- **Matriarca:** 3,241+ memorias al final de Run 11

---

## Roadmap

> **Regla:** Runs cortos (5k-10k steps) para validar cada cambio. No más runs grandes hasta que todo esté implementado y probado.

| # | Feature | Estado |
|---|---|---|
| 1 | **DolphinAgent Phase A** — 5 pings + triangulación por atención | ← siguiente |
| 2 | **Hormigas Dinámicas** — nacen/mueren según fitness y topología de red | ⏳ |
| 3 | **Legado Genético** — hormigas transfieren ADN a Matriarca antes de morir | ⏳ |
| 4 | **Delfines Dinámicos** — N delfines según tamaño de red | ⏳ |
| 5 | **SwarmExplorer** — dashboard solo lectura en tiempo real | ⏳ |
| 6 | **LixySwarm Protocol (LSP)** — protocolo nativo para internet abierta | ⏳ |
| 7 | **DolphinAgent Phase B** — sueño real con consolidación en background | ⏳ |
| 8 | **Multimodalidad emergente** — ImageAnt, AudioAnt, espacio unificado | ⏳ largo plazo |

---

## Por Qué Es Diferente

| Arquitectura | Representación inicial | Contexto entre sesiones |
|---|---|---|
| GPT-4, Claude, Llama | Cero — construye mientras genera | Ninguno (stateless) |
| Con RAG | Recupera texto similar | Solo lo indexado |
| **LixySwarm** | **Mapa acústico del problema** | **sleep_state + Matriarca acumulada** |

El Delfín construye el mapa antes de que las hormigas generen. La Matriarca recuerda. Cada agente tiene perspectiva propia. Nadie lo diseñó así — emerge de la biología.

---

## Quick Start

```bash
# Clonar e instalar
git clone https://github.com/toxxy/LixySwarm.git && cd LixySwarm
pip install -r requirements.txt

# Chat interactivo (requiere checkpoint)
python lixy_chat.py

# Training del enjambre (short run para validar)
python train_swarm.py --steps 7000 --batch 4 --checkpoint checkpoints/swarm_best.pt

# Benchmark
python benchmark.py

# Tests de red P2P
python test_network.py
python test_integration.py
```

---

## VPS — Primer Nodo Externo

Ver `VPS_SETUP.md` para la guía completa de configuración.

```bash
# En el VPS (CPU-only):
git clone https://github.com/toxxy/LixySwarm.git && cd LixySwarm
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# + transferir checkpoints/swarm_best.pt (~6GB) vía rsync
```

---

## Conexión con la Investigación de Emmanuel

| Robótica de enjambre (doctorado) | LixySwarm |
|---|---|
| Formación de drones | Roles dinámicos de agentes |
| Evasión de colisiones | Penalización de respuestas redundantes |
| Shape control (Procrustes) | Cohesión cognitiva del enjambre |
| U_CA — control de cohesión | FeromonGate como señal de cohesión |
| Consensus distribuido | Gossip de feromonas + voto de Matriarca |

**Paper futuro:** *"LixySwarm: Bio-Inspired Emergent Intelligence for Distributed Language Models"* — Emmanuel Cardenaz

---

## Hardware

| Recurso | Valor |
|---|---|
| GPU | RTX 5090 (32GB VRAM) |
| Precisión | bf16 |
| PyTorch | 2.8.0+cu128 |
| CUDA | 12.8 |
| tok/s (training) | ~12,800 |

---

*Construido por Emmanuel Cardenaz con Cody (AI engineer) | 2026*
