# LixySwarm — Arquitectura Bio-Inspirada
**"AntElephantDolphin — A Bio-Inspired P2P Architecture for Distributed Language Intelligence"**
*Última actualización: 2026-05-30 | Estado: Fase 2 completa — 146/146 tests | ~568M params*

---

## Visión en 5 minutos

LixySwarm **no es un transformer monolítico**. Es una **colonia distribuida**: la inteligencia emerge de la interacción entre tres capas bio-inspiradas que se coordinan en cada forward pass, en cada nodo de la red.

```
────────────────── RED P2P (LSP v2) ──────────────────
  Nodo A (GPU)            Nodo B (CPU)       VPS Relay
  ┌────────────┐          ┌──────────────┐   ┌───────┐
  │ 🐜 Enjambre│◄─feromon─►│ 🐜 Enjambre  │   │ relay │
  │ 🐘 Matriarca│         │ 🐘 Matriarca  │   │ gossip│
  │ 🐬 Delfín  │         │ 🐬 Delfín    │   └───────┘
  └────────────┘          └──────────────┘
       │                        │
       └──────── feromonas ──────┘  (UDP 7337, LSP v2 binario)
       └──────── gossip   ──────┘  (TCP 7338)

────────────── DENTRO DE CADA NODO ──────────────────
INPUT
  │
  ▼
🐬 EcholocationRouter ── construye mapa acústico del problema (5 pings × 128d)
  │  (DolphinPool: 1..N delfines según complejidad)
  ▼
🐘 Matriarca ─────────── emite infrasónidos: orienta con 3000+ memorias acumuladas
  │  (MatriarcaDual: banco Personal encriptado + banco Global distribuido)
  ▼
🐜 Enjambre de Hormigas ─ procesan en paralelo con feromonas cruzadas
  │  (NodeManager: 1 hormiga = 1 nodo físico; SectManager: especialidades vivas)
  ▼
Agregación ponderada (fitness × confianza × voto Matriarca × legado genético)
  │
  ▼
OUTPUT
```

**Total params activos:** ~568M
- Cada AntAgent: 125.7M params × 3 = 377M
- DolphinPool: 49.1M params
- Swarm mechanisms (FeromonGate, aggregation): 75.3M
- Matriarca: 5.6M params

---

## 🐜 Capa 1 — El Enjambre de Hormigas

### Principio
Ningún agente es inteligente solo. La inteligencia **emerge** de la interacción.

### NodeManager — Hormiga = Nodo físico (`src/swarm/node_manager.py`)

Cada nodo físico en la red P2P **es** una hormiga. No hay N fijo — el enjambre crece y se contrae con la red.

```python
@dataclass
class NodeRecord:
    node_id: str              # identidad Ed25519 única
    host: str                 # IP o hostname
    feromon_port: int         # UDP — feromonas
    gossip_port: int          # TCP — gossip de estado
    contribution_mode: ContributionMode  # MAXIMUM / MODERATE / RELAY
    hardware_profile: dict    # GPU VRAM, CPU cores, RAM
    effective_gpu_fraction: float
    effective_sect_capacity: int
```

**ContributionMode** — cada nodo contribuye según su capacidad:
| Modo | GPU | Sectas máx | Rol |
|------|-----|------------|-----|
| `MAXIMUM` | 100% | sin límite | Training completo |
| `MODERATE` | 50% | 4 | Training parcial |
| `RELAY` | 0% | 0 | Solo relay de feromonas |

### SectManager — Especialidades vivas (`src/swarm/sect_manager.py`)

Las sectas son **agrupaciones dinámicas de nodos con la misma especialización**. Nacen, mutan, y mueren según fitness.

```python
@dataclass
class SectRecord:
    sect_id: str
    role_type: str      # explorador / refinador / integrador / ...
    n_agents: int
    avg_fitness: float
    alive: bool
    # Cuando muere → transfiere legado genético a la Matriarca
    legacy_transferred: bool
```

**Ciclo de vida de una secta:**
```
Nacimiento: fitness nueva especialidad detectada
    ↓
Crecimiento: más nodos se unen (mismo role_type)
    ↓
Madurez: fitness estabilizado, legado generado
    ↓
Muerte: fitness bajo por N ciclos → transfiere legado a Matriarca
```

**Legado genético**: antes de morir, una secta transfiere a la Matriarca su `role_type`, `avg_fitness`, y los patrones de feromona más fuertes. Las nuevas sectas del mismo tipo heredan ese ADN.

### AgentBase — El transformer individual (`src/agents/agent_base.py`)

```
AgentBase (125.7M params):
├── GPT-2 style transformer (12L × 12H × 768d)
├── FeromonGate: ajusta activaciones según feromona entrante
└── IdentityVec: [256d] único e inmutable — "personalidad" del agente
```

**FeromonGate**: antes de cada forward, el agente lee la feromona del enjambre y ajusta sus activaciones. Como una hormiga que "huele" el camino antes de decidir.

**IdentityVec**: el mismo input produce representaciones diferentes en cada agente — garantiza diversidad sin requerir architecturas distintas.

### DynamicRoleAdapter — Especialización por tarea (`src/swarm/dynamic_roles.py`)

Clasifica cada query en 6 tipos y ajusta temperatura + pesos de confianza dinámicamente:

| Rol | Temperatura | Cuando se activa |
|-----|-------------|------------------|
| `explorador` | alta | preguntas abiertas, creatividad |
| `refinador` | baja | pulir respuestas, calidad |
| `integrador` | media | síntesis, combinar perspectivas |
| `narrativo` | alta | historias, continuación |
| `tecnico` | muy baja | código, matemáticas |
| `aprendiendo` | media | contexto nuevo, adaptación |

---

## 🐘 Capa 2 — La Matriarca

### Principio
La elefanta que recuerda todo. Memoria a largo plazo del enjambre.

### MatriarcaDual — Banco Personal + Global (`src/matriarca/matriarca.py`)

```
MatriarcaDual:
├── Banco Personal (local, encriptado)
│   └── Memorias privadas del nodo, nunca salen a la red
└── Banco Global (distribuido, via LSP)
    └── Memorias compartidas con otros nodos (consentidas)
```

**Elephant layer**: memoria esparsa con `block_size=512`, almacena memorias como pares (embedding, texto). Las más relevantes se refuerzan positivamente en cada uso (`update_importance=True` por defecto).

**`emit_infrasound()`**: durante el forward, la Matriarca emite una señal de orientación (`infrasound_vec`) que se inyecta en el camino de las hormigas via feromona. Como el infrasónido real — frecuencia demasiado baja para escuchar, pero orienta el movimiento de la manada.

**Importancia dinámica** — 5 métricas por turno:
1. Longitud de respuesta (`len`)
2. Type-Token Ratio de diversidad léxica (`TTR`)
3. Penalización bigram repetitivo (`bigram-rep`)
4. Coherencia semántica con el contexto
5. Continuidad temática entre turnos

**Retroactive feedback**: si el usuario continúa el tema del turno anterior, el turno anterior recibe +15% de importancia. Si el usuario cambia de tema bruscamente, -20%.

**Compresión generacional**: al 90% de capacidad, los recuerdos menos importantes se comprimen (averaging) para liberar espacio sin perder el patrón general.

**Legado genético de sectas**: cuando una secta muere, sus patrones van al banco de legados de la Matriarca (`sect_legacy_bank.json`). Las nuevas sectas del mismo tipo heredan ese banco genético.

### RuntimeSession — Estado cross-turn (`src/swarm/runtime_session.py`)

Persiste el estado del enjambre entre turnos (y entre reinicios):

```python
{
  "feromon_state":     [...],   # feromona activa del enjambre
  "turn_history":      [...],   # pares Q+A con importancia
  "session_step":      int,     # contador de turnos
  "matriarca_hits":    {...},   # qué memorias se usaron
}
```

Al cerrar: `penalize_unused()` — memorias que nunca se usaron bajan importancia. Guarda resumen de sesión como nueva memoria (importancia=0.85).

---

## 🐬 Capa 3 — El Delfín

### Principio
El delfín explora antes de comprometerse. Ecolocalización = claridad antes de generar.

### EcholocationRouter — Enrutamiento inteligente (`src/swarm/echolocation_router.py`)

```
EcholocationRouter:
├── DolphinPool (1..N delfines según complejidad de red)
│   └── DolphinAgent: transformer 49.1M params + 5 pings acústicos
├── AdaptiveSleepController
│   └── Ciclos de consolidación en silencio (sin input)
└── Enrutamiento por similitud acústica
    └── Elige qué secta procesa cada query
```

**5 pings de ecolocalización** (Phase A):
- `topic` — ¿de qué trata?
- `intent` — ¿qué quiere el usuario?
- `need` — ¿qué necesita realmente?
- `context` — ¿cuál es el estado del diálogo?
- `emotion` — ¿cuál es el tono esperado?

Cada ping es un vector 128d. Los 5 juntos construyen el "mapa acústico" del problema antes de que las hormigas procesen.

**AdaptiveSleepController**: regula los ciclos de sueño unihemisférico del delfín. En silencio (sin input por N tokens), consolida representaciones — análogo a la consolidación de memoria en sueño real.

**Escala con la red**: 1 delfín en red pequeña → más delfines en red grande, cada uno con su frecuencia de identidad única (`whistle_id`). Varios delfines construyen imagen más rica del problema.

---

## 🌐 Capa 4 — Red P2P (LSP v2)

### LixySwarm Protocol — Wire format propio

LSP no es un wrapper de TCP/UDP. Es un **protocolo nativo** para enjambre distribuido:

```
Paquete LSP v2 (528 bytes fijos):
┌──────────────────────────────────────────────────────┐
│ Magic: 0x4C595357 (LYSW)              [4 bytes]      │
│ Version: 2                            [1 byte]       │
│ Message type: FEROMON/GOSSIP/PING/ACK [1 byte]       │
│ TTL: 0-255                            [1 byte]       │
│ Reserved                              [1 byte]       │
│ Node ID (Ed25519 public key prefix)   [32 bytes]     │
│ Sequence number                       [4 bytes]      │
│ Timestamp (ms)                        [8 bytes]      │
│ Fitness score (float16)               [2 bytes]      │
│ Reserved                              [2 bytes]      │
│ Feromon vector (256d × float16)       [512 bytes]    │
└──────────────────────────────────────────────────────┘
Total: 568 bytes → 528B payload, ~5× más compacto que JSON (~2.5KB)
```

**Semántica nativa de feromona**:
- **Merge-on-transit**: `FeromonMergeBuffer` combina feromonas en vuelo (fitness-weighted average) antes de entregar al destino
- **TTL decay**: cada hop reduce TTL; al llegar a 0, la feromona se descarta
- **Temporal decay**: `strength *= decay_factor^elapsed_seconds`

**Topología**:
- Descubrimiento LAN: mDNS (`_lixyswarm._udp.local`)
- Internet: DHT / relay VPS
- Modo dual: mismo protocolo, diferente capa de descubrimiento

**Identidad criptográfica**: cada nodo tiene un key Ed25519 generado al arranque. No hay servidor central de identidades. Las contribuciones se firman con esta clave.

### SwarmNetwork (`src/network/swarm_network.py`)

```python
SwarmNetwork(
    protocol="v1"  # default, backward-compatible
    # protocol="v2" → activa LSP v2 en puerto feromon+10
)
```

- `broadcast_feromon(feromon, fitness)` → v2 si disponible, fallback a v1
- `merge_remote_feromons()` → recibe y mergea feromonas de la red
- Compatible con nodos v1 y v2 simultáneamente

---

## 🔄 Auto-Training Loop

### `auto_train.py` — El enjambre aprende solo

```
Ciclo infinito:
┌─────────────────────────────────────────────────────┐
│  1. Cargar estado (training_state.json)              │
│  2. Detectar mejor checkpoint para resume            │
│  3. Correr N steps (chunk_steps, default 1000)       │
│     → train_swarm.py como subprocess                 │
│  4. Evaluar val_loss del ciclo                       │
│  5. check_plateau():                                 │
│     - Mejora → reset plateau_count, update best      │
│     - Sin mejora N ciclos → LR *= decay_factor (0.7) │
│     - LR floor: nunca baja de lr_min (5e-6)          │
│  6. Guardar checkpoint ciclo N                       │
│  7. Rotar: mantener solo últimos K checkpoints       │
│  8. Actualizar training_state.json                   │
│  goto 1                                              │
└─────────────────────────────────────────────────────┘
```

**SIGTERM-safe**: al recibir señal, termina el ciclo actual y guarda estado limpio.

**Estado persistido** (`checkpoints/training_state.json`):
```json
{
  "cycles_completed": 5,
  "total_steps": 5000,
  "best_val_loss": 3.4376,
  "lr_current": 1e-4,
  "plateau_count": 0,
  "last_checkpoint": "swarm_auto_cycle5.pt",
  "cycle_history": [[ciclo, steps, val_loss, lr], ...]
}
```

---

## 📊 SwarmExplorer

Dashboard de solo lectura en tiempo real (`frontend/swarm-explorer.html`).

**Lo que muestra:**
- Nodos conectados en la red P2P
- Sectas activas y legados genéticos
- Velocidad agregada (tok/s)
- Diversidad genética del enjambre (0-1)
- Val loss actual + step
- Matriarca: count memorias, importancia media, % activas, legados por rol
- Auto-Loop: ciclos completados, steps totales, LR, plateau, trend val_loss últimos 5 ciclos
- Red LSP v2: protocol, feromonas recibidas
- Nodos: contribution_mode, GPU fraction, sectas máx

**Arquitectura del stack de monitoreo:**
```
RTX 5090 machine
  └── swarm_publisher.py  (cada 15s)
        → lee checkpoints/, logs, training_state.json
        → sube swarm_status.json al VPS via scp
VPS (31.97.9.54)
  └── nginx sirve /opt/lixyswarm/swarm_status.json
        → frontend/swarm-explorer.html consulta cada 15s
```

---

## 🗂️ Estructura del Proyecto

```
lixy-llm/
├── src/
│   ├── agents/
│   │   └── agent_base.py         # AgentBase 125.7M: transformer + FeromonGate + IdentityVec
│   ├── swarm/
│   │   ├── orchestrator.py       # LixySwarm v3: 3 ants + Matriarca + DolphinPool
│   │   ├── node_manager.py       # NodeManager: hormiga=nodo, ContributionMode, hardware
│   │   ├── sect_manager.py       # SectManager: sectas vivas, legado genético, ciclo de vida
│   │   ├── echolocation_router.py # EcholocationRouter: DolphinPool + AdaptiveSleep
│   │   ├── dynamic_roles.py      # DynamicRoleAdapter: 6 tipos, temperatura dinámica
│   │   └── runtime_session.py    # RuntimeSession: estado cross-turn, warm-start
│   ├── matriarca/
│   │   └── matriarca.py          # MatriarcaDual: Personal+Global, emit_infrasound, legado
│   └── network/
│       ├── swarm_network.py      # SwarmNetwork: v1/v2, mDNS, broadcast, merge
│       ├── lsp_node_v2.py        # LSPNodeV2: wire format binario, TTL, Ed25519
│       ├── feromon_payload.py    # FeromonV2Payload: 528B packing/unpacking
│       └── feromon_merge_buffer.py # FeromonMergeBuffer: merge-on-transit
├── train_swarm.py                # Training conjunto (invocado por auto_train)
├── train_matriarca.py            # Loop evolutivo Matriarca
├── auto_train.py                 # Loop de auto-entrenamiento infinito
├── lixy_orchestrator.py          # Orquestador principal (producción)
├── lixy_chat.py                  # CLI interactiva
├── swarm_publisher.py            # Publica estado al VPS cada 15s
├── benchmark.py                  # Suite de benchmarks
├── frontend/
│   └── swarm-explorer.html       # Dashboard solo lectura
├── checkpoints/
│   ├── swarm_best.pt             # Mejor checkpoint (step 54500, val_loss 3.4376)
│   ├── swarm_latest.pt           # Último checkpoint
│   ├── matriarca.pt              # Matriarca: 3131 memorias
│   ├── training_state.json       # Estado del auto-loop
│   ├── ant_specialization.json   # Especialización actual de las hormigas
│   └── runtime_session.json      # Estado cross-turn de la última sesión
├── test_ant_lifecycle.py         # 11 tests
├── test_node_sect.py             # 14 tests
├── test_dolphin_router.py        # 13 tests
├── test_matriarca_legacy.py      # 15 tests
├── test_lsp_v2.py                # 12 tests
├── test_integration.py           # 15 tests
├── test_network.py               # 34 tests
└── test_auto_train.py            # 32 tests  [146/146 total ✅]
```

---

## 🚀 Arranque en producción

```bash
# 1. Training autónomo (RTX 5090)
python3 auto_train.py

# 2. Monitor en tiempo real (publica al VPS cada 15s)
python3 swarm_publisher.py --vps-host root@31.97.9.54

# 3. Chat interactivo local
python3 lixy_chat.py

# 4. Test de red P2P (requiere 2 máquinas en LAN)
# Máquina A:
python3 -c "
from src.network.swarm_network import SwarmNetwork
net = SwarmNetwork(mode='lan', protocol='v2')
net.start()
"
# Máquina B: idem — se descubren vía mDNS automáticamente
```

---

## 📈 Estado actual (2026-05-30)

| Métrica | Valor |
|---------|-------|
| Val loss (mejor checkpoint) | **3.4376** (step 54500) |
| Parámetros totales | **568.2M** |
| Memorias Matriarca | **3131** |
| Tests | **146/146 ✅** |
| Tokens vistos (training) | ~400M (FineWeb 90% + personal 10%) |

### Curva de aprendizaje

```
Run  1-8: val_loss 5.3 → 3.9   (fundamentos del enjambre)
Run  9-10: val_loss 3.9 → 4.27  (ajuste de hiperparámetros)
Run  11:  val_loss 4.27 → 3.5687 (step 53920)
Phase A:  val_loss 3.5687 → 3.4376 (record) ← posición actual
```

---

## 🧬 Diseño filosófico

Este proyecto nació de una pregunta: ¿puede la inteligencia artificial ser **ecológica** en lugar de monolítica?

- **Hormiga** = especialización modular, sin coordinador central
- **Elefante** = memoria acumulada que no se olvida, que orienta sin ordenar
- **Delfín** = exploración antes de comprometerse, escucha activa

El resultado no es un modelo que "sabe más". Es un enjambre que **aprende a aprender** — donde el conocimiento circula como feromona, se consolida como memoria, y se transmite como legado genético entre generaciones.

---

*Documentación mantenida por Cody — AI/ML Engineer de LixySwarm*
*Repo: https://github.com/toxxy/LixySwarm*
