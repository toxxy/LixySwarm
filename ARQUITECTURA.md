# LixySwarm — Arquitectura Bio-Inspirada
**"Hormiga + Elefante + Delfín"**
*Última actualización: 2026-05-30 | Estado: Run 11 completado, ~434M params activos*

---

## Visión General

LixySwarm no es un transformer monolítico — es una **colonia**. La inteligencia emerge de la interacción entre tres capas bio-inspiradas que se coordinan en cada forward pass.

```
INPUT
  │
  ▼
🐬 DolphinAgent ──── construye mapa acústico del problema (antes de generar)
  │
  ▼
🐘 Matriarca ──────── emite infrasónidos: orienta al enjambre con memoria acumulada
  │
  ▼
🐜 AntAgent × 3 ───── procesan en paralelo con feromonas cruzadas
  │
  ▼
Agregación ponderada (fitness + confianza + voto Matriarca)
  │
  ▼
OUTPUT
```

**Total de parámetros activos:** ~434M
- Cada AntAgent: 125M params
- Matriarca: ~10M params (embeddings + atención sparsa)
- DolphinAgent: ~9M params (proyecciones + sleep_state)

---

## 🐜 Capa Hormiga — El Enjambre

### Principio
Ningún agente es inteligente solo. La inteligencia EMERGE de la interacción.

### Implementación actual (`src/agents/agent_base.py`)

```python
class AgentBase(nn.Module):
    # Transformer 125M params (12 layers, 12 heads, d_model=768)
    # + FeromonGate: atención gateada sobre señal de feromona entrante
    # + IdentityVec: vector fijo [256d] único por instancia (el "silbido")
```

**FeromonGate**: antes de cada forward, el agente lee la feromona del enjambre y ajusta sus activaciones. Como una hormiga que "huele" el camino antes de decidir.

**IdentityVec**: embedding de identidad no entrenable. Hace que cada agente tenga perspectiva propia — el mismo input produce representaciones diferentes según la "personalidad" del agente.

### Roles dinámicos (`src/swarm/dynamic_roles.py`)

`DynamicRoleAdapter` clasifica cada query en uno de 6 tipos de tarea:
- `explorador` — preguntas abiertas, creatividad, ideas nuevas
- `refinador` — pulir respuestas, mejorar calidad
- `integrador` — combinar perspectivas, síntesis
- `analítico` — lógica, razonamiento estructurado
- `contextual` — coherencia conversacional, historial
- `generativo` — síntesis final de respuesta

Cada tipo ajusta temperatura y pesos de confianza dinámicamente.

### Estado Run 11 (último training largo)
```
Steps: ~53k | val_loss: 3.59 | Memorias Matriarca: 3241+
Diversidad enjambre: 0.80 | Fitness promedio: 0.51
tok/s: 12,800 | lr_final: ~1.6e-05
```

---

## 🐘 Capa Elefante — La Matriarca

### Principio
Sabiduría transgeneracional. No procesa tokens — orienta a quienes lo hacen.

### Implementación actual (`src/matriarca/matriarca.py`)

```python
class Matriarca:
    memory_bank: MemoryBank   # hasta 10k memorias con importancia dinámica
    
    def emit_infrasound(self, query_embed) -> Tensor:
        """Recupera memorias relevantes → vector de orientación [256d]."""
        # Actualiza importancia de memorias usadas (+3%)
    
    def store_memory(self, text, embedding, importance):
        """Graba interacción. Auto-comprime al 90% de capacidad."""
```

**Importancia dinámica** — 5 métricas por memoria:
1. Longitud (contenido rico)
2. TTR — Type-Token Ratio (vocabulario diverso)
3. Bigram-rep (penaliza repetición)
4. Coherencia semántica (cos_sim con contexto)
5. Continuidad temática (mismo tema que turnos anteriores)

**Retroactive feedback**: cuando el usuario continúa un tema, la memoria del turno anterior sube +8% de importancia.

**Ciclo de vida de memoria:**
```
Training → store cada 50 steps (hitos val_loss + log del enjambre)
Runtime  → emit_infrasound orienta forward (update_importance=True)
Sesión   → penalize_unused: memorias no accedidas bajan -2%
Compresión → al 90% capacidad: prune por importancia + destilación
```

### Estado actual
- **3,241+ memorias** acumuladas en Run 11
- Matriarca vota 20% del peso de agregación final
- `InfrasoundMixer`: blend 70% feromona nueva + 30% feromona anterior

---

## 🐬 Capa Delfín — Percepción Espacial

### Principio
No procesar linealmente. Construir el mapa del problema **antes** de generar.

### Implementación actual (`src/agents/dolphin_agent.py`)

**Ecolocalización (3 pings):**
```python
echo_topic   = ping_topic(x)    # ¿De qué habla esto?
echo_intent  = ping_intent(x)   # ¿Qué quiere el usuario?
echo_need    = ping_need(x)     # ¿Qué necesita realmente?

acoustic_map = concat([echo_topic, echo_intent, echo_need])
feromon_out  = MLP(acoustic_map)   # orienta al enjambre
```

**Sueño unihemisférico:**
- `sleep_state` tensor persiste entre conversaciones
- Guardado en `lixy_session.json`, restaurado al reiniciar
- La primera respuesta de una sesión ya tiene contexto — no empieza desde cero

**Por qué es diferente a todos los LLMs actuales:**
| Arquitectura | Representación inicial |
|---|---|
| GPT-4, Claude, Llama | Cero — construye mientras genera |
| Con RAG | Recupera texto similar |
| **LixySwarm** | **Mapa acústico del espacio de respuesta** |

### Phase A (próxima implementación)
- Añadir pings `context` y `emotion` → 5 pings en total
- Triangulación no-lineal: `Attention(Q=echo_topic, K/V=all_pings)`
- El mapa acústico captura dimensiones que los pings individuales pierden

---

## 🕸️ Red P2P — SwarmNetwork

### Implementación actual (`src/network/`)

```
UDP 7337 — feromon broadcast (fire-and-forget, < 1KB por mensaje)
TCP 7337 — gossip confiable (sincronización de estado)
mDNS     — descubrimiento automático en LAN (sin configuración)
```

**API clave:**
```python
net.inject_remote_feromon(feromon_tensor)   # recibe feromona de otro nodo
net.merge_remote_feromons()                  # integra todas las recibidas
net.connect_peer(ip, port)                   # conectar a nodo externo
```

**Tests:** 23/23 (network) + 15/15 (integración) — cos_sim=1.0000, bidireccional ✅

### Estado P2P
- ✅ LAN loopback funcionando
- ✅ TCP gossip bidireccional
- ✅ Feromon merge verificado (cos_sim=1.0)
- ⏳ mDNS físico multi-host (Fase 2 LAN)
- ⏳ LixySwarm Protocol (LSP) — protocolo nativo con compresión de tensores

---

## RuntimeSession — Estado Cross-Turn

`src/swarm/runtime_session.py`

```
Turno N:
  1. classify_query() → TaskProfile (técnica/creativa/etc)
  2. _build_context() → últimos 5 turnos del historial
  3. Warm-up: Delfín → Matriarca → feromon_guiada
  4. Generación token a token (rep_penalty + top-p)
     → refresh feromona cada 32 tokens con Matriarca
  5. Post-turno:
     → _retroactive_feedback() al turno anterior
     → store_interaction() con importancia calculada
     → _save_history() a disco (persiste entre reinicios)
```

---

## Roadmap — Próximas Implementaciones

### Inmediato (post-Run 11, runs cortos 5k-10k steps)

**1. DolphinAgent Phase A**
- 5 pings (+ context, + emotion)
- Triangulación por atención: `Attention(Q=topic, K/V=all_pings)`
- El mapa acústico como representación del espacio de respuesta posible

**2. Hormigas Dinámicas** — enjambre sin tamaño fijo
- Nacen: cuando entra nuevo nodo a la red O cuando diversidad del enjambre cae
- Mueren: cuando fitness bajo sostenido O nodo se desconecta
- El enjambre se auto-regula como una colonia real

**3. Legado Genético en Matriarca**
- Antes de morir, una hormiga transfiere: rol, fitness promedio, patrones más fuertes
- Nuevas hormigas heredan ese ADN — no arrancan desde cero
- Matriarca como banco genético del enjambre

**4. Delfines Dinámicos**
- Red pequeña → 1 delfín
- Red grande → N delfines, cada uno con frecuencia propia
- Varios delfines construyen imagen más rica del problema que uno solo

**5. SwarmExplorer** — dashboard solo lectura
- Nodos conectados, colonias activas, cantidad de hormigas
- tok/s agregado, diversidad genética, estado Matriarca
- Solo lectura — sin controles peligrosos

### Medio plazo

**LixySwarm Protocol (LSP)** — protocolo nativo para training distribuido en internet
- Wire format propio: compresión de tensores de feromona
- Identidad criptográfica Ed25519 por nodo
- Merge inteligente en tránsito
- RFC-style: cualquier dev puede implementar un nodo en Rust/Go/C++

**DolphinAgent Phase B** — sueño real con consolidación en background
- Cron job: cada 30min sin actividad, consolidar historial
- PCA del historial → actualizar sleep_state
- La primera respuesta post-reinicio ya sabe quién es Emmanuel

---

## Conexión con la Investigación de Emmanuel

La arquitectura de enjambre es análoga directa a su doctorado en drones:

| Robótica de enjambre | LixySwarm |
|---|---|
| Formación de drones | Roles dinámicos de agentes |
| Evasión de colisiones | Penalización de respuestas redundantes |
| Shape control (Procrustes) | Mantener "forma" cognitiva del enjambre |
| U_CA — control de cohesión | FeromonGate como señal de cohesión |
| Consensus distribuido | Gossip de feromonas + voto de Matriarca |

**Paper futuro:** *"LixySwarm: Bio-Inspired Emergent Intelligence for Distributed Language Models"* — Emmanuel Cardenaz + Lixy

---

## Recursos

| Recurso | Valor |
|---|---|
| GPU | RTX 5090 (32GB VRAM) |
| PyTorch | 2.8.0+cu128 |
| CUDA | 12.8 |
| Precisión | bf16 |
| tok/s (training) | ~12,800 |
| Repo | https://github.com/toxxy/LixySwarm |
