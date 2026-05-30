# CODY_MEMORY.md — Estado del Proyecto LixySwarm
*Última actualización: 2026-05-30 | Run 11 completado*

---

## Estado General

**LixySwarm** — LLM bio-inspirado con arquitectura swarm (Hormiga + Elefante + Delfín)
- **Repo oficial:** https://github.com/toxxy/LixySwarm.git
- Workspace local: `/home/toxxy/Dropbox/Lixy/clawd/workspace/lixy-llm/`
- Orquestador: **LixySwarm v3** (DolphinAgent + 3 AntAgents + Matriarca)
- **Parámetros totales:** ~434M

---

## Checkpoints

| Archivo | Val Loss | Steps | Descripción |
|---|---|---|---|
| `checkpoints/swarm_best.pt` | 3.59 | ~53k | Run 11 — checkpoint principal actual |
| `checkpoints/swarm_latest.pt` | — | — | Último paso guardado |
| `checkpoints/swarm_final.pt` | — | — | Final de run anterior |
| `checkpoints/matriarca.pt` | 0.008 | 300 | Matriarca entrenada |

**Matriarca:** 3,241+ memorias acumuladas en Run 11

---

## Historial de Training

| Run | Steps totales | val_loss final | Notas |
|---|---|---|---|
| Runs 1-8 | 0 → 11k | 4.8 → 4.27 | Progresión continua |
| Run 10 | 11k → 11.9k | 4.27 | lr=2e-4, grad_accum=8 |
| Run 11 | 12k → ~53k | **3.59** | 40k steps, lr=1e-4→1.6e-05, batch=4, grad_accum=16 |

**Scaling law fit:** R²=0.93 — loss sigue power law como se esperaba.

**Bug crítico resuelto:** `load_swarm()` usaba `block_size=1024` en vez de 512 → ppl 78K → 51 post-fix.

---

## Arquitectura Implementada

### `src/agents/agent_base.py` — AntAgent (125M params)
- Transformer GPT-style: 12 layers, 12 heads, d_model=768
- **FeromonGate**: atención gateada sobre señal de feromona entrante
- **IdentityVec**: vector fijo [256d] único por instancia

### `src/agents/dolphin_agent.py` — DolphinAgent (~9M params)
- 3 pings de ecolocalización: topic, intent, need
- sleep_state persistente entre sesiones
- Primer en el forward pass — mapea antes de que las hormigas procesen

### `src/matriarca/matriarca.py` — Matriarca (~10M params)
- MemoryBank: hasta 10k memorias con importancia dinámica (5 métricas)
- `emit_infrasound()`: orienta feromonas antes del forward
- `store_memory()`: guarda interacciones con importancia calculada
- Auto-compresión al 90% de capacidad
- Retroactive feedback: +8% si usuario continúa el tema

### `src/swarm/orchestrator.py` — LixySwarm v3
- Orquesta: Delfín → Matriarca → 3 AntAgents → Agregación
- InfrasoundMixer: blend 70/30 feromona nueva/anterior
- Voto de Matriarca: 20% del peso de agregación

### `src/swarm/dynamic_roles.py` — DynamicRoleAdapter
- 6 tipos de tarea: explorador, refinador, integrador, analítico, contextual, generativo
- Temperatura dinámica por tipo + pesos de confianza

### `src/swarm/runtime_session.py` — RuntimeSession
- Estado cross-turn: feromona, historial (últimos 5 turnos), fitness
- Persistencia a disco (`lixy_session.json`)
- `penalize_unused()`: memorias no accedidas en sesión bajan -2%

### `src/network/swarm_network.py` — SwarmNetwork
- UDP 7337: feromon broadcast
- TCP 7337: gossip confiable
- mDNS: descubrimiento LAN automático
- `inject_remote_feromon()` + `merge_remote_feromons()`
- **Tests:** 23/23 + 15/15 ✅

### `src/utils/sampling.py` — sample_token
- Repetition penalty (ventana reciente, más agresivo en cercanos)
- Top-k + top-p sampling

---

## Scripts Principales

| Script | Función |
|---|---|
| `train_swarm.py` | Training del enjambre completo (FineWeb 90% + personal 10%) |
| `train_matriarca.py` | Training dedicado de la Matriarca |
| `train.py` | Training base de un agente solo |
| `lixy_orchestrator.py` | Runtime orquestador completo |
| `lixy_chat.py` | CLI interactiva con /eval, /status, /exit |
| `generate.py` | Generación libre + benchmark |
| `benchmark.py` | Perplexity FineWeb/personal, rep@5/10, TTR, comparativa |

---

## Resultados de Benchmarks

### post-Run 11 (step ~53k)
- **val_loss:** 3.59
- **tok/s training:** 12,800
- **Diversidad enjambre:** 0.80
- **Memorias Matriarca:** 3,241+
- Agente individual: ppl=51.5 (con block_size fix), missing=0
- Generación multi-agente: en progreso (fix agregación pendiente)

### Scaling law
- R² = 0.93 entre steps y val_loss
- Loss sigue power law — modelo escalará bien

---

## Roadmap (en orden de prioridad)

> **Regla:** Runs cortos (5k-10k steps) para validar cada cambio. No más runs grandes hasta que todo esté implementado y probado.

### 1. DolphinAgent Phase A ← SIGUIENTE
- Añadir pings `context` + `emotion` → 5 pings total
- Triangulación: `Attention(Q=topic, K/V=all_pings)` — no lineal
- Comparar calidad de feromona vs implementación actual
- Validar con run 7k steps

### 2. Hormigas Dinámicas
- Enjambre sin tamaño fijo — es un ecosistema vivo
- **Nacen:** nuevo nodo en la red OR diversidad del enjambre cae por debajo de umbral
- **Mueren:** fitness bajo sostenido (N steps) OR nodo se desconecta
- Gestión de ciclo de vida en el orquestador

### 3. Legado Genético en Matriarca
- Antes de morir, hormiga transfiere a Matriarca: rol, fitness promedio, patrones top-K
- Nuevas hormigas heredan ese ADN → no arrancan desde cero
- Matriarca como banco genético del enjambre

### 4. Delfines Dinámicos
- Red pequeña → 1 delfín
- Red grande → N delfines, cada uno con frecuencia de ecolocalización propia
- N delfines = imagen más rica del problema que 1 solo

### 5. SwarmExplorer — Dashboard Solo Lectura
- Nodos conectados, colonias activas, cantidad de hormigas
- tok/s agregado, diversidad genética, estado Matriarca y memorias
- Solo lectura — sin controles, sin botones peligrosos
- Útil tanto para Emmanuel como para Cody al monitorear

### 6. LixySwarm Protocol (LSP)
- Protocolo nativo para distribución en internet abierta
- Wire format: compresión de tensores, identidad Ed25519, merge en tránsito
- RFC-style: spec para que cualquier dev implemente un nodo en otro lenguaje
- Matriarca Dual: Personal (privada, local) + Global (distribuida, compartida)

### 7. DolphinAgent Phase B
- Sueño real: cron job de consolidación cada 30min sin actividad
- PCA del historial → actualizar sleep_state
- Primera respuesta post-reinicio ya tiene contexto acumulado

---

## VPS — Primer Nodo Externo

- **Repo:** https://github.com/toxxy/LixySwarm.git (código subido ✅)
- **Guía:** `VPS_SETUP.md` en el repo
- **Qué falta:** transferir `checkpoints/swarm_best.pt` (~6GB) vía rsync
- **Puerto requerido:** UDP/TCP 7337 abierto en firewall

---

## Reglas de Trabajo

1. **Todo cambio va al repo** `https://github.com/toxxy/LixySwarm.git`
2. **Runs cortos** (5k-10k steps) para validar cada feature nueva
3. Commit + push después de cada cambio significativo
4. No modificar archivos fuera de `lixy-llm/` para código del proyecto
