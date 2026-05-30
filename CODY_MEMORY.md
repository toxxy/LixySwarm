# CODY_MEMORY.md — Estado del Proyecto Lixy-0.1
*Última actualización: 2026-05-28*

## Estado General del Proyecto

**Lixy-0.1** — LLM bio-inspirado con arquitectura swarm (Hormiga + Elefante + Delfín)
- Workspace: `/home/toxxy/Dropbox/Lixy/clawd/workspace/lixy-llm/`
- Orquestador actual: **v3** (DolphinAgent integrado como agente real)

---

## Checkpoints Existentes

| Archivo | Descripción | Val Loss |
|---|---|---|
| `checkpoints/best.pt` | Pre-training base | ~4.5 |
| `checkpoints/finetune_best.pt` | Fine-tuning corpus personal | 0.14 |
| `checkpoints/swarm_best.pt` | Training conjunto del enjambre | 0.4385 (step 450) |
| `checkpoints/matriarca.pt` | Matriarca entrenada | loss 0.008 |
| `checkpoints/matriarca_next.pt` | Nueva generación (destilación) | — |

---

## Trabajo Completado (2026-05-28)

### Bugs Arreglados
1. **Importancia negativa en MemoryBank** (`matriarca.py`, `orchestrator.py`)
   - `add()` ahora clampea importancia a `[0.0, 1.0]`
   - `_prune()` usa `max(0.0, score)` 
   - Importancia en orchestrator escalada: `max(0, min(1, 1.0 - avg_loss/10.0))`

2. **Off-by-one checkpoint resume** (`train_swarm.py`)
   - El loop ahora hace eval final en `step == start_step + max_steps` antes del break
   - Salta eval del primer step al reanudar (ya evaluado)

3. **DataLoader freeze con FineWeb 900M tokens** (`train_swarm.py`)
   - `shuffle=True` generaba índice permutado de ~3.6GB → timeout
   - Fix: `shuffle=False` + randomización interna en `__getitem__` con hash del índice
   - Primer batch: 14ms ✅

### Archivos Nuevos
- `train_matriarca.py` — Training dedicado para la Matriarca
- `train_swarm.py` — Training del LixySwarm completo (con soporte mixto FineWeb/personal)
- `src/agents/dolphin_agent.py` — DolphinAgent completo (3 componentes)
- `DISTRIBUTED_PROTOCOL.md` — Diseño protocolo P2P para red distribuida

### Trainings Ejecutados
1. **Matriarca 300 pasos**: loss 0.19→0.008 (↓95%), destilación 99.5%
2. **Swarm 500 pasos** (solo personal): val_loss 7.83→0.44 (↓94.4%), 37 memorias
3. **Swarm 500 pasos** (90% FineWeb + 10% personal): 64 memorias Matriarca, val_loss ~4.78

---

## Arquitectura Actual — LixySwarm v3

```
input (tokens)
    ↓
🐬 DolphinAgent (29.5M)
   - Ecolocalización: 3 pings (topic, intent, need) → feromon
   - Sueño unihemisférico: estado persistente entre turnos
   - Silbido único: IdentityVec no-entrenable
    ↓
🐘 Matriarca (5.6M) → infrasónidos
    ↓
InfrasoundMixer (feromon + infrasónidos)
    ↓
🐜 AgentBase ×3 (125.7M cada uno) — con feromonas guiadas
    ↓
FeromonPool → nuevo feromon (ronda siguiente)
    ↓
Aggregación por confianza → logits_final
    ↓
🐘 Matriarca.store() + 🐬 DolphinSleep.update()
```

**Total params: ~434M**

---

## DolphinAgent — Componentes

### 1. Ecolocalización (`Echolocation`)
- 3 `PingEncoder` independientes: topic, intent, need
- Fusión → feromona de salida (feromon_dim=256)
- Cabeza de confianza (Sigmoid)

### 2. Sueño Unihemisférico (`HalfSleepState`)
- Buffer circular de 16 contextos pasados
- Estado acumulado con decay=0.95
- Thread-safe (`threading.Lock`)
- Persiste en save/load

### 3. Silbido Único (`IdentityProjector`)
- `identity_vec`: buffer no-entrenable (normalizado en esfera unitaria)
- Proyección entrenable: identity_dim → feromon_dim
- Gate suave para modular feromona final

---

## Integración con Orquestador

`EcholocationHead` reemplazada por `DolphinSwarmBridge` en `LixySwarm.__init__()`:
- El estado de sueño del Delfín alimenta la Matriarca directamente
- `dolphin_info["sleep_for_matriarca"]` → `matriarca.store_interaction()`
- Parámetros del Delfín incluidos en los 434M totales del swarm

---

## Protocolo Distribuido — Diseño

Ver `DISTRIBUTED_PROTOCOL.md`. Decisiones clave:
- Node ID: `SHA256(IdentityVec)` — reutiliza arquitectura existente
- Feromonas: UDP, float16, ~550 bytes/mensaje
- Matriarca distribuida: Gossip + CRDT merge (sin coordinador central)
- Federated learning: solo deltas comprimidos, nunca datos privados
- `SwarmNetwork`: abstracción transparente single/multi-node

---

## Integración Matriarca-Orquestador (2026-05-28 noche)

### Gaps identificados y corregidos:

1. **orquestador post-sesión**: `penalize_unused` cada 10 mensajes (penalty=-0.005, suave)
2. **auto post-training**: `train_from_swarm_log` se ejecuta automáticamente al terminar `train_swarm.py`
3. **Matriarca en selección**: `matriarca_conf_proj` (infrasound_dim → n_agents) da bias 20% a confidence_heads. Decisión: 80/20 split para no dominar la selección.
4. **Device fix**: `.to(feromon.device)` en `_get_infrasound()` para CPU/CUDA

### Flujo completo Matriarca:
```
Forward:
  infrasound = emit_infrasound(state, use_retrieval=True, update_importance=True)
  feromon = InfrasoundMixer(feromon, infrasound)        # orienta input de agentes
  conf_bias = matriarca_conf_proj(infrasound)           # orienta selección de agentes
  weights = 0.8 * conf_heads + 0.2 * conf_bias          # agrega con bias

Post-training:
  auto-run train_from_swarm_log(log)                   # actualiza desde log

Periodicamente:
  penalize_unused(penalty=-0.01)                       # limpia memorias no usadas
```

---

## Estado Post-Crash (2026-05-28 noche)

- `swarm_latest.pt` → **CORRUPTO** (zip incompleto, crash OOM en step ~490)
- `swarm_best.pt` → **VÁLIDO**, step=450, val_loss=0.4385 ✅
- GPU libre: 1.1GB / 30.9GB disponible
- Para re-lanzar training v3: `batch=2` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

---

## Fixes Aplicados (2026-05-28 — Cody, segunda sesión)

### Bugs DolphinAgent (dolphin_agent.py)

1. **Device mismatch en `DolphinAgent.forward()`**
   - `sleep_ctx` era `.to(self.device)` → forzaba a CPU cuando `feromon_echo` estaba en CUDA
   - Fix: `.to(idx.device, dtype=feromon_echo.dtype)` — sigue device Y dtype del input

2. **dtype mismatch float32/bfloat16 en `HalfSleepState.update()`**
   - `_projector` inicializado en float32, pero en training bfloat16 recibía tensores bf16
   - Fix: casteo explícito al dtype del projector + sync del `awake_state`

3. **dtype mismatch en `DolphinSwarmBridge.forward()`**
   - `sleep_to_matriarca` linear recibía float32 input en contexto bf16
   - Fix: `.to(idx.device, dtype=next(self.sleep_to_matriarca.parameters()).dtype)`

4. **Proceso zombie de training anterior** usaba 22.8 GB de GPU
   - Matado PID 26302, liberando 30.5 GB

### Resultado
- Forward del LixySwarm v3 (434M params) OK en bfloat16 CUDA
- Training re-lanzado desde `swarm_best.pt` (step 450, val_loss 0.4385)
- **Running**: PID 26517, step 460+, loss ~0.75, ~13k tok/s
- Log: `/tmp/swarm_v3_train.log`

---

## Plan Inmediato

### PRIORIDAD 1 — Loop Evolutivo Completo Matriarca

**4 componentes a implementar en `src/matriarca/matriarca.py` + `train_matriarca.py`:**

1. **Selección/Retrieval activo** — `MemoryBank.retrieve(query, top_k)`: buscar memorias relevantes durante forward, no solo almacenar
2. **Actualización de importancia** — memorias usadas en inferencia correcta suben importancia; inútiles bajan
3. **Compresión generacional** — al llegar al 90% de capacidad, comprimir las menos importantes en "memorias sintéticas"
4. **`train_matriarca.py --from-swarm-log`** — ciclo standalone que lee interacciones del training del swarm y re-entrena la Matriarca

### PRIORIDAD 2 — Re-lanzar Swarm v3
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 -u train_swarm.py --steps 500 --batch 2 \
  --checkpoint checkpoints/swarm_best.pt > /tmp/swarm_v3_train.log 2>&1 &
```

---

## Próximos Pasos Sugeridos

1. **Training del enjambre v3** — entrenar el swarm con DolphinAgent integrado desde cero o desde fine-tuning
2. **Implementar `src/network/`** — Fase 1 del protocolo distribuido (SwarmNetwork local)
3. **Loop evolutivo avanzado** — selección/compresión/expansión del banco de memorias (pendiente de Emmanuel)
4. **Más corpus** — el corpus personal de 41k tokens es muy pequeño; necesita expansión

---

## Notas Técnicas

- **RTX 5090**: ~16,600 tok/s en training del swarm, bf16
- **FineWeb**: 900M tokens train, 100M val (1.8GB total)
- **Personal corpus**: 41,986 tokens — overfitting visible con loss <0.1
- Training mixto val_loss ~4.78 después de 500p sobre FineWeb — necesita más pasos para generalizar
- `torch.compile` desactivado en swarm training (arquitectura compleja con Matriarca externa)
