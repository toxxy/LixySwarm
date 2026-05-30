# ORCHESTRATOR_RUNTIME.md
# Orquestador ↔ Matriarca ↔ Delfín: Estado y Flujo Runtime
*Última actualización: 2026-05-30 | LixySwarm v3*

---

## Estado del Ciclo Completo ✅

| Componente | Estado | Descripción |
|---|---|---|
| Training feedback | ✅ | `store_memory=True` durante training, cada 50 steps |
| Infrasónidos en forward | ✅ | Matriarca orienta feromonas en cada forward pass |
| Voto Matriarca en agregación | ✅ | 20% del peso de agregación |
| Sueño del Delfín → Matriarca | ✅ | `sleep_for_matriarca` almacenado si `store_memory=True` |
| Compresión generacional | ✅ | Auto-compresión al 90% de capacidad |
| Feromon refresh | ✅ | Refresh cada 32 tokens con Matriarca activa |
| Blend feromona | ✅ | 70% nueva + 30% anterior (InfrasoundMixer) |
| Feedback post-generación | ✅ | Feedback final con Q+A completo |
| Estado cross-turn | ✅ | `RuntimeSession` mantiene feromon entre turnos |
| Contexto acumulado | ✅ | `_build_context()` incluye últimos 5 turnos |
| Resumen de sesión | ✅ | `end_session()` almacena resumen con importancia |
| `penalize_unused()` | ✅ | Memorias no usadas en sesión: -2% importancia |
| Importancia dinámica | ✅ | 5 métricas: len, TTR, bigram-rep, coherencia semántica, continuidad |
| Retroactive feedback | ✅ | Turno anterior sube +8% si usuario continúa el tema |
| Sesiones persistentes | ✅ | `session_file` serializa history a disco entre reinicios |
| `emit_infrasound` refuerzo | ✅ | `update_importance=True` — memorias usadas suben +3% |
| `train_from_swarm_log` v2 | ✅ | Embeddings reales + chunks + hitos val_loss |
| `matriarca_eval()` | ✅ | Diagnóstico: importancia, diversidad, acceso, tipos |
| Roles dinámicos | ✅ | `DynamicRoleAdapter` — 6 tipos de tarea, temperatura adaptada |
| Feedback explícito usuario | ✅ | 22/22 casos correctos (ES+EN), +15%/-20% importancia |

---

## Flujo Runtime Completo

```
Usuario → lixy_orchestrator.py
              │
              ▼
         LixyOrchestrator.chat(message)
              │
              ▼
         RuntimeSession.turn(user_input)
              │
              ├── 1. classify_query() → TaskProfile
              │      (técnica / exploratoria / creativa / contextual / analítica)
              │      → temperatura dinámica por tipo
              │      → role_weights para agentes (explorador vs refinador vs integrador)
              │
              ├── 2. _build_context()
              │      → últimos 5 turnos del historial
              │      → incluye respuestas anteriores del enjambre
              │
              ├── 3. Warm-up forward (construye feromon orientada)
              │      │
              │      ├── 🐬 DolphinAgent.echolocate(context)
              │      │     → ping_topic + ping_intent + ping_need
              │      │     → acoustic_map [256d]
              │      │     → feromon_dolphin
              │      │
              │      ├── 🐘 Matriarca.emit_infrasound(context_embed)
              │      │     → recupera top-K memorias relevantes
              │      │     → update_importance=True (+3% a memorias usadas)
              │      │     → infrasound [256d]
              │      │
              │      └── InfrasoundMixer
              │            → feromon_guiada = 0.7 * nueva + 0.3 * anterior
              │            → se fija para toda la generación
              │
              ├── 4. Generación token a token
              │      │
              │      ├── 🐜 AntAgent × 3 en paralelo (rol dinámico)
              │      │     → cada uno lee feromon_guiada via FeromonGate
              │      │     → produce logits con perspectiva propia
              │      │
              │      ├── Agregación: fitness × confianza × role_weight + 0.2×matriarca_vote
              │      │
              │      ├── sample_token(): rep_penalty + top-k + top-p
              │      │
              │      └── Refresh feromona cada 32 tokens (Matriarca activa)
              │
              └── 5. Post-turno
                     ├── _retroactive_feedback() → +8% al turno anterior si mismo tema
                     ├── _compute_response_importance() → 5 métricas
                     ├── store_interaction() → Matriarca.add()
                     ├── _save_history() → disco (lixy_session.json)
                     └── update DynamicRoleAdapter fitness scores
```

---

## Importancia Dinámica — 5 Métricas

```python
def _compute_response_importance(text, context):
    score = 0.0
    
    # 1. Longitud (contenido rico)
    score += min(len(text.split()) / 100, 1.0) * 0.2
    
    # 2. TTR — Type-Token Ratio (vocabulario diverso)
    tokens = text.lower().split()
    ttr = len(set(tokens)) / max(len(tokens), 1)
    score += ttr * 0.25
    
    # 3. Anti bigram-rep (penaliza repetición)
    bigrams = list(zip(tokens, tokens[1:]))
    unique_bigrams = len(set(bigrams)) / max(len(bigrams), 1)
    score += unique_bigrams * 0.2
    
    # 4. Coherencia semántica (cos_sim con contexto)
    cos_sim = compute_semantic_similarity(text, context)
    score += cos_sim * 0.2
    
    # 5. Continuidad temática
    topic_continuity = topic_overlap(text, context)
    score += topic_continuity * 0.15
    
    return max(0.0, min(1.0, score))  # siempre en [0, 1]
```

---

## Feedback del Usuario

El orquestador detecta señales implícitas y explícitas:

| Señal | Efecto |
|---|---|
| Usuario continúa el mismo tema | +8% importancia turno anterior |
| Señal explícita positiva ("bien", "correcto", "sí") | +15% importancia |
| Señal explícita negativa ("mal", "no", "incorrecto") | -20% importancia |
| Memoria usada en runtime | +3% (emit_infrasound) |
| Memoria no usada en sesión completa | -2% (penalize_unused) |

**Precisión de detección:** 22/22 casos correctos en ES+EN (benchmark interno).

---

## Roles Dinámicos — DynamicRoleAdapter

```python
TASK_TYPES = {
    "explorador":   {"temp": 0.9,  "conf_weight": 0.8},  # creatividad, ideas
    "refinador":    {"temp": 0.6,  "conf_weight": 1.1},  # precisión, pulir
    "integrador":   {"temp": 0.7,  "conf_weight": 1.0},  # síntesis, combinar
    "analítico":    {"temp": 0.5,  "conf_weight": 1.2},  # lógica, estructura
    "contextual":   {"temp": 0.65, "conf_weight": 0.9},  # historial, coherencia
    "generativo":   {"temp": 0.8,  "conf_weight": 1.0},  # síntesis final
}
```

Cada agente puede tener su rol asignado dinámicamente según el tipo de query detectado.

---

## Estado Futuro — Cambios en el Orquestador

### Con Hormigas Dinámicas
```
El orquestador gestionará ciclo de vida de agentes:
- ant_pool: lista viva de agentes activos
- spawn_ant(parent_dna=None): crea nueva hormiga
- kill_ant(ant_id, transfer_legacy=True): mata hormiga, transfiere ADN a Matriarca
- Matriarca.store_genetic_legacy(role, fitness_avg, top_patterns)
```

### Con Delfines Dinámicos
```
dolphin_pool: lista de delfines activos
- Uno por defecto (red pequeña)
- Escala con número de nodos conectados
- Cada delfín tiene su propia frecuencia de ecolocalización
- El acoustic_map final = promedio ponderado de todos los delfines
```

### Con DolphinAgent Phase A
```
# 5 pings en lugar de 3
echo_topic   = ping_topic(x)
echo_intent  = ping_intent(x)
echo_need    = ping_need(x)
echo_context = ping_context(x)    ← nuevo
echo_emotion = ping_emotion(x)    ← nuevo

# Triangulación por atención (no lineal)
acoustic_map = Attention(Q=echo_topic, K=all_echos, V=all_echos)
```
