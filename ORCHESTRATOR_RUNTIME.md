# ORCHESTRATOR_RUNTIME.md
# Orquestador ↔ Matriarca: Plan de Integración Runtime

## Estado Final (post Run 10) — 2026-05-29

### Ciclo completo de memoria ✅
| Componente | Estado | Descripción |
|-----------|--------|-------------|
| Training feedback | ✅ | `store_memory=True` durante training, cada 50 steps |
| Infrasónidos en forward | ✅ | Matriarca orienta feromonas en cada forward pass |
| Voto de confianza Matriarca | ✅ | 20% del peso de agregación viene de la Matriarca |
| Sueño del Delfín → Matriarca | ✅ | `sleep_for_matriarca` almacenado si `store_memory=True` |
| Compresión generacional | ✅ | Auto-compresión al 90% de capacidad |
| Feromon refresh con Matriarca | ✅ | Refresh cada 32 tokens incluye Matriarca + blend 70/30 |
| Feedback post-generación | ✅ | Feedback al FINAL con Q+A completo |
| Estado cross-turn | ✅ | `RuntimeSession` mantiene feromon entre turnos |
| Contexto acumulado | ✅ | `_build_context()` incluye últimos 5 turnos |
| Resumen de sesión | ✅ | `end_session()` almacena resumen importancia=f(turnos,longitud) |
| `penalize_unused()` | ✅ | Memorias no usadas en sesión: -2% importancia |
| Importancia dinámica | ✅ | 5 métricas: len, TTR, bigram-rep, coherencia semántica, continuidad |
| Retroactive feedback | ✅ | Turno anterior sube +8% si usuario continúa el tema |
| Sesiones persistentes | ✅ | `session_file` serializa history a disco entre reinicios |
| `emit_infrasound` refuerzo | ✅ | `update_importance=True` por defecto — memorias usadas suben +3% |
| `train_from_swarm_log` v2 | ✅ | Embeddings reales + chunks + hitos val_loss + eventos spec |
| `matriarca_eval()` | ✅ | Diagnóstico: importancia, diversidad, acceso, tipos |
| Roles dinámicos | ✅ | `DynamicRoleAdapter` — 6 tipos de tarea, temperatura adaptada |

---

## Arquitectura del Flujo Runtime

```
Usuario → lixy_orchestrator.py (load: matriarca_eval si --eval-matriarca)
              │
              ▼
         LixyOrchestrator.chat(message)
              │
              ▼
         RuntimeSession.turn(user_input)
              │
              ├── 1. classify_query() → TaskProfile (técnica/exploratoria/etc)
              │   → temperatura dinámica + role_weights para agentes
              │
              ├── 2. _build_context() → últimos 5 turnos
              │
              ├── 3. Warm-up: swarm.forward(context_with_history)
              │   ├── Delfín 🐬 → ecolocalización
              │   ├── Matriarca 🐘 → emit_infrasound (update_importance=True)
              │   └── InfrasoundMixer → feromon_guiada
              │   (blend 70% nuevo + 30% feromon anterior si hay)
              │
              ├── 4. Generación token a token (feromon fijo)
              │   ├── sample_token(): rep_penalty + top-k + top-p
              │   └── Refresh cada 32 tokens con Matriarca activa
              │
              ├── 5. Post-turno:
              │   ├── _retroactive_feedback() → +8% al turno anterior si mismo tema
              │   ├── _compute_response_importance() → 5 métricas
              │   ├── store_interaction() → memoria con importancia calculada
              │   └── _save_history() → disco
              │
              ▼
         LixyOrchestrator.close() / end_session()
              ├── Resumen de sesión → Matriarca (imp=0.6-0.95)
              ├── penalize_unused() → -2% a memorias no accedidas
              └── matriarca.save()

Training (train_swarm.py):
  ├── store_memory cada 50 steps durante training
  └── Al terminar: train_from_swarm_log() automático
      ├── Chunks de 50 pasos → memorias semánticas reales
      ├── Hito de mejor val_loss (imp=0.98)
      ├── Eventos de especialización de agentes
      └── evolutionary_loop() → síntesis + checkpoint versionado
```

---

## API Pública

```python
# Orquestador completo
from lixy_orchestrator import LixyOrchestrator, OrchestratorConfig

cfg = OrchestratorConfig(
    eval_matriarca=True,   # diagnóstico al cargar
)
lixy = LixyOrchestrator(cfg)
response = lixy.chat("¿Cómo funciona un transformer?")
lixy.close()   # penaliza, guarda, limpia

# RuntimeSession directa
from src.swarm.runtime_session import RuntimeSession
session = RuntimeSession(swarm, enc, session_file="checkpoints/runtime_session.json")
r1 = session.turn("pregunta")    # detecta tipo, adapta temperatura
r2 = session.turn("seguimiento") # retroactive feedback al turno anterior
session.end_session()

# Eval banco de memorias
python3 train_matriarca.py --eval

# Loop evolutivo desde log
python3 train_matriarca.py --from-swarm-log /tmp/swarm_run10.log

# Training con control fino
python3 train_swarm.py --steps 5000 --lr 2e-4 --warmup 200 \
    --grad-accum 8 --eval-steps 50 --min-lr 5e-6 --triple
```

---

## Métricas del banco (post Run 10)

```
🐘 Matriarca Eval (2817 memorias)
  Importancia: media=0.373  mediana=0.351  activas=70%  degradadas=8%
  Diversidad:  cosine_sim=0.448  diversity_score=0.552
  Acceso:      408 recientes (24h) | 86% nunca accedidas
  Tipos:       runtime=22, training=1607, spec=4, hito=1, sintética=349, otro=834
```

**Interpretación:**
- 70% activas (imp > 0.3) — banco en buen estado
- 8% degradadas (imp < 0.1) — serán eliminadas en próxima compresión
- diversity_score=0.552 — banco diverso (0=idénticas, 1=ortogonales)
- 86% nunca accedidas — candidatas a penalización en próximas sesiones

---

## Pendientes (prioridad baja)

- [ ] **Feedback explícito del usuario** — si el usuario dice "bien" o "no entendí", ajustar importancia de última memoria
- [ ] **Multi-usuario** — etiquetar memorias con `session_id` para distinguir contextos
- [ ] **Protocolo P2P** — SwarmNetwork distribuida (diseño en DISTRIBUTED_PROTOCOL.md)
- [ ] **Benchmarks formales** — perplexity, coherencia, comparativa vs modelos base
