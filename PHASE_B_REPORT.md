# DolphinAgent Phase B — Sueño unihemisférico real

## Estado

Implementado el 2026-06-01.

Phase B convierte el sueño del Delfín de un buffer pasivo a un proceso de consolidación:

- Trigger por defecto: **30 minutos sin actividad** (`sleep_consolidation_idle_s = 1800`).
- Requisito mínimo: 3 contextos recientes en `HalfSleepState.context_buffer`.
- Método: PCA/SVD sobre historial de embeddings de feromona proyectados a `sleep_dim`.
- Resultado: `awake_state` se mezcla con un vector consolidado (`mean + principal_component`).
- Export: `DolphinSwarmBridge.maybe_consolidate_sleep()` puede devolver `sleep_for_matriarca` proyectado a 512d.

## Archivos tocados

- `src/agents/dolphin_agent.py`
  - Config de consolidación.
  - `HalfSleepState.should_consolidate()`.
  - `HalfSleepState.consolidate_pca()`.
  - Persistencia de timestamps, contador de consolidaciones y projector.
  - `DolphinAgent.maybe_consolidate_sleep()`.
  - `DolphinSwarmBridge.maybe_consolidate_sleep()`.

- `src/swarm/dolphin_pool.py`
  - Consolidación para todos los delfines del pool.
  - Status incluye `sleep_consolidations`.

- `src/swarm/orchestrator.py`
  - `tick_lifecycle()` dispara consolidación si hubo inactividad suficiente.
  - Emite evento `dolphin_sleep_consolidated`.

- `test_dolphin_router.py`
  - +2 tests Phase B.
  - Suite del delfín sube de 13/13 a 15/15.

## Seguridad / guardrails

- No hay sleeps ni polling bloqueante.
- El forward normal no ejecuta PCA salvo llamada explícita/oportunista.
- No consolida repetidamente la misma ventana: requiere actividad nueva desde la última consolidación.
- SVD tiene fallback a media si falla.
- No promueve checkpoints ni modifica training automático.

## Tests

Suite completa local:

```text
  test_ant_lifecycle.py: Tests: 11/11 passed ✅
  test_node_sect.py: Tests: 14/14 passed ✅
  test_dolphin_router.py: Tests: 15/15 passed ✅
  test_matriarca_legacy.py: Tests 1d: 15/15 passed ✅
  test_lsp_v2.py: tests_passed=12/12
  test_integration.py: RESULTADO: 15/15 tests pasaron ✅
  test_network.py: RESULTADO: 34/34 tests pasaron ✅
  test_auto_train.py: Tests: 32/32 passed ✅
  test_edge_cases.py: Tests: 33/33 passed ✅
```

Total: **181/181 tests verdes**.

## Próximo paso

Preparar Run 12 corto (5k-10k steps) para validar Phase B + arquitectura nueva.

Reglas de Run 12:

- No run grande.
- Evaluación antes/después.
- No sobrescribir `swarm_best.pt` sin mejora validada.
- Guardar candidate/latest separado.
- Reportar métricas: val_loss, rep ratio, diversidad, sleep_consolidations, estabilidad.
