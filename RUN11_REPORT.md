# Run 11 — Reporte Final
*Fecha: 2026-05-30 | Parado limpio en step 53,920*

---

## Resumen Ejecutivo

| Métrica | Valor |
|---|---|
| **Steps entrenados** | ~53,920 (target original: 52,100 — 1,820 extra por bug de resume) |
| **val_loss final** | 3.5687 (checkpoint best, step 53,600) |
| **Perplexity FineWeb** | **35.2** |
| **Perplexity personal** | **11.1** |
| **GPT-2 baseline ppl** | 14,870 (comparativa) |
| **Rep@5** | 0.2% ✅ |
| **Rep@10** | 0.0% ✅ |
| **TTR promedio** | 0.790 |
| **Muestras sin bucles** | 100% |
| **Idioma correcto** | 100% (ES + EN) |
| **Memorias Matriarca** | 3,662 al cierre |
| **tok/s** | ~12,800 |

---

## Benchmark Completo

### Perplexity por checkpoint (30 batches cada uno)
| Checkpoint | Step | PPL |
|---|---|---|
| `swarm_best.pt` | 53,600 | **35.22** |
| `swarm_latest.pt` | 53,900 | 35.41 |
| `swarm_final.pt` | 12,500 | 97.96 |

### Generación — 10 muestras
```
✅ [en_tech]  rep@5=0% ttr=0.73 | ' able to predict the behaviour of neurons in humans and'
✅ [en_tech]  rep@5=0% ttr=0.78 | ' the UK as part of their project. The design process in'
✅ [en_tech]  rep@5=0% ttr=0.83 | ' software development, which provides a programming lan'
✅ [en_fact]  rep@5=0% ttr=0.76 | ' the capital for all French colonies. This city was fou'
✅ [en_fact]  rep@5=0% ttr=0.74 | ' 1 second: an increase in the velocity of light from on'
✅ [en_creat] rep@5=0% ttr=0.88 | ' an automatic (in short order) transmission, there is n'
✅ [en_creat] rep@5=0% ttr=0.81 | ' be very different for all types of systems. The future'
✅ [es_gen]   rep@5=0% ttr=0.80 | ' la inteligencia artificial (AIM) que afecte al desarro'
✅ [es_gen]   rep@5=0% ttr=0.78 | ' ser de tipo en el mapa. En otros países, la formación '
✅ [es_gen]   rep@5=2% ttr=0.79 | ' la nueva imagen: "Ya que está haciendo el mundial, pue'
```

---

## Estado del Enjambre

### Arquitectura
| Parámetro | Valor |
|---|---|
| n_agents | 3 |
| n_layer | 12 |
| n_head | 12 |
| n_embd | 768 |
| block_size | 512 |
| vocab_size | 50,304 |
| feromon_dim | 256 |
| identity_dim | 64 |
| Params en checkpoint | 548.9M |

### Estado Matriarca
| Métrica | Valor |
|---|---|
| Total memorias | 3,662 |
| Importancia media | 0.313 |
| Importancia mediana | 0.300 |
| Activas (>0.2) | 50% |
| Degradadas (<0.1) | 22% |
| Diversidad | 0.531 |
| Accedidas en 24h | 125 |
| Nunca accedidas | 97% |

### Tipos de memoria
| Tipo | Cantidad |
|---|---|
| `otro` | 2,683 |
| `sintética` | 870 |
| `training` | 102 |
| `runtime` | 5 |
| `hito` | 1 |
| `spec` | 1 |

---

## Nota sobre el bug de steps

Run 11 fue planeado para 40,000 steps desde el step 12,100 → target 52,100.
El proceso se reinició en step ~23,400 y por el bug (relativo vs absoluto), 
corrió 40,000 steps más desde ahí → llegó a ~53,920 en vez de 52,100.

**Bug fixeado** en `train_swarm.py` antes de Run 12. Ver commit `3553adb`.

---

## Observaciones

1. **Generación limpia:** 100% sin bucles, 100% idioma correcto — la repetition penalty funciona bien
2. **Bilingüe natural:** muestras en ES y EN sin configuración especial
3. **Matriarca saludable:** 3,662 memorias, diversidad 0.531 — listo para evolución futura
4. **Rendimientos decrecientes:** lr=1.5e-05 muy bajo, cada 1k steps solo baja ~0.01 en val_loss
5. **Próximo paso:** Features nuevas antes de correr más steps — DolphinAgent Phase A primero

---

## Próximo: DolphinAgent Phase A

Ver `DOLPHIN_ROADMAP.md` para spec completo.

```python
# 5 pings en lugar de 3
echo_topic   = ping_topic(x)
echo_intent  = ping_intent(x)
echo_need    = ping_need(x)
echo_context = ping_context(x)   # ← nuevo
echo_emotion = ping_emotion(x)   # ← nuevo

# Triangulación por atención (no lineal)
acoustic_map = Attention(Q=echo_topic, K/V=all_echos)
```

Validar con run corto: 7,000 steps desde `swarm_best.pt`.
