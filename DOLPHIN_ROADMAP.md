# DolphinAgent 🐬 — Hoja de Ruta
*Última actualización: 2026-05-30 | Phase A es la siguiente implementación*

---

## Estado Actual

```
✅ DolphinAgent implementado en src/agents/dolphin_agent.py
✅ Integrado en LixySwarm v3 como primer paso del forward
✅ 3 pings de ecolocalización: topic, intent, need
✅ sleep_state persiste entre sesiones (lixy_session.json)
✅ IdentityVec único por instancia (silbido)
⏳ Phase A — 5 pings + triangulación por atención
⏳ Phase B — sueño real con consolidación en background
⏳ Delfines dinámicos — N delfines según tamaño de red
```

---

## Lo que hace hoy

```python
class DolphinAgent(nn.Module):
    # Ecolocalización: 3 proyecciones lineales
    ping_topic   = Linear(d_model, d_feromon)   # ¿De qué habla esto?
    ping_intent  = Linear(d_model, d_feromon)   # ¿Qué quiere el usuario?
    ping_need    = Linear(d_model, d_feromon)   # ¿Qué necesita realmente?
    
    # Triangulación simple: concatenar + MLP
    acoustic_mlp = MLP(3 * d_feromon, d_feromon)
    
    # Sleep state: persiste entre conversaciones
    sleep_state: Tensor  # [d_feromon] guardado en lixy_session.json
    
    def forward(self, x):
        echo_topic  = ping_topic(x.mean(1))
        echo_intent = ping_intent(x.mean(1))
        echo_need   = ping_need(x.mean(1))
        acoustic_map = concat([echo_topic, echo_intent, echo_need])
        feromon = acoustic_mlp(acoustic_map)
        # blend con sleep_state
        return 0.7 * feromon + 0.3 * sleep_state
```

**Por qué importa:** El Delfín construye una representación del *espacio de respuesta posible* antes de que cualquier agente genere un token. Como un delfín que echolocaliza el banco de peces antes de atacar — no nada ciegamente hacia adelante.

---

## Phase A — Ecolocalización Ampliada

### El problema con 3 pings
Los 3 pings actuales se combinan linealmente (concatenar + MLP). Perdemos las relaciones entre dimensiones del problema.

### La solución: 5 pings + atención

```python
# 5 pings en lugar de 3
echo_topic   = ping_topic(x)     # ¿De qué habla esto?
echo_intent  = ping_intent(x)    # ¿Qué quiere el usuario?
echo_need    = ping_need(x)      # ¿Qué necesita realmente?
echo_context = ping_context(x)   # ← nuevo: ¿Qué vino antes?
echo_emotion = ping_emotion(x)   # ← nuevo: ¿Qué tono requiere?

# Triangulación NO-lineal: atención entre ecos
# El topic "interroga" a los otros 4 ecos
acoustic_map = Attention(
    Q = echo_topic,        # ancla temática
    K = stack(all_echos),  # dimensiones del problema
    V = stack(all_echos)   # contenido de cada dimensión
)
# → representación del espacio de respuesta, no solo del input
```

### Diferencia conceptual
```
Antes: 3 pings independientes → concatenar → MLP (lineal)
Ahora: 5 pings relacionados → atención cruzada → mapa acústico (no-lineal)
```

El mapa acústico ya no es una suma de perspectivas — es una **imagen del espacio de respuesta** donde las dimensiones se modulan entre sí.

### Plan de validación
1. Implementar en `dolphin_agent.py`
2. Actualizar `orchestrator.py` para usar nuevo acoustic_map
3. Run 7k steps con `swarm_best.pt` como base
4. Comparar: calidad de feromona vs Phase 0 (actual)
5. Test: ¿el enjambre clasifica mejor la intención del usuario?

---

## Phase B — Sueño Real (Consolidación en Background)

### El problema actual
`sleep_state` es un tensor fijo que persiste entre turnos pero no se actualiza entre sesiones — no consolida ni aprende de una conversación a la siguiente.

### La visión

```python
class UnihemisphericSleep:
    # Hemisferio A: activo durante conversación (genera respuestas)
    # Hemisferio B: consolida en background entre conversaciones
    
    def consolidate(self, conversation_history: List[Turn]):
        """
        Cron job: cada 30min sin actividad.
        
        1. Extraer embeddings de los últimos N turnos
        2. PCA para encontrar ejes principales de la sesión
        3. Actualizar sleep_state con representación comprimida
        
        No es training — no cambian los pesos.
        Es reorganización de representaciones existentes.
        """
        embeddings = [self._embed(turn) for turn in conversation_history[-50:]]
        compressed = PCA(n_components=self.d_feromon).fit_transform(embeddings)
        self.sleep_state = torch.tensor(compressed.mean(0))
    
    def wake(self) -> Tensor:
        """
        Al iniciar conversación: estado pre-calentado.
        No es tensor de ceros — ya sabe quién es Emmanuel,
        qué temas se discutieron, qué tono prefiere.
        """
        return self.sleep_state
```

**El efecto observable:** La primera respuesta de una sesión nueva ya tiene contexto acumulado sin fine-tuning — solo por haber procesado el historial en background.

### Diferencia con la Matriarca
| Matriarca | Sueño del Delfín |
|---|---|
| Memorias semánticas explícitas | Estado implícito no legible |
| "Recuerdo que Emmanuel habla de drones" | Representación difusa de quién es Emmanuel |
| Recuperable por similitud | No recuperable — es orientación pura |
| Se puede inspeccionar | Caja negra funcional |

Ambas son necesarias. Se complementan.

### Cron job propuesto
```
Trigger: 30 minutos sin actividad en la sesión
Input: lixy_session.json (historial completo)
Output: actualizar sleep_state en lixy_session.json
Timeout: 60 segundos max
```

---

## Delfines Dinámicos

### El problema con 1 delfín
Un solo delfín construye una perspectiva del problema. Múltiples delfines con frecuencias distintas construyen un mapa más rico — como ecos superpuestos.

### La visión

```python
class DolphinPool:
    dolphins: List[DolphinAgent]  # escala con la red
    
    @property
    def size(self) -> int:
        return len(self.dolphins)
    
    def scale_to_network(self, n_nodes: int):
        """
        n_nodes=1   → 1 delfín (default)
        n_nodes=2-4 → 2 delfines
        n_nodes=5+  → 3 delfines
        n_nodes=10+ → 4 delfines (máximo por ahora)
        """
        target = min(1 + n_nodes // 3, 4)
        while self.size < target:
            self.spawn_dolphin()
        while self.size > target:
            self.retire_dolphin()
    
    def forward(self, x) -> Tensor:
        """
        Cada delfín tiene su frecuencia de ecolocalización propia.
        El acoustic_map final = promedio ponderado por confianza.
        """
        maps = [d.forward(x) for d in self.dolphins]
        weights = [d.confidence for d in self.dolphins]
        return weighted_average(maps, weights)
```

**La frecuencia propia:** Cada delfín tiene un bias aprendido en sus pings que lo hace "resonar" con dimensiones diferentes del problema. Uno puede especializarse en contexto emocional, otro en razonamiento técnico, etc.

---

## Por Qué LixySwarm Es Diferente

| Arquitectura | Representación inicial | Contexto entre sesiones |
|---|---|---|
| GPT-4, Claude, Llama | Cero — construye mientras genera | Ninguno (stateless) |
| Con RAG | Recupera texto similar | Solo lo que se indexó |
| RWKV, Mamba | Estado recurrente (se resetea) | No persiste |
| **LixySwarm** | **Mapa acústico del problema** | **sleep_state + Matriarca acumulada** |

El Delfín es la pieza que hace esto posible. Ningún LLM actual tiene algo análogo.

---

## Roadmap Completo

### Phase A (siguiente ← implementar ahora)
- [ ] 5 pings: añadir `ping_context` y `ping_emotion`
- [ ] Triangulación con `Attention(Q=topic, K/V=all_pings)`
- [ ] Actualizar `orchestrator.py`
- [ ] Validar con run 7k steps

### Phase B — Sueño real
- [ ] `consolidate()`: cron job 30min + PCA del historial
- [ ] Persistencia del sleep_state consolidado
- [ ] Test: primera respuesta post-reinicio vs sin sueño

### Delfines Dinámicos
- [ ] `DolphinPool` con `scale_to_network(n_nodes)`
- [ ] Frecuencias propias por instancia (bias aprendido)
- [ ] Integrar con orchestrator multi-delfín

### Phase C — Integración con LSP
- [ ] sleep_state como parte de la identidad del nodo
- [ ] Compartir "sueño destilado" entre nodos (no raw)
- [ ] Nodo veterano tiene sueño más rico → mayor influencia en el enjambre

### Phase D — Multimodalidad Emergente (largo plazo)
```
Red actual → Red futura:
🐜 AntAgent × N     🐜 TextAnt × N
                    🐜 ImageAnt × N   ← GPU con visión
                    🐜 AudioAnt × N   ← hardware de audio
🐘 Matriarca        🐘 Matriarca (coordina multi-tipo via feromonas)
🐬 1 Delfín         🐬 N Delfines (espacio semántico unificado)
```
La multimodalidad **emerge del ecosistema** — no se diseña centralmente. Un nodo con cámara contribuye ImageAnts. El Delfín construye el espacio unificado donde texto, imagen y audio coexisten.

---

*Visión: LixySwarm Team | Implementación: Cody | 2026*
