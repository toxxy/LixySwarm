# DolphinAgent 🐬 — Hoja de Ruta de Implementación Profunda

> Estado: Implementación básica existe (`src/agents/dolphin_agent.py`).
> Este doc captura la visión completa de Emmanuel para la siguiente fase.

---

## Estado actual

El `DolphinAgent` actual implementa:
- ✅ Ecolocalización con 3 pings paralelos (topic, intent, need)
- ✅ Sueño unihemisférico: `sleep_state` persiste entre turnos en memoria
- ✅ Identificación de agente: vector de identidad único por instancia
- ✅ Integrado en LixySwarm v3 como primer paso del forward

Lo que falta es llevar estas ideas a su potencial completo.

---

## Visión 1: Ecolocalización Real — Representación No-Lineal

### El problema con los transformers actuales
Los transformers procesan tokens izquierda→derecha (o bidireccional en BERT, pero aun así posición por posición). El espacio del problema se construye incrementalmente.

### La idea del Delfín
Un delfín no "lee" su entorno token a token — lanza un pulso y recibe el mapa completo de una vez. En LixySwarm esto significa:

**Fase 1 — Ping múltiple**
```
input → ping_topic(x)    → echo_topic    [¿de qué habla esto?]
input → ping_intent(x)   → echo_intent   [¿qué quiere el usuario?]
input → ping_need(x)     → echo_need     [¿qué necesita realmente?]
input → ping_context(x)  → echo_context  [¿qué vino antes?]  ← nuevo
input → ping_emotion(x)  → echo_emotion  [¿qué tono requiere?]  ← nuevo
```

**Fase 2 — Triangulación**
Los 5 ecos se combinan para construir una "imagen acústica" del espacio del problema:
```
acoustic_map = Attention(Q=echo_topic, K=echo_all, V=echo_all)
```
Esto no es una representación lineal del texto — es una representación del *espacio de respuesta posible* antes de generar nada.

**Fase 3 — Feromona de orientación**
```
feromon_out = MLP(acoustic_map)   # orienta al enjambre antes de que procese
```

### Por qué es diferente
Todos los LLMs actuales construyen su representación mientras generan. El Delfín construye **primero** el mapa, **después** delega a las hormigas. Como un delfín que echolocaliza un banco de peces antes de atacar — no nada ciegamente hacia adelante.

---

## Visión 2: Sueño Unihemisférico Real — Contexto Persistente

### Estado actual
El `sleep_state` es un tensor que persiste entre llamadas en memoria. Se guarda en `lixy_session.json` y se restaura al reiniciar.

### La visión completa

**Problema real**: Entre conversaciones (días, reinicios), el contexto debería acumularse gradualmente, no resetearse. Como el hemisferio despierto de un delfín que sigue procesando mientras el otro "duerme".

**Implementación propuesta:**

```python
class UnihemisphericSleep:
    # Hemisferio A: activo durante conversación
    # Hemisferio B: consolidando en background entre conversaciones
    
    def consolidate(self, conversation_history):
        """Mientras no hay conversación activa, procesar en background."""
        # Cron job: cada 30min sin conversación
        # Extraer patrones de las últimas N conversaciones
        # Actualizar el 'sueño' con memorias consolidadas
        # No es training — es reorganización de representaciones existentes
    
    def wake(self) -> torch.Tensor:
        """Al iniciar conversación: estado pre-calentado con contexto acumulado."""
        return self.consolidated_state   # no zeros — ya sabe quién es Emmanuel
```

**El efecto observable**: La primera respuesta de una sesión nueva ya tiene contexto. El modelo "recuerda" sin fine-tuning — solo por haber procesado el historial en background.

**Diferencia con la Matriarca**: La Matriarca acumula memorias semánticas explícitas. El sueño del Delfín acumula estado implícito — representaciones que no puedes leer directamente pero que orientan la generación.

---

## Por qué esto hace a LixySwarm genuinamente diferente

| Arquitectura | Contexto entre sesiones | Representación inicial |
|--------------|------------------------|----------------------|
| GPT-4, Claude | Ninguno (stateless) | Cero — empieza sin saber nada |
| Con RAG | Textual explícito | Recupera docs similares |
| **LixySwarm** | **Implícito acumulado** | **Mapa acústico del problema** |

El Delfín es la pieza que hace esto posible. Ningún transformer actual tiene algo análogo.

---

## Roadmap de implementación

### Fase A (después de Run 11) — Ecolocalización ampliada
- [ ] Añadir pings de `context` y `emotion` al DolphinAgent
- [ ] Implementar triangulación con `Attention(Q=topic, K/V=all_pings)`
- [ ] Comparar calidad de feromona vs implementación actual
- [ ] Test: ¿el mapa acústico captura mejor la intención del usuario?

### Fase B — Sueño real (consolidación en background)
- [ ] Cron job de consolidación: cada 30min sin actividad
- [ ] Algoritmo de consolidación: PCA del historial → actualizar sleep_state
- [ ] Persistencia cifrada: sleep_state encriptado con privkey del usuario (AD-002)
- [ ] Test: ¿la primera respuesta tras reinicio es mejor con sueño vs sin sueño?

### Fase C — Integración completa con LSP
- [ ] El sueño del Delfín es parte de la identidad del nodo
- [ ] En red distribuida: los nodos comparten "sueño destilado" (no raw) para alinear perspectivas
- [ ] Un nodo que ha procesado millones de conversaciones tiene un sueño más rico

---

*Última actualización: 2026-05-29 | Visión: Emmanuel Cardenaz | Doc: Cody*

---

## Fase D — Multimodalidad Emergente (visión a largo plazo)

### La idea
Expandir LixySwarm más allá de texto mediante hormigas especializadas por modalidad. La clave: **emerge del uso, no se diseña explícitamente**.

```
Red actual:         Red futura:
🐜 AntAgent × 3    🐜 TextAnt × N
🐘 Matriarca       🐜 ImageAnt × N   ← usuarios con GPUs visuales
🐬 Delfín          🐜 AudioAnt × N   ← usuarios con hardware de audio
                   🐜 VideoAnt × N   ← usuarios con mucho compute
                   🐘 Matriarca (coordina todos los tipos via feromonas)
                   🐬 Delfín (espacio semántico unificado texto+imagen+audio)
```

### Por qué emerge naturalmente
- Un nodo con webcam contribuye ImageAnts
- Un nodo con micrófono contribuye AudioAnts
- Un nodo con GPU potente contribuye VideoAnts
- Nadie asigna roles — cada nodo contribuye lo que puede
- La Matriarca aprende a coordinar feromonas multi-tipo automáticamente
- El Delfín construye representaciones unificadas en un espacio compartido

### Arquitectura de feromonas multi-modal
```python
# Cada tipo de hormiga emite feromonas en el mismo espacio de dimensión D
# Aunque el input sea diferente, el espacio de feromona es compartido
feromon_text  = TextAnt(text_tokens)    # [D]
feromon_image = ImageAnt(image_pixels)  # [D]  ← mismo espacio
feromon_audio = AudioAnt(audio_frames)  # [D]  ← mismo espacio

# La Matriarca puede orientar sin saber qué tipo de input procesó cada hormiga
infrasound = Matriarca(feromon_pool([feromon_text, feromon_image, feromon_audio]))
```

### El Delfín como capa de fusion
El mapa acústico del Delfín se vuelve el espacio semántico unificado donde texto, imagen y audio coexisten. Similar a CLIP (OpenAI) pero emergente y distribuido — ninguna empresa central lo diseñó.

### Roadmap de implementación
- [ ] **Fase D.1** — ImageAnt: encoder visual (ViT o similar) → feromona [256d]
- [ ] **Fase D.2** — AudioAnt: encoder de audio (Whisper features) → feromona [256d]
- [ ] **Fase D.3** — Matriarca aprende a distinguir y coordinar tipos via metadata de feromona
- [ ] **Fase D.4** — VideoAnt: ImageAnt + temporal encoding entre frames
- [ ] **Fase D.5** — Delfín multimodal: pings visuales + auditivos + textuales en un mapa acústico unificado

### Por qué es diferente a GPT-4V / Gemini
- Centralizado vs **descentralizado** — ninguna empresa controla los nodos
- Diseñado vs **emergente** — la multimodalidad emerge del ecosistema de contribuidores
- Privado vs **soberano** — tu ImageAnt en tu máquina, tu privacidad
- Fijo vs **evolutivo** — nuevas modalidades (sensores, etc.) se integran sin rediseñar la arquitectura

---
