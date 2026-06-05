# Pendientes LixySwarm — 2026-06-05

Este documento resume lo pendiente frente al paper `paper/LixySwarm_AntElephantDolphin.pdf` y al estado actual del repo al 2026-06-05.

## Prioridad 0 — Seguridad y Privacidad

- [x] Mover credenciales hardcodeadas de `swarm_publisher.py`; ahora publica por API HTTP con `LIXYSWARM_PUBLISH_TOKEN`.
- [ ] Rotar cualquier secreto que haya estado en código, logs, shell history o commits.
- [x] Implementar cifrado AES-256-GCM para Personal Matriarca en reposo.
- [x] Separar físicamente rutas de memoria personal/global por defecto en runtime.
- [x] Añadir tests que prueben que memoria personal nunca se exporta, ni como texto ni como embedding.
- [x] Documentar política de secretos: nunca escribir passwords/tokens dentro del repo.

## Prioridad 1 — Matriarca Global

- [x] Hacer que `LixySwarm` use `MatriarcaDual` como ruta runtime principal, no solo en `train_swarm.py`.
- [x] Conectar `lixy_chat.py`/`lixy_orchestrator.py` a la Matriarca dual sin mezclar memoria privada con global.
- [ ] Convertir las contribuciones globales en memorias sintéticas estrictas, no texto crudo de conversación.
- [ ] Firmar contribuciones globales por memoria/delta y guardar `source_node_id`, timestamp y firma verificable.
- [ ] Agregar anti-replay para `GOSSIP_DELTA`: nonce/sequence + ventana temporal.
- [ ] Implementar reputación de nodos para ponderar memoria global recibida.
- [x] Actualizar `LSP_SPEC.md` para declarar que la integración real usa `GOSSIP_DELTA` global-safe, no feromonas como memoria cruda.

## Prioridad 2 — LSP v2 y Red P2P

- [ ] Alinear `FeromonV2Payload` con el wire format prometido en el paper o actualizar el paper/spec.
- [ ] Añadir sequence number real al header/payload para dedupe y anti-replay.
- [ ] Usar timestamp de 64 bits real en LSP v2; hoy el timestamp va truncado.
- [ ] Decidir si `fitness` debe ser `float16` como paper o `float32` como implementación.
- [ ] Implementar DHT/Kademlia para descubrimiento internet-scale.
- [ ] Añadir DNS seeds reales o mecanismo de bootstrap configurable sin IP hardcodeada.
- [ ] Reintegrar/validar mDNS LAN en el flujo moderno de `SwarmNetwork`.
- [ ] Medir relay multipunto con 3+ máquinas reales y registrar resultados.

## Prioridad 3 — Autonomía del Organismo

- [ ] Convertir metabolic hunger en daemon/autoloop real, no solo flag opt-in.
- [ ] Aprender pesos de hambre desde la Matriarca usando resultados de ciclos previos.
- [ ] Guardar en Matriarca cada ciclo de alimentación: señales, steps, mejora, diversidad y saciedad.
- [ ] Implementar criterio de saciedad por utilidad marginal de loss por step.
- [ ] Ejecutar Dolphin sleep consolidation automáticamente tras inactividad real.
- [ ] Crear `GrowthGate`: métricas, thresholds y transición Infant → Child.
- [ ] Persistir etapa actual del organismo y razones de promoción/rollback.

## Prioridad 4 — Sects, Nodos y Especialización

- [ ] Conectar nodos remotos reales al `NodeManager`, no solo al peer table de red.
- [ ] Propagar hardware profile/contribution mode por LSP.
- [ ] Permitir que un nodo remoto anuncie si es `maximum`, `moderate` o `relay`.
- [ ] Usar disponibilidad real de nodos para nacimiento/muerte de sectas.
- [ ] Medir fitness por contribución remota efectiva, uptime y calidad de feromonas.
- [ ] Completar legado genético con top-K patrones de feromona reales por secta.
- [ ] Probar bifurcación de sectas en entrenamiento largo con métricas guardadas.

## Prioridad 5 — Dashboard y Observabilidad

- [ ] Mostrar Matriarca global: memoria global, deltas enviados/recibidos, imports y reputación.
- [ ] Mostrar separación `personal/global` sin revelar contenido privado.
- [ ] Mostrar estado de LSP v2: peers activos, known peers, relay mode, gossip deltas y feromonas.
- [ ] Mostrar metabolic hunger: score, señales, nivel y decisión.
- [ ] Mostrar Dolphin sleep: consolidaciones, idle time y norm por delfín.
- [ ] Añadir alerta visual si el publisher no puede subir estado al VPS.
- [x] Eliminar dependencia de password directo en `swarm_publisher.py`.

## Prioridad 6 — Benchmarks y Paper

- [ ] Recalcular métricas después de Matriarca global: val_loss, perplexity, repetición y bilingüe.
- [ ] Separar métricas verificadas vs aspiracionales en README/paper.
- [ ] Actualizar conteo real de tests después de cambios recientes.
- [ ] Añadir benchmark multi-nodo de colaboración por internet.
- [ ] Añadir benchmark de privacidad: prueba negativa de fuga personal.
- [ ] Documentar limitaciones actuales de DHT, reputación, cifrado y GrowthGate.

## Prioridad 7 — Visión Largo Plazo

- [ ] Diseñar sandbox aislado para self-modification.
- [ ] Implementar evaluación de cambios generados por el organismo con rollback.
- [ ] Añadir multimodal ants para imagen/audio como nodos especializados.
- [ ] Diseñar gateways entre colonias con Matriarcas soberanas.
- [ ] Definir consenso distribuido para decisiones del enjambre sin nodo privilegiado.

## Ya Implementado o Parcialmente Implementado

- [x] FeromonGate en `AgentBase`.
- [x] Dolphin Phase A con 5 pings y triangulación por atención.
- [x] Dolphin Phase B con consolidación PCA/SVD.
- [x] NodeManager con contribution modes `maximum`, `moderate`, `relay`.
- [x] SectManager con ciclo de vida, muerte, bifurcación y legado.
- [x] Matriarca base con top-k retrieval, refuerzo de importancia y compresión generacional.
- [x] MatriarcaDual runtime por defecto con rutas separadas personal/global.
- [x] Cifrado AES-256-GCM opcional para Personal Matriarca vía `LIXYSWARM_MATRIARCA_KEY`.
- [x] Matriarca global inicial vía `GOSSIP_DELTA`.
- [x] Tests de privacidad/dedupe para delta global inicial.
- [x] Relay VPS para feromonas LSP v2.

## Próximo Corte Recomendado

1. Rotar secretos expuestos fuera del repo y purgar historial si aplica.
2. Hacer `MatriarcaDual` runtime principal.
3. Añadir cifrado personal AES-256-GCM.
4. Fortalecer `GOSSIP_DELTA` con firma/nonce/reputación.
5. Actualizar `LSP_SPEC.md` y README con el estado real.
