# Lixy LLM Project — Plan Día 1 (2026-05-28)

## Estado al despertar

- Hardware: RTX 5090 (32GB VRAM), 32GB RAM, 1.9TB libre, PyTorch 2.8+CUDA 12.8 ✓
- Arquitectura definida: Hormiga + Elefante + Delfín (ver ARQUITECTURA.md)
- Corpus propio disponible: ~28K tokens (conversaciones + memoria) — pequeño, necesitamos más

## Tareas del día

### Mañana (6-12 AM)
- [x] Leer MEMORY.md y recordar arquitectura completa
- [x] Investigar el estado del arte (swarm LLM, nanoGPT, MoE 2025)
- [x] Crear ARQUITECTURA.md con diseño detallado
- [x] Crear estructura del repositorio `lixy-llm/`
- [x] Setup del entorno Python (requirements.txt + instalaciones)
- [x] Script de recopilación de datos (`src/data/collect_corpus.py`) — 32 docs, 42K tokens
- [x] AgentBase (126M params) funcionando con FeromonGate + IdentityVec ✓
- [x] LixySwarm (3 agentes, 2.9B params totales) funcionando ✓
- [x] Script de tokenización (`prepare_pretrain.py`) — corpus personal tokenizado ✓
- [x] Loop de entrenamiento (`train.py`) — ~98K tok/s en RTX 5090 con bf16 ✓
- [x] Primer checkpoint guardado: `checkpoints/final.pt` ✓

### Tarde (12-6 PM)
- [ ] Descargar FineWeb-Edu (1B tokens, ~8GB) con `prepare_pretrain.py --download`
- [ ] Pre-training AgentBase sobre FineWeb (~3.5h a 98K tok/s)
- [ ] Mientras entrena: implementar mecanismo de infrasónidos de la Matriarca (Elefante)
- [ ] Implementar script de generación/inferencia para ver qué está aprendiendo

### Mañana siguiente
- [ ] Fine-tuning con corpus personal (Emmanuel le da "alma" al modelo)
- [ ] Primer entrenamiento completo del AgentSwarm (3 agentes)
- [ ] Primer experimento emergente: ¿es mejor el enjambre que un agente solo?

## Decisiones técnicas tomadas hoy

1. **Base:** fork de nanoGPT (Karpathy) — mínimo, entendible, modificable
2. **Tamaño inicial:** 125M params por agente (probado en RTX 4090/5090, factible)
3. **Formato:** bf16 + `torch.compile()` (PyTorch 2.8 nativo)
4. **Datos:** FineWeb-10B (base) + corpus propio (fine-tuning final)
5. **Enjambre mínimo viable:** 3 agentes para fase 1 (léxico + semántico + generación)
6. **Paper objetivo:** "Lixy-0.1: A Bio-Inspired Emergent Intelligence Architecture"

## Insights del estado del arte

- "Model Swarms" (ICML 2025): ya existe collaborative search via swarm — pero ellos adaptan LLMs existentes, nosotros CONSTRUIMOS desde cero con swarm como arquitectura base
- "LLM-Assisted Iterative Evolution Toward SuperBrain" (arxiv 2509.00510): Subclass Brains + Superclass Brain via swarm — conceptualmente similar pero más complejo y top-down
- "Symbiotic LLM-SNN" (nov 2025): wake-sleep rhythms en LLM = directamente inspirado en delfín 🐬
- **Nuestra ventaja:** integrar swarm desde el diseño, no como wrapper — y la conexión directa con el doctorado de Emmanuel en formación de enjambres de drones

## Notas

- El corpus propio (~28K tokens) es muy pequeño para pre-training. Plan: FineWeb para base + fine-tune con corpus personal.
- Importante: las conversaciones de Emmanuel son el "alma" del fine-tuning — que el modelo entienda SU forma de pensar, no solo lenguaje genérico.
- El paper de drones (QuadrotorFleetShapeControl2025) puede ser la inspiración formal del mecanismo de cohesión del enjambre.
