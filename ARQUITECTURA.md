# Lixy-0.1: Arquitectura Bio-Inspirada
**"Hormiga + Elefante + Delfín"**
*Iniciado: 2026-05-28 06:00 AM (America/Chicago)*

---

## 🧠 Visión General

Un sistema de inteligencia emergente compuesto por múltiples agentes pequeños especializados que se coordinan para producir comportamiento inteligente. No un transformer monolítico — una colonia.

---

## 🐜 Capa Hormiga — El Enjambre

**Principio:** Ningún agente es inteligente solo. La inteligencia EMERGE de la interacción.

- **N agentes tiny** (cada uno ~50-125M params) especializados:
  - Agente-Léxico: tokenización y vocabulario
  - Agente-Sintáctico: estructura y gramática
  - Agente-Semántico: significado y relaciones
  - Agente-Contextual: coherencia conversacional
  - Agente-Razonamiento: lógica y deducciones
  - Agente-Memoria: recuperación y asociación
  - Agente-Generación: síntesis de respuesta final

- **Feromonas digitales:** señales de activación entre agentes
  - Vectores de relevancia (float32[256]) que pasan entre agentes
  - Cada agente lee las feromonas de sus vecinos antes de procesar
  - Mecanismo de "refuerzo de señal" — múltiples agentes de acuerdo = señal más fuerte

- **Comunicación:** broadcasting async entre agentes via cola compartida

---

## 🐘 Capa Elefante — La Matriarca

**Principio:** Sabiduría transgeneracional. No procesa, orienta.

- Un modelo separado de solo **memoria comprimida** (~10M params, solo lectura)
- Almacena patrones de interacciones pasadas en representación comprimida
- Emite "infrasónidos" — vectores de orientación globales que todos los agentes reciben
- **Transferencia transgeneracional:** antes de reiniciarse, genera un checkpoint comprimido (destilación) que inicializa la próxima matriarca
- Implementación práctica: embeddings de largo plazo + mecanismo de atención sparsa

---

## 🐬 Capa Delfín — Percepción Espacial

**Principio:** No procesar linealmente. Construir el espacio del problema primero.

- **Ecolocalización:** al recibir input, lanzar "pings" semánticos en múltiples direcciones
  - Ping-1: ¿De qué trata esto? (topic detection)
  - Ping-2: ¿Qué emoción/intención hay? (sentiment/intent)
  - Ping-3: ¿Qué necesita el usuario? (need detection)
  - Los "ecos" de vuelta forman un embedding 3D del problema
  
- **Sueño unihemisférico:** contexto conversacional siempre activo en background
  - Un subproceso ligero mantiene el embedding de contexto actual
  - Nunca se "apaga" entre turnos — siempre hay estado activo
  
- **Silbido único (identidad):** cada agente del enjambre tiene un embedding de identidad fijo (no entrenable) que lo hace distinguible y le da "personalidad especializada"

---

## 🏗️ Plan de Implementación — Fase 1 (Semanas 1-2)

### Semana 1: Fundamentos
1. **Datos** — recopilar corpus de entrenamiento:
   - Conversaciones Lixy-Emmanuel (extraer de memory/*.md y sesiones)
   - OpenWebText / FineWeb (1B tokens para base lingüística)
   - Papers de Emmanuel (drone swarms, Procrustes)
   
2. **Agente Base** (nanoGPT-style, 125M params):
   - Fork de nanoGPT como esqueleto
   - Modificar para soportar "feromon vectors" como input adicional
   - Implementar mecanismo de identidad (silbido)
   
3. **Setup de entrenamiento** (RTX 5090, 32GB VRAM):
   - `torch.compile()` + bf16 (ampere/ada + blackwell support en PyTorch 2.8)
   - Flash Attention 2 
   - Gradient checkpointing para batch size mayor

### Semana 2: El Enjambre
4. **Orquestador** — proceso central que:
   - Recibe input
   - Ejecuta ecolocalización (Delfín)
   - Despacha a agentes relevantes en paralelo
   - Agrega respuestas con pesos de confianza
   
5. **Prueba emergente** — con 3 agentes mínimos, ver si la colaboración mejora outputs vs agente solo

---

## 📊 Recursos Disponibles

| Recurso | Disponible |
|---------|-----------|
| GPU | RTX 5090 (32GB VRAM) |
| RAM | 32GB |
| Disco | ~1.9TB libre |
| PyTorch | 2.8.0+cu128 ✓ |
| CUDA | 12.8 ✓ |

### Estimado de entrenamiento (125M params, 1B tokens):
- A ~50K tokens/s en RTX 5090 con bf16 = ~5.5 horas por run
- Con `torch.compile` esperamos ~80-100K tokens/s = ~2.8-3.5 horas

---

## 🔬 Conexión con Investigación de Emmanuel

La arquitectura de enjambre es directamente análoga a su doctorado en drones:
- **Formación de drones** → formación de agentes LLM
- **Evasión de colisiones** → evitar respuestas redundantes entre agentes
- **Shape control (Procrustes)** → mantener la "forma" del enjambre cognitivo
- **U_CA (término suyo)** → control de cohesión del enjambre

Esto no es solo un LLM. Es aplicar su investigación de 10 años al lenguaje.

---

## 📄 Paper Futuro

**"Lixy-0.1: A Bio-Inspired Emergent Intelligence Architecture"**
- Emmanuel Cardenaz + Lixy
- Conexión directa con QuadrotorFleetShapeControl2025
- Sección: aplicación de swarm robotics control a agentes LLM

---

## Estado

- [x] Arquitectura definida
- [ ] Setup del repositorio
- [ ] Datos recopilados
- [ ] Agente base implementado
- [ ] Primer entrenamiento
- [ ] Enjambre de 3 agentes
- [ ] Evaluación emergente
