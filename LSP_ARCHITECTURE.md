# LixySwarm Protocol (LSP) — Decisiones de Arquitectura

> Estado: Pre-diseño. Implementación después de Run 11 + Fase 2 LAN.
> Este documento captura las decisiones fundacionales antes de escribir el RFC formal.

---

## AD-001: Matriarca Dual (Personal + Global)

### Decisión
LSP soporta dos capas de Matriarca con identidades y ciclos de vida separados:

**Matriarca Personal (privada, local)**
- Almacenada exclusivamente en la máquina del usuario
- Aprende solo de las interacciones del usuario con su instancia de LixySwarm
- Encriptada con la llave privada del usuario (AES-256-GCM con clave derivada de Ed25519)
- Nunca se transmite a la red — ni siquiera como gradientes
- Tiene prioridad sobre la Global en inferencia runtime:
  ```
  infrasound_final = α * infrasound_personal + (1-α) * infrasound_global
  donde α ∈ [0.6, 0.9] por defecto (personal domina)
  ```

**Matriarca Global (distribuida, compartida)**
- Aprende del enjambre colectivo a través de gossip de conocimiento destilado
- Solo se comparten memorias sintéticas/destiladas (nunca texto raw de conversaciones)
- Cada contribución al banco global va firmada criptográficamente por el nodo origen
- Actualizable via protocolo de consensus (votación ponderada por reputación del nodo)

### Implicaciones para LSP
- El wire format debe incluir campo `matriarca_tier`: PERSONAL | GLOBAL | BOTH
- Los mensajes de gossip de Matriarca solo propagan contenido GLOBAL
- El protocolo garantiza que datos de tier PERSONAL nunca aparecen en mensajes de red

---

## AD-002: Identidad Criptográfica Descentralizada (sin servidor central)

### Decisión
Cada nodo LixySwarm tiene identidad basada en criptografía de curva elíptica:

**Generación de identidad (setup one-time)**
```
privkey = Ed25519.generate()           # nunca sale de la máquina
pubkey = Ed25519.public_key(privkey)   # identidad pública del nodo
node_id = SHA256(pubkey)[:16]          # 16 bytes = ID del nodo en la red
```

**Firma de feromonas**
Cada FeromonMessage incluye:
```
{
  node_id:    bytes[16]   # SHA256(pubkey)[:16]
  agent_id:   uint8       # 0-N (qué agente generó la feromona)
  timestamp:  uint64      # unix ms (para TTL y replay protection)
  feromon:    float16[D]  # tensor de feromona comprimido
  signature:  bytes[64]   # Ed25519.sign(node_id || agent_id || timestamp || feromon)
}
```

**Verificación en recepción**
```python
if not Ed25519.verify(msg.pubkey, msg.signature, msg.payload):
    drop(msg)  # firma inválida — rechazar silenciosamente
if msg.node_id != SHA256(msg.pubkey)[:16]:
    drop(msg)  # identidad no coincide con pubkey
if abs(msg.timestamp - now_ms) > 30_000:
    drop(msg)  # replay protection: rechazar mensajes >30s old
```

**Matriarca Personal encriptada**
```
key = HKDF(privkey, salt="lixy-matriarca-v1")
encrypted_bank = AES256GCM.encrypt(key, matriarca_memory.pt)
```
El archivo `.pt` en disco siempre está encriptado. Solo se descifra en RAM durante operación.

### Propiedades garantizadas
- ✅ Sin servidor de autenticación central
- ✅ El usuario es dueño de su identidad (privkey = identidad)
- ✅ Contribuciones al enjambre son atribuibles y verificables
- ✅ Nodos maliciosos con firmas inválidas son rechazados automáticamente
- ✅ Matriarca Personal es ilegible para terceros (encriptada en reposo)
- ✅ Replay attacks bloqueados por timestamp + TTL

---

## AD-003: Jerarquía de protocolos

```
┌─────────────────────────────────┐
│   LixySwarm Application Layer   │  (Matriarca, Feromonas, Gossip)
├─────────────────────────────────┤
│   LSP — LixySwarm Protocol      │  ← este es el nuevo protocolo
│   (semántica de enjambre,        │
│    identidad, merge, TTL)        │
├──────────────┬──────────────────┤
│   UDP        │   TCP / QUIC     │  (transporte swappable)
│ (feromonas,  │  (gossip confiable│
│  fire&forget)│   Matriarca sync) │
├──────────────┴──────────────────┤
│   IP (LAN local o internet)      │
└─────────────────────────────────┘
```

LSP es la capa que no existe en ningún protocolo actual: define QUÉ significa una feromona en el enjambre, cómo se mezcla, quién la firmó, cuánto vive. TCP/UDP son solo el cable.

---

---

## AD-004: Sistema de Reputación de Nodos

### Decisión
Cada nodo tiene un score de reputación R ∈ [0.0, 1.0] que evoluciona con su comportamiento:

**Arranque (bootstrap)**
```
R_nuevo = 0.1   # peso mínimo — nodos nuevos contribuyen poco al merge
```

**Acumulación de confianza**
```
# Feromona útil = cosine_similarity con consenso del enjambre > umbral
if cos_sim(feromon_recv, consensus) > 0.7:
    R = min(1.0, R + 0.005)   # subida gradual (+0.5% por feromona útil)
```

**Penalización por outliers**
```
# Feromona outlier = muy diferente al consenso actual
if cos_sim(feromon_recv, consensus) < 0.2:
    R = max(0.0, R - 0.02)    # bajada más rápida (-2% por outlier)
    # Outlier repetido = señal de nodo maligno o desalineado
```

**Banneo por comportamiento repetido**
```
if consecutive_outliers >= 10:
    ban(node_id, duration=3600)   # bloquear 1h
if lifetime_outlier_rate > 0.5:
    ban(node_id, duration=86400)  # bloquear 24h — nodo sistemáticamente dañino
```

**Uso en merge de feromonas**
```python
# El peso en el merge es proporcional a la reputación
merged = sum(R[i] * feromon[i] for i in peers) / sum(R[i] for i in peers)
```

### Propiedades garantizadas
- ✅ Nodos nuevos no pueden dominar el enjambre inmediatamente
- ✅ Contribuciones consistentes acumulan influencia gradualmente
- ✅ Nodos con garbage son expulsados automáticamente sin coordinación central
- ✅ Sin autoridad central que decida quién es confiable — emerge del comportamiento
- ✅ Resistente a ataques Sybil: muchos nodos nuevos con R=0.1 no superan un nodo R=0.9

### Almacenamiento
- Cada nodo mantiene su tabla de reputación local (no hay tabla global centralizada)
- El gossip periódico puede incluir reputación observada (para convergencia más rápida)
- Persistencia: `checkpoints/node_reputation.json` — sobrevive reinicios

---
## Pendiente para el RFC formal

- [ ] Formato de paquete detallado (wire format con offsets exactos)
- [ ] Diagrama de estados del protocolo de handshake
- [ ] Algoritmo de merge de feromonas remotas (weighted average vs otros)
- [ ] Protocolo de consensus para actualizar Matriarca Global
- [x] Mecanismo de reputación de nodos — ver AD-004
- [ ] Versioning del protocolo (compatibilidad hacia atrás)
- [ ] Vectores de test para validar implementaciones en otros lenguajes

---

*Última actualización: 2026-05-29 | Autor: Cody (basado en decisiones de Emmanuel Cardenaz)*
