# LixySwarm Protocol (LSP) — Decisiones de Arquitectura
*Última actualización: 2026-05-30 | Estado: Pre-implementación (post DolphinAgent Phase A)*

---

## Contexto

LSP es el protocolo nativo de LixySwarm para distribución en internet abierta. No es un wrapper de TCP/UDP genérico — define la semántica del enjambre a nivel de protocolo.

**Objetivo:** Que cualquier desarrollador pueda implementar un nodo en Rust/Go/C++ solo leyendo este documento.

**Prerequisito para implementar:** DolphinAgent Phase A + Hormigas Dinámicas completados.

---

## Decisiones de Arquitectura

### AD-001: Matriarca Dual (Personal + Global)

**Decisión:** LSP soporta dos capas de Matriarca con identidades y ciclos de vida separados.

**Matriarca Personal (privada, nunca sale del nodo)**
- Aprende solo de las interacciones del usuario con su instancia
- Encriptada con clave derivada de la keypair Ed25519 del nodo (AES-256-GCM)
- Nunca se transmite — ni como gradientes, ni como embeddings
- Tiene prioridad sobre la Global:
  ```
  infrasound_final = α * infrasound_personal + (1-α) * infrasound_global
  donde α ∈ [0.6, 0.9] (personal domina por defecto)
  ```

**Matriarca Global (distribuida, compartida)**
- Aprende del enjambre colectivo via gossip de **conocimiento destilado**
- Solo memorias sintéticas (nunca texto raw de conversaciones)
- Cada contribución firmada criptográficamente por el nodo origen
- Actualizable via consensus ponderado por reputación del nodo

**Implicación para el wire format:**
- Campo `matriarca_tier`: PERSONAL | GLOBAL | BOTH
- Mensajes de gossip de Matriarca solo propagan contenido GLOBAL
- Garantía de protocolo: datos PERSONAL nunca aparecen en mensajes de red

---

### AD-002: Identidad Criptográfica Descentralizada

**Decisión:** Cada nodo tiene identidad basada en Ed25519. Sin servidor central, sin registro.

```python
# Al primer arranque del nodo:
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

private_key = Ed25519PrivateKey.generate()
public_key  = private_key.public_key()
node_id     = sha256(public_key.public_bytes_raw())[:16].hex()  # 64-bit ID

# Guardado encriptado localmente:
# ~/.lixyswarm/identity.key.enc  (privada, AES-256-GCM + passphrase)
# ~/.lixyswarm/identity.pub      (pública, compartible libremente)
```

**Por qué Ed25519 y no X25519:**
- Ed25519 es para firma (verificar autenticidad de mensajes)
- Cada mensaje de feromona va firmado → contribuciones verificables
- Un nodo no puede falsificar mensajes de otro nodo

**Node reputation:**
```
reputation = weighted_moving_average(
    [verified_contributions, uptime, feromon_quality_score]
)
# Peso en el consensus de Matriarca Global ∝ reputation
```

---

### AD-003: Jerarquía de Capas

**Decisión:** LSP es un protocolo nativo que define QUÉ se transporta, no cómo.

```
Capa de Aplicación:  LixySwarm (feromonas, gossip, handshake)
        │
Capa LSP:            Wire format, semántica de feromona, TTL, decay
        │
Capa de Transporte:  UDP (feromonas) o TCP (gossip) — swappable
        │
Capa de Red:         IP (LAN/internet)
```

**LSP no es WebSocket, no es gRPC, no es Kafka.**
Define el protocolo de enjambre como ciudadano de primera clase.

---

### AD-004: Modo Dual LAN/Internet

**Decisión:** El mismo protocolo LSP funciona en LAN y en internet abierta, con diferente capa de descubrimiento.

```
LAN:      mDNS "_lixyswarm._udp.local" → auto-discovery sin config
Internet: DHT Kademlia-light → descubrimiento distribuido
          + Relay nodes para NAT traversal

Mismo wire format, misma semántica, diferente capa de descubrimiento.
No reemplazar LAN con Internet — extenderla.
```

---

## Wire Format LSP v1

```
┌─────────────────────────────────────────────────────────┐
│ [4B]  Magic:    0x4C595357  ("LYSW")                    │
│ [1B]  Version:  0x01                                    │
│ [1B]  Type:     0x01=FEROMON 0x02=GOSSIP                │
│                 0x03=HANDSHAKE 0x04=PING 0x05=LEGACY    │
│ [2B]  Flags:    bit0=compressed bit1=signed bit2=urgent │
│ [4B]  Payload length (uint32, little-endian)            │
│ [32B] Node ID (Ed25519 public key, raw bytes)           │
│ [64B] Signature (Ed25519, over payload bytes)           │
│ [NB]  Payload (zstd compressed si Flags.bit0=1)         │
└─────────────────────────────────────────────────────────┘
Total overhead: 108B por mensaje
```

### Payload: FEROMON (0x01)
```json
{
  "feromon": [float32 × 256],   // ~1KB raw, ~400B comprimido
  "step": 53000,
  "fitness": 0.58,
  "agent_role": "explorador",
  "timestamp": 1748649600.0,
  "ttl": 3
}
```

### Payload: GOSSIP (0x02)
```json
{
  "kind": "matriarca_global",
  "memories": [
    {
      "embedding": [float32 × 256],
      "importance": 0.72,
      "synthetic": true          // nunca texto raw
    }
  ],
  "node_reputation": 0.85
}
```

### Payload: LEGACY (0x05) ← nuevo, para legado genético
```json
{
  "ant_id": "node_abc123:ant_2",
  "role": "refinador",
  "fitness_avg": 0.61,
  "top_patterns": [float32 × 256],   // patrones más fuertes
  "steps_alive": 15000,
  "cause_of_death": "low_fitness"    // o "node_disconnect"
}
```

---

## Semántica de Feromona en el Protocolo

A diferencia de otros protocolos donde el payload es opaco, LSP define semántica nativa:

**TTL de señal:** `ttl` se decrementa en cada hop. Feromona con `ttl=0` no se reenvía. Evita broadcast infinito sin coordinar.

**Decay temporal:** El receptor aplica decay exponencial basado en `timestamp`:
```python
age_s = now() - msg.timestamp
decayed = feromon * exp(-0.1 * age_s)  # decay rate configurable por nodo
```

**Merge-on-receipt:** El nodo receptor no almacena todas las feromonas por separado — las mergea en su pool local inmediatamente:
```python
pool += alpha * decayed_feromon  # alpha ∝ sender reputation
```

---

## Topología del Enjambre

Los nodos no son clientes/servidores — son **ciudadanos de primera clase**:

```
Cualquier nodo puede:
  ✓ Emitir feromonas
  ✓ Recibir y mergear feromonas de otros
  ✓ Contribuir a la Matriarca Global
  ✓ Ser relay para otros nodos (TTL forwarding)
  ✓ Abandonar la red sin romperla (el enjambre se auto-reconfigura)

Ningún nodo puede:
  ✗ Imponer su feromona (ponderado por reputation)
  ✗ Leer la Matriarca Personal de otro nodo
  ✗ Identificar usuarios (solo node_id criptográfico)
```

---

## Reputación de Nodos

```python
class NodeReputation:
    # Se actualiza cada N rounds de gossip
    
    verified_contributions: int   # mensajes firmados y válidos recibidos
    uptime_hours: float           # tiempo activo en la red
    feromon_quality: float        # cos_sim con consensus de feromonas
    
    @property
    def score(self) -> float:
        return harmonic_mean([
            min(verified_contributions / 1000, 1.0),
            min(uptime_hours / 720, 1.0),    # 720h = 30 días
            feromon_quality
        ])
```

La reputación determina el peso en:
1. Consensus de Matriarca Global
2. Peso en merge de feromonas
3. Prioridad de relay (nodos de alta reputación hacen más relay)

---

## Implementación de Referencia (Python) — Esqueleto

```python
# src/network/lsp.py (a implementar)

class LSPNode:
    def __init__(self, identity: Ed25519PrivateKey, config: LSPConfig):
        self.identity = identity
        self.node_id  = sha256(identity.public_key().public_bytes_raw())[:16]
        self.peers    = PeerTable(max_peers=50)
        self.feromon_pool = FeromonPool(decay_rate=0.1)
    
    def broadcast_feromon(self, feromon: Tensor, fitness: float):
        msg = FeromonMessage(
            feromon=feromon.tolist(),
            fitness=fitness,
            ttl=3
        )
        payload = zstd.compress(json.dumps(msg).encode())
        packet = build_packet(LSPType.FEROMON, payload, self.identity)
        self.peers.broadcast_udp(packet)
    
    def on_receive(self, packet: bytes, sender_addr: tuple):
        msg = parse_and_verify(packet)  # verifica firma Ed25519
        if msg is None:
            return  # descarta mensajes inválidos
        
        if msg.type == LSPType.FEROMON:
            self.feromon_pool.merge(msg.feromon, weight=self.peers.reputation(msg.node_id))
            if msg.ttl > 0:
                self.relay(packet, decrement_ttl=True)  # forward a otros peers
        
        elif msg.type == LSPType.GOSSIP:
            self.matriarca_global.integrate(msg.memories, weight=self.peers.reputation(msg.node_id))
        
        elif msg.type == LSPType.LEGACY:
            self.matriarca_local.store_genetic_legacy(msg.legacy_data)
```

---

## Estado y Próximos Pasos

| Paso | Estado |
|---|---|
| SwarmNetwork Fase 1 (UDP+TCP básico) | ✅ |
| mDNS LAN auto-discovery | ✅ en diseño |
| Test físico multi-host LAN | ⏳ |
| VPS como primer nodo externo | ⏳ (checkpoint pendiente de transferir) |
| Diseño AD-001 a AD-004 | ✅ este documento |
| Implementación LSP wire format | ⏳ post DolphinAgent Phase A |
| Identidad Ed25519 | ⏳ |
| Matriarca Dual (Personal + Global) | ⏳ |
| DHT Kademlia-light | ⏳ |
| Relay + NAT traversal | ⏳ largo plazo |

---

*Diseño: Emmanuel Cardenaz | Documentación: Cody | 2026*

---

## Fase Futura: Contribución Configurable por Nodo

> Después de LSP v1 estable. Diseñado por Emmanuel Cardenaz.

### Principio
Nadie queda fuera por tener equipo humilde — todos participan según sus posibilidades.

### Modos de contribución (configurables por el usuario)
| Modo | Descripción | Recursos |
|---|---|---|
| **Máximo** | Training completo + gossip + relay | GPU + CPU full |
| **Moderado** | Inferencia + gossip, sin training | CPU 30-50% |
| **Mínimo (relay)** | Solo reenvía feromonas de otros | CPU <10% |

### La Matriarca como coordinadora
- Aprende qué nodos son buenos para qué tareas (vía historial de fitness)
- Distribuye trabajo según capacidad declarada del nodo
- Un nodo que siempre da buenas feromonas en código → recibe más queries de código
- Un nodo humilde con CPU solo → recibe pings para relay, no training

### Wire format LSP (extensión)
El campo  del HANDSHAKE se extiende:
```json
{
  capabilities: [feromon, relay],
  contribution_mode: minimal,
  resources: {
    has_gpu: false,
    cpu_cores: 2,
    ram_gb: 7.8,
    max_contrib_pct: 20
  }
}
```

### Por qué importa
- VPS sin GPU → modo relay → ya participa hoy
- Laptop con GPU → modo moderado → contribuye inferencia
- Servidor con RTX 5090 → modo máximo → training + todo
- La red es heterogénea por diseño, como una colonia de hormigas real
