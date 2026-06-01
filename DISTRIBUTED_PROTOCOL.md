# LixySwarm Distributed Protocol — Diseño v0.2
*Última actualización: 2026-05-30 | Estado: SwarmNetwork Fase 1 implementada y testeada*

---

## Estado de Implementación

| Componente | Estado | Notas |
|---|---|---|
| SwarmNetwork básica | ✅ | UDP + TCP, 23/23 tests |
| Feromon broadcast UDP | ✅ | Fire-and-forget, < 1KB/msg |
| TCP gossip bidireccional | ✅ | Confiable, 15/15 integración |
| `inject_remote_feromon()` | ✅ | Recibe feromona de otro nodo |
| `merge_remote_feromons()` | ✅ | Integra todas las recibidas, cos_sim=1.0 |
| mDNS descubrimiento LAN | ✅ en diseño | Implementado, test físico multi-host pendiente |
| LixySwarm Protocol (LSP) | ⏳ | Post DolphinAgent Phase A |
| Identidad criptográfica | ⏳ | Ed25519, parte de LSP |
| Matriarca Dual | ⏳ | Personal + Global, parte de LSP |

---

## Principios de Diseño

1. **Single-node primero** — funciona perfectamente en una sola PC hoy
2. **P2P verdadero** — sin servidor central, sin punto único de fallo
3. **Ligero** — feromonas son tensores pequeños (256 floats = ~1KB), funciona en laptop/VPS
4. **Extensible** — agregar nodo no rompe el enjambre existente
5. **Privacy-preserving** — solo se comparten feromonas destiladas, nunca datos raw
6. **Soberanía** — cada nodo controla su Matriarca Personal

---

## Arquitectura Actual (Fase 1)

```
[Nodo Local — Emmanuel]         [VPS — Nodo Externo]
┌──────────────────────┐        ┌──────────────────────┐
│ AntAgent × 3         │        │ AntAgent × N         │
│ DolphinAgent         │        │ DolphinAgent         │
│ Matriarca Personal   │        │ Matriarca Personal   │
│ SwarmNetwork         │◄───────│ SwarmNetwork         │
│   UDP 7337 (feromon) │        │   UDP 7337 (feromon) │
│   TCP 7337 (gossip)  │        │   TCP 7337 (gossip)  │
└──────────────────────┘        └──────────────────────┘
         ▲                                ▲
         └──────────── LAN/Internet ──────┘
                  (feromonas + gossip)
```

---

## SwarmNetwork — API Actual

```python
# src/network/swarm_network.py

class SwarmNetwork:
    def start(host, port):
        """Levanta UDP + TCP listeners."""
    
    def connect_peer(ip, port):
        """Conectar a nodo externo explícitamente (VPS, nodo remoto)."""
    
    def broadcast_feromon(feromon_tensor: Tensor):
        """Envía feromona a todos los peers vía UDP."""
    
    def inject_remote_feromon(feromon_tensor: Tensor):
        """Recibe feromona de otro nodo y la encola."""
    
    def merge_remote_feromons() -> Tensor:
        """Promedia todas las feromonas recibidas en la cola."""
    
    def gossip_state(state_dict: dict):
        """Sincroniza estado del enjambre via TCP (confiable)."""
```

---

## Formato de Mensaje de Feromona (actual)

```python
@dataclass
class FeromonMessage:
    node_id: str          # ID del nodo origen (hash del IdentityVec)
    feromon: Tensor       # [256] float32 = 1KB
    step: int             # paso de training del nodo origen
    fitness: float        # fitness del agente emisor
    timestamp: float      # unix timestamp
    ttl: int = 3          # Time-To-Live: hops máximos
```

**Tamaño total por mensaje:** ~1.1KB (feromona 1KB + metadata 100B)
**Frecuencia:** cada forward pass en training, cada turno en runtime

---

## Descubrimiento de Nodos

### LAN (implementado)
```
mDNS service: "_lixyswarm._udp.local"
Puerto: 7337
Sin configuración manual — auto-discovery en red local
```

### Internet (diseño)
```
Fase A: IPs explícitas en config (VPS actual)
Fase B: DHT Kademlia-light
Fase C: DHT + relay nodes para NAT traversal
```

### Node ID (actual)
```python
identity_bytes = agent.identity_vec.numpy().tobytes()
node_id = hashlib.sha256(identity_bytes).hexdigest()[:16]
```

---

## LixySwarm Protocol (LSP) — Diseño (próxima implementación)

> Protocolo nativo para enjambre distribuido en internet. No un wrapper de TCP/UDP genérico — define semántica de feromona a nivel de protocolo.

### Por qué LSP y no WebSocket/gRPC
- **Wire format propio**: magic bytes, versión, tipo, payload comprimido
- **Semántica de feromona nativa**: merge-on-receipt, TTL de señal, decay temporal
- **Topología de enjambre**: nodos son ciudadanos de primera clase
- **Eficiencia**: < 1KB por feromona 256d con compresión
- **Cualquier dev puede implementar**: spec RFC-style → nodo en Rust/Go/C++ solo leyendo el doc

### Wire Format Propuesto
```
[4B  magic: 0x4C595357 ("LYSW")]
[1B  version]
[1B  message_type: FEROMON=0x01, GOSSIP=0x02, HANDSHAKE=0x03, PING=0x04]
[2B  payload_len]
[32B node_id (Ed25519 pubkey)]
[64B signature]
[NB  payload (zstd compressed)]
```

### Decisiones de Arquitectura (LSP_ARCHITECTURE.md)

**AD-001: Matriarca Dual**
- Matriarca Personal: encriptada, nunca sale del nodo
- Matriarca Global: distribuida, solo memorias destiladas/sintéticas
- `infrasound_final = α * personal + (1-α) * global` donde α ∈ [0.6, 0.9]

**AD-002: Identidad Criptográfica**
- Ed25519 keypair generado localmente al primer arranque
- `node_id = pubkey[:16]` — sin servidor central, sin registro
- Cada mensaje de feromona va firmado — contributions verificables

**AD-003: Jerarquía de Capas**
- LSP define QUÉ se transporta, no cómo
- Puede correr sobre UDP (feromonas) o TCP (gossip)
- La capa de transporte es swappable

**AD-004: Modo Dual LAN/Internet**
- Mismo protocolo, diferente capa de descubrimiento
- LAN: mDNS (sin config)
- Internet: DHT/relay (con config mínima)

---

## Próximos Pasos P2P

1. **Test físico LAN multi-host** (Fase 2) — dos máquinas en la misma red
2. **VPS como primer nodo externo** — IP explícita, transferir checkpoint
3. **Implementar LSP** — post DolphinAgent Phase A
4. **Compresión de tensores** — zstd sobre float32[256] → ~400B
5. **Identidad Ed25519** — keypair local, firma de mensajes

---

## VPS — Configuración Inicial

Ver `VPS_SETUP.md` para guía completa.

```bash
# Clonar repo
git clone https://github.com/<user>/LixySwarm.git && cd LixySwarm

# Instalar dependencias (CPU-only)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Transferir checkpoint (desde máquina local)
rsync -avz checkpoints/swarm_best.pt usuario@VPS_IP:~/LixySwarm/checkpoints/

# Abrir puertos
sudo ufw allow 7337/udp && sudo ufw allow 7337/tcp

# Conectar al nodo local
python3 -c "
from src.network.swarm_network import SwarmNetwork
net = SwarmNetwork(host='0.0.0.0', port=7337)
net.start()
net.connect_peer('EMMANUEL_IP', 7337)
"
```
