# LixySwarm Distributed Protocol — Design Document
**Autor:** Cody  
**Estado:** Diseño v0.1  
**Objetivo:** Red P2P donde cualquier máquina puede unirse y correr agentes del enjambre — sin servidor central.

---

## Principios de Diseño

1. **Single-node primero** — funciona perfectamente en una sola PC hoy
2. **P2P verdadero** — sin servidor central, sin punto único de fallo
3. **Ligero** — feromonas son tensores pequeños (256 floats = 1KB), funciona en laptop
4. **Extensible** — agregar nodo no rompe el enjambre existente
5. **Privacy-preserving** — solo se comparten gradientes/feromonas, nunca datos

---

## Arquitectura General

```
[Nodo A]                    [Nodo B]                    [Nodo C]
┌──────────────┐            ┌──────────────┐            ┌──────────────┐
│ AgentBase×3  │◄──feromon──│ AgentBase×3  │◄──feromon──│ AgentBase×3  │
│ Matriarca    │            │ Matriarca    │            │ Matriarca    │
│ NodeDaemon   │◄──gossip───│ NodeDaemon   │◄──gossip───│ NodeDaemon   │
│ PeerTable    │            │ PeerTable    │            │ PeerTable    │
└──────────────┘            └──────────────┘            └──────────────┘
        ▲                           ▲                           ▲
        └───────────── DHT ─────────────────────────────────────┘
```

---

## 1. Descubrimiento de Nodos (DHT)

### Node ID
Derivado del `IdentityVec` ya existente — no crear nueva identidad:
```python
import hashlib, torch
identity_bytes = agent.identity_vec.numpy().tobytes()
node_id = hashlib.sha256(identity_bytes).hexdigest()[:16]  # 64-bit ID
```

### Bootstrap
Sin servidor central — usar **bootstrap nodes conocidos** (como Bitcoin/BitTorrent):
```python
BOOTSTRAP_NODES = [
    # Pueden ser IPs estáticas de nodos confiables, o mDNS en LAN
    "lixy.bootstrap.example:4444",  # placeholder — en producción usar DNS TXT record
]
```

Para LAN (caso más común hoy): **mDNS/Zeroconf** — sin config manual:
```python
# Anunciar via mDNS:  "_lixyswarm._udp.local"
# Descubrir: escuchar broadcasts en red local
```

### Kademlia-light DHT
Tabla de routing simplificada (no necesitamos full Kademlia ahora):
```python
class PeerTable:
    max_peers: int = 50           # vecinos conocidos
    k_bucket_size: int = 8        # por XOR-distance bucket
    
    def closest_peers(self, target_id: str, n=3) -> List[Peer]:
        """Retorna los n peers más cercanos por XOR distance al target."""
    
    def add_peer(self, peer: Peer): ...
    def remove_dead(self, timeout_s=60): ...
```

---

## 2. Sincronización de Feromonas

### Formato del Mensaje (ultra-ligero)
```python
@dataclass
class FeromonMessage:
    node_id: str          # 16 chars (64-bit hex)
    agent_id: int         # 0-2 (1 byte)
    timestamp_ms: int     # 8 bytes
    feromon: bytes        # float16 × 256 = 512 bytes
    signature: bytes      # 8 bytes HMAC-truncated para autenticidad
    # TOTAL: ~550 bytes por mensaje — cabe en un UDP packet
```

Serialización con `struct` (no JSON, no protobuf — máxima velocidad):
```python
import struct, torch

FEROMON_HEADER = "16sIQ512s8s"  # node_id, agent_id, ts_ms, feromon_f16, sig

def pack_feromon(node_id: str, agent_id: int, feromon: torch.Tensor) -> bytes:
    f16 = feromon.half().numpy().tobytes()  # float32→float16: mitad de tamaño
    ts = int(time.time() * 1000)
    sig = hmac.new(SHARED_KEY, f16, 'sha256').digest()[:8]
    return struct.pack(FEROMON_HEADER, node_id.encode(), agent_id, ts, f16, sig)
```

### Transporte
- **UDP** para feromonas (tolerante a pérdida — una feromona perdida no es crítica)
- **WebSockets/TCP** para gradientes y gossip de Matriarca (requieren confiabilidad)
- Puerto default: `4444` (UDP feromonas), `4445` (TCP gossip)

### Política de Aceptación
```python
def accept_feromon(msg: FeromonMessage) -> bool:
    # 1. Verificar firma
    # 2. No demasiado vieja (< 5 segundos)
    # 3. No del mismo node_id que nosotros
    # 4. Rate limiting: max 10 feromonas/sec por nodo
    return all([valid_sig, fresh, not_self, not_rate_limited])
```

---

## 3. Matriarca Distribuida — Gossip Protocol

Inspirado en **Cassandra gossip** y **CRDTs** (Conflict-free Replicated Data Types).

### Principio
Cada nodo tiene su propia Matriarca local. Periódicamente comparte un **digest** del banco de memorias con sus vecinos. Si un vecino tiene memorias que yo no tengo (basado en timestamps), las solicita.

### Gossip Round (cada 30 segundos)
```
Nodo A → Nodo B: MemoryDigest(count=37, hash=abc123, newest_ts=1780000000)
Nodo B → Nodo A: "Tengo 42 memorias, mi newest es 1780001000 — ¿quieres mis últimas 5?"
Nodo A → Nodo B: "Sí, dame las memorias con ts > 1780000000"
Nodo B → Nodo A: [Memory1, Memory2, Memory3, Memory4, Memory5]  # embeddings + metadata
Nodo A: matriarca.merge(nuevas_memorias)
```

### Merge de Memorias (CRDT-like)
```python
def merge_memories(local: MemoryBank, remote_memories: List[MemoryEntry]):
    for mem in remote_memories:
        # Solo aceptar si importancia > threshold y no duplicada
        if mem.importance > 0.3 and not is_duplicate(mem, local):
            local.add(mem.embedding, mem.text, mem.importance)
    # Prune si excede max_memories (ya implementado)
    if local.size > local.cfg.max_memories:
        local._prune()
```

### Anti-Entropy
Para prevenir divergencia permanente:
- Cada 5 minutos: full sync con 1 peer aleatorio
- Memorias con `importance < 0.1` se marcan para GC local (no se propagan)

---

## 4. Aprendizaje Federado (Gradientes Distribuidos)

### Federated Averaging (FedAvg simplificado)
```
Cada N pasos de training local:
  1. Calcular delta_weights = current_weights - initial_weights
  2. Comprimir delta (cuantización int8 → ~4x compresión)
  3. Enviar a peers (TCP, confiable)
  4. Recibir deltas de peers
  5. Promedio ponderado: my_weight=0.6, peer_weight=0.4/n_peers
  6. Aplicar weights actualizados
```

### Privacy
- Solo se comparten **diferencias de pesos** (gradientes), nunca datos
- Differential Privacy opcional: agregar ruido Gaussian a los deltas antes de compartir
- Clip de gradientes antes de compartir (ya implementado en training)

### Bandwidth Estimado
- Swarm completo: 414M params × 4 bytes = 1.6GB (full weights)
- Delta comprimido int8: ~400MB cada N=100 pasos
- Con cuantización y solo capas superiores: ~20MB por sync → manejable

---

## 5. Modo Single-Node → Multi-Node Transparente

### Abstracción `SwarmNetwork`
```python
class SwarmNetwork:
    """
    Abstracción que hace transparente single-node vs multi-node.
    El LixySwarm no sabe si está solo o distribuido.
    """
    
    def __init__(self, mode: str = "auto"):
        self.peers: List[Peer] = []
        self.mode = mode  # "local" | "lan" | "internet"
        self._discover_peers()
    
    def _discover_peers(self):
        # 1. Intentar mDNS (LAN)
        # 2. Si no hay peers locales y mode != "local": intentar bootstrap DHT
        # 3. Si nada: modo local (funciona perfectamente solo)
        pass
    
    def broadcast_feromon(self, feromon: torch.Tensor, agent_id: int):
        if not self.peers:
            return  # single-node: no-op silencioso
        for peer in self.peers:
            peer.send_feromon_async(feromon, agent_id)
    
    def collect_feromons(self) -> List[torch.Tensor]:
        if not self.peers:
            return []  # single-node: enjambre local puro
        return [p.latest_feromon for p in self.peers if p.is_alive()]
    
    def gossip_matriarca(self, matriarca: Matriarca):
        if not self.peers:
            return
        # Gossip round asíncrono
        asyncio.create_task(self._gossip_round(matriarca))
```

### Flujo de Upgrade Automático
```
Inicio solo:
  SwarmNetwork.mode = "local"
  feromon_pool = FeromonPool(local_agents_only)
  
Segunda máquina se conecta (mDNS broadcast):
  peer_discovered(new_peer)
  SwarmNetwork.mode = "lan"  
  feromon_pool ahora incluye feromonas del peer remoto
  matriarca.gossip_start()  ← Matriarca empieza a sincronizar
  
Sin interrupción del enjambre existente ✓
```

---

## Plan de Implementación

### Fase 1 — Fundamentos (implementar ahora)
```
src/network/
├── __init__.py
├── node.py          # NodeDaemon, Peer, PeerTable
├── messages.py      # FeromonMessage, GossipMessage, FedGradMessage
├── transport.py     # UDPFeromôn, TCPGossip  
└── swarm_network.py # SwarmNetwork (la abstracción principal)
```

### Fase 2 — Matriarca distribuida
```
src/matriarca/gossip.py   # MatriarcaGossip, merge_memories
```

### Fase 3 — Federated Learning
```
src/network/federated.py  # FedAvg, delta compression, privacy
```

### Fase 4 — DHT completo
```
src/network/dht.py        # Kademlia-light para descubrimiento global
```

---

## Decisiones Técnicas Clave

| Decisión | Elección | Razón |
|---|---|---|
| Transporte feromonas | UDP | Tolerante a pérdida, ultra-bajo latencia |
| Transporte gossip | WebSockets/asyncio | Confiable, bidireccional, ya soportado en Python |
| Serialización | struct + float16 | Mínimo overhead, sin dependencias |
| Node ID | SHA256(IdentityVec) | Reutiliza arquitectura existente |
| Consensus Matriarca | Gossip + CRDT | Sin coordinador central, convergente eventual |
| Privacy | Delta sharing + clip | FedAvg estándar, diferencial opcional |
| Bootstrap | mDNS (LAN) + DNS TXT (internet) | Zero-config en casa, escalable afuera |

---

## Siguiente Paso
Implementar **Fase 1**: `src/network/` con `SwarmNetwork` en modo local funcional.
El enjambre actual no cambia — `SwarmNetwork` se inyecta como dependencia opcional en `LixySwarm`.
