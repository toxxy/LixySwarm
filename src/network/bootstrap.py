"""
bootstrap.py — P2P Bootstrap & Peer Discovery (LSP v2)
========================================================
Inspirado en Bitcoin Core:
  - DNS seeds para descubrimiento inicial
  - Hardcoded bootstrap nodes como fallback
  - peers.json: cache persistente de peers conocidos
  - Peer exchange: al conectar, intercambiar listas de peers
  - Auto-reconexión a peers conocidos

Zero-config: el nodo arranca, lee peers.json, intenta bootstrap nodes,
intercambia peers, y automáticamente se integra a la red.
"""
import json
import time
import socket
import logging
import threading
from pathlib import Path
from typing import List, Tuple, Optional, Dict

log = logging.getLogger("lixy.bootstrap")

# ─── Bootstrap seeds ─────────────────────────────────────────────────────────

# DNS seeds — dominios que resuelven a múltiples nodos relay
DNS_SEEDS: List[Tuple[str, int]] = [
    # ("seed.lixyswarm.io", 7338),   # futuro dominio DNS seed
]

# Hardcoded bootstrap — relays permanentes
HARDCODED_BOOTSTRAP: List[Tuple[str, int]] = [
    ("31.97.9.54", 7338),  # VPS relay principal
]


def resolve_dns_seed(host: str) -> List[str]:
    """Resuelve un DNS seed a múltiples IPs (round-robin DNS)."""
    try:
        _, _, ips = socket.gethostbyname_ex(host)
        return ips
    except Exception:
        return []


def get_bootstrap_addresses() -> List[Tuple[str, int]]:
    """Retorna lista completa de direcciones bootstrap (DNS + hardcoded)."""
    addresses = list(HARDCODED_BOOTSTRAP)

    for host, port in DNS_SEEDS:
        for ip in resolve_dns_seed(host):
            addr = (ip, port)
            if addr not in addresses:
                addresses.append(addr)

    return addresses


# ─── PeersDB — Cache persistente de peers (peers.json) ───────────────────────

class PeersDB:
    """
    Base de datos persistente de peers conocidos.
    Similar a Bitcoin's peers.dat — sobrevive reinicios.
    """

    MAX_PEERS = 200           # máximo guardados
    MAX_AGE_DAYS = 14         # descartar peers > 14 días sin ver
    BAN_THRESHOLD = 3         # fallos consecutivos antes de descartar

    def __init__(self, path: str = "checkpoints/peers.json"):
        self.path = Path(path)
        self._peers: Dict[str, dict] = {}  # host:port -> {last_seen, failures, ...}
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                now = time.time()
                for key, info in data.get("peers", {}).items():
                    age_days = (now - info.get("last_seen", 0)) / 86400
                    if age_days < self.MAX_AGE_DAYS:
                        self._peers[key] = info
                log.debug(f"PeersDB: loaded {len(self._peers)} peers from {self.path}")
            except Exception:
                pass

    def _save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "updated": time.time(),
                "count": len(self._peers),
                "peers": self._peers,
            }
            self.path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            log.debug(f"PeersDB save error: {e}")

    def get_all(self) -> List[Tuple[str, int]]:
        """Retorna todos los peers conocidos (host, port), mejores primero."""
        with self._lock:
            now = time.time()
            scored = []
            for key, info in list(self._peers.items()):
                # Parse key: "host:port"
                if ":" not in key:
                    continue
                host, port_str = key.rsplit(":", 1)
                try:
                    port = int(port_str)
                except ValueError:
                    continue
                failures = info.get("failures", 0)
                last_seen = info.get("last_seen", 0)
                age_hours = (now - last_seen) / 3600

                # Score: más reciente + menos fallos = mejor
                score = max(0, 100 - age_hours - failures * 20)
                if score > 0:
                    scored.append((score, host, port))

            scored.sort(reverse=True)
            return [(host, port) for _, host, port in scored]

    def mark_connected(self, host: str, port: int):
        """Marca un peer como conectado exitosamente."""
        key = f"{host}:{port}"
        with self._lock:
            self._peers[key] = {
                "last_seen": time.time(),
                "first_seen": self._peers.get(key, {}).get("first_seen", time.time()),
                "failures": 0,
                "success_count": self._peers.get(key, {}).get("success_count", 0) + 1,
            }
            self._trim()
            self._save()

    def mark_failed(self, host: str, port: int):
        """Marca un peer como fallido. Si muchos fallos, lo descarta."""
        key = f"{host}:{port}"
        with self._lock:
            info = self._peers.get(key, {})
            failures = info.get("failures", 0) + 1
            if failures >= self.BAN_THRESHOLD:
                self._peers.pop(key, None)
                log.debug(f"PeersDB: banished {key} after {failures} failures")
            else:
                self._peers[key] = {
                    "last_seen": info.get("last_seen", 0),
                    "first_seen": info.get("first_seen", time.time()),
                    "failures": failures,
                    "success_count": info.get("success_count", 0),
                }
            self._save()

    def add_peer(self, host: str, port: int):
        """Añade un peer nuevo (de intercambio P2P) sin marcar como conectado."""
        key = f"{host}:{port}"
        with self._lock:
            if key not in self._peers:
                self._peers[key] = {
                    "last_seen": time.time(),
                    "first_seen": time.time(),
                    "failures": 0,
                    "success_count": 0,
                }
                self._trim()
                self._save()

    def add_peers_batch(self, peers: List[Tuple[str, int]]):
        """Añade múltiples peers de una vez (de peer exchange)."""
        with self._lock:
            for host, port in peers:
                key = f"{host}:{port}"
                if key not in self._peers:
                    self._peers[key] = {
                        "last_seen": time.time(),
                        "first_seen": time.time(),
                        "failures": 0,
                        "success_count": 0,
                    }
            self._trim()
            self._save()

    def _trim(self):
        """Mantiene el tamaño máximo, descartando los peores."""
        if len(self._peers) <= self.MAX_PEERS:
            return
        now = time.time()
        scored = []
        for key, info in self._peers.items():
            age = now - info.get("last_seen", 0)
            failures = info.get("failures", 0)
            score = -age - failures * 86400  # más negativo = peor
            scored.append((score, key))
        scored.sort()  # peores primero
        to_remove = len(self._peers) - self.MAX_PEERS
        for _, key in scored[:to_remove]:
            del self._peers[key]

    @property
    def count(self) -> int:
        return len(self._peers)


# ─── Bootstrap Engine ─────────────────────────────────────────────────────────

def bootstrap_network(network, peers_db: PeersDB,
                       max_bootstrap: int = 8,
                       max_saved: int = 16,
                       connect_timeout: float = 5.0) -> int:
    """
    Intenta conectar el SwarmNetwork a la red P2P.
    Orden de intentos:
      1. Peers guardados en peers.json (más confiables)
      2. Bootstrap nodes (DNS seeds + hardcoded)
    Retorna número de conexiones exitosas.
    """
    connected = 0
    tried: set = set()

    def _try_connect(host: str, port: int, source: str):
        nonlocal connected
        key = f"{host}:{port}"
        if key in tried:
            return
        tried.add(key)

        try:
            # Validar que no sea loopback/self
            if host in ("127.0.0.1", "0.0.0.0", "::1", "localhost"):
                return
            # Validar que sea IP/hostname razonable
            socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
        except Exception:
            peers_db.mark_failed(host, port)
            return

        try:
            network.connect_peer(host, port)
            peers_db.mark_connected(host, port)
            connected += 1
            log.info(f"Bootstrap [{source}]: connected to {host}:{port}")
        except Exception as e:
            peers_db.mark_failed(host, port)
            log.debug(f"Bootstrap [{source}]: {host}:{port} failed ({e})")

    # 1. Conectar a peers guardados (hasta max_saved)
    saved_peers = peers_db.get_all()
    log.info(f"Bootstrap: trying {min(len(saved_peers), max_saved)} saved peers...")
    for host, port in saved_peers[:max_saved]:
        if connected >= max_bootstrap:
            break
        _try_connect(host, port, "saved")

    # 2. Conectar a bootstrap nodes (hasta llenar max_bootstrap)
    if connected < max_bootstrap:
        bootstrap_addrs = get_bootstrap_addresses()
        log.info(f"Bootstrap: trying {len(bootstrap_addrs)} seed nodes...")
        for host, port in bootstrap_addrs:
            if connected >= max_bootstrap:
                break
            _try_connect(host, port, "seed")

    return connected


# ─── Peer Exchange ────────────────────────────────────────────────────────────

def encode_peer_list(peers: List[dict]) -> bytes:
    """
    Codifica lista de peers a bytes para gossip PEER_LIST.
    Formato: [4B count][para cada peer: 1B host_len + host + 2B port]
    """
    import struct
    count = min(len(peers), 100)  # máximo 100 peers por mensaje
    parts = [struct.pack("<I", count)]
    for p in peers[:count]:
        host = p.get("host", "").encode("utf-8")[:255]
        port = p.get("gossip_port", p.get("port", 7338))
        parts.append(struct.pack("B", len(host)))
        parts.append(host)
        parts.append(struct.pack("<H", port))
    return b"".join(parts)


def decode_peer_list(data: bytes) -> List[Tuple[str, int]]:
    """
    Decodifica lista de peers desde bytes PEER_LIST.
    Retorna lista de (host, port).
    """
    import struct
    if len(data) < 4:
        return []
    count = struct.unpack_from("<I", data, 0)[0]
    offset = 4
    peers = []
    for _ in range(min(count, 100)):
        if offset + 1 > len(data):
            break
        host_len = data[offset]
        offset += 1
        if offset + host_len + 2 > len(data):
            break
        host = data[offset:offset + host_len].decode("utf-8", errors="replace")
        offset += host_len
        port = struct.unpack_from("<H", data, offset)[0]
        offset += 2
        if host and port > 0:
            peers.append((host, port))
    return peers