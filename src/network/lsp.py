"""
LixySwarm Protocol (LSP) v1
Wire format: LYSW magic + versión + tipo + Ed25519 identity + firma
Corre sobre UDP (feromonas) o TCP (gossip, handshake)

Wire format (108 bytes header):
    [4B]  Magic:    0x4C595357  ("LYSW")
    [1B]  Version:  0x01
    [1B]  Type:     FEROMON=0x01 GOSSIP=0x02 HANDSHAKE=0x03 PING=0x04 PONG=0x05
    [2B]  Flags:    bit0=compressed bit1=signed bit2=urgent
    [4B]  Payload length (uint32, little-endian)
    [32B] Node ID (primeros 32 bytes del public key Ed25519)
    [64B] Signature (Ed25519 sobre los bytes del payload)
    [NB]  Payload (zlib compressed si Flags.bit0=1)
"""

import os
import json
import zlib
import struct
import socket
import threading
import time
import logging
from pathlib import Path
from typing import Callable, Optional, Dict, List

log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

MAGIC = b"LYSW"
VERSION = 0x01

class PacketType:
    FEROMON   = 0x01
    GOSSIP    = 0x02
    HANDSHAKE = 0x03
    PING      = 0x04
    PONG      = 0x05

class Flags:
    COMPRESSED = 0x01
    SIGNED     = 0x02
    URGENT     = 0x04

HEADER_SIZE = 4 + 1 + 1 + 2 + 4 + 32 + 64  # = 108 bytes
MAX_UDP_SIZE = 65507


# ─── LSPIdentity ──────────────────────────────────────────────────────────────

class LSPIdentity:
    """Identidad Ed25519 de un nodo LSP."""

    def __init__(self, private_key, public_key):
        self._private_key = private_key
        self._public_key = public_key
        self.node_id: bytes = public_key.public_bytes_raw()[:32]
        self.node_id_hex: str = self.node_id.hex()

    @classmethod
    def generate(cls) -> "LSPIdentity":
        """Genera un nuevo keypair Ed25519."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        return cls(private_key, public_key)

    @classmethod
    def load(cls, path: str) -> Optional["LSPIdentity"]:
        """Carga identidad desde archivo. Retorna None si no existe."""
        p = Path(path)
        if not p.exists():
            return None
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
        with open(p, "rb") as f:
            pem_data = f.read()
        private_key = load_pem_private_key(pem_data, password=None)
        public_key = private_key.public_key()
        return cls(private_key, public_key)

    def save(self, path: str) -> None:
        """Guarda identidad en archivo PEM."""
        from cryptography.hazmat.primitives.serialization import (
            Encoding, PrivateFormat, NoEncryption
        )
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        pem_data = self._private_key.private_bytes(
            Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
        )
        with open(p, "wb") as f:
            f.write(pem_data)
        p.chmod(0o600)
        log.debug(f"Identity saved to {path}")

    def sign(self, data: bytes) -> bytes:
        """Firma datos con clave privada Ed25519. Retorna 64 bytes."""
        return self._private_key.sign(data)

    def verify(self, data: bytes, sig: bytes, pubkey: bytes) -> bool:
        """Verifica firma Ed25519. pubkey = 32 bytes raw."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.exceptions import InvalidSignature
        try:
            pk = Ed25519PublicKey.from_public_bytes(pubkey)
            pk.verify(sig, data)
            return True
        except (InvalidSignature, Exception):
            return False

    @property
    def public_key_bytes(self) -> bytes:
        """Retorna 32 bytes de la clave pública raw."""
        return self.node_id  # ya es los 32 bytes del pubkey


# ─── LSPPacket ────────────────────────────────────────────────────────────────

class LSPPacket:
    """Un paquete LSP serializable con header LYSW."""

    magic = MAGIC
    version = VERSION

    def __init__(self):
        self.type: int = 0
        self.flags: int = 0
        self.node_id: bytes = b"\x00" * 32
        self.signature: bytes = b"\x00" * 64
        self.payload: bytes = b""

    @classmethod
    def create(cls, ptype: int, payload: bytes,
               compress: bool = True, urgent: bool = False) -> "LSPPacket":
        pkt = cls()
        pkt.type = ptype
        pkt.flags = 0
        if urgent:
            pkt.flags |= Flags.URGENT
        if compress:
            compressed = zlib.compress(payload, level=6)
            if len(compressed) < len(payload):
                pkt.payload = compressed
                pkt.flags |= Flags.COMPRESSED
            else:
                pkt.payload = payload
        else:
            pkt.payload = payload
        return pkt

    def pack(self, identity: "LSPIdentity") -> bytes:
        """Serializa + firma el paquete. Retorna bytes listos para enviar."""
        self.node_id = identity.node_id
        self.flags |= Flags.SIGNED
        self.signature = identity.sign(self.payload)

        header = struct.pack(
            "<4sBBHI",
            self.magic,
            self.version,
            self.type,
            self.flags,
            len(self.payload)
        )
        # header is 4+1+1+2+4 = 12 bytes, then node_id (32) + signature (64)
        return header + self.node_id + self.signature + self.payload

    @classmethod
    def unpack(cls, data: bytes) -> "LSPPacket":
        """Deserializa un paquete LSP desde bytes raw."""
        if len(data) < HEADER_SIZE:
            raise ValueError(f"Packet too short: {len(data)} < {HEADER_SIZE}")

        magic, version, ptype, flags, payload_len = struct.unpack_from("<4sBBHI", data, 0)
        if magic != MAGIC:
            raise ValueError(f"Bad magic: {magic!r}")
        if version != VERSION:
            raise ValueError(f"Unsupported version: {version}")

        offset = 12  # 4+1+1+2+4
        node_id = data[offset:offset+32]
        offset += 32
        signature = data[offset:offset+64]
        offset += 64

        raw_payload = data[offset:offset+payload_len]
        if len(raw_payload) != payload_len:
            raise ValueError(f"Truncated payload: {len(raw_payload)} != {payload_len}")

        pkt = cls()
        pkt.type = ptype
        pkt.flags = flags
        pkt.node_id = node_id
        pkt.signature = signature

        # Decompress if needed (signature is over compressed bytes)
        if flags & Flags.COMPRESSED:
            pkt.payload = zlib.decompress(raw_payload)
            pkt._raw_payload = raw_payload  # keep for signature verification
        else:
            pkt.payload = raw_payload
            pkt._raw_payload = raw_payload

        return pkt

    def verify(self, known_pubkeys: dict = None) -> bool:
        """Verifica la firma del paquete."""
        if not (self.flags & Flags.SIGNED):
            return True  # no firmado, aceptar
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.exceptions import InvalidSignature
        # Use the node_id as the public key (32 bytes raw)
        pubkey_bytes = self.node_id
        # If we have a known pubkey for this node, prefer it
        if known_pubkeys and self.node_id in known_pubkeys:
            pubkey_bytes = known_pubkeys[self.node_id]
        try:
            pk = Ed25519PublicKey.from_public_bytes(pubkey_bytes)
            pk.verify(self.signature, self._raw_payload)
            return True
        except (InvalidSignature, Exception):
            return False

    def payload_json(self) -> dict:
        """Decodifica el payload como JSON."""
        return json.loads(self.payload.decode("utf-8"))


# ─── LSPNode ──────────────────────────────────────────────────────────────────

class LSPNode:
    """
    Nodo LSP que puede conectar con peers.
    Gestiona UDP (feromonas, ping/pong) y TCP (handshake, gossip).
    """

    def __init__(self, identity: LSPIdentity,
                 feromon_port: int = 7337,
                 gossip_port: int = 7338):
        self.identity = identity
        self.feromon_port = feromon_port
        self.gossip_port = gossip_port
        self._peers: Dict[str, dict] = {}  # node_id_hex -> {host, feromon_port, gossip_port, ...}
        self._running = False
        self._udp_sock: Optional[socket.socket] = None
        self._tcp_sock: Optional[socket.socket] = None
        self._threads: List[threading.Thread] = []
        self._feromon_callbacks: List[Callable] = []
        self._peer_callbacks: List[Callable] = []
        self._step = 0
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Levanta UDP + TCP listeners."""
        if self._running:
            return
        self._running = True
        self._start_udp()
        self._start_tcp()
        log.info(f"LSPNode started | id={self.identity.node_id_hex[:16]}... "
                 f"| UDP:{self.feromon_port} TCP:{self.gossip_port}")

    def stop(self):
        """Para el nodo limpiamente."""
        self._running = False
        if self._udp_sock:
            try:
                self._udp_sock.close()
            except Exception:
                pass
        if self._tcp_sock:
            try:
                self._tcp_sock.close()
            except Exception:
                pass
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        log.info("LSPNode stopped")

    def connect_peer(self, host: str, port: int):
        """Handshake TCP con un peer y lo registra."""
        try:
            payload = self._make_handshake_payload()
            pkt = LSPPacket.create(PacketType.HANDSHAKE, payload, compress=False)
            data = pkt.pack(self.identity)
            with socket.create_connection((host, port), timeout=5.0) as s:
                # Send length-prefixed
                s.sendall(struct.pack("<I", len(data)) + data)
                # Read response
                resp_len_bytes = s.recv(4)
                if len(resp_len_bytes) == 4:
                    resp_len = struct.unpack("<I", resp_len_bytes)[0]
                    resp_data = self._recv_exact(s, resp_len)
                    if resp_data:
                        resp_pkt = LSPPacket.unpack(resp_data)
                        if resp_pkt.verify():
                            info = json.loads(resp_pkt.payload)
                            self._register_peer(resp_pkt.node_id.hex(), host,
                                                info.get("feromon_port", self.feromon_port),
                                                info.get("gossip_port", self.gossip_port))
            log.info(f"Connected peer {host}:{port}")
        except Exception as e:
            log.warning(f"connect_peer {host}:{port} failed: {e}")

    def send_feromon(self, feromon, fitness: float = 0.5):
        """Envía tensor de feromona a todos los peers conocidos."""
        try:
            import torch
            if isinstance(feromon, torch.Tensor):
                feromon_list = feromon.cpu().float().tolist()
            else:
                feromon_list = list(feromon)
        except ImportError:
            feromon_list = list(feromon)

        self._step += 1
        payload_dict = {
            "feromon": feromon_list,
            "fitness": float(fitness),
            "step": self._step,
            "ttl": 3,
            "timestamp": time.time(),
        }
        payload = json.dumps(payload_dict).encode("utf-8")
        pkt = LSPPacket.create(PacketType.FEROMON, payload, compress=True)
        data = pkt.pack(self.identity)

        if len(data) > MAX_UDP_SIZE:
            log.warning(f"Feromon packet too large for UDP: {len(data)} bytes")
            return

        with self._lock:
            peers = list(self._peers.values())

        for peer in peers:
            try:
                self._udp_sock.sendto(data, (peer["host"], peer["feromon_port"]))
            except Exception as e:
                log.debug(f"send_feromon to {peer['host']}: {e}")

    def send_ping(self, host: str, port: int):
        """Envía PING UDP para descubrimiento."""
        payload_dict = {
            "version": "1.0",
            "node_id": self.identity.node_id_hex,
            "feromon_port": self.feromon_port,
            "gossip_port": self.gossip_port,
            "timestamp": time.time(),
        }
        payload = json.dumps(payload_dict).encode("utf-8")
        pkt = LSPPacket.create(PacketType.PING, payload, compress=False)
        data = pkt.pack(self.identity)
        try:
            self._udp_sock.sendto(data, (host, port))
            log.debug(f"PING sent to {host}:{port}")
        except Exception as e:
            log.warning(f"send_ping {host}:{port}: {e}")

    def on_feromon_received(self, callback: Callable):
        """Registra callback(feromon_tensor_or_list, from_node_id_hex)."""
        if callable(callback):
            self._feromon_callbacks.append(callback)
        return callback  # allows use as decorator

    def on_peer_connected(self, callback: Callable):
        """Registra callback(node_id_hex, host, port)."""
        if callable(callback):
            self._peer_callbacks.append(callback)
        return callback  # allows use as decorator

    def peers(self) -> List[dict]:
        with self._lock:
            return list(self._peers.values())

    # ── Internal ──────────────────────────────────────────────────────────────

    def _start_udp(self):
        self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._udp_sock.bind(("0.0.0.0", self.feromon_port))
        self._udp_sock.settimeout(1.0)
        t = threading.Thread(target=self._udp_loop, daemon=True, name="lsp-udp")
        t.start()
        self._threads.append(t)

    def _start_tcp(self):
        self._tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._tcp_sock.bind(("0.0.0.0", self.gossip_port))
        self._tcp_sock.listen(16)
        self._tcp_sock.settimeout(1.0)
        t = threading.Thread(target=self._tcp_loop, daemon=True, name="lsp-tcp")
        t.start()
        self._threads.append(t)

    def _udp_loop(self):
        while self._running:
            try:
                data, addr = self._udp_sock.recvfrom(MAX_UDP_SIZE)
                self._handle_udp(data, addr)
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as e:
                log.debug(f"UDP loop error: {e}")

    def _tcp_loop(self):
        while self._running:
            try:
                conn, addr = self._tcp_sock.accept()
                t = threading.Thread(target=self._handle_tcp_conn,
                                     args=(conn, addr), daemon=True)
                t.start()
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as e:
                log.debug(f"TCP loop error: {e}")

    def _handle_udp(self, data: bytes, addr):
        try:
            pkt = LSPPacket.unpack(data)
            if not pkt.verify():
                log.debug(f"Invalid signature from {addr}")
                return

            if pkt.type == PacketType.FEROMON:
                info = pkt.payload_json()
                feromon = info["feromon"]
                try:
                    import torch
                    feromon = torch.tensor(feromon, dtype=torch.float32)
                except ImportError:
                    pass
                node_id_hex = pkt.node_id.hex()
                for cb in self._feromon_callbacks:
                    try:
                        cb(feromon, node_id_hex)
                    except Exception as e:
                        log.debug(f"feromon callback error: {e}")

            elif pkt.type == PacketType.PING:
                info = pkt.payload_json()
                remote_node_id = info.get("node_id", pkt.node_id.hex())
                remote_feromon_port = info.get("feromon_port", addr[1])
                remote_gossip_port  = info.get("gossip_port", self.gossip_port)
                self._register_peer(remote_node_id, addr[0],
                                    remote_feromon_port, remote_gossip_port)
                # Send PONG
                self._send_pong(addr[0], addr[1])

            elif pkt.type == PacketType.PONG:
                info = pkt.payload_json()
                remote_node_id = info.get("node_id", pkt.node_id.hex())
                self._register_peer(remote_node_id, addr[0],
                                    info.get("feromon_port", self.feromon_port),
                                    info.get("gossip_port", self.gossip_port))

        except Exception as e:
            log.debug(f"_handle_udp error from {addr}: {e}")

    def _handle_tcp_conn(self, conn: socket.socket, addr):
        try:
            conn.settimeout(10.0)
            len_bytes = conn.recv(4)
            if len(len_bytes) < 4:
                return
            pkt_len = struct.unpack("<I", len_bytes)[0]
            data = self._recv_exact(conn, pkt_len)
            if not data:
                return
            pkt = LSPPacket.unpack(data)
            if not pkt.verify():
                log.debug(f"Invalid signature from TCP {addr}")
                return

            if pkt.type == PacketType.HANDSHAKE:
                info = pkt.payload_json()
                remote_node_id = info.get("node_id", pkt.node_id.hex())
                self._register_peer(remote_node_id, addr[0],
                                    info.get("feromon_port", self.feromon_port),
                                    info.get("gossip_port", self.gossip_port))
                # Send response handshake
                resp_payload = self._make_handshake_payload()
                resp_pkt = LSPPacket.create(PacketType.HANDSHAKE, resp_payload, compress=False)
                resp_data = resp_pkt.pack(self.identity)
                conn.sendall(struct.pack("<I", len(resp_data)) + resp_data)

        except Exception as e:
            log.debug(f"_handle_tcp_conn error from {addr}: {e}")
        finally:
            conn.close()

    def _send_pong(self, host: str, port: int):
        payload_dict = {
            "version": "1.0",
            "node_id": self.identity.node_id_hex,
            "feromon_port": self.feromon_port,
            "gossip_port": self.gossip_port,
            "timestamp": time.time(),
        }
        payload = json.dumps(payload_dict).encode("utf-8")
        pkt = LSPPacket.create(PacketType.PONG, payload, compress=False)
        data = pkt.pack(self.identity)
        try:
            self._udp_sock.sendto(data, (host, port))
        except Exception as e:
            log.debug(f"_send_pong to {host}:{port}: {e}")

    def _make_handshake_payload(self) -> bytes:
        return json.dumps({
            "version": "1.0",
            "node_id": self.identity.node_id_hex,
            "feromon_port": self.feromon_port,
            "gossip_port": self.gossip_port,
            "swarm_step": self._step,
            "capabilities": ["feromon", "gossip"],
        }).encode("utf-8")

    def _register_peer(self, node_id_hex: str, host: str,
                        feromon_port: int, gossip_port: int):
        # Don't register ourselves
        if node_id_hex == self.identity.node_id_hex:
            return
        with self._lock:
            is_new = node_id_hex not in self._peers
            self._peers[node_id_hex] = {
                "node_id": node_id_hex,
                "host": host,
                "feromon_port": feromon_port,
                "gossip_port": gossip_port,
                "last_seen": time.time(),
            }
        if is_new:
            log.info(f"Peer registered: {node_id_hex[:16]}...@{host}:{feromon_port}")
            for cb in self._peer_callbacks:
                try:
                    cb(node_id_hex, host, feromon_port)
                except Exception as e:
                    log.debug(f"peer_connected callback error: {e}")

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
        """Lee exactamente n bytes del socket."""
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf
