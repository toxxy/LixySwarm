"""
LixySwarm Network — Capa de Transporte
========================================
UDP para feromonas (tolerante a pérdida, baja latencia)
TCP para gossip de Matriarca (confiable)
mDNS para descubrimiento LAN (zero-config)
"""
import socket
import struct
import threading
import time
import logging
from typing import Callable, Optional, List

from .messages import FeromonMessage, GossipMessage
from .node import NodeIdentity, Peer, PeerTable

log = logging.getLogger("lixy.network")

MDNS_GROUP = "224.0.0.251"
MDNS_PORT = 5353
LIXY_SERVICE = "_lixyswarm._udp.local"


class FeromonUDP:
    """
    Transporte UDP para feromonas.
    Envío: fire-and-forget (no importa si se pierde una feromona)
    Recepción: callback asíncrono en thread dedicado
    """

    def __init__(self, identity: NodeIdentity, on_receive: Callable[[FeromonMessage, str], None]):
        self.identity = identity
        self.on_receive = on_receive
        self._sock: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Abre socket UDP y arranca thread de recepción."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1) if hasattr(socket, 'SO_REUSEPORT') else None
        self._sock.bind(("0.0.0.0", self.identity.feromon_port))
        self._sock.settimeout(1.0)
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True, name="feromon-udp")
        self._thread.start()
        log.info(f"FeromonUDP escuchando en :{self.identity.feromon_port}")

    def stop(self):
        self._running = False
        if self._sock:
            self._sock.close()
        if self._thread:
            self._thread.join(timeout=2)

    def send(self, peer: Peer, msg: FeromonMessage):
        """Envía feromona a un peer (UDP, fire-and-forget)."""
        if self._sock is None:
            return
        try:
            data = msg.pack()
            self._sock.sendto(data, peer.feromon_addr)
        except Exception as e:
            log.debug(f"FeromonUDP send error: {e}")

    def broadcast_lan(self, msg: FeromonMessage, peers: List[Peer]):
        """Envía feromona a todos los peers conocidos."""
        for peer in peers:
            self.send(peer, msg)

    def _recv_loop(self):
        while self._running:
            try:
                data, addr = self._sock.recvfrom(4096)
                msg = FeromonMessage.unpack(data)
                if msg and msg.valid and msg.is_fresh() and msg.node_id != self.identity.node_id:
                    self.on_receive(msg, addr[0])
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    log.debug(f"FeromonUDP recv error: {e}")


class GossipTCP:
    """
    Transporte TCP para gossip de Matriarca y descubrimiento.
    Cada conexión es corta (connect → send → recv → close).
    """

    def __init__(self, identity: NodeIdentity, on_message: Callable[[GossipMessage, str], Optional[GossipMessage]]):
        self.identity = identity
        self.on_message = on_message  # retorna respuesta opcional
        self._server: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("0.0.0.0", self.identity.gossip_port))
        self._server.listen(10)
        self._server.settimeout(1.0)
        self._running = True
        self._thread = threading.Thread(target=self._accept_loop, daemon=True, name="gossip-tcp")
        self._thread.start()
        log.info(f"GossipTCP escuchando en :{self.identity.gossip_port}")

    def stop(self):
        self._running = False
        if self._server:
            self._server.close()
        if self._thread:
            self._thread.join(timeout=2)

    def send(self, peer: Peer, msg: GossipMessage, recv_response: bool = False) -> Optional[GossipMessage]:
        """Envía un mensaje gossip a un peer y opcionalmente espera respuesta."""
        try:
            with socket.create_connection(peer.gossip_addr, timeout=5) as conn:
                conn.sendall(msg.to_bytes())
                if recv_response:
                    return self._recv_msg(conn)
        except Exception as e:
            log.debug(f"GossipTCP send error to {peer}: {e}")
        return None

    def _recv_msg(self, conn: socket.socket) -> Optional[GossipMessage]:
        """Lee un mensaje TCP con length-prefix framing."""
        try:
            header = self._recvall(conn, 4)
            if not header:
                return None
            length = struct.unpack("!I", header)[0]
            if length > 10 * 1024 * 1024:  # max 10MB
                return None
            body = self._recvall(conn, length)
            if not body:
                return None
            return GossipMessage.from_bytes(header + body)
        except Exception:
            return None

    def _recvall(self, conn: socket.socket, n: int) -> Optional[bytes]:
        data = b""
        while len(data) < n:
            chunk = conn.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def _accept_loop(self):
        while self._running:
            try:
                conn, addr = self._server.accept()
                t = threading.Thread(
                    target=self._handle_conn,
                    args=(conn, addr[0]),
                    daemon=True,
                )
                t.start()
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    log.debug(f"GossipTCP accept error: {e}")

    def _handle_conn(self, conn: socket.socket, addr: str):
        try:
            msg = self._recv_msg(conn)
            if msg:
                response = self.on_message(msg, addr)
                if response:
                    conn.sendall(response.to_bytes())
        except Exception as e:
            log.debug(f"GossipTCP handle error: {e}")
        finally:
            conn.close()


class MDNSDiscovery:
    """
    Descubrimiento de nodos en LAN via mDNS multicast.
    Zero-config — no requiere servidor central.
    Anuncia presencia y escucha anuncios de otros nodos.
    """
    ANNOUNCE_INTERVAL = 10   # segundos entre anuncios
    MULTICAST_TTL = 2

    def __init__(self, identity: NodeIdentity, on_peer_found: Callable[[Peer], None]):
        self.identity = identity
        self.on_peer_found = on_peer_found
        self._running = False
        self._announce_thread: Optional[threading.Thread] = None
        self._listen_thread: Optional[threading.Thread] = None
        self._recv_sock: Optional[socket.socket] = None

    def start(self):
        """Arranca anuncios periódicos y escucha pasiva."""
        self._running = True

        # Socket multicast para recibir
        try:
            self._recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            self._recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1) if hasattr(socket, 'SO_REUSEPORT') else None
            self._recv_sock.bind(("", MDNS_PORT))
            group = socket.inet_aton(MDNS_GROUP)
            mreq = struct.pack("4sL", group, socket.INADDR_ANY)
            self._recv_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            self._recv_sock.settimeout(1.0)
            self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True, name="mdns-listen")
            self._listen_thread.start()
        except Exception as e:
            log.warning(f"mDNS listen no disponible: {e}")

        self._announce_thread = threading.Thread(target=self._announce_loop, daemon=True, name="mdns-announce")
        self._announce_thread.start()
        log.info(f"mDNS activo — anunciando cada {self.ANNOUNCE_INTERVAL}s")

    def stop(self):
        self._running = False
        if self._recv_sock:
            self._recv_sock.close()

    def _make_announce(self) -> bytes:
        """Payload del anuncio mDNS: JSON con identidad del nodo."""
        import json
        msg = {
            "service": LIXY_SERVICE,
            "node_id": self.identity.node_id,
            "host": self.identity.host,
            "feromon_port": self.identity.feromon_port,
            "gossip_port": self.identity.gossip_port,
            "ts": time.time(),
        }
        return json.dumps(msg).encode("utf-8")

    def _announce_loop(self):
        """Envía anuncios periódicos al grupo multicast."""
        send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        send_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, self.MULTICAST_TTL)
        while self._running:
            try:
                data = self._make_announce()
                send_sock.sendto(data, (MDNS_GROUP, MDNS_PORT))
            except Exception as e:
                log.debug(f"mDNS announce error: {e}")
            time.sleep(self.ANNOUNCE_INTERVAL)
        send_sock.close()

    def _listen_loop(self):
        """Escucha anuncios mDNS de otros nodos."""
        import json
        while self._running:
            try:
                data, addr = self._recv_sock.recvfrom(4096)
                msg = json.loads(data.decode("utf-8"))
                if msg.get("service") != LIXY_SERVICE:
                    continue
                if msg.get("node_id") == self.identity.node_id:
                    continue  # soy yo mismo
                peer = Peer(
                    node_id=msg["node_id"],
                    host=msg.get("host", addr[0]),
                    feromon_port=msg.get("feromon_port", 7337),
                    gossip_port=msg.get("gossip_port", 7338),
                )
                log.info(f"🌐 Peer descubierto via mDNS: {peer}")
                self.on_peer_found(peer)
            except socket.timeout:
                continue
            except Exception as e:
                log.debug(f"mDNS listen error: {e}")
