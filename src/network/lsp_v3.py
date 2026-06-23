"""LixySwarm Protocol v3: persistent, signed, outbound-friendly P2P sessions.

LSP v3 intentionally keeps bootstrap separate from data transport. DNS seeds
only provide initial addresses; peer exchange and persistent outbound sessions
allow the network to continue when every seed is offline.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import struct
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .lsp import LSPIdentity
from .lsp_v2 import FeromonV2Payload
from .peer_manager import AddressBook, PeerManager, PeerReputation, network_group


log = logging.getLogger("lixy.network.lsp3")

MAGIC = b"LYS3"
VERSION = 3
SIGNED_FLAG = 0x01
ENCRYPTED_FLAG = 0x02
DEFAULT_NETWORK_ID = b"LIXYMAIN"
MAX_PAYLOAD_SIZE = 1024 * 1024
MAX_CLOCK_SKEW_MS = 5 * 60 * 1000
MAX_MESSAGES_PER_WINDOW = 500
MAX_BYTES_PER_WINDOW = 8 * 1024 * 1024
RATE_WINDOW_S = 10.0
MAX_PEERS_PER_MESSAGE = 100
DEFAULT_MAX_CONNECTIONS = 64
DEFAULT_MAX_INBOUND_PER_IP = 4

# magic, version, type, flags, ttl, payload_len, timestamp_ms, sequence,
# session_id, message_id, sender public key, network id
_PREFIX = struct.Struct("!4sBBBBIQQ16s16s32s8s")
SIGNATURE_SIZE = 64
HEADER_SIZE = _PREFIX.size + SIGNATURE_SIZE
MAX_FRAME_SIZE = HEADER_SIZE + MAX_PAYLOAD_SIZE


class PacketType:
    HELLO = 0x01
    PEERS = 0x02
    PHEROMONE = 0x03
    GLOBAL_DELTA = 0x04
    PING = 0x05
    PONG = 0x06
    WORK_OFFER = 0x07
    WORK_RESULT = 0x08


_VALID_PACKET_TYPES = {
    PacketType.HELLO,
    PacketType.PEERS,
    PacketType.PHEROMONE,
    PacketType.GLOBAL_DELTA,
    PacketType.PING,
    PacketType.PONG,
    PacketType.WORK_OFFER,
    PacketType.WORK_RESULT,
}


class ProtocolError(ValueError):
    pass


@dataclass
class V3Packet:
    packet_type: int
    payload: bytes
    ttl: int
    timestamp_ms: int
    sequence: int
    session_id: bytes
    message_id: bytes
    sender_id: bytes
    network_id: bytes = DEFAULT_NETWORK_ID
    flags: int = SIGNED_FLAG
    signature: bytes = b""

    @classmethod
    def create(
        cls,
        packet_type: int,
        payload: bytes,
        *,
        sequence: int,
        session_id: bytes,
        ttl: int = 8,
        network_id: bytes = DEFAULT_NETWORK_ID,
    ) -> "V3Packet":
        if packet_type not in _VALID_PACKET_TYPES:
            raise ProtocolError("Unknown LSP v3 packet type")
        if not isinstance(payload, bytes) or len(payload) > MAX_PAYLOAD_SIZE:
            raise ProtocolError("LSP v3 payload exceeds the maximum size")
        if len(session_id) != 16:
            raise ProtocolError("LSP v3 session ID must be 16 bytes")
        if len(network_id) != 8:
            raise ProtocolError("LSP v3 network ID must be 8 bytes")
        return cls(
            packet_type=packet_type,
            payload=payload,
            ttl=max(0, min(int(ttl), 255)),
            timestamp_ms=int(time.time() * 1000),
            sequence=int(sequence),
            session_id=session_id,
            message_id=os.urandom(16),
            sender_id=b"\x00" * 32,
            network_id=network_id,
        )

    def _prefix(self, payload_len: Optional[int] = None) -> bytes:
        return _PREFIX.pack(
            MAGIC,
            VERSION,
            self.packet_type,
            self.flags,
            self.ttl,
            len(self.payload) if payload_len is None else int(payload_len),
            self.timestamp_ms,
            self.sequence,
            self.session_id,
            self.message_id,
            self.sender_id,
            self.network_id,
        )

    def pack(self, identity: LSPIdentity) -> bytes:
        self.flags |= SIGNED_FLAG
        self.sender_id = identity.node_id
        prefix = self._prefix()
        self.signature = identity.sign(prefix + self.payload)
        return prefix + self.signature + self.payload

    @classmethod
    def unpack(
        cls,
        data: bytes,
        *,
        expected_network_id: bytes = DEFAULT_NETWORK_ID,
        now_ms: Optional[int] = None,
    ) -> "V3Packet":
        if len(data) < HEADER_SIZE:
            raise ProtocolError("Truncated LSP v3 packet")
        (
            magic,
            version,
            packet_type,
            flags,
            ttl,
            payload_len,
            timestamp_ms,
            sequence,
            session_id,
            message_id,
            sender_id,
            network_id,
        ) = _PREFIX.unpack_from(data, 0)

        if magic != MAGIC or version != VERSION:
            raise ProtocolError("Unsupported LSP v3 envelope")
        if network_id != expected_network_id:
            raise ProtocolError("LSP v3 network ID mismatch")
        if packet_type not in _VALID_PACKET_TYPES:
            raise ProtocolError("Unknown LSP v3 packet type")
        if not (flags & SIGNED_FLAG):
            raise ProtocolError("Unsigned LSP v3 packet")
        if payload_len > MAX_PAYLOAD_SIZE:
            raise ProtocolError("LSP v3 payload exceeds the maximum size")
        if len(data) != HEADER_SIZE + payload_len:
            raise ProtocolError("LSP v3 payload length mismatch")
        if sequence <= 0 or session_id == b"\x00" * 16 or message_id == b"\x00" * 16:
            raise ProtocolError("Invalid LSP v3 replay fields")

        current_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
        if abs(current_ms - timestamp_ms) > MAX_CLOCK_SKEW_MS:
            raise ProtocolError("Stale LSP v3 packet")

        signature_offset = _PREFIX.size
        signature = data[signature_offset:signature_offset + SIGNATURE_SIZE]
        payload = data[HEADER_SIZE:]
        prefix = data[:_PREFIX.size]
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            from cryptography.exceptions import InvalidSignature

            Ed25519PublicKey.from_public_bytes(sender_id).verify(
                signature, prefix + payload
            )
        except (ValueError, InvalidSignature) as exc:
            raise ProtocolError("Invalid LSP v3 signature") from exc

        return cls(
            packet_type=packet_type,
            payload=payload,
            ttl=ttl,
            timestamp_ms=timestamp_ms,
            sequence=sequence,
            session_id=session_id,
            message_id=message_id,
            sender_id=sender_id,
            network_id=network_id,
            flags=flags,
            signature=signature,
        )


class ReplayGuard:
    """Bounded message-ID cache plus monotonic sequence enforcement."""

    def __init__(self, max_messages: int = 50_000):
        self.max_messages = max(100, int(max_messages))
        self._seen: OrderedDict[tuple[bytes, bytes], float] = OrderedDict()
        self._max_sequence: dict[tuple[bytes, bytes], int] = {}

    def accept(self, packet: V3Packet) -> bool:
        message_key = (packet.sender_id, packet.message_id)
        sequence_key = (packet.sender_id, packet.session_id)
        if message_key in self._seen:
            return False
        if packet.sequence <= self._max_sequence.get(sequence_key, 0):
            return False

        self._seen[message_key] = time.time()
        self._seen.move_to_end(message_key)
        self._max_sequence[sequence_key] = packet.sequence
        while len(self._seen) > self.max_messages:
            self._seen.popitem(last=False)
        return True


@dataclass
class PeerSession:
    node_id: str
    host: str
    port: int
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    outbound: bool
    hello: dict
    encryption_key: bytes = field(repr=False)
    connected_at: float = field(default_factory=time.time)
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    task: Optional[asyncio.Task] = None
    _message_times: deque = field(default_factory=deque)
    _byte_events: deque = field(default_factory=deque)

    def allow_frame(self, frame_size: int) -> bool:
        now = time.monotonic()
        threshold = now - RATE_WINDOW_S
        while self._message_times and self._message_times[0] < threshold:
            self._message_times.popleft()
        while self._byte_events and self._byte_events[0][0] < threshold:
            self._byte_events.popleft()
        if len(self._message_times) >= MAX_MESSAGES_PER_WINDOW:
            return False
        if sum(size for _, size in self._byte_events) + frame_size > MAX_BYTES_PER_WINDOW:
            return False
        self._message_times.append(now)
        self._byte_events.append((now, frame_size))
        return True

    async def send_raw(self, raw: bytes):
        if self.writer.is_closing():
            raise ConnectionError("Peer session is closed")
        framed = struct.pack("!I", len(raw)) + raw
        async with self.write_lock:
            self.writer.write(framed)
            await self.writer.drain()


class LSPNodeV3:
    """Persistent LSP v3 node with outbound bootstrap and peer exchange."""

    def __init__(
        self,
        identity: LSPIdentity,
        *,
        host: str = "0.0.0.0",
        port: int = 7338,
        advertised_host: Optional[str] = None,
        seeds: Optional[list[tuple[str, int]]] = None,
        address_book_path: str | Path = "checkpoints/peers_v3.json",
        reputation_path: str | Path | None = None,
        target_outbound: int = 8,
        allow_private: bool = False,
        capabilities: Optional[dict] = None,
        resource_profile: Optional[dict] = None,
        network_id: bytes = DEFAULT_NETWORK_ID,
        maintenance_interval: float = 2.0,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        max_inbound_per_ip: int = DEFAULT_MAX_INBOUND_PER_IP,
    ):
        self.identity = identity
        self.host = host
        self.port = int(port)
        self.advertised_host = advertised_host
        self.network_id = network_id
        self.capabilities = dict(capabilities or {
            "pheromone": True,
            "global_memory": True,
            "peer_exchange": True,
        })
        self.resource_profile = dict(resource_profile or {})
        self.max_connections = max(1, min(int(max_connections), 4096))
        self.max_inbound_per_ip = max(1, min(int(max_inbound_per_ip), 256))
        self.session_id = b""
        self._key_agreement_private = None
        self._key_agreement_public = b""
        self._reset_process_session()

        self.replay_guard = ReplayGuard()
        self.address_book = AddressBook(
            address_book_path, allow_private=allow_private
        )
        self.reputation = PeerReputation(
            reputation_path
            or Path(address_book_path).with_name("peer_reputation_v3.json")
        )
        self.peer_manager = PeerManager(
            self,
            address_book=self.address_book,
            seeds=seeds,
            target_outbound=target_outbound,
            maintenance_interval=maintenance_interval,
        )

        self._sequence = 0
        self._sessions: dict[str, PeerSession] = {}
        self._snapshot: dict[str, dict] = {}
        self._snapshot_lock = threading.RLock()
        self._server: Optional[asyncio.AbstractServer] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._startup_error: Optional[BaseException] = None

        self._peer_connected_callbacks: list[Callable] = []
        self._peer_lost_callbacks: list[Callable] = []
        self._pheromone_callbacks: list[Callable] = []
        self._global_delta_callbacks: list[Callable] = []
        self._work_offer_callbacks: list[Callable] = []
        self._work_result_callbacks: list[Callable] = []

    def _reset_process_session(self):
        self.session_id = os.urandom(16)
        self._key_agreement_private = x25519.X25519PrivateKey.generate()
        self._key_agreement_public = self._key_agreement_private.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )

    # Public lifecycle -----------------------------------------------------

    def start(self, timeout: float = 10.0):
        if self._thread and self._thread.is_alive():
            return
        self._sequence = 0
        self.replay_guard = ReplayGuard()
        self._reset_process_session()
        self._ready.clear()
        self._startup_error = None
        self._thread = threading.Thread(
            target=self._thread_main,
            daemon=True,
            name=f"lsp-v3-{self.port}",
        )
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("LSP v3 startup timed out")
        if self._startup_error:
            raise RuntimeError("LSP v3 failed to start") from self._startup_error

    def stop(self, timeout: float = 10.0):
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                raise TimeoutError("LSP v3 shutdown timed out")
        self._thread = None
        self._loop = None
        self._key_agreement_private = None
        self._key_agreement_public = b""

    def connect_peer(self, host: str, port: int, timeout: float = 10.0) -> bool:
        if not self._loop:
            return False
        future = asyncio.run_coroutine_threadsafe(
            self._connect_address(host, int(port)), self._loop
        )
        return bool(future.result(timeout=timeout))

    def broadcast_feromon(self, feromon, *, fitness: float = 0.5, step: int = 0) -> int:
        payload = FeromonV2Payload(
            feromon=feromon,
            ttl=8,
            step=step,
            fitness=float(fitness),
            timestamp_ms=int(time.time() * 1000),
        ).pack()
        return self._broadcast_sync(PacketType.PHEROMONE, payload, ttl=8)

    def broadcast_global_delta(self, delta: dict) -> int:
        if not isinstance(delta, dict):
            return 0
        payload = json.dumps(delta, ensure_ascii=False, separators=(",", ":")).encode()
        return self._broadcast_sync(PacketType.GLOBAL_DELTA, payload, ttl=8)

    def ping(self) -> int:
        payload = json.dumps({"sent_at": time.time()}).encode()
        return self._broadcast_sync(PacketType.PING, payload, ttl=1)

    def send_work_offer(self, node_id: str, offer: dict) -> bool:
        return self._send_json_to_peer(PacketType.WORK_OFFER, node_id, offer)

    def send_work_result(self, node_id: str, result: dict) -> bool:
        return self._send_json_to_peer(PacketType.WORK_RESULT, node_id, result)

    def peers(self) -> list[dict]:
        with self._snapshot_lock:
            return [dict(peer) for peer in self._snapshot.values()]

    @property
    def peer_count(self) -> int:
        with self._snapshot_lock:
            return len(self._snapshot)

    @property
    def outbound_count(self) -> int:
        with self._snapshot_lock:
            return sum(1 for peer in self._snapshot.values() if peer["outbound"])

    @property
    def connected_node_ids(self) -> set[str]:
        with self._snapshot_lock:
            return set(self._snapshot)

    @property
    def connected_network_groups(self) -> set[str]:
        with self._snapshot_lock:
            return {
                network_group(str(peer["host"]))
                for peer in self._snapshot.values()
            }

    def is_address_connected(self, host: str, port: int) -> bool:
        with self._snapshot_lock:
            return any(
                peer["host"] == host and int(peer["port"]) == int(port)
                for peer in self._snapshot.values()
            )

    # Callback registration ------------------------------------------------

    def on_peer_connected(self, callback: Callable):
        self._peer_connected_callbacks.append(callback)
        return callback

    def on_peer_lost(self, callback: Callable):
        self._peer_lost_callbacks.append(callback)
        return callback

    def on_feromon_received(self, callback: Callable):
        self._pheromone_callbacks.append(callback)
        return callback

    def on_gossip_delta_received(self, callback: Callable):
        self._global_delta_callbacks.append(callback)
        return callback

    def on_work_offer_received(self, callback: Callable):
        self._work_offer_callbacks.append(callback)
        return callback

    def on_work_result_received(self, callback: Callable):
        self._work_result_callbacks.append(callback)
        return callback

    # Event-loop lifecycle --------------------------------------------------

    def _thread_main(self):
        try:
            asyncio.run(self._run())
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            log.exception("LSP v3 event loop failed")

    async def _run(self):
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        try:
            self._server = await asyncio.start_server(
                self._handle_inbound, self.host, self.port
            )
            sockets = self._server.sockets or []
            if sockets:
                self.port = int(sockets[0].getsockname()[1])
            await self.peer_manager.start()
            self._ready.set()
            await self._stop_event.wait()
        finally:
            await self.peer_manager.stop()
            if self._server:
                self._server.close()
            for session in list(self._sessions.values()):
                await self._close_session(session, notify=False)
            self._sessions.clear()
            if self._server:
                try:
                    await asyncio.wait_for(self._server.wait_closed(), timeout=2.0)
                except asyncio.TimeoutError:
                    log.debug("Timed out waiting for LSP v3 listener connections")
                self._server = None
            with self._snapshot_lock:
                self._snapshot.clear()

    # Handshake and session management -------------------------------------

    async def _handle_inbound(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        session: Optional[PeerSession] = None
        remote_host = "unknown"
        try:
            remote_host = str(writer.get_extra_info("peername")[0])
            if self.reputation.is_banned("ip", remote_host):
                writer.close()
                await writer.wait_closed()
                return
            if len(self._sessions) >= self.max_connections:
                raise ProtocolError("LSP v3 connection limit reached")
            inbound_from_host = sum(
                1 for current in self._sessions.values()
                if not current.outbound and current.host == remote_host
            )
            if inbound_from_host >= self.max_inbound_per_ip:
                raise ProtocolError("LSP v3 per-IP connection limit reached")
            packet = await asyncio.wait_for(self._read_packet(reader), timeout=10.0)
            if packet.packet_type != PacketType.HELLO or not self.replay_guard.accept(packet):
                raise ProtocolError("Expected a fresh LSP v3 HELLO")
            hello = self._parse_hello(packet.payload)
            if self.reputation.is_banned("node", packet.sender_id.hex()):
                raise ProtocolError("Peer identity is locally banned")
            remote_port = int(hello["listen_port"])
            session = PeerSession(
                node_id=packet.sender_id.hex(),
                host=remote_host,
                port=remote_port,
                reader=reader,
                writer=writer,
                outbound=False,
                hello=hello,
                encryption_key=self._derive_session_key(
                    hello, packet.sender_id, packet.session_id
                ),
                task=asyncio.current_task(),
            )
            if not await self._register_session(session):
                return
            await session.send_raw(self._new_packet(PacketType.HELLO, self._hello_payload(), ttl=1))
            await self._after_connected(session)
            await self._read_loop(session)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        except ProtocolError as exc:
            self.reputation.report("ip", remote_host)
            if session is not None:
                self.reputation.report("node", session.node_id)
            log.debug("Rejected inbound LSP v3 protocol: %s", exc)
        except Exception as exc:
            log.debug("Rejected inbound LSP v3 connection: %s", exc)
        finally:
            if session:
                await self._close_session(session)
            else:
                writer.close()
                try:
                    await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
                except Exception:
                    pass

    async def _connect_address(self, host: str, port: int) -> bool:
        if not self._loop or self.is_address_connected(host, port):
            return False
        if self.reputation.is_banned("ip", host):
            return False
        if port == self.port and host in {"127.0.0.1", "::1", "localhost", self.host}:
            return False
        writer: Optional[asyncio.StreamWriter] = None
        session: Optional[PeerSession] = None
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=5.0
            )
            await self._write_raw(writer, self._new_packet(
                PacketType.HELLO, self._hello_payload(), ttl=1
            ))
            packet = await asyncio.wait_for(self._read_packet(reader), timeout=10.0)
            if packet.packet_type != PacketType.HELLO or not self.replay_guard.accept(packet):
                raise ProtocolError("Peer returned an invalid LSP v3 HELLO")
            node_id = packet.sender_id.hex()
            if node_id == self.identity.node_id_hex:
                raise ProtocolError("Refusing self connection")
            if self.reputation.is_banned("node", node_id):
                raise ProtocolError("Peer identity is locally banned")
            hello = self._parse_hello(packet.payload)
            session = PeerSession(
                node_id=node_id,
                host=host,
                port=int(hello["listen_port"]),
                reader=reader,
                writer=writer,
                outbound=True,
                hello=hello,
                encryption_key=self._derive_session_key(
                    hello, packet.sender_id, packet.session_id
                ),
            )
            if not await self._register_session(session):
                return False
            await self._after_connected(session)
            session.task = asyncio.create_task(
                self._run_outbound_session(session),
                name=f"lsp-v3-read-{node_id[:8]}",
            )
            return True
        except ProtocolError as exc:
            self.reputation.report("ip", host)
            if session is not None:
                self.reputation.report("node", session.node_id)
            log.debug("LSP v3 protocol rejection %s:%s: %s", host, port, exc)
            if session:
                await self._close_session(session)
            elif writer:
                writer.close()
                try:
                    await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
                except Exception:
                    pass
            return False
        except Exception as exc:
            log.debug("LSP v3 connect %s:%s failed: %s", host, port, exc)
            if session:
                await self._close_session(session)
            elif writer:
                writer.close()
                try:
                    await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
                except Exception:
                    pass
            return False

    async def _register_session(self, session: PeerSession) -> bool:
        if session.node_id == self.identity.node_id_hex:
            await self._close_session(session, notify=False)
            return False
        existing = self._sessions.get(session.node_id)
        if existing is None and len(self._sessions) >= self.max_connections:
            await self._close_session(session, notify=False)
            return False
        if existing:
            prefer_outbound = self.identity.node_id_hex < session.node_id
            if existing.outbound == prefer_outbound or session.outbound != prefer_outbound:
                await self._close_session(session, notify=False)
                return False
            await self._close_session(existing)

        self._sessions[session.node_id] = session
        with self._snapshot_lock:
            self._snapshot[session.node_id] = {
                "node_id": session.node_id,
                "host": session.host,
                "port": session.port,
                "outbound": session.outbound,
                "connected_at": session.connected_at,
                "encrypted": True,
                "capabilities": dict(session.hello.get("capabilities", {})),
                "resources": dict(session.hello.get("resources", {})),
            }
        return True

    async def _after_connected(self, session: PeerSession):
        self.reputation.reward("ip", session.host)
        self.reputation.reward("node", session.node_id)
        self.peer_manager.connected(
            session.node_id,
            session.host,
            session.port,
            outbound=session.outbound,
        )
        for callback in self._peer_connected_callbacks:
            try:
                callback(session.node_id, session.host, session.port, dict(session.hello))
            except Exception as exc:
                log.debug("LSP v3 peer callback failed: %s", exc)
        peers = self.peer_manager.addresses_for_exchange(MAX_PEERS_PER_MESSAGE)
        if peers:
            payload = json.dumps({"peers": peers}, separators=(",", ":")).encode()
            await session.send_raw(self._new_packet(
                PacketType.PEERS,
                payload,
                ttl=1,
                encryption_key=session.encryption_key,
            ))

    async def _close_session(self, session: PeerSession, *, notify: bool = True):
        current = self._sessions.get(session.node_id)
        removed = current is session
        if removed:
            self._sessions.pop(session.node_id, None)
            with self._snapshot_lock:
                self._snapshot.pop(session.node_id, None)
        if not session.writer.is_closing():
            session.writer.close()
            try:
                await asyncio.wait_for(session.writer.wait_closed(), timeout=1.0)
            except Exception:
                pass
        session.encryption_key = b""
        if removed and notify:
            self.peer_manager.disconnected()
            for callback in self._peer_lost_callbacks:
                try:
                    callback(session.node_id)
                except Exception:
                    pass

    # Message processing ----------------------------------------------------

    async def _run_outbound_session(self, session: PeerSession):
        try:
            await self._read_loop(session)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        except ProtocolError as exc:
            self.reputation.report("ip", session.host)
            self.reputation.report("node", session.node_id)
            log.debug("Closing misbehaving LSP v3 peer %s: %s", session.node_id[:12], exc)
        except Exception as exc:
            log.debug("Closing LSP v3 peer %s: %s", session.node_id[:12], exc)
        finally:
            await self._close_session(session)

    async def _read_loop(self, session: PeerSession):
        while not session.writer.is_closing():
            packet = await self._read_packet(session.reader)
            if packet.sender_id.hex() != session.node_id:
                raise ProtocolError("Session sender identity changed")
            if not session.allow_frame(HEADER_SIZE + len(packet.payload)):
                raise ProtocolError("LSP v3 peer exceeded rate limits")
            if not self.replay_guard.accept(packet):
                raise ProtocolError("Replayed or out-of-order LSP v3 packet")
            if not (packet.flags & ENCRYPTED_FLAG):
                raise ProtocolError("Established LSP v3 packet is not encrypted")
            packet = self._decrypt_packet(packet, session.encryption_key)
            await self._process_packet(session, packet)

    async def _process_packet(self, session: PeerSession, packet: V3Packet):
        if packet.packet_type == PacketType.HELLO:
            raise ProtocolError("Unexpected HELLO on established session")
        if packet.packet_type == PacketType.PEERS:
            payload = self._parse_json(packet.payload, max_size=128 * 1024)
            peers = payload.get("peers", [])
            if not isinstance(peers, list) or len(peers) > MAX_PEERS_PER_MESSAGE:
                raise ProtocolError("Invalid LSP v3 peer list")
            filtered = [
                peer for peer in peers
                if isinstance(peer, dict)
                and str(peer.get("node_id", "")) != self.identity.node_id_hex
            ]
            self.peer_manager.add_discovered(filtered)
            return
        if packet.packet_type == PacketType.PHEROMONE:
            pheromone = FeromonV2Payload.unpack(packet.payload)
            size = pheromone.feromon.numel() if hasattr(pheromone.feromon, "numel") else len(pheromone.feromon)
            if size <= 0 or size > 4096:
                raise ProtocolError("Invalid LSP v3 pheromone dimensions")
            for callback in self._pheromone_callbacks:
                callback(pheromone.feromon, session.node_id)
            return
        if packet.packet_type == PacketType.GLOBAL_DELTA:
            delta = self._parse_json(packet.payload, max_size=MAX_PAYLOAD_SIZE)
            for callback in self._global_delta_callbacks:
                callback(delta, session.node_id)
            return
        if packet.packet_type == PacketType.PING:
            await session.send_raw(self._new_packet(
                PacketType.PONG,
                packet.payload,
                ttl=1,
                encryption_key=session.encryption_key,
            ))
            return
        if packet.packet_type == PacketType.PONG:
            return
        if packet.packet_type == PacketType.WORK_OFFER:
            offer = self._parse_json(packet.payload, max_size=256 * 1024)
            for callback in self._work_offer_callbacks:
                callback(offer, session.node_id)
            return
        if packet.packet_type == PacketType.WORK_RESULT:
            result = self._parse_json(packet.payload, max_size=256 * 1024)
            for callback in self._work_result_callbacks:
                callback(result, session.node_id)
            return

    # Serialization helpers -------------------------------------------------

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _new_packet(
        self,
        packet_type: int,
        payload: bytes,
        *,
        ttl: int,
        encryption_key: Optional[bytes] = None,
    ) -> bytes:
        packet = V3Packet.create(
            packet_type,
            payload,
            sequence=self._next_sequence(),
            session_id=self.session_id,
            ttl=ttl,
            network_id=self.network_id,
        )
        if encryption_key is not None:
            if packet_type == PacketType.HELLO:
                raise ProtocolError("LSP v3 HELLO cannot be encrypted")
            if len(payload) + 16 > MAX_PAYLOAD_SIZE:
                raise ProtocolError("Encrypted LSP v3 payload exceeds the maximum size")
            packet.flags |= ENCRYPTED_FLAG
            packet.sender_id = self.identity.node_id
            nonce = packet.session_id[:4] + packet.sequence.to_bytes(8, "big")
            aad = packet._prefix(payload_len=len(payload) + 16)
            packet.payload = ChaCha20Poly1305(encryption_key).encrypt(
                nonce, payload, aad
            )
        return packet.pack(self.identity)

    @staticmethod
    def _decrypt_packet(packet: V3Packet, encryption_key: bytes) -> V3Packet:
        if not (packet.flags & ENCRYPTED_FLAG):
            raise ProtocolError("Encrypted LSP v3 flag is missing")
        nonce = packet.session_id[:4] + packet.sequence.to_bytes(8, "big")
        try:
            packet.payload = ChaCha20Poly1305(encryption_key).decrypt(
                nonce, packet.payload, packet._prefix()
            )
        except Exception as exc:
            raise ProtocolError("Invalid LSP v3 encrypted payload") from exc
        return packet

    def _derive_session_key(
        self,
        hello: dict,
        remote_node_id: bytes,
        remote_session_id: bytes,
    ) -> bytes:
        try:
            encoded_key = hello["key_agreement"]["ephemeral_key"]
            remote_public_bytes = base64.b64decode(encoded_key, validate=True)
            remote_public = x25519.X25519PublicKey.from_public_bytes(
                remote_public_bytes
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("Invalid LSP v3 key agreement") from exc
        local = (self.identity.node_id, self.session_id)
        remote = (remote_node_id, remote_session_id)
        ordered = sorted((local, remote), key=lambda item: item[0])
        transcript = b"".join(node_id + session_id for node_id, session_id in ordered)
        shared_secret = self._key_agreement_private.exchange(remote_public)
        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.network_id,
            info=b"LixySwarm LSP v3 session encryption\x00" + transcript,
        ).derive(shared_secret)

    async def _read_packet(self, reader: asyncio.StreamReader) -> V3Packet:
        length_bytes = await reader.readexactly(4)
        length = struct.unpack("!I", length_bytes)[0]
        if length < HEADER_SIZE or length > MAX_FRAME_SIZE:
            raise ProtocolError("Invalid LSP v3 frame size")
        raw = await reader.readexactly(length)
        return V3Packet.unpack(raw, expected_network_id=self.network_id)

    @staticmethod
    async def _write_raw(writer: asyncio.StreamWriter, raw: bytes):
        if len(raw) > MAX_FRAME_SIZE:
            raise ProtocolError("LSP v3 frame exceeds the maximum size")
        writer.write(struct.pack("!I", len(raw)) + raw)
        await writer.drain()

    def _hello_payload(self) -> bytes:
        payload = {
            "protocol": 3,
            "listen_port": self.port,
            "capabilities": self.capabilities,
            "resources": self.resource_profile,
            "user_agent": "LixySwarm/3",
            "key_agreement": {
                "algorithm": "X25519-HKDF-SHA256+ChaCha20-Poly1305",
                "ephemeral_key": base64.b64encode(
                    self._key_agreement_public
                ).decode("ascii"),
            },
        }
        if self.advertised_host:
            payload["advertised_host"] = self.advertised_host
        return json.dumps(payload, separators=(",", ":")).encode()

    def _parse_hello(self, payload: bytes) -> dict:
        hello = self._parse_json(payload, max_size=16 * 1024)
        if hello.get("protocol") != 3:
            raise ProtocolError("Incompatible LSP protocol")
        try:
            port = int(hello["listen_port"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("HELLO is missing a valid listen port") from exc
        if not 0 < port < 65536:
            raise ProtocolError("HELLO listen port is out of range")
        if not isinstance(hello.get("capabilities", {}), dict):
            raise ProtocolError("HELLO capabilities must be an object")
        if not isinstance(hello.get("resources", {}), dict):
            raise ProtocolError("HELLO resources must be an object")
        key_agreement = hello.get("key_agreement")
        if (
            not isinstance(key_agreement, dict)
            or key_agreement.get("algorithm")
            != "X25519-HKDF-SHA256+ChaCha20-Poly1305"
        ):
            raise ProtocolError("HELLO key agreement is missing or unsupported")
        try:
            ephemeral = base64.b64decode(
                key_agreement.get("ephemeral_key", ""), validate=True
            )
        except (TypeError, ValueError) as exc:
            raise ProtocolError("HELLO ephemeral key is invalid") from exc
        if len(ephemeral) != 32:
            raise ProtocolError("HELLO ephemeral key is invalid")
        # Bound untrusted metadata even though the complete HELLO is also capped.
        hello["capabilities"] = dict(list(hello.get("capabilities", {}).items())[:32])
        hello["resources"] = dict(list(hello.get("resources", {}).items())[:32])
        return hello

    @staticmethod
    def _parse_json(payload: bytes, *, max_size: int) -> dict:
        if len(payload) > max_size:
            raise ProtocolError("LSP v3 JSON payload is too large")
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError("Invalid LSP v3 JSON") from exc
        if not isinstance(value, dict):
            raise ProtocolError("LSP v3 JSON payload must be an object")
        return value

    def _broadcast_sync(self, packet_type: int, payload: bytes, *, ttl: int) -> int:
        if not self._loop:
            return 0
        future = asyncio.run_coroutine_threadsafe(
            self._broadcast(packet_type, payload, ttl=ttl), self._loop
        )
        return int(future.result(timeout=15.0))

    def _send_json_to_peer(self, packet_type: int, node_id: str, value: dict) -> bool:
        if not self._loop or not isinstance(value, dict):
            return False
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
        if len(payload) > 256 * 1024:
            raise ProtocolError("Work payload exceeds 256 KiB")
        future = asyncio.run_coroutine_threadsafe(
            self._send_to_peer(packet_type, node_id, payload), self._loop
        )
        return bool(future.result(timeout=15.0))

    async def _send_to_peer(
        self, packet_type: int, node_id: str, payload: bytes
    ) -> bool:
        session = self._sessions.get(node_id)
        if session is None:
            return False
        try:
            await session.send_raw(self._new_packet(
                packet_type,
                payload,
                ttl=1,
                encryption_key=session.encryption_key,
            ))
            return True
        except Exception:
            await self._close_session(session)
            return False

    async def _broadcast(self, packet_type: int, payload: bytes, *, ttl: int) -> int:
        delivered = 0
        for session in list(self._sessions.values()):
            try:
                await session.send_raw(self._new_packet(
                    packet_type,
                    payload,
                    ttl=ttl,
                    encryption_key=session.encryption_key,
                ))
                delivered += 1
            except Exception:
                await self._close_session(session)
        return delivered
