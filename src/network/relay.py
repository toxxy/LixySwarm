"""Bounded byte tunnels for end-to-end LSP sessions behind NAT.

The relay forwards framed ciphertext. Each endpoint still performs the ordinary
signed HELLO and X25519/AEAD handshake with the other endpoint, and work retains
the worker/requester identities. A relay does not become a compute authority.
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import time
from dataclasses import dataclass

from .lsp_v3 import MAX_FRAME_SIZE, PacketType, ProtocolError
from .peer_manager import validate_peer_address

OPEN, OFFER, ACCEPT, DATA, CLOSE, REJECT = range(1, 7)
HEADER = struct.Struct("!B16s32s")
MAX_CHUNK = 64 * 1024
MAX_BUFFER = 2 * MAX_FRAME_SIZE + 8
ROUTE_IDLE_SECONDS = 3600


@dataclass
class Route:
    source: str
    target: str
    touched: float


class RelayWriter:
    """StreamWriter surface consumed by the existing LSP handshake/read loop."""

    def __init__(self, manager, relay_id, stream_id, remote_id, host, port):
        self.manager = manager
        self.relay_id = relay_id
        self.stream_id = stream_id
        self.expected_node_id = remote_id
        self.host = host
        self.port = port
        self.reader = asyncio.StreamReader(limit=MAX_FRAME_SIZE)
        self.opened = asyncio.get_running_loop().create_future()
        self.buffer = bytearray()
        self.closed = False

    def get_extra_info(self, name, default=None):
        if name == "peername":
            return self.host, self.port
        return default

    def is_closing(self):
        return self.closed

    def write(self, data):
        if self.closed:
            raise ConnectionError("Relay stream is closed")
        if len(self.buffer) + len(data) > MAX_FRAME_SIZE + 4:
            raise ProtocolError("Relay write exceeds the LSP frame bound")
        self.buffer.extend(data)

    async def drain(self):
        while self.buffer:
            chunk = bytes(self.buffer[:MAX_CHUNK])
            del self.buffer[:MAX_CHUNK]
            if self.closed or not await self.manager.send(
                self.relay_id, DATA, self.stream_id, self.expected_node_id, chunk
            ):
                self.close()
                raise ConnectionError("Relay disconnected during write")

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.buffer.clear()
        self.reader.feed_eof()
        if not self.opened.done():
            self.opened.set_result(False)
        self.manager.streams.pop((self.relay_id, self.stream_id), None)
        asyncio.create_task(self.manager.send(
            self.relay_id, CLOSE, self.stream_id, self.expected_node_id
        ))

    async def wait_closed(self):
        return


class RelayManager:
    def __init__(self, node, *, enabled=False, max_streams=64, per_peer=4):
        self.node = node
        self.enabled = bool(enabled)
        self.max_streams = max(1, min(int(max_streams), 256))
        self.per_peer = max(1, min(int(per_peer), self.max_streams))
        self.routes: dict[bytes, Route] = {}
        self.streams: dict[tuple[str, bytes], RelayWriter] = {}
        self.tasks: set[asyncio.Task] = set()
        self.forwarded_bytes = 0

    async def send(self, peer_id, operation, stream_id, remote_id, data=b""):
        payload = HEADER.pack(operation, stream_id, bytes.fromhex(remote_id)) + data
        try:
            return await asyncio.wait_for(
                self.node._send_to_peer(PacketType.RELAY, peer_id, payload), timeout=5
            )
        except (ConnectionError, asyncio.TimeoutError):
            return False

    def _stream(self, relay_id, stream_id, remote_id, host, port):
        if len(self.streams) >= self.max_streams:
            raise ConnectionError("Local relay stream capacity reached")
        if sum(key[0] == relay_id for key in self.streams) >= self.per_peer:
            raise ConnectionError("Local per-relay capacity reached")
        if (relay_id, stream_id) in self.streams:
            raise ProtocolError("Duplicate relay stream")
        writer = RelayWriter(self, relay_id, stream_id, remote_id, host, port)
        self.streams[relay_id, stream_id] = writer
        return writer

    async def open(self, relay_id, target_id, host, port):
        writer = self._stream(relay_id, os.urandom(16), target_id, host, port)
        try:
            if not await self.send(relay_id, OPEN, writer.stream_id, target_id):
                raise ConnectionError("Relay offer could not be sent")
            if not await asyncio.wait_for(asyncio.shield(writer.opened), timeout=5):
                raise ConnectionError("Relay declined the stream")
            return writer.reader, writer
        except BaseException:
            writer.close()
            raise

    async def handle(self, session, payload):
        if len(payload) < HEADER.size:
            raise ProtocolError("Truncated relay control")
        operation, stream_id, remote_bytes = HEADER.unpack_from(payload)
        remote_id = remote_bytes.hex()
        data = payload[HEADER.size:]
        if operation not in {OPEN, OFFER, ACCEPT, DATA, CLOSE, REJECT}:
            raise ProtocolError("Unknown relay operation")
        if len(data) > MAX_CHUNK:
            raise ProtocolError("Relay chunk exceeds bound")
        if operation in {OPEN, ACCEPT, CLOSE} and data:
            raise ProtocolError("Unexpected relay control data")
        if operation == OPEN:
            await self._offer(session, stream_id, remote_id)
            return
        if operation == OFFER:
            await self._accept_offer(session, stream_id, remote_id, data)
            return

        # Only the two authenticated physical endpoints may use a relay route.
        route = self.routes.get(stream_id)
        if route is not None and session.node_id in {route.source, route.target}:
            opposite = route.target if session.node_id == route.source else route.source
            if remote_id != opposite:
                raise ProtocolError("Relay route destination changed")
            if operation == ACCEPT and session.node_id != route.target:
                raise ProtocolError("Only the destination can accept a relay offer")
            if operation == REJECT and session.node_id != route.target:
                raise ProtocolError("Only the destination can reject a relay offer")
            route.touched = time.monotonic()
            if operation in {CLOSE, REJECT}:
                self.routes.pop(stream_id, None)
            if not await self.send(opposite, operation, stream_id, session.node_id, data):
                await self._drop_route(stream_id)
            elif operation == DATA:
                self.forwarded_bytes += len(data)
            return
        if route is not None:
            raise ProtocolError("Relay stream belongs to different endpoints")

        writer = self.streams.get((session.node_id, stream_id))
        if writer is None:
            # In-flight frames may arrive after a close; they cannot allocate state.
            return
        if writer.expected_node_id != remote_id:
            raise ProtocolError("Relay stream remote identity changed")
        if operation == ACCEPT:
            if not writer.opened.done():
                writer.opened.set_result(True)
        elif operation == DATA:
            if len(writer.reader._buffer) + len(data) > MAX_BUFFER:
                writer.close()
                raise ProtocolError("Relay receive buffer exceeds bound")
            writer.reader.feed_data(data)
        elif operation in {CLOSE, REJECT}:
            writer.close()

    async def _offer(self, session, stream_id, target_id):
        now = time.monotonic()
        for existing, route in list(self.routes.items()):
            if now - route.touched > ROUTE_IDLE_SECONDS:
                await self._drop_route(existing)
        target = self.node._sessions.get(target_id)
        reason = None
        if not self.enabled:
            reason = "relay_disabled"
        elif session.relay_node_id or target is None or target.relay_node_id:
            reason = "destination_unavailable"
        elif target_id == session.node_id or target_id == self.node.identity.node_id_hex:
            reason = "invalid_destination"
        elif target.hello.get("capabilities", {}).get("relay_client_v1") is not True:
            reason = "destination_incompatible"
        elif stream_id in self.routes:
            reason = "duplicate_stream"
        elif len(self.routes) >= self.max_streams:
            reason = "relay_capacity"
        elif any(sum(peer in {r.source, r.target} for r in self.routes.values()) >= self.per_peer
                 for peer in (session.node_id, target_id)):
            reason = "peer_capacity"
        if reason:
            await self.send(session.node_id, REJECT, stream_id, target_id, reason.encode())
            return
        self.routes[stream_id] = Route(session.node_id, target_id, now)
        metadata = json.dumps({"host": session.host, "port": session.port}, separators=(",", ":")).encode()
        if not await self.send(target_id, OFFER, stream_id, session.node_id, metadata):
            await self._drop_route(stream_id)

    async def _accept_offer(self, session, stream_id, remote_id, data):
        if session.relay_node_id or session.hello.get("capabilities", {}).get("relay_v1") is not True:
            raise ProtocolError("Unadvertised relay offered a stream")
        if remote_id == self.node.identity.node_id_hex or len(data) > 512:
            raise ProtocolError("Invalid relay offer")
        try:
            metadata = json.loads(data)
            host, port = metadata["host"], metadata["port"]
            if not validate_peer_address(host, port, allow_private=self.node.address_book.allow_private):
                raise ValueError("Invalid endpoint")
        except (KeyError, TypeError, ValueError) as error:
            raise ProtocolError("Invalid relay endpoint metadata") from error
        try:
            writer = self._stream(session.node_id, stream_id, remote_id, host, port)
        except ConnectionError:
            await self.send(session.node_id, REJECT, stream_id, remote_id, b"local_capacity")
            return
        writer.opened.set_result(True)
        task = asyncio.create_task(self.node._handle_inbound(writer.reader, writer))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        if not await self.send(session.node_id, ACCEPT, stream_id, remote_id):
            writer.close()

    async def _drop_route(self, stream_id):
        route = self.routes.pop(stream_id, None)
        if route is not None:
            await self.send(route.source, CLOSE, stream_id, route.target)
            await self.send(route.target, CLOSE, stream_id, route.source)

    async def peer_closed(self, node_id):
        for stream_id, route in list(self.routes.items()):
            if node_id in {route.source, route.target}:
                await self._drop_route(stream_id)
        for (relay_id, _), writer in list(self.streams.items()):
            if relay_id == node_id:
                writer.close()

    async def stop(self):
        for writer in list(self.streams.values()):
            writer.close()
        self.routes.clear()
        for task in list(self.tasks):
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
