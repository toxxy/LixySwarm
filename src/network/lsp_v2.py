"""
LixySwarm Protocol (LSP) v2
Wire format extendido con soporte de:
  - Compresión nativa float16 (no JSON list)
  - Merge-on-transit inteligente
  - TTL de señal + decay temporal
  - TYPE_FEROMON_V2=0x10 (nuevo, v1 tipos 0x01-0x05 intactos)
  - Backward compat total con LSPNode v1
"""

from __future__ import annotations

import struct
import socket
import threading
import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .lsp import (
    LSPIdentity, LSPNode, LSPPacket, PacketType as PacketTypeV1,
    Flags, MAGIC, VERSION, HEADER_SIZE, MAX_UDP_SIZE
)

log = logging.getLogger(__name__)

# ─── PacketType v2 (nuevos tipos, adicionales a v1) ───────────────────────────

class PacketType(PacketTypeV1):
    FEROMON_V2   = 0x10  # feromon binario nativo
    GOSSIP_DELTA = 0x11  # gossip incremental
    MERGE_HINT   = 0x12  # sugerencia de merge para intermediarios
    PEER_LIST    = 0x13  # intercambio de peers (addr gossip)


# ─── DimType constants ────────────────────────────────────────────────────────

DIM_FLOAT16  = 0x01
DIM_FLOAT32  = 0x02
DIM_BFLOAT16 = 0x03

_BYTES_PER_DIM = {
    DIM_FLOAT16:  2,
    DIM_FLOAT32:  4,
    DIM_BFLOAT16: 2,
}

# ─── FeromonV2Payload ─────────────────────────────────────────────────────────

@dataclass
class FeromonV2Payload:
    """
    Payload binario eficiente para feromonas v2.

    Wire format (body del LSPPacket payload):
        [1B]  dim_type: 0x01=float16, 0x02=float32, 0x03=bfloat16
        [2B]  n_dims (uint16 LE)
        [1B]  ttl: saltos restantes (empieza en 3)
        [4B]  step (uint32 LE)
        [4B]  fitness (float32 LE)
        [4B]  timestamp_delta (uint32 LE): ms desde epoch truncado a 32 bits
        [NB]  tensor_bytes: n_dims × bytes_per_type
    """
    feromon: "torch.Tensor"          # float32 en memoria
    ttl: int = 3
    step: int = 0
    fitness: float = 0.5
    timestamp_ms: int = 0            # ms origen (0 = now on pack)
    dim_type: int = DIM_FLOAT16      # float16 por defecto

    # ─── Serialización ────────────────────────────────────────────────────────

    def pack(self) -> bytes:
        """Serializa a bytes binarios eficientes (sin overhead JSON)."""
        import torch

        tensor = self.feromon.cpu().float()  # float32 en CPU
        n_dims = tensor.numel()

        bpd = _BYTES_PER_DIM[self.dim_type]

        if self.dim_type == DIM_FLOAT16:
            tensor_bytes = tensor.half().numpy().tobytes()
        elif self.dim_type == DIM_FLOAT32:
            tensor_bytes = tensor.numpy().tobytes()
        elif self.dim_type == DIM_BFLOAT16:
            tensor_bytes = tensor.bfloat16().numpy().tobytes()
        else:
            raise ValueError(f"Unknown dim_type: {self.dim_type}")

        ts_ms = self.timestamp_ms if self.timestamp_ms else int(time.time() * 1000)
        ts_delta = ts_ms & 0xFFFFFFFF  # truncar a 32 bits

        header = struct.pack(
            "<BHBII f I",
            self.dim_type,         # 1B
            n_dims,                # 2B
            max(0, self.ttl),      # 1B
            self.step,             # 4B
            0,                     # placeholder, fitness va separado
            float(self.fitness),   # 4B float32
            ts_delta,              # 4B
        )
        # Repack limpio — struct format sin alias confuso:
        header = struct.pack(
            "<B H B I f I",
            self.dim_type,
            n_dims,
            max(0, self.ttl),
            self.step,
            float(self.fitness),
            ts_delta,
        )
        return header + tensor_bytes

    @classmethod
    def unpack(cls, data: bytes) -> "FeromonV2Payload":
        """Deserializa desde bytes binarios. No requiere torch (VPS-safe)."""
        import numpy as np

        FIXED_SIZE = 1 + 2 + 1 + 4 + 4 + 4  # = 16 bytes header
        if len(data) < FIXED_SIZE:
            raise ValueError(f"FeromonV2Payload too short: {len(data)}")

        dim_type, n_dims, ttl, step, fitness, ts_delta = struct.unpack_from(
            "<B H B I f I", data, 0
        )
        tensor_bytes = data[FIXED_SIZE:]

        bpd = _BYTES_PER_DIM.get(dim_type)
        if bpd is None:
            raise ValueError(f"Unknown dim_type: {dim_type}")

        expected = n_dims * bpd
        if len(tensor_bytes) != expected:
            raise ValueError(
                f"tensor_bytes length mismatch: {len(tensor_bytes)} != {expected}"
            )

        if dim_type == DIM_FLOAT16:
            arr = np.frombuffer(tensor_bytes, dtype=np.float16).astype(np.float32)
        elif dim_type == DIM_FLOAT32:
            arr = np.frombuffer(tensor_bytes, dtype=np.float32)
        elif dim_type == DIM_BFLOAT16:
            # bfloat16: reinterpret uint16 → shift left 16 → float32
            raw = np.frombuffer(tensor_bytes, dtype=np.uint16).astype(np.uint32)
            arr = (raw << 16).view(np.float32)
        else:
            raise ValueError(f"Unknown dim_type: {dim_type}")

        # torch optional — numpy ndarray is fine for relay/VPS
        try:
            import torch
            feromon = torch.tensor(arr.copy(), dtype=torch.float32)
        except ImportError:
            feromon = arr.copy()  # numpy float32 array — callback-safe

        return cls(
            feromon=feromon,
            ttl=ttl,
            step=step,
            fitness=float(fitness),
            timestamp_ms=int(ts_delta),
            dim_type=dim_type,
        )

    def apply_decay(self, decay: float = 0.95) -> "FeromonV2Payload":
        """Retorna nuevo payload con feromon * decay y TTL - 1."""
        return FeromonV2Payload(
            feromon=self.feromon * decay,
            ttl=max(0, self.ttl - 1),
            step=self.step,
            fitness=self.fitness,
            timestamp_ms=self.timestamp_ms,
            dim_type=self.dim_type,
        )

    def merge(self, other: "FeromonV2Payload", alpha: float = 0.5) -> "FeromonV2Payload":
        """Weighted average: self * (1 - alpha) + other * alpha."""
        import torch
        merged = self.feromon * (1.0 - alpha) + other.feromon * alpha
        avg_fitness = self.fitness * (1.0 - alpha) + other.fitness * alpha
        min_ttl = min(self.ttl, other.ttl)
        return FeromonV2Payload(
            feromon=merged,
            ttl=min_ttl,
            step=max(self.step, other.step),
            fitness=avg_fitness,
            timestamp_ms=max(self.timestamp_ms, other.timestamp_ms),
            dim_type=self.dim_type,
        )

    @property
    def wire_size(self) -> int:
        """Tamaño en bytes cuando empaquetado."""
        return 16 + self.feromon.numel() * _BYTES_PER_DIM[self.dim_type]


# ─── FeromonMergeBuffer ───────────────────────────────────────────────────────

class FeromonMergeBuffer:
    """
    Acumula feromonas por nodo y hace merge inteligente.
    Descarta entradas > MAX_AGE_MS ms de antigüedad.
    """

    MAX_AGE_MS:  int = 2000
    MAX_PER_NODE: int = 4

    def __init__(self):
        self._buf: Dict[str, List[Tuple[int, FeromonV2Payload]]] = defaultdict(list)
        self._lock = threading.Lock()

    def push(self, node_id: str, payload: FeromonV2Payload) -> None:
        """Agrega feromona de un nodo al buffer."""
        now_ms = int(time.time() * 1000)
        if payload.timestamp_ms == 0:
            payload = FeromonV2Payload(
                feromon=payload.feromon,
                ttl=payload.ttl,
                step=payload.step,
                fitness=payload.fitness,
                timestamp_ms=now_ms,
                dim_type=payload.dim_type,
            )
        with self._lock:
            entries = self._buf[node_id]
            # Evict old entries
            entries = [(ts, p) for ts, p in entries
                       if (now_ms - ts) <= self.MAX_AGE_MS]
            entries.append((now_ms, payload))
            # Trim to MAX_PER_NODE (keep newest)
            if len(entries) > self.MAX_PER_NODE:
                entries = entries[-self.MAX_PER_NODE:]
            self._buf[node_id] = entries

    def flush(self) -> List[Tuple[str, FeromonV2Payload]]:
        """
        Retorna lista (node_id, merged_payload) para cada nodo.
        Hace merge fitness-weighted de todas las feromonas acumuladas.
        Limpia el buffer.
        """
        now_ms = int(time.time() * 1000)
        result = []
        with self._lock:
            for node_id, entries in list(self._buf.items()):
                # Filter stale
                fresh = [(ts, p) for ts, p in entries
                         if (now_ms - ts) <= self.MAX_AGE_MS]
                if not fresh:
                    del self._buf[node_id]
                    continue
                payloads = [p for _, p in fresh]
                merged = self._merge_payloads(payloads)
                result.append((node_id, merged))
            self._buf.clear()
        return result

    def _merge_payloads(self, payloads: List[FeromonV2Payload]) -> FeromonV2Payload:
        """Fitness-weighted average de múltiples feromonas."""
        import torch

        if len(payloads) == 1:
            p = payloads[0]
            if p.ttl < 3:
                p = p.apply_decay(0.95)
            return p

        total_fitness = sum(p.fitness for p in payloads)
        if total_fitness <= 0:
            weights = [1.0 / len(payloads)] * len(payloads)
        else:
            weights = [p.fitness / total_fitness for p in payloads]

        base = payloads[0].feromon.clone()
        merged_vec = torch.zeros_like(base)
        for w, p in zip(weights, payloads):
            merged_vec += p.feromon * w

        avg_fitness = sum(p.fitness for p in payloads) / len(payloads)
        min_ttl = min(p.ttl for p in payloads)
        max_step = max(p.step for p in payloads)
        max_ts = max(p.timestamp_ms for p in payloads)

        result = FeromonV2Payload(
            feromon=merged_vec,
            ttl=min_ttl,
            step=max_step,
            fitness=avg_fitness,
            timestamp_ms=max_ts,
            dim_type=payloads[0].dim_type,
        )

        # Apply decay si ya saltó hops (TTL < 3)
        if min_ttl < 3:
            result = result.apply_decay(0.95)

        return result


# ─── LSPNodeV2 ────────────────────────────────────────────────────────────────

class LSPNodeV2(LSPNode):
    """
    Extensión de LSPNode con soporte LSP v2:
    - send_feromon_v2: envía FeromonV2Payload binario
    - _handle_feromon_v2: recibe + merge buffer + dispara callbacks
    - Backward compat: v1 FEROMON (0x01) sigue funcionando
    """

    def __init__(self, identity: LSPIdentity,
                 feromon_port: int = 7337,
                 gossip_port: int = 7338):
        super().__init__(identity, feromon_port, gossip_port)
        self._merge_buffer = FeromonMergeBuffer()
        self._peer_list_callbacks: list = []

    # ─── Peer Exchange (addr gossip estilo Bitcoin) ───────────────────────

    def on_peer_list_received(self, callback):
        """callback(peers: List[Tuple[str,int]]) cuando llega PEER_LIST."""
        if callable(callback):
            self._peer_list_callbacks.append(callback)
        return callback

    def send_peer_list(self, peers: List[dict]):
        """Envía lista de peers a todos los peers conectados (TCP handshake socket)."""
        from .bootstrap import encode_peer_list
        payload = encode_peer_list(peers)
        pkt = LSPPacket.create(PacketType.PEER_LIST, payload, compress=True)
        data = pkt.pack(self.identity)
        with self._lock:
            peer_list = list(self._peers.values())
        for peer in peer_list:
            try:
                # Enviar por TCP (el socket de gossip/handshake)
                with socket.create_connection((peer["host"], peer["gossip_port"]), timeout=5.0) as s:
                    s.sendall(struct.pack("<I", len(data)) + data)
            except Exception as e:
                log.debug(f"send_peer_list to {peer['host']}: {e}")

    def request_peer_list(self, host: str, port: int):
        """Solicita lista de peers a un nodo específico (handshake incluye flag)."""
        payload = json.dumps({
            "version": "2.0",
            "node_id": self.identity.node_id_hex,
            "feromon_port": self.feromon_port,
            "gossip_port": self.gossip_port,
            "request_peers": True,
        }).encode("utf-8")
        pkt = LSPPacket.create(PacketType.HANDSHAKE, payload, compress=False)
        data = pkt.pack(self.identity)
        try:
            with socket.create_connection((host, port), timeout=5.0) as s:
                s.sendall(struct.pack("<I", len(data)) + data)
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
                            # Peer list viene en la respuesta
                            if info.get("peers"):
                                from .bootstrap import decode_peer_list
                                peer_list = [(p["host"], p.get("gossip_port", 7338))
                                             for p in info["peers"]]
                                for cb in self._peer_list_callbacks:
                                    try:
                                        cb(peer_list)
                                    except Exception:
                                        pass
        except Exception as e:
            log.debug(f"request_peer_list {host}:{port}: {e}")

    def send_feromon_v2(self, feromon_tensor, fitness: float = 0.5, step: int = 0):
        """Envía FeromonV2Payload binario a todos los peers."""
        import torch
        if not isinstance(feromon_tensor, torch.Tensor):
            feromon_tensor = torch.tensor(feromon_tensor, dtype=torch.float32)

        self._step += 1
        actual_step = step if step > 0 else self._step

        payload_obj = FeromonV2Payload(
            feromon=feromon_tensor,
            ttl=3,
            step=actual_step,
            fitness=float(fitness),
            timestamp_ms=int(time.time() * 1000),
            dim_type=DIM_FLOAT16,
        )
        payload_bytes = payload_obj.pack()
        pkt = LSPPacket.create(PacketType.FEROMON_V2, payload_bytes, compress=False)
        data = pkt.pack(self.identity)

        if len(data) > MAX_UDP_SIZE:
            log.warning(f"FeromonV2 packet too large for UDP: {len(data)} bytes")
            return

        with self._lock:
            peers = list(self._peers.values())

        for peer in peers:
            try:
                self._udp_sock.sendto(data, (peer["host"], peer["feromon_port"]))
            except Exception as e:
                log.debug(f"send_feromon_v2 to {peer['host']}: {e}")

    def _handle_feromon_v2(self, pkt: LSPPacket, addr):
        """Recibe FeromonV2Payload, push al merge buffer, flush y dispara callbacks."""
        try:
            payload_obj = FeromonV2Payload.unpack(pkt.payload)

            # TTL=0 → descartar sin callback
            if payload_obj.ttl <= 0:
                log.info(f"FeromonV2 TTL=0, discarding from {addr}")
                return

            node_id_hex = pkt.node_id.hex()
            self._merge_buffer.push(node_id_hex, payload_obj)

            # Flush inmediato
            merged_list = self._merge_buffer.flush()
            for nid, merged in merged_list:
                for cb in self._feromon_callbacks:
                    try:
                        cb(merged.feromon, nid)
                    except Exception as e:
                        log.info(f"feromon_v2 callback error: {e}")

        except Exception as e:
            log.info(f"_handle_feromon_v2 error from {addr}: {e}")

    def _handle_udp(self, data: bytes, addr):
        """Override para manejar también paquetes v2."""
        try:
            pkt = LSPPacket.unpack(data)
            if not pkt.verify():
                log.info(f"Invalid signature from {addr}")
                return

            if pkt.type == PacketType.FEROMON_V2:
                log.info(f"FEROMON_V2 received from {addr}, handling...")
                self._handle_feromon_v2(pkt, addr)
            else:
                # Delegar a v1 handler
                super()._handle_udp(data, addr)

        except Exception as e:
            log.info(f"LSPNodeV2._handle_udp error from {addr}: {e}")
