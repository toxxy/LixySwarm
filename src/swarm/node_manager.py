"""
Lixy — NodeManager: Ciclo de Vida de Nodos Físicos (Hormigas) 🐜
=================================================================
CONCEPTO CORRECTO:
  Hormiga = Nodo físico (la máquina)
    - Laptop vieja        → hormiga débil (poco RAM/CPU)
    - RTX 5090            → hormiga fuerte (mucha GPU)
    - VPS                 → hormiga media

  Las hormigas NO son agentes individuales.
  Los agentes viven dentro de Sectas, y las Sectas corren EN hormigas.

  Hormiga fuerte → puede hospedar múltiples Sectas simultáneamente
  Hormiga débil  → solo 1-2 Sectas

Ciclo de vida:
  - Nacen: cuando un nuevo nodo se conecta a la red
  - Mueren: cuando el nodo se desconecta
  - Fitness: basado en hardware + uptime + contribución al enjambre
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.swarm.sect_manager import SectManager


class ContributionMode(Enum):
    """
    Modo de contribución de un nodo al enjambre.

    MAXIMUM  → Todo el hardware disponible: GPU al 100%, sin límites de sectas
    MODERATE → Mitad del hardware: GPU al 50%, máx la mitad de sectas
    RELAY    → Solo reenvia feromonas: sin GPU, sin sectas propias
    """
    MAXIMUM  = "maximum"
    MODERATE = "moderate"
    RELAY    = "relay"


@dataclass
class HardwareProfile:
    """Capacidades de hardware de un nodo físico."""
    cpu_cores: int = 1
    ram_gb: float = 4.0
    gpu_vram_gb: float = 0.0       # 0 = sin GPU
    disk_gb: float = 50.0
    has_gpu: bool = False

    @classmethod
    def from_local(cls) -> "HardwareProfile":
        """Detecta el hardware de este nodo."""
        cpu_cores = 1
        ram_gb = 4.0
        gpu_vram_gb = 0.0
        has_gpu = False

        try:
            import psutil
            cpu_cores = psutil.cpu_count(logical=False) or 1
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                has_gpu = True
                gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except ImportError:
            pass

        return cls(
            cpu_cores=cpu_cores,
            ram_gb=ram_gb,
            gpu_vram_gb=gpu_vram_gb,
            has_gpu=has_gpu,
        )

    @property
    def compute_score(self) -> float:
        """Puntuación relativa de cómputo del nodo (0-10+)."""
        gpu_score = self.gpu_vram_gb * 0.8 if self.has_gpu else 0.0
        cpu_score = self.cpu_cores * 0.3
        ram_score = self.ram_gb * 0.1
        return gpu_score + cpu_score + ram_score

    @property
    def max_concurrent_sects(self) -> int:
        """Cuántas sectas puede correr este nodo simultáneamente."""
        if self.has_gpu and self.gpu_vram_gb >= 20:
            return 4   # nodo muy fuerte (RTX 5090+)
        elif self.has_gpu and self.gpu_vram_gb >= 8:
            return 3   # nodo fuerte (RTX 3090+)
        elif self.has_gpu:
            return 2   # nodo con GPU modesta
        elif self.ram_gb >= 16:
            return 2   # nodo CPU fuerte
        else:
            return 1   # nodo débil


@dataclass
class NodeRecord:
    """Registro de un nodo físico (hormiga) en el enjambre."""
    node_id: str
    hardware: HardwareProfile
    connected_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    connected_sects: List[str] = field(default_factory=list)   # sect_id[]
    contribution_score: float = 0.0   # acumulado
    is_local: bool = False             # True = este mismo proceso
    contribution_mode: ContributionMode = ContributionMode.MAXIMUM   # modo de contribución

    @property
    def uptime(self) -> float:
        """Segundos conectado."""
        return time.time() - self.connected_at

    @property
    def fitness(self) -> float:
        """
        Fitness del nodo [0, 1]:
        - hardware compute_score normalizado
        - uptime (nodos estables > nodos nuevos)
        - contribución al enjambre
        """
        hw_score = min(1.0, self.hardware.compute_score / 10.0)
        uptime_score = min(1.0, self.uptime / 3600.0)    # satura a 1h
        contrib_score = min(1.0, self.contribution_score / 1000.0)
        return 0.5 * hw_score + 0.3 * uptime_score + 0.2 * contrib_score

    def can_host_sect(self, sect_id: str) -> bool:
        """¿Puede este nodo hospedar una secta más?"""
        if self.contribution_mode == ContributionMode.RELAY:
            return False  # modo relay: nunca hospeda sectas
        if sect_id in self.connected_sects:
            return False  # ya la tiene
        cap = self._effective_sect_capacity()
        return len(self.connected_sects) < cap

    def _effective_sect_capacity(self) -> int:
        """Capacidad real de sectas según el modo de contribución."""
        base = self.hardware.max_concurrent_sects
        if self.contribution_mode == ContributionMode.MAXIMUM:
            return base
        elif self.contribution_mode == ContributionMode.MODERATE:
            return max(1, base // 2)
        else:  # RELAY
            return 0

    @property
    def effective_gpu_fraction(self) -> float:
        """Fracción de GPU disponible según el modo de contribución."""
        if self.contribution_mode == ContributionMode.MAXIMUM:
            return 1.0
        elif self.contribution_mode == ContributionMode.MODERATE:
            return 0.5
        else:  # RELAY
            return 0.0

    def attach_sect(self, sect_id: str) -> bool:
        if not self.can_host_sect(sect_id):
            return False
        self.connected_sects.append(sect_id)
        return True

    def detach_sect(self, sect_id: str):
        self.connected_sects = [s for s in self.connected_sects if s != sect_id]

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "cpu_cores": self.hardware.cpu_cores,
            "ram_gb": self.hardware.ram_gb,
            "gpu_vram_gb": self.hardware.gpu_vram_gb,
            "has_gpu": self.hardware.has_gpu,
            "uptime": round(self.uptime, 1),
            "fitness": round(self.fitness, 3),
            "connected_sects": list(self.connected_sects),
            "is_local": self.is_local,
            "contribution_mode": self.contribution_mode.value,
            "effective_gpu_fraction": self.effective_gpu_fraction,
            "effective_sect_capacity": self._effective_sect_capacity(),
        }


class NodeManager:
    """
    Gestiona el ciclo de vida de los nodos físicos (hormigas) del enjambre.

    Las hormigas son máquinas. Nacen cuando se conectan, mueren cuando
    se desconectan. No hay límites artificiales: el enjambre es tan grande
    como la red que lo sostiene.
    """

    # Timeout para declarar un nodo muerto (segundos sin heartbeat)
    NODE_TIMEOUT_SECONDS = 300    # 5 min sin ping → nodo muerto

    def __init__(self, matriarca=None):
        self.matriarca = matriarca
        self._nodes: Dict[str, NodeRecord] = {}
        self._join_count = 0
        self._leave_count = 0

    # ─── API pública ─────────────────────────────────────────────────────────

    def node_joined(self, node_id: str, hardware: Optional[HardwareProfile] = None,
                    is_local: bool = False) -> NodeRecord:
        """
        Registra un nuevo nodo que se une a la red.
        Si el nodo ya existía (reconexión), actualiza su timestamp.
        """
        if hardware is None:
            hardware = HardwareProfile()  # defaults conservadores

        if node_id in self._nodes:
            # Reconexión
            record = self._nodes[node_id]
            record.last_seen = time.time()
            record.connected_at = time.time()  # reset uptime
            return record

        record = NodeRecord(
            node_id=node_id,
            hardware=hardware,
            is_local=is_local,
        )
        self._nodes[node_id] = record
        self._join_count += 1
        return record

    def node_left(self, node_id: str, reason: str = "disconnect") -> Optional[dict]:
        """
        Elimina un nodo que se desconectó.
        Transfiere su legado de sectas a la Matriarca.
        """
        if node_id not in self._nodes:
            return None

        record = self._nodes.pop(node_id)
        self._leave_count += 1

        # Transferir legado a la Matriarca
        event = {
            "type": "node_left",
            "node_id": node_id,
            "reason": reason,
            "uptime": record.uptime,
            "fitness": record.fitness,
            "sects_hosted": list(record.connected_sects),
            "legacy_stored": False,
        }

        if self.matriarca is not None:
            self._store_node_legacy(record, reason)
            event["legacy_stored"] = True

        return event

    def heartbeat(self, node_id: str, contribution_delta: float = 0.0) -> bool:
        """
        Actualiza el timestamp de un nodo (señal de vida).
        Retorna False si el nodo no era conocido.
        """
        if node_id not in self._nodes:
            return False
        record = self._nodes[node_id]
        record.last_seen = time.time()
        record.contribution_score += contribution_delta
        return True

    def prune_dead_nodes(self) -> List[dict]:
        """
        Elimina nodos que no han dado señal de vida en NODE_TIMEOUT_SECONDS.
        Retorna lista de eventos de muerte.
        """
        now = time.time()
        dead = [
            nid for nid, rec in self._nodes.items()
            if (now - rec.last_seen) > self.NODE_TIMEOUT_SECONDS
        ]
        events = []
        for nid in dead:
            event = self.node_left(nid, reason="timeout")
            if event:
                events.append(event)
        return events

    def get_node(self, node_id: str) -> Optional[NodeRecord]:
        return self._nodes.get(node_id)

    def all_nodes(self) -> List[NodeRecord]:
        return list(self._nodes.values())

    def set_contribution_mode(self, node_id: str, mode: ContributionMode) -> bool:
        """
        Cambia el modo de contribución de un nodo.
        Retorna True si el nodo existe y el modo fue aplicado.

        MAXIMUM  → GPU al 100%, todas las sectas disponibles
        MODERATE → GPU al 50%, mitad de sectas
        RELAY    → 0% GPU, 0 sectas, solo relay de feromonas
        """
        node = self._nodes.get(node_id)
        if node is None:
            return False
        old_mode = node.contribution_mode
        node.contribution_mode = mode
        # Si bajamos a RELAY, desconectar sectas que ya no puede hospedar
        if mode == ContributionMode.RELAY:
            node.connected_sects.clear()
        elif mode == ContributionMode.MODERATE:
            cap = node._effective_sect_capacity()
            while len(node.connected_sects) > cap:
                node.connected_sects.pop()
        import logging
        logging.getLogger(__name__).info(
            f"Node {node_id[:8]} contribution_mode: {old_mode.value} → {mode.value}"
        )
        return True

    def best_node_for_sect(self, sect_id: str) -> Optional[NodeRecord]:
        """Nodo con más capacidad disponible que pueda hospedar la secta."""
        candidates = [
            n for n in self._nodes.values()
            if n.can_host_sect(sect_id)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda n: n.hardware.compute_score)

    def n_nodes(self) -> int:
        return len(self._nodes)

    def stats(self) -> dict:
        return {
            "total_nodes": len(self._nodes),
            "joins": self._join_count,
            "leaves": self._leave_count,
            "nodes": [n.to_dict() for n in self._nodes.values()],
        }

    # ─── Legado ──────────────────────────────────────────────────────────────

    def _store_node_legacy(self, record: NodeRecord, reason: str):
        """Guarda el perfil del nodo en la Matriarca como legado."""
        import torch
        import torch.nn.functional as F

        embd_dim = self.matriarca.cfg.embd_dim

        # Embedding del nodo: encode hardware como vector simple
        hw = record.hardware
        raw = torch.tensor([
            hw.cpu_cores / 32.0,
            hw.ram_gb / 128.0,
            hw.gpu_vram_gb / 80.0,
            float(hw.has_gpu),
            record.fitness,
        ], dtype=torch.float32)

        # Pad/truncar a embd_dim
        if raw.shape[0] < embd_dim:
            embedding = F.pad(raw, (0, embd_dim - raw.shape[0]))
        else:
            embedding = raw[:embd_dim]

        text = (
            f"[NODE_LEGACY] node={record.node_id} "
            f"gpu={hw.gpu_vram_gb:.0f}GB cpu={hw.cpu_cores} ram={hw.ram_gb:.0f}GB "
            f"fitness={record.fitness:.3f} uptime={record.uptime:.0f}s "
            f"sects={','.join(record.connected_sects)} reason={reason}"
        )

        self.matriarca.add(
            embedding=embedding,
            text=text,
            importance=record.fitness,
        )
