"""
Lixy — SectManager: Ciclo de Vida de Sectas (Especialidades) 🏛️
================================================================
CONCEPTO CORRECTO:
  Secta = Grupo de agentes con una especialidad concreta.
    - "Explorador"   → agentes que navegan espacio creativo/divergente
    - "Refinador"    → agentes que precisan y convergen respuestas
    - "Delfín"       → agentes de ecolocalización / síntesis distribuida
    - (futuras sectas nacen cuando el enjambre detecta nueva necesidad)

  Las sectas NO son nodos físicos. Son capacidades que corren EN nodos.
  Un nodo fuerte puede hospedar varias sectas a la vez.
  Un nodo débil solo puede hospedar 1-2 sectas.

Ciclo de vida de una secta:
  - Nace cuando: diversidad baja, nueva capacidad detectada, nueva necesidad
  - Muere cuando: fitness bajo sostenido, sin nodos disponibles, utilidad = 0
  - Transferencia: antes de morir → legado en Matriarca

La Matriarca guarda el legado de todas las sectas que existieron y
puede orientar el nacimiento de nuevas sectas basándose en ese conocimiento.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.swarm.node_manager import NodeManager


# Tipos de secta conocidos (no exhaustivos — el enjambre puede crear nuevos)
KNOWN_SECT_ROLES = {
    "explorador":     "Alta divergencia, exploración del espacio de soluciones",
    "refinador":      "Alta precisión, convergencia y corrección",
    "delfín":         "Síntesis distribuida, ecolocalización, puente entre sectas",
    "aprendiendo":    "Secta en formación, rol aún no definido",
    "generalista":    "Distribución equilibrada, sin especialidad clara",
}

# Recursos mínimos para que una secta pueda vivir en un nodo (en unidades de compute_score)
SECT_RESOURCE_REQUIREMENTS = {
    "explorador":  2.0,
    "refinador":   2.0,
    "delfín":      3.0,
    "aprendiendo": 1.0,
    "generalista": 1.5,
}


@dataclass
class AgentSlot:
    """Un agente dentro de una secta."""
    agent_id: int
    node_id: str
    fitness: float = 0.5
    low_fitness_steps: int = 0


@dataclass
class SectRecord:
    """Registro de una secta en el enjambre."""
    sect_id: str
    role_type: str                                      # explorador, refinador, delfín, ...
    priority: float = 0.5                              # [0, 1] — importancia en el enjambre
    created_at: float = field(default_factory=time.time)
    agents: List[AgentSlot] = field(default_factory=list)
    fitness_history: List[float] = field(default_factory=list)
    required_compute: float = 2.0                      # unidades de compute_score mínimas

    # Ciclo de vida
    LOW_FITNESS_THRESHOLD: float = 0.3
    LOW_FITNESS_STEPS_TO_DIE: int = 500

    @property
    def avg_fitness(self) -> float:
        if not self.agents:
            return 0.0
        return sum(a.fitness for a in self.agents) / len(self.agents)

    @property
    def n_agents(self) -> int:
        return len(self.agents)

    @property
    def age(self) -> float:
        return time.time() - self.created_at

    def record_fitness(self, fitness: float):
        self.fitness_history.append(fitness)
        if len(self.fitness_history) > 100:
            self.fitness_history = self.fitness_history[-100:]

    def is_dying(self) -> bool:
        """
        True si la secta lleva demasiados pasos con fitness bajo.
        Nota: también muere si no hay nodos disponibles — eso lo verifica SectManager.
        """
        if len(self.fitness_history) < self.LOW_FITNESS_STEPS_TO_DIE:
            return False
        recent = self.fitness_history[-self.LOW_FITNESS_STEPS_TO_DIE:]
        return sum(1 for f in recent if f < self.LOW_FITNESS_THRESHOLD) >= self.LOW_FITNESS_STEPS_TO_DIE

    def to_dict(self) -> dict:
        return {
            "sect_id": self.sect_id,
            "role_type": self.role_type,
            "priority": self.priority,
            "n_agents": self.n_agents,
            "avg_fitness": round(self.avg_fitness, 3),
            "age_s": round(self.age, 1),
            "agents": [{"agent_id": a.agent_id, "node_id": a.node_id, "fitness": round(a.fitness, 3)}
                       for a in self.agents],
        }


class SectManager:
    """
    Gestiona el ciclo de vida de las Sectas del enjambre.

    Las sectas son capacidades especializadas que nacen cuando el enjambre
    lo necesita y mueren cuando pierden utilidad. Sin límites artificiales.
    """

    # Umbrales de diversidad para disparar spawn
    DIVERSITY_THRESHOLD = 0.4

    def __init__(self, node_manager: Optional["NodeManager"] = None, matriarca=None):
        self.node_manager = node_manager
        self.matriarca = matriarca
        self._sects: Dict[str, SectRecord] = {}
        self._birth_count = 0
        self._death_count = 0
        self._next_agent_id = 0

    # ─── API pública ─────────────────────────────────────────────────────────

    def spawn_sect(self, role_type: str, priority: float = 0.5) -> Optional[SectRecord]:
        """
        Crea una nueva secta si hay nodos disponibles para hospedarla.
        Sin límite de cuántas sectas pueden existir.

        Returns:
            SectRecord si fue creada, None si no hay recursos disponibles.
        """
        required = SECT_RESOURCE_REQUIREMENTS.get(role_type, 2.0)
        sect_id = f"{role_type}_{self._birth_count}"

        # Verificar que haya al menos un nodo capaz
        if self.node_manager is not None:
            host = self.node_manager.best_node_for_sect(sect_id)
            if host is None:
                return None  # sin nodos disponibles

        sect = SectRecord(
            sect_id=sect_id,
            role_type=role_type,
            priority=priority,
            required_compute=required,
        )
        self._sects[sect_id] = sect
        self._birth_count += 1

        # Asignar al nodo más fuerte disponible
        if self.node_manager is not None and host is not None:
            host.attach_sect(sect_id)

        return sect

    def kill_sect(self, sect_id: str, reason: str = "low_fitness") -> Optional[dict]:
        """
        Elimina una secta. Transfiere legado a la Matriarca.
        Returns evento de muerte o None si no existía.
        """
        if sect_id not in self._sects:
            return None

        sect = self._sects.pop(sect_id)
        self._death_count += 1

        # Liberar de los nodos
        if self.node_manager is not None:
            for node in self.node_manager.all_nodes():
                node.detach_sect(sect_id)

        event = {
            "type": "sect_death",
            "sect_id": sect_id,
            "role_type": sect.role_type,
            "reason": reason,
            "avg_fitness": sect.avg_fitness,
            "n_agents": sect.n_agents,
            "age_s": sect.age,
            "legacy_stored": False,
        }

        if self.matriarca is not None:
            self._store_sect_legacy(sect, reason)
            event["legacy_stored"] = True

        return event

    def tick(self, step: int, swarm_diversity: float) -> List[dict]:
        """
        Evalúa el ciclo de vida de las sectas.
        Llama esto periódicamente (cada N steps durante training o runtime).

        - Registra fitness de cada secta
        - Mata sectas moribundas (si hay más de 1)
        - Spawnea sectas nuevas si la diversidad es baja y hay recursos

        Returns lista de eventos.
        """
        events = []

        # 1. Actualizar fitness de cada secta
        for sect in list(self._sects.values()):
            sect.record_fitness(sect.avg_fitness)

        # 2. Matar sectas moribundas (proteger al menos 1)
        for sect in list(self._sects.values()):
            if len(self._sects) <= 1:
                break   # nunca matar la última secta
            if sect.is_dying():
                event = self.kill_sect(sect.sect_id, reason="low_fitness")
                if event:
                    events.append(event)

        # 3. Spawn si diversidad baja y hay recursos
        if swarm_diversity < self.DIVERSITY_THRESHOLD:
            # Intentar spawnear el rol menos representado
            role = self._most_needed_role()
            new_sect = self.spawn_sect(role)
            if new_sect is not None:
                events.append({
                    "type": "sect_birth",
                    "sect_id": new_sect.sect_id,
                    "role_type": role,
                    "trigger": "low_diversity",
                })

        return events

    def add_agent_to_sect(self, sect_id: str, node_id: str) -> Optional[AgentSlot]:
        """Añade un agente a una secta en el nodo especificado."""
        if sect_id not in self._sects:
            return None
        slot = AgentSlot(
            agent_id=self._next_agent_id,
            node_id=node_id,
        )
        self._sects[sect_id].agents.append(slot)
        self._next_agent_id += 1
        return slot

    def update_agent_fitness(self, sect_id: str, agent_id: int, fitness: float):
        """Actualiza el fitness de un agente en su secta."""
        if sect_id not in self._sects:
            return
        for slot in self._sects[sect_id].agents:
            if slot.agent_id == agent_id:
                slot.fitness = fitness
                if fitness < SectRecord.LOW_FITNESS_THRESHOLD:
                    slot.low_fitness_steps += 1
                else:
                    slot.low_fitness_steps = 0
                break

    def get_sect(self, sect_id: str) -> Optional[SectRecord]:
        return self._sects.get(sect_id)

    def all_sects(self) -> List[SectRecord]:
        return list(self._sects.values())

    def n_sects(self) -> int:
        return len(self._sects)

    def stats(self) -> dict:
        return {
            "total_sects": len(self._sects),
            "births": self._birth_count,
            "deaths": self._death_count,
            "sects": [s.to_dict() for s in self._sects.values()],
        }

    # ─── Internos ─────────────────────────────────────────────────────────────

    def _most_needed_role(self) -> str:
        """Determina qué rol está más sub-representado."""
        existing_roles = [s.role_type for s in self._sects.values()]
        for role in ["explorador", "refinador", "delfín"]:
            if role not in existing_roles:
                return role
        # Todos presentes → spawnear explorador por defecto (máxima diversidad)
        return "explorador"

    def _store_sect_legacy(self, sect: SectRecord, reason: str):
        """Guarda el legado de una secta en la Matriarca."""
        import torch
        import torch.nn.functional as F

        embd_dim = self.matriarca.cfg.embd_dim

        # Codificar la secta como vector
        role_map = {r: i for i, r in enumerate(KNOWN_SECT_ROLES.keys())}
        role_idx = role_map.get(sect.role_type, len(role_map))

        raw = torch.zeros(max(8, embd_dim), dtype=torch.float32)
        raw[0] = float(role_idx) / max(len(role_map), 1)
        raw[1] = sect.priority
        raw[2] = sect.avg_fitness
        raw[3] = float(sect.n_agents) / 10.0
        raw[4] = min(1.0, sect.age / 3600.0)
        embedding = raw[:embd_dim]

        text = (
            f"[SECT_LEGACY] sect={sect.sect_id} role={sect.role_type} "
            f"fitness={sect.avg_fitness:.3f} n_agents={sect.n_agents} "
            f"age={sect.age:.0f}s reason={reason}"
        )

        self.matriarca.add(
            embedding=embedding,
            text=text,
            importance=max(sect.avg_fitness, 0.1),
        )
