"""
Lixy-0.1 — Gestor del Ciclo de Vida de Hormigas 🐜
=====================================================
Las hormigas nacen, viven, y mueren — como una colonia real.

Características:
- Hormigas con bajo fitness durante mucho tiempo → mueren
- Antes de morir → transfieren legado genético a Matriarca
- Nuevas hormigas heredan patrones del mejor legado disponible
- El enjambre nunca baja de MIN_ANTS ni sube de MAX_ANTS
- Diversidad baja → spawn de nueva hormiga
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from src.swarm.orchestrator import LixySwarm
    from src.matriarca.matriarca import Matriarca


class AntLifecycleManager:
    """
    Gestiona el ciclo de vida de las hormigas del enjambre.
    Las hormigas nacen, viven, y mueren — como una colonia real.
    """

    # Umbrales
    LOW_FITNESS_THRESHOLD = 0.3      # fitness mínimo para sobrevivir
    LOW_FITNESS_STEPS = 500          # steps con fitness bajo antes de morir
    MIN_ANTS = 2                     # nunca menos de 2 hormigas
    MAX_ANTS = 8                     # máximo de hormigas
    DIVERSITY_THRESHOLD = 0.4        # diversidad mínima del enjambre

    # Prefijo para identificar memorias de legado en la Matriarca
    LEGACY_PREFIX = "[LEGACY]"

    def __init__(self, swarm: "LixySwarm", matriarca: Optional["Matriarca"] = None):
        self.swarm = swarm
        self.matriarca = matriarca
        self._low_fitness_counters: Dict[str, int] = {}  # ant_id → steps con bajo fitness
        self._birth_count = 0
        self._death_count = 0

    # ─── API pública ──────────────────────────────────────────────────────────────

    def tick(self, step: int, swarm_diversity: float, n_connected_nodes: int = 1) -> List[dict]:
        """
        Llamar cada N steps durante training o runtime.
        Evalúa si deben nacer o morir hormigas.

        Args:
            step: step global de entrenamiento
            swarm_diversity: diversidad del enjambre [0, 1]  (e.g. std de feromonas)
            n_connected_nodes: nodos distribuidos conectados

        Returns:
            list of events [{"type": "birth"|"death", "ant_id": str, ...}]
        """
        events = []

        # 1. Actualizar contadores de bajo fitness
        for ant in list(self.swarm.agents):
            fitness = self._get_fitness(ant)
            aid = str(ant.config.agent_id)
            if fitness < self.LOW_FITNESS_THRESHOLD:
                self._low_fitness_counters[aid] = self._low_fitness_counters.get(aid, 0) + 1
            else:
                self._low_fitness_counters[aid] = 0

        # 2. Matar hormigas débiles (si hay más del mínimo)
        if len(self.swarm.agents) > self.MIN_ANTS:
            for ant in list(self.swarm.agents):
                aid = str(ant.config.agent_id)
                if self._low_fitness_counters.get(aid, 0) >= self.LOW_FITNESS_STEPS:
                    event = self._kill_ant(ant, reason="low_fitness")
                    events.append(event)
                    if len(self.swarm.agents) <= self.MIN_ANTS:
                        break

        # 3. Nacer hormigas nuevas si:
        #    - Diversidad baja Y hay espacio
        #    - Nuevos nodos conectados
        should_spawn = (
            (swarm_diversity < self.DIVERSITY_THRESHOLD and len(self.swarm.agents) < self.MAX_ANTS)
            or (n_connected_nodes > 1 and len(self.swarm.agents) < min(3 + n_connected_nodes, self.MAX_ANTS))
        )
        if should_spawn:
            event = self._spawn_ant(step)
            events.append(event)

        return events

    def stats(self) -> dict:
        """Estadísticas del lifecycle manager."""
        return {
            "total_ants": len(self.swarm.agents),
            "births": self._birth_count,
            "deaths": self._death_count,
            "low_fitness_counters": dict(self._low_fitness_counters),
        }

    # ─── Fitness ──────────────────────────────────────────────────────────────────

    def _get_fitness(self, ant) -> float:
        """Lee fitness del SpecializationTracker."""
        aid = str(ant.config.agent_id)
        spec = getattr(self.swarm, "specialization", None)
        if spec is not None:
            current = getattr(spec, "current", {})
            if aid in current:
                entry = current[aid]
                # SpecializationTracker.current tiene AgentFitness objects
                if hasattr(entry, "fitness"):
                    return float(entry.fitness)
                if isinstance(entry, dict):
                    return float(entry.get("fitness", 0.5))
        return 0.5

    # ─── Muerte ───────────────────────────────────────────────────────────────────

    def _kill_ant(self, ant, reason: str) -> dict:
        """
        Mata una hormiga:
        1. Extrae y transfiere su legado a la Matriarca
        2. La elimina del enjambre
        """
        aid = str(ant.config.agent_id)
        fitness = self._get_fitness(ant)

        # Legado genético → Matriarca
        legacy = self._extract_legacy(ant, fitness, reason)
        if self.matriarca is not None:
            # Proyectar el pattern_embedding a embd_dim de la Matriarca si hace falta
            embd_dim = self.matriarca.cfg.embd_dim
            pat = legacy["pattern_embedding"]  # [D]
            if pat.shape[0] != embd_dim:
                if pat.shape[0] < embd_dim:
                    pat = F.pad(pat, (0, embd_dim - pat.shape[0]))
                else:
                    pat = pat[:embd_dim]

            text = (
                f"{self.LEGACY_PREFIX} ant_{aid} "
                f"role={legacy['role']} "
                f"fitness={legacy['fitness_avg']:.3f} "
                f"reason={reason}"
            )
            self.matriarca.add(
                embedding=pat,
                text=text,
                importance=legacy["fitness_avg"],
            )

        # Eliminar del enjambre (ModuleList no soporta remove directo)
        import torch.nn as nn
        self.swarm.agents = nn.ModuleList(
            [a for a in self.swarm.agents if str(a.config.agent_id) != aid]
        )
        self._low_fitness_counters.pop(aid, None)
        self._death_count += 1

        return {
            "type": "death",
            "ant_id": aid,
            "reason": reason,
            "fitness": fitness,
            "legacy_stored": self.matriarca is not None,
        }

    def _extract_legacy(self, ant, fitness: float, reason: str) -> dict:
        """Extrae la esencia de una hormiga antes de morir."""
        aid = str(ant.config.agent_id)
        spec = getattr(self.swarm, "specialization", None)
        role = "unknown"
        if spec is not None:
            current = getattr(spec, "current", {})
            entry = current.get(aid, {})
            if hasattr(entry, "fitness"):  # AgentFitness object — infer label separately
                role = spec._infer_label(aid) if hasattr(spec, "_infer_label") else "unknown"
            elif isinstance(entry, dict):
                role = entry.get("label", "unknown")

        # Patrón genético: identity_vec de la hormiga
        pattern = ant.identity_vec.detach().cpu()  # [identity_dim]

        return {
            "role": role,
            "fitness_avg": fitness,
            "pattern_embedding": pattern,
            "reason": reason,
        }

    # ─── Nacimiento ───────────────────────────────────────────────────────────────

    def _spawn_ant(self, step: int) -> dict:
        """
        Crea una nueva hormiga.
        Si hay legado en la Matriarca, la nueva hormiga hereda el mejor patrón.
        Copia los pesos del agente padre (mayor fitness) — sin entrenar desde cero.
        """
        import torch.nn as nn
        from src.agents.agent_base import AgentBase, AgentConfig

        # Nuevo ID: max actual + 1
        existing_ids = [a.config.agent_id for a in self.swarm.agents]
        new_id = max(existing_ids) + 1 if existing_ids else 0

        if not self.swarm.agents:
            return {"type": "birth", "ant_id": str(new_id), "inherited": False, "parent_id": None}

        ref_cfg = self.swarm.agents[0].config

        cfg = AgentConfig(
            block_size=ref_cfg.block_size,
            vocab_size=ref_cfg.vocab_size,
            n_layer=ref_cfg.n_layer,
            n_head=ref_cfg.n_head,
            n_embd=ref_cfg.n_embd,
            dropout=0.0,
            bias=ref_cfg.bias,
            feromon_dim=ref_cfg.feromon_dim,
            identity_dim=ref_cfg.identity_dim,
            agent_id=new_id,
            n_agents=len(self.swarm.agents) + 1,
        )

        device = next(self.swarm.agents[0].parameters()).device
        new_ant = AgentBase(cfg).to(device)

        # Herencia genética: buscar el mejor legado en la Matriarca
        inherited = False
        if self.matriarca is not None:
            legacy_metas = [
                (i, m) for i, m in enumerate(self.matriarca.metadata)
                if m.get("text", "").startswith(self.LEGACY_PREFIX)
                and m.get("importance", 0.0) > 0.3
            ]
            if legacy_metas:
                best_idx, best_meta = max(legacy_metas, key=lambda x: x[1]["importance"])
                # Recuperar el embedding del banco
                embs = self.matriarca.get_embeddings("cpu")
                if best_idx < embs.shape[0]:
                    legacy_emb = embs[best_idx]  # [embd_dim]
                    id_dim = new_ant.identity_vec.shape[0]
                    if legacy_emb.shape[0] >= id_dim:
                        legacy_slice = legacy_emb[:id_dim]
                    else:
                        legacy_slice = F.pad(legacy_emb, (0, id_dim - legacy_emb.shape[0]))
                    with torch.no_grad():
                        new_ant.identity_vec.copy_(legacy_slice.to(device))
                    inherited = True

        # Copiar pesos del padre (agente de mayor fitness) — excluye identity_vec
        parent = max(self.swarm.agents, key=lambda a: self._get_fitness(a))
        parent_state = {k: v.clone() for k, v in parent.state_dict().items()}
        parent_state.pop("identity_vec", None)  # identidad única — no copiar
        new_ant.load_state_dict(parent_state, strict=False)

        # Añadir al enjambre
        self.swarm.agents = nn.ModuleList(list(self.swarm.agents) + [new_ant])
        self._birth_count += 1

        return {
            "type": "birth",
            "ant_id": str(new_id),
            "inherited": inherited,
            "parent_id": str(parent.config.agent_id),
        }
