"""
DolphinPool — Delfines Dinámicos 🐬
=====================================
El número de delfines escala con el tamaño de la red.

Red pequeña (1 nodo)  → 1 delfín (comportamiento actual)
Red mediana (2-4)     → 2 delfines
Red grande (5+)       → 3 delfines
Red muy grande (10+)  → 4 delfines (máximo)

Cada delfín tiene su propia frecuencia de ecolocalización (bias aprendido).
El acoustic_map final = promedio ponderado por confianza de todos los delfines.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Callable

from src.agents.dolphin_agent import DolphinAgent, DolphinConfig, DolphinSwarmBridge


def _target_pool_size(n_nodes: int) -> int:
    """Cuántos delfines debe haber según nodos conectados."""
    if n_nodes <= 1:
        return 1
    elif n_nodes <= 4:
        return 2
    elif n_nodes <= 9:
        return 3
    else:
        return 4


class DolphinPool(nn.Module):
    """
    Pool de delfines que escala dinámicamente con la red.

    Uso:
        pool = DolphinPool(base_cfg, device="cuda")
        pool.scale_to_network(n_nodes=3)   # añade delfín si hace falta
        feromon, info = pool.forward(input_ids)
    """

    MAX_DOLPHINS = 4
    MIN_DOLPHINS = 1

    def __init__(self, base_cfg: DolphinConfig, device: str = "cpu"):
        super().__init__()
        self.base_cfg = base_cfg
        self.device = device

        # Empezamos con 1 delfín (igual que antes)
        self.dolphins = nn.ModuleList([
            DolphinSwarmBridge(self._make_cfg(0), device=device)
        ])

        self._n_nodes = 1
        self._on_scale_callbacks: List[Callable] = []

    def _make_cfg(self, dolphin_idx: int) -> DolphinConfig:
        """Config por delfín — cada uno tiene agent_id único."""
        cfg = DolphinConfig(
            vocab_size=self.base_cfg.vocab_size,
            n_embd=self.base_cfg.n_embd,
            feromon_dim=self.base_cfg.feromon_dim,
            identity_dim=self.base_cfg.identity_dim,
            echo_dim=self.base_cfg.echo_dim,
            n_pings=self.base_cfg.n_pings,
            echo_layers=self.base_cfg.echo_layers,
            sleep_dim=self.base_cfg.sleep_dim,
            sleep_buffer_size=self.base_cfg.sleep_buffer_size,
            sleep_decay=self.base_cfg.sleep_decay,
            agent_id=100 + dolphin_idx,  # IDs 100+ para delfines, evita colisión con hormigas
            n_agents=self.base_cfg.n_agents,
            dropout=0.0,  # inference mode
        )
        return cfg

    def scale_to_network(self, n_nodes: int) -> List[dict]:
        """
        Ajusta el número de delfines según nodos conectados.
        Returns: lista de eventos {"type": "spawn"|"retire", "dolphin_idx": int}
        """
        self._n_nodes = n_nodes
        target = _target_pool_size(n_nodes)
        target = max(self.MIN_DOLPHINS, min(self.MAX_DOLPHINS, target))
        events = []

        while len(self.dolphins) < target:
            idx = len(self.dolphins)
            new_cfg = self._make_cfg(idx)
            new_dolphin = DolphinSwarmBridge(new_cfg, device=self.device).to(self.device)
            # Hereda pesos del primer delfín (misma base, distinta identidad)
            if len(self.dolphins) > 0:
                state = {k: v.clone() for k, v in self.dolphins[0].state_dict().items()
                         if "identity" not in k}  # identidad no se copia
                new_dolphin.load_state_dict(state, strict=False)
            self.dolphins.append(new_dolphin)
            events.append({"type": "spawn", "dolphin_idx": idx})

        while len(self.dolphins) > target:
            retired_idx = len(self.dolphins) - 1
            self.dolphins = nn.ModuleList(list(self.dolphins)[:-1])
            events.append({"type": "retire", "dolphin_idx": retired_idx})

        for cb in self._on_scale_callbacks:
            cb(len(self.dolphins), events)

        return events

    def on_scale(self, fn: Callable):
        """Registrar callback cuando cambia el pool."""
        self._on_scale_callbacks.append(fn)
        return fn

    def forward(self, idx: torch.Tensor) -> tuple:
        """
        Todos los delfines procesan en paralelo.
        El resultado es el promedio ponderado por confianza.

        Returns:
            feromon: (B, feromon_dim) — mapa acústico combinado
            info: dict con detalles de cada delfín
        """
        if len(self.dolphins) == 1:
            # Fast path: un solo delfín, sin overhead
            return self.dolphins[0](idx)

        feromons = []
        confidences = []
        all_info = []

        for dolphin in self.dolphins:
            f, info = dolphin(idx)
            conf = info.get("confidence", 0.5)
            if isinstance(conf, torch.Tensor):
                conf = conf.mean().item()
            feromons.append(f)
            confidences.append(conf)
            all_info.append(info)

        # Promedio ponderado por confianza
        weights = torch.softmax(torch.tensor(confidences, dtype=torch.float32), dim=0)
        feromon = sum(w * f for w, f in zip(weights.tolist(), feromons))

        combined_info = {
            "n_dolphins": len(self.dolphins),
            "confidences": confidences,
            "weights": weights.tolist(),
            "feromon_norm": feromon.norm().item(),
            # sleep_for_matriarca del primer delfín (más veterano)
            "sleep_for_matriarca": all_info[0].get("sleep_for_matriarca"),
            "confidence": max(confidences),
        }

        return feromon, combined_info

    @property
    def n_dolphins(self) -> int:
        return len(self.dolphins)

    @property
    def primary(self) -> DolphinSwarmBridge:
        """El delfín principal (primero, más veterano)."""
        return self.dolphins[0]

    def save_sleep_states(self, path_prefix: str):
        """Guarda el sleep_state de todos los delfines."""
        for i, d in enumerate(self.dolphins):
            d.dolphin.sleep_state.state_dict()
            # Se persiste via el checkpoint normal del swarm

    def status(self) -> dict:
        return {
            "n_dolphins": self.n_dolphins,
            "n_nodes": self._n_nodes,
            "target": _target_pool_size(self._n_nodes),
            "dolphins": [
                {
                    "idx": i,
                    "agent_id": d.dolphin.cfg.agent_id,
                    "sleep_norm": d.dolphin.sleep_state.get_state().norm().item(),
                }
                for i, d in enumerate(self.dolphins)
            ],
        }
