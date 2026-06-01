"""
lsp_merge.py — Merge standalone para FeromonV2Payload.
Útil en tests y en el relay VPS.
"""

from __future__ import annotations

from typing import List

from .lsp_v2 import FeromonV2Payload, DIM_FLOAT16


def merge_feromons(
    payloads: List[FeromonV2Payload],
    strategy: str = "fitness_weighted",
) -> FeromonV2Payload:
    """
    Merge de múltiples FeromonV2Payload.

    Estrategias:
        "fitness_weighted" — promedio ponderado por fitness (default)
        "equal_weight"     — promedio simple sin ponderar
    """
    import torch

    if not payloads:
        raise ValueError("payloads list is empty")
    if len(payloads) == 1:
        return payloads[0]

    if strategy == "fitness_weighted":
        total_fitness = sum(p.fitness for p in payloads)
        if total_fitness <= 0:
            weights = [1.0 / len(payloads)] * len(payloads)
        else:
            weights = [p.fitness / total_fitness for p in payloads]
    elif strategy == "equal_weight":
        weights = [1.0 / len(payloads)] * len(payloads)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    merged_vec = torch.zeros_like(payloads[0].feromon)
    for w, p in zip(weights, payloads):
        merged_vec += p.feromon * w

    avg_fitness = sum(p.fitness for p in payloads) / len(payloads)
    min_ttl = min(p.ttl for p in payloads)
    max_step = max(p.step for p in payloads)
    max_ts = max(p.timestamp_ms for p in payloads)

    return FeromonV2Payload(
        feromon=merged_vec,
        ttl=min_ttl,
        step=max_step,
        fitness=avg_fitness,
        timestamp_ms=max_ts,
        dim_type=payloads[0].dim_type,
    )


def decay_feromon(
    payload: FeromonV2Payload,
    hops: int = 1,
    decay: float = 0.95,
) -> FeromonV2Payload:
    """
    Aplica decay^hops al vector de feromona y decrementa TTL en hops.
    factor = decay ^ hops
    """
    factor = decay ** hops
    return FeromonV2Payload(
        feromon=payload.feromon * factor,
        ttl=max(0, payload.ttl - hops),
        step=payload.step,
        fitness=payload.fitness,
        timestamp_ms=payload.timestamp_ms,
        dim_type=payload.dim_type,
    )
