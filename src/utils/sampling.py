"""
Lixy-0.1 — Utilidades de Sampling
====================================
Sampling avanzado para generación de texto.

Implementa:
- Top-k sampling
- Top-p (nucleus) sampling — corta la cola larga de probabilidades
- Repetition penalty — penaliza tokens ya generados
- Combined: top-k + top-p + rep_penalty en un solo paso eficiente

Uso:
    from src.utils.sampling import sample_token

    next_tok = sample_token(
        logits,             # [B, vocab] — logits del último paso
        temperature=0.8,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.3,
        generated_ids=x,   # [B, T] — todos los tokens ya generados
    )
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Optional


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    penalty: float = 1.3,
    recent_penalty: float = None,
    recent_window: int = 8,
) -> torch.Tensor:
    """
    Penaliza tokens que ya aparecen en generated_ids.

    Mecanismo:
    - Si logit > 0: dividir por penalty (reduce probabilidad)
    - Si logit < 0: multiplicar por penalty (la hace más negativa)
    - recent_penalty: penalización adicional para los últimos `recent_window` tokens
      (más agresiva para evitar loops inmediatos)

    Referencia: CTRL paper (Keskar et al., 2019)
    """
    if penalty == 1.0 and (recent_penalty is None or recent_penalty == 1.0):
        return logits

    _recent_penalty = recent_penalty if recent_penalty is not None else penalty * 2.0

    B, vocab = logits.shape
    for b in range(B):
        context = generated_ids[b, -64:].tolist()
        recent = set(generated_ids[b, -recent_window:].tolist())
        unique_ids = set(context)
        for tid in unique_ids:
            if 0 <= tid < vocab:
                p = _recent_penalty if tid in recent else penalty
                if logits[b, tid] > 0:
                    logits[b, tid] /= p
                else:
                    logits[b, tid] *= p

    return logits


def apply_top_p(
    logits: torch.Tensor,
    top_p: float = 0.92,
) -> torch.Tensor:
    """
    Nucleus (top-p) sampling: mantiene solo los tokens cuya probabilidad
    acumulada <= top_p. Elimina la cola larga de tokens raros.

    Args:
        logits: [B, vocab] — ya divididos por temperatura
        top_p:  0.0-1.0; 1.0 = desactivado; 0.9 = núcleo 90%

    Returns:
        logits con -inf en tokens fuera del núcleo
    """
    if top_p >= 1.0:
        return logits

    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Eliminar tokens donde la prob acumulada ya supera top_p
    # Shift para incluir el token que cruza el umbral
    sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
    sorted_indices_to_remove[:, 0] = False  # siempre mantener el top-1

    # Mapear de vuelta al orden original
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, float("-inf"))

    return logits


def sample_token(
    logits: torch.Tensor,
    generated_ids: Optional[torch.Tensor] = None,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: float = 0.92,
    repetition_penalty: float = 1.3,
    recent_penalty: float = None,
    recent_window: int = 8,
    greedy: bool = False,
) -> torch.Tensor:
    """
    Sampling completo: rep_penalty → temperatura → top_k → top_p → multinomial.

    Args:
        logits:             [B, vocab] o [B, T, vocab] — si T dims, usa último token
        generated_ids:      [B, T] — tokens ya generados (para rep_penalty)
        temperature:        float > 0; 0.8 = balanceado; 1.0 = sin cambio
        top_k:              int o None; 50 = mantener top-50 tokens
        top_p:              float 0-1; 0.92 = nucleus 92%
        repetition_penalty: float >= 1.0; 1.3 = moderada; 1.0 = sin efecto
        greedy:             si True, devuelve argmax (determinístico)

    Returns:
        next_token: [B, 1] — índice del token seleccionado
    """
    # Asegurar shape [B, vocab]
    if logits.dim() == 3:
        logits = logits[:, -1, :]  # [B, T, vocab] → [B, vocab]

    logits = logits.float().clone()

    # 1. Repetition penalty (antes de temperatura para escala correcta)
    if generated_ids is not None and repetition_penalty != 1.0:
        logits = apply_repetition_penalty(
            logits, generated_ids, repetition_penalty,
            recent_penalty=recent_penalty, recent_window=recent_window,
        )

    # 2. Temperatura
    if temperature != 1.0 and temperature > 0:
        logits = logits / temperature

    # 3. Greedy (bypass todo el sampling)
    if greedy:
        return logits.argmax(dim=-1, keepdim=True)

    # 4. Top-k
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits[logits < v[:, [-1]]] = float("-inf")

    # 5. Top-p (nucleus)
    if top_p < 1.0:
        logits = apply_top_p(logits, top_p)

    # 6. Softmax + multinomial
    probs = F.softmax(logits, dim=-1)

    # Seguridad: si todos son -inf (no debería pasar, pero por robustez)
    if probs.isnan().any() or (probs == 0).all():
        return logits.new_zeros(logits.shape[0], 1, dtype=torch.long)

    return torch.multinomial(probs, num_samples=1)


# ─── Defaults recomendados ─────────────────────────────────────────────────────

SAMPLING_DEFAULTS = {
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.92,
    "repetition_penalty": 1.3,
    "recent_penalty": 3.0,      # penalizar los últimos 8 tokens más agresivamente
    "recent_window": 8,
}

SAMPLING_CREATIVE = {
    "temperature": 1.0,
    "top_k": 100,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "recent_penalty": 2.0,
    "recent_window": 4,
}

SAMPLING_FOCUSED = {
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.85,
    "repetition_penalty": 1.5,
    "recent_penalty": 5.0,
    "recent_window": 12,
}
