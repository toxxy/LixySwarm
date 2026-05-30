"""
Lixy-0.1 — Especialización Dinámica de Hormigas
=================================================
Módulo que adapta el rol activo de cada AntAgent según el tipo de query
detectado en runtime, guiado por la Matriarca.

Principio bio-inspirado:
  Las hormigas de un enjambre no hacen siempre lo mismo — reclutan
  según las necesidades actuales. Una hormiga exploradora puede
  temporalmente actuar como refinadora si la tarea lo requiere.

En LixySwarm esto se traduce en:
  - Analizar la query del usuario → clasificar tipo de tarea
  - La Matriarca emite "infrasónidos de rol" — ajusta el bias de
    confianza por agente según el tipo de tarea detectado
  - En el forward, los pesos de agregación reflejan el rol emergente

Tipos de tarea detectados:
  "técnica"      → refinador lidera (alta precisión)
  "exploratoria" → explorador lidera (alta diversidad)
  "creativa"     → explorador lidera (máxima divergencia)
  "factual"      → refinador/explotador lidera (recuperación exacta)
  "conversacional" → distribución equilibrada
  "razonamiento" → refinador lidera + temperatura baja
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


# ─── Clasificador de Query ────────────────────────────────────────────────────

# Patrones léxicos para clasificar el tipo de tarea
# (lightweight, sin dependencias extra)
_TASK_PATTERNS = {
    "técnica": [
        r'\b(código|code|función|function|error|bug|implementa|implement|'
        r'algorithm|algoritmo|compile|syntax|debug|class|def |import|'
        r'python|javascript|sql|docker|api|json|xml|regex)\b',
    ],
    "exploratoria": [
        r'\b(qué es|what is|qué son|cómo funciona|how does|explain|explica|'
        r'describe|cuéntame|tell me|overview|introduce|concept|concepto)\b',
    ],
    "creativa": [
        r'\b(escribe|write|crea|create|genera|generate|poem|poema|story|'
        r'historia|inventa|imagine|diseña|design|compone|compose)\b',
    ],
    "factual": [
        r'\b(quién|who|cuándo|when|dónde|where|cuánto|how many|how much|'
        r'capital|year|año|fecha|date|número|number|lista|list)\b',
    ],
    "razonamiento": [
        r'\b(por qué|why|razón|reason|causa|cause|porque|analiza|analyze|'
        r'compara|compare|diferencia|difference|ventaja|advantage|'
        r'debería|should|mejor|mejor opción)\b',
    ],
}

# Pesos de confianza por rol para cada tipo de tarea
# Formato: [agente_0_weight, agente_1_weight, agente_2_weight]
# (se aplican como multiplicadores sobre los confidence heads — no hardcoded)
_TASK_ROLE_WEIGHTS = {
    "técnica":        [1.4, 0.7, 0.9],   # refinador lidera
    "exploratoria":   [0.8, 1.5, 1.0],   # explorador lidera
    "creativa":       [0.7, 1.6, 1.0],   # explorador max
    "factual":        [1.3, 0.8, 1.1],   # refinador + aprendiendo
    "razonamiento":   [1.5, 0.9, 0.8],   # refinador dominante
    "conversacional": [1.0, 1.0, 1.0],   # equilibrado
}

# Temperatura recomendada por tipo de tarea
_TASK_TEMPERATURES = {
    "técnica":        0.5,
    "exploratoria":   0.85,
    "creativa":       1.0,
    "factual":        0.6,
    "razonamiento":   0.65,
    "conversacional": 0.8,
}


@dataclass
class TaskProfile:
    """Perfil de una query clasificada."""
    task_type: str           # tipo principal
    confidence: float        # confianza en la clasificación (0-1)
    role_weights: list       # [w0, w1, w2] multiplicadores de confianza
    temperature: float       # temperatura recomendada
    description: str         # descripción legible


def classify_query(text: str) -> TaskProfile:
    """
    Clasifica una query en un tipo de tarea usando patrones léxicos.

    Retorna un TaskProfile con el tipo, confianza y pesos de rol recomendados.
    Usa un enfoque ensemble: suma de matches por categoría.
    """
    text_lower = text.lower()
    scores = {}

    for task_type, patterns in _TASK_PATTERNS.items():
        score = 0
        for pattern in patterns:
            matches = len(re.findall(pattern, text_lower))
            score += matches
        scores[task_type] = score

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    if best_score == 0:
        # Sin señal clara → conversacional por defecto
        return TaskProfile(
            task_type="conversacional",
            confidence=0.3,
            role_weights=_TASK_ROLE_WEIGHTS["conversacional"],
            temperature=_TASK_TEMPERATURES["conversacional"],
            description="sin señal clara → distribución equilibrada",
        )

    # Confianza: score normalizado + diferencia con el segundo mejor
    total = sum(scores.values()) or 1
    second_best = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
    confidence = min(1.0, (best_score / total) * 2 + (best_score - second_best) * 0.1)

    role_descriptions = {
        "técnica":        "refinador lidera (precisión técnica)",
        "exploratoria":   "explorador lidera (amplitud de respuesta)",
        "creativa":       "explorador max (máxima diversidad)",
        "factual":        "refinador activo (recuperación exacta)",
        "razonamiento":   "refinador dominante (lógica rigurosa)",
        "conversacional": "distribución equilibrada",
    }

    return TaskProfile(
        task_type=best_type,
        confidence=confidence,
        role_weights=_TASK_ROLE_WEIGHTS[best_type],
        temperature=_TASK_TEMPERATURES[best_type],
        description=role_descriptions[best_type],
    )


# ─── Mixer de Confianza Dinámica ──────────────────────────────────────────────

class DynamicRoleAdapter:
    """
    Adapta los pesos de confianza de los agentes en runtime según el tipo de query.

    Se integra en RuntimeSession.turn() para ajustar la temperatura y los
    pesos de confianza antes de la generación.

    No modifica los pesos del modelo — solo sesga el sampling de confianza.
    """

    def __init__(self, n_agents: int = 3, verbose: bool = False):
        self.n_agents = n_agents
        self.verbose = verbose
        self._last_profile: Optional[TaskProfile] = None
        self._profile_history: list[TaskProfile] = []

    def get_weights_for_query(
        self,
        query: str,
        base_weights: Optional[torch.Tensor] = None,
        mix_ratio: float = 0.35,
    ) -> tuple[TaskProfile, torch.Tensor]:
        """
        Clasifica la query y devuelve los pesos de confianza ajustados.

        Args:
            query:        texto del usuario
            base_weights: [B, n_agents] o [n_agents] — pesos base de los confidence heads
                          Si None, usa distribución uniforme
            mix_ratio:    cuánto peso tiene el perfil dinámico vs. los confidence heads
                          0.0 = solo confidence heads; 1.0 = solo perfil dinámico; 0.35 = mezcla

        Returns:
            (profile, adjusted_weights) donde adjusted_weights tiene shape [n_agents]
        """
        profile = classify_query(query)
        self._last_profile = profile
        self._profile_history.append(profile)

        # Pesos dinámicos del perfil de tarea
        dynamic_weights = torch.tensor(
            profile.role_weights[:self.n_agents],
            dtype=torch.float32,
        )
        dynamic_weights = F.softmax(dynamic_weights, dim=0)  # normalizar

        if base_weights is None:
            # Sin pesos base → usar solo el perfil dinámico
            final_weights = dynamic_weights
        else:
            bw = base_weights.float()
            if bw.dim() > 1:
                bw = bw.mean(dim=0)  # [n_agents]
            bw = bw[:self.n_agents]
            bw = F.softmax(bw, dim=0)

            # Mezcla: (1-mix_ratio) * confidence_heads + mix_ratio * task_profile
            final_weights = (1 - mix_ratio) * bw + mix_ratio * dynamic_weights.to(bw.device)
            final_weights = F.softmax(final_weights, dim=0)

        if self.verbose:
            w = final_weights.tolist()
            print(
                f"  🐜 Rol dinámico: [{profile.task_type}] conf={profile.confidence:.2f} "
                f"→ pesos=[{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}] "
                f"temp={profile.temperature:.2f}"
            )

        return profile, final_weights

    @property
    def last_task_type(self) -> str:
        return self._last_profile.task_type if self._last_profile else "desconocido"

    def task_distribution(self) -> dict:
        """Distribución de tipos de tarea en el historial."""
        if not self._profile_history:
            return {}
        counts = {}
        for p in self._profile_history:
            counts[p.task_type] = counts.get(p.task_type, 0) + 1
        total = len(self._profile_history)
        return {k: round(v/total, 2) for k, v in sorted(counts.items(), key=lambda x: -x[1])}


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    adapter = DynamicRoleAdapter(n_agents=3, verbose=True)

    test_queries = [
        "¿Cómo implemento un transformer en Python con PyTorch?",
        "¿Qué es la inteligencia artificial y cómo funciona?",
        "Escribe un poema sobre el mar al atardecer",
        "¿Cuándo nació Einstein?",
        "¿Por qué es mejor usar Adam que SGD?",
        "Hola, ¿cómo estás?",
    ]

    print("=== Test DynamicRoleAdapter ===\n")
    for q in test_queries:
        profile, weights = adapter.get_weights_for_query(q)
        print(f"Query: '{q[:50]}'")
        print(f"  → {profile.task_type} ({profile.description})")
        print()

    print("Distribución:", adapter.task_distribution())
