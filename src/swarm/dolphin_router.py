"""
Lixy — EcholocationRouter: El Delfín como Sistema de Enrutamiento 🐬
====================================================================
El Delfín NO es una secta. Es el SONAR del enjambre.

Función:
  1. Llega una tarea al enjambre
  2. El Delfín lanza pings de ecolocalización (ya existente)
  3. Los ecos "rebotan" en las sectas disponibles → PingResponse
  4. EcholocationRouter construye el mapa acústico y ELIGE la secta correcta
  5. La tarea se deriva a esa secta
  6. El Delfín integra la respuesta final

Métricas para el enrutamiento:
  - Especialización de secta: ¿qué tan buena en este tipo de tarea?
  - Fitness actual: ¿está en forma?
  - Carga: ¿tiene capacidad disponible?
  - Historial: ¿ha resuelto problemas similares? (Matriarca)
  - Afinidad acústica: ¿qué tan bien resuena con el acoustic_map del delfín?

Phase B (sueño unihemisférico):
  - diversity > 0.7 → modo reposo (procesamiento ligero)
  - diversity < 0.3 → modo agresivo (máxima activación)
  - Siempre un hemisferio activo: buffer circular de acoustic_maps persistente
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from src.swarm.sect_manager import SectRecord, SectManager


# ─── PingResponse ─────────────────────────────────────────────────────────────

@dataclass
class PingResponse:
    """
    Respuesta de una secta al ping del delfín.
    'Yo puedo', 'no puedo', 'parcialmente'.
    """
    sect_id: str
    role_type: str
    can_handle: float        # [0, 1] — confianza en poder manejar la tarea
    fitness: float           # fitness actual de la secta
    load_factor: float       # [0, 1] — 0 = libre, 1 = saturada
    latency_ms: float        # latencia histórica promedio
    acoustic_affinity: float # similitud con el acoustic_map del delfín

    @property
    def score(self) -> float:
        """Score compuesto para comparar sectas."""
        availability = 1.0 - self.load_factor
        return (
            0.35 * self.can_handle
            + 0.25 * self.fitness
            + 0.20 * availability
            + 0.20 * self.acoustic_affinity
        )


@dataclass
class RouteDecision:
    """Decisión de enrutamiento del delfín."""
    primary_sect: str                    # secta principal seleccionada
    secondary_sects: List[str]           # sectas secundarias para multi-secta
    confidence: float                    # confianza en la decisión
    acoustic_map: Optional[torch.Tensor] # mapa acústico usado para la decisión
    ping_responses: List[PingResponse]   # respuestas de todas las sectas
    mode: str = "single"                 # "single" | "multi" | "broadcast"
    reason: str = ""


# ─── SectPingEncoder ──────────────────────────────────────────────────────────

class SectPingEncoder(nn.Module):
    """
    Encoder que transforma el acoustic_map del delfín + perfil de secta
    en una puntuación de afinidad acústica.

    El delfín "pregunta" a cada secta: ¿puedes manejar este problema?
    La secta responde con un score de afinidad basado en su identidad.
    """

    def __init__(self, acoustic_map_dim: int = 128, sect_embed_dim: int = 64):
        super().__init__()
        self.acoustic_map_dim = acoustic_map_dim
        self.sect_embed_dim = sect_embed_dim

        # Embeddings aprendidos por tipo de secta (roles conocidos)
        self.role_embeddings = nn.Embedding(16, sect_embed_dim)  # hasta 16 roles

        # Scorer: combina acoustic_map + embedding de secta → afinidad
        self.scorer = nn.Sequential(
            nn.Linear(acoustic_map_dim + sect_embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Mapa de roles conocidos → índice
        self._role_to_idx: Dict[str, int] = {}
        self._next_idx = 0

    def _role_idx(self, role_type: str) -> int:
        if role_type not in self._role_to_idx:
            self._role_to_idx[role_type] = self._next_idx % 16
            self._next_idx += 1
        return self._role_to_idx[role_type]

    def forward(
        self,
        acoustic_map: torch.Tensor,     # (feromon_dim,) — señal del delfín
        role_type: str,
    ) -> float:
        """Retorna afinidad acústica [0, 1] para una secta."""
        role_idx = torch.tensor(
            self._role_idx(role_type),
            device=acoustic_map.device,
        )
        sect_emb = self.role_embeddings(role_idx)  # (sect_embed_dim,)

        # Aplanar si acoustic_map tiene dimensión extra (B, feromon_dim) → (feromon_dim,)
        am = acoustic_map.float()
        if am.dim() > 1:
            am = am.squeeze(0)

        # Concatenar y puntuar
        combined = torch.cat([
            am,
            sect_emb.float(),
        ], dim=0).unsqueeze(0)  # (1, feromon_dim + sect_embed_dim)

        score = self.scorer(combined)
        return score.item()


# ─── EcholocationRouter ───────────────────────────────────────────────────────

class EcholocationRouter(nn.Module):
    """
    El corazón del delfín como enrutador.

    Recibe el acoustic_map del DolphinAgent y las sectas disponibles,
    y decide a qué secta(s) derivar la tarea.

    Flujo:
        acoustic_map (del DolphinAgent)
            ↓
        Ping a cada secta → PingResponse[]
            ↓
        Scorer: combina can_handle + fitness + load + latency + affinity
            ↓
        RouteDecision: primary + secondary (multi-secta si complejidad alta)
    """

    # Umbral para activar multi-secta (complejidad del problema)
    MULTI_SECT_THRESHOLD = 0.75
    # Umbral mínimo de score para incluir una secta secundaria
    SECONDARY_MIN_SCORE = 0.4

    def __init__(self, acoustic_map_dim: int = 128):
        super().__init__()
        self.acoustic_map_dim = acoustic_map_dim
        self.ping_encoder = SectPingEncoder(acoustic_map_dim=acoustic_map_dim)

        # Detector de complejidad: ¿necesita multi-secta?
        self.complexity_head = nn.Sequential(
            nn.Linear(acoustic_map_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def ping_sect(
        self,
        acoustic_map: torch.Tensor,
        sect: "SectRecord",
    ) -> PingResponse:
        """
        Lanza un ping a una secta y obtiene su PingResponse.
        En una red distribuida real, esto sería un mensaje de red.
        Aquí simulamos la respuesta con las métricas de la secta.
        """
        # Afinidad acústica: qué tan bien resuena el acoustic_map con esta secta
        acoustic_affinity = self.ping_encoder(acoustic_map, sect.role_type)

        # Fitness de la secta
        fitness = sect.avg_fitness

        # Load factor: basado en número de agentes vs capacidad
        # (simple heurística: más agentes = más carga, capped at 1)
        load_factor = min(1.0, sect.n_agents / 10.0)

        # Can handle: combinación de fitness + afinidad acústica
        can_handle = min(1.0, fitness * 0.6 + acoustic_affinity * 0.4)

        return PingResponse(
            sect_id=sect.sect_id,
            role_type=sect.role_type,
            can_handle=can_handle,
            fitness=fitness,
            load_factor=load_factor,
            latency_ms=0.0,         # se actualizará con historial real
            acoustic_affinity=acoustic_affinity,
        )

    def route(
        self,
        acoustic_map: torch.Tensor,
        sects: List["SectRecord"],
        matriarca=None,
    ) -> RouteDecision:
        """
        Decide a qué secta(s) derivar la tarea.

        Args:
            acoustic_map: señal del delfín (feromon_dim,)
            sects: lista de sectas disponibles
            matriarca: consulta historial de resoluciones similares

        Returns:
            RouteDecision con primary + secondary sects
        """
        # Normalizar: (B, D) → (D,) — el router opera sobre un único acoustic_map
        am = acoustic_map.float()
        if am.dim() > 1:
            am = am.squeeze(0)

        if not sects:
            return RouteDecision(
                primary_sect="none",
                secondary_sects=[],
                confidence=0.0,
                acoustic_map=acoustic_map,
                ping_responses=[],
                mode="broadcast",
                reason="no sects available",
            )

        # 1. Ping a cada secta
        responses = []
        with torch.no_grad():
            for sect in sects:
                resp = self.ping_sect(am, sect)
                responses.append(resp)

        # 2. Ordenar por score compuesto
        responses.sort(key=lambda r: r.score, reverse=True)

        # 3. Detectar complejidad del problema (multi-secta?)
        with torch.no_grad():
            complexity = self.complexity_head(
                am.unsqueeze(0)
            ).item()

        # 4. Seleccionar sectas
        primary = responses[0]
        secondary = []

        if complexity >= self.MULTI_SECT_THRESHOLD and len(responses) > 1:
            # Problema complejo: incluir sectas secundarias que superen el umbral
            for resp in responses[1:]:
                if resp.score >= self.SECONDARY_MIN_SCORE:
                    secondary.append(resp.sect_id)

        mode = "multi" if secondary else "single"
        confidence = primary.score

        reason = (
            f"primary={primary.role_type}(score={primary.score:.2f}) "
            f"complexity={complexity:.2f} "
            f"mode={mode}"
        )

        return RouteDecision(
            primary_sect=primary.sect_id,
            secondary_sects=secondary,
            confidence=confidence,
            acoustic_map=acoustic_map.clone(),
            ping_responses=responses,
            mode=mode,
            reason=reason,
        )

    def integrate_responses(
        self,
        primary_output: torch.Tensor,
        secondary_outputs: Optional[List[torch.Tensor]] = None,
        decision: Optional[RouteDecision] = None,
    ) -> torch.Tensor:
        """
        Integra respuestas de múltiples sectas en una respuesta final.
        En modo single: devuelve el output de la secta principal.
        En modo multi: promedio ponderado por score.
        """
        if not secondary_outputs:
            return primary_output

        # Pesos de integración basados en scores del RouteDecision
        if decision and decision.ping_responses:
            score_map = {r.sect_id: r.score for r in decision.ping_responses}
            weights = [score_map.get(sid, 0.5) for sid in decision.secondary_sects]
            primary_weight = score_map.get(decision.primary_sect, 1.0)
        else:
            weights = [0.5] * len(secondary_outputs)
            primary_weight = 1.0

        all_outputs = [primary_output] + secondary_outputs
        all_weights = torch.tensor(
            [primary_weight] + weights,
            dtype=primary_output.dtype,
            device=primary_output.device,
        )
        all_weights = F.softmax(all_weights, dim=0)

        return sum(w * out for w, out in zip(all_weights.tolist(), all_outputs))


# ─── Phase B: Sueño Unihemisférico Adaptativo ────────────────────────────────

class AdaptiveSleepController:
    """
    Phase B: Controla el modo de sueño del delfín según la diversidad del enjambre.

    diversity > 0.7 → modo REPOSO    (procesamiento ligero, 1 hemisferio)
    diversity < 0.3 → modo AGRESIVO  (máxima activación, full processing)
    0.3 ≤ d ≤ 0.7   → modo NORMAL

    El delfín NUNCA duerme completamente — siempre hay un hemisferio activo.
    Los acoustic_maps se guardan en buffer circular persistente.
    """

    # Umbrales de modo
    REST_THRESHOLD = 0.7     # diversidad alta → reposo
    ACTIVE_THRESHOLD = 0.3   # diversidad baja → agresivo

    def __init__(self, buffer_size: int = 32, persist_path: Optional[str] = None):
        self.persist_path = persist_path
        self._acoustic_buffer: deque = deque(maxlen=buffer_size)
        self._diversity_history: deque = deque(maxlen=100)
        self._current_mode = "normal"
        self._mode_changes = 0

    @property
    def mode(self) -> str:
        return self._current_mode

    def update_diversity(self, diversity: float) -> str:
        """
        Actualiza el modo basándose en la diversidad actual.
        Retorna el modo resultante.
        """
        self._diversity_history.append(diversity)

        if len(self._diversity_history) >= 5:
            # Usar promedio de últimas 5 lecturas para estabilidad
            recent_avg = sum(list(self._diversity_history)[-5:]) / 5
        else:
            recent_avg = diversity

        old_mode = self._current_mode
        if recent_avg > self.REST_THRESHOLD:
            self._current_mode = "rest"
        elif recent_avg < self.ACTIVE_THRESHOLD:
            self._current_mode = "aggressive"
        else:
            self._current_mode = "normal"

        if old_mode != self._current_mode:
            self._mode_changes += 1

        return self._current_mode

    def should_process_full(self) -> bool:
        """¿Debe el delfín procesar completamente (no en reposo)?"""
        return self._current_mode != "rest"

    def activation_scale(self) -> float:
        """
        Factor de escala para la activación del delfín.
        rest=0.4 (ligero), normal=1.0, aggressive=1.5 (máximo).
        """
        scales = {"rest": 0.4, "normal": 1.0, "aggressive": 1.5}
        return scales.get(self._current_mode, 1.0)

    def store_acoustic_map(self, acoustic_map: torch.Tensor, metadata: Optional[dict] = None):
        """Guarda un acoustic_map en el buffer circular persistente."""
        entry = {
            "map": acoustic_map.detach().cpu().clone(),
            "ts": time.time(),
            "mode": self._current_mode,
            "meta": metadata or {},
        }
        self._acoustic_buffer.append(entry)

    def recent_acoustic_maps(self, n: int = 5) -> List[torch.Tensor]:
        """Últimos N acoustic_maps del buffer."""
        buf = list(self._acoustic_buffer)
        return [e["map"] for e in buf[-n:]]

    def save(self, path: str):
        """Persiste el buffer a disco."""
        import json
        import pathlib
        data = {
            "mode": self._current_mode,
            "mode_changes": self._mode_changes,
            "acoustic_buffer": [
                {
                    "map": e["map"].tolist(),
                    "ts": e["ts"],
                    "mode": e["mode"],
                    "meta": e["meta"],
                }
                for e in self._acoustic_buffer
            ],
        }
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        """Carga el buffer desde disco."""
        import json
        try:
            with open(path) as f:
                data = json.load(f)
            self._current_mode = data.get("mode", "normal")
            self._mode_changes = data.get("mode_changes", 0)
            for e in data.get("acoustic_buffer", []):
                self._acoustic_buffer.append({
                    "map": torch.tensor(e["map"]),
                    "ts": e["ts"],
                    "mode": e["mode"],
                    "meta": e.get("meta", {}),
                })
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # Primer arranque — buffer vacío

    def stats(self) -> dict:
        return {
            "mode": self._current_mode,
            "mode_changes": self._mode_changes,
            "buffer_size": len(self._acoustic_buffer),
            "diversity_history": len(self._diversity_history),
        }
