"""
Lixy — Matriarca: Legado de Sectas + Arquitectura Dual
=======================================================
La Matriarca es la memoria transgeneracional del enjambre.

Este módulo añade:

1. SectLegacyRecord — ADN completo de una secta muerta/bifurcada
   - Huella vectorial real (embedding 512d) del estado final
   - Historial de fitness (min, max, promedio, tendencia)
   - Linaje genético (quién fue su padre, si se bifurcó)
   - Causa de muerte: low_fitness | bifurcation | dissolved | manual
   - Hijos generados (si bifurcó antes de morir)

2. SectLegacyBank — banco de ADN genético de sectas
   - Persistente en disco (JSON + .pt para embeddings)
   - Recuperación por similaridad de rol/embedding
   - Compresión generacional cuando supera capacidad

3. MatriarcaDual — arquitectura lista para distribución
   - PersonalMatriarca: memoria privada local (encriptada en futuro)
   - GlobalMatriarca: memoria compartida del enjambre distribuido
   - Hoy ambas son locales; la red P2P se enchufa sin cambiar la API

4. MatriarcaEnriched — extiende Matriarca con el layer genético
   - store_sect_legacy(): versión enriquecida del método del SectManager
   - query_sect_history(): busca legados similares (para orientar spawn)
   - suggest_bifurcation(): dado un sect_id, sugiere cómo bifurcarse
   - merge_global_update(): fusión de conocimiento global (preparación LSP)
"""

from __future__ import annotations

import json
import time
import math
import hashlib
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── SectLegacyRecord ─────────────────────────────────────────────────────────

@dataclass
class SectLegacyRecord:
    """
    ADN completo de una secta. Se guarda cuando la secta muere o se bifurca.
    Este es el 'fósil' que la Matriarca preserva para la siguiente generación.

    La próxima vez que nazca una secta del mismo rol, puede consultar
    este registro para no repetir los mismos errores.
    """
    # Identidad
    sect_id: str
    role_type: str
    parent_sect_id: Optional[str] = None   # None si es secta original
    children_sect_ids: List[str] = field(default_factory=list)

    # Tiempo de vida
    born_at: float = field(default_factory=time.time)
    died_at: float = field(default_factory=time.time)

    # Causa de extinción
    death_reason: str = "low_fitness"      # low_fitness | bifurcation | dissolved | manual

    # Métricas genéticas
    fitness_history: List[float] = field(default_factory=list)  # todos los registros
    peak_fitness: float = 0.0
    final_fitness: float = 0.0
    n_agents_peak: int = 0
    n_agents_final: int = 0

    # Contexto del enjambre al morir
    swarm_diversity_at_death: float = 0.5
    total_interactions: int = 0

    # Huella vectorial (se guarda por separado como tensor)
    embedding_id: Optional[str] = None     # hash para lookup en el tensor bank

    @property
    def lifespan_s(self) -> float:
        return self.died_at - self.born_at

    @property
    def fitness_trend(self) -> float:
        """Tendencia del fitness: positiva = mejorando, negativa = degradando."""
        if len(self.fitness_history) < 4:
            return 0.0
        recent = self.fitness_history[-4:]
        early = self.fitness_history[:4]
        return sum(recent) / len(recent) - sum(early) / len(early)

    @property
    def fitness_avg(self) -> float:
        if not self.fitness_history:
            return self.final_fitness
        return sum(self.fitness_history) / len(self.fitness_history)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Redondear para legibilidad
        d["fitness_avg"] = round(self.fitness_avg, 4)
        d["fitness_trend"] = round(self.fitness_trend, 4)
        d["lifespan_s"] = round(self.lifespan_s, 1)
        return d

    def to_summary(self) -> str:
        """Texto corto para almacenar en el banco de memorias general."""
        trend_sym = "↑" if self.fitness_trend > 0.05 else ("↓" if self.fitness_trend < -0.05 else "→")
        parent_str = f"←{self.parent_sect_id}" if self.parent_sect_id else "original"
        return (
            f"[SECT_FOSSIL] {self.role_type} ({parent_str}) "
            f"fitness={self.final_fitness:.3f}{trend_sym} "
            f"life={self.lifespan_s:.0f}s n_agents={self.n_agents_final} "
            f"reason={self.death_reason}"
        )


# ─── SectLegacyBank ───────────────────────────────────────────────────────────

class SectLegacyBank:
    """
    Banco de ADN genético: almacena y recupera SectLegacyRecord.

    Dos capas de almacenamiento:
    - JSON: metadatos completos (SectLegacyRecord.to_dict())
    - .pt:  embeddings vectoriales (huella del estado final de la secta)

    La recuperación es por similitud vectorial + filtro por role_type.
    """

    def __init__(
        self,
        legacy_path: str = "checkpoints/sect_legacy.json",
        max_records: int = 512,
        embd_dim: int = 512,
        device: str = "cpu",
    ):
        self.legacy_path = Path(legacy_path)
        self.emb_path = self.legacy_path.with_suffix(".pt")
        self.max_records = max_records
        self.embd_dim = embd_dim
        self.device = device

        self.records: List[SectLegacyRecord] = []
        self.embeddings: Optional[torch.Tensor] = None  # (N, embd_dim)

        self._load()

    def _load(self):
        """Carga desde disco si existe."""
        if self.legacy_path.exists():
            with open(self.legacy_path) as f:
                raw = json.load(f)
            self.records = []
            for r in raw:
                # Reconstruir SectLegacyRecord desde dict
                try:
                    self.records.append(SectLegacyRecord(**{
                        k: v for k, v in r.items()
                        if k in SectLegacyRecord.__dataclass_fields__
                    }))
                except Exception:
                    pass

        if self.emb_path.exists() and self.records:
            try:
                self.embeddings = torch.load(
                    self.emb_path, map_location=self.device, weights_only=True
                )
            except Exception:
                self.embeddings = None

    def save(self):
        self.legacy_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.legacy_path, "w") as f:
            json.dump([r.to_dict() for r in self.records], f, ensure_ascii=False, indent=2)
        if self.embeddings is not None:
            torch.save(self.embeddings.cpu(), self.emb_path)

    def add(self, record: SectLegacyRecord, embedding: Optional[torch.Tensor] = None):
        """
        Añade un registro de legado al banco.
        Si no se pasa embedding, se genera uno desde los metadatos del record.
        """
        emb = embedding if embedding is not None else self._encode_record(record)
        emb = emb.detach().cpu().unsqueeze(0)  # (1, embd_dim)

        # Generar ID del embedding
        record.embedding_id = hashlib.md5(
            f"{record.sect_id}:{record.died_at}".encode()
        ).hexdigest()[:8]

        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = torch.cat([self.embeddings.cpu(), emb], dim=0)

        self.records.append(record)

        # Poda si supera capacidad
        if len(self.records) > self.max_records:
            self._prune()

        self.save()

    def query(
        self,
        role_type: Optional[str] = None,
        embedding: Optional[torch.Tensor] = None,
        top_k: int = 5,
    ) -> List[Tuple[SectLegacyRecord, float]]:
        """
        Recupera los legados más relevantes.

        Args:
            role_type: filtrar por tipo de rol (None = todos)
            embedding: vector de referencia para similitud (None = por role solo)
            top_k: máximo de resultados

        Returns:
            Lista de (SectLegacyRecord, score) ordenada por score desc.
        """
        if not self.records:
            return []

        # Filtrar por rol
        if role_type is not None:
            indices = [i for i, r in enumerate(self.records) if r.role_type == role_type]
        else:
            indices = list(range(len(self.records)))

        if not indices:
            return []

        # Si hay embedding: ordenar por similitud
        if embedding is not None and self.embeddings is not None:
            bank = self.embeddings[indices].to(self.device)
            q = embedding.to(self.device).float()
            if q.dim() > 1:
                q = q.squeeze(0)
            q_norm = F.normalize(q.unsqueeze(0), dim=-1)
            b_norm = F.normalize(bank.float(), dim=-1)
            sims = torch.mm(q_norm, b_norm.t()).squeeze(0)  # (M,)
            sorted_local = torch.argsort(sims, descending=True)[:top_k]
            return [
                (self.records[indices[i]], sims[i].item())
                for i in sorted_local.tolist()
            ]
        else:
            # Sin embedding: ordenar por fitness * recencia
            def score(r: SectLegacyRecord) -> float:
                recency = 1.0 / (1.0 + (time.time() - r.died_at) / 3600.0)
                return r.fitness_avg * 0.7 + recency * 0.3
            sorted_records = sorted(
                [(self.records[i], score(self.records[i])) for i in indices],
                key=lambda x: x[1],
                reverse=True,
            )
            return sorted_records[:top_k]

    def _encode_record(self, record: SectLegacyRecord) -> torch.Tensor:
        """
        Codifica un SectLegacyRecord como vector (embd_dim,) desde sus metadatos.
        Esto es un proxy hasta que tengamos el embedding real del modelo.
        """
        from src.swarm.sect_manager import KNOWN_SECT_ROLES
        role_map = {r: i for i, r in enumerate(KNOWN_SECT_ROLES.keys())}

        v = torch.zeros(self.embd_dim, dtype=torch.float32)
        v[0] = float(role_map.get(record.role_type, 0)) / max(len(role_map), 1)
        v[1] = record.final_fitness
        v[2] = record.fitness_avg
        v[3] = record.fitness_trend * 0.5 + 0.5    # normalizar [-1,1] → [0,1]
        v[4] = min(1.0, record.lifespan_s / 3600.0)
        v[5] = float(record.n_agents_peak) / 20.0
        v[6] = 1.0 if record.parent_sect_id else 0.0
        v[7] = float(len(record.children_sect_ids)) / 5.0

        # Añadir ruido del fitness history como firma
        if record.fitness_history:
            hist = torch.tensor(record.fitness_history[-16:], dtype=torch.float32)
            hist_padded = F.pad(hist, (0, max(0, 16 - len(hist))))
            available = max(0, self.embd_dim - 8)
            if available:
                n = min(16, available)
                v[8:8 + n] = hist_padded[:n]

        return v

    def _prune(self):
        """Elimina los registros de menor fitness cuando supera capacidad."""
        scores = [r.fitness_avg for r in self.records]
        keep = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.max_records]
        keep = sorted(keep)
        self.records = [self.records[i] for i in keep]
        if self.embeddings is not None and self.embeddings.shape[0] > len(keep):
            self.embeddings = self.embeddings[keep]

    @property
    def size(self) -> int:
        return len(self.records)


# ─── MatriarcaDual ────────────────────────────────────────────────────────────

class MatriarcaDual:
    """
    Arquitectura dual para la Matriarca:

    - PersonalMatriarca: memoria privada local
      * Memoria de interacciones del usuario (conversaciones, preferencias)
      * Solo existe en el nodo local
      * En el futuro: encriptada con clave del usuario (Ed25519)
      * NO se comparte con la red

    - GlobalMatriarca: memoria compartida del enjambre
      * Conocimiento técnico, legados de sectas, ADN genético
      * Se sincroniza con otros nodos vía LSP (cuando Fase 2 esté lista)
      * Hoy: solo local, API lista para enchufar P2P

    Ambas son instancias de Matriarca. Esta clase las coordina.
    """

    def __init__(
        self,
        personal_cfg,
        global_cfg,
        device: str = "cuda",
    ):
        from src.matriarca.matriarca import Matriarca
        self.personal = Matriarca(personal_cfg, device=device)
        self.global_ = Matriarca(global_cfg, device=device)
        self.device = device

    def emit_combined(
        self,
        state_embedding: torch.Tensor,
        personal_weight: float = 0.7,
        global_weight: float = 0.3,
    ) -> torch.Tensor:
        """
        Emite infrasónidos combinando conocimiento personal y global.

        personal: contexto del usuario/sesión actual
        global: sabiduría técnica acumulada del enjambre
        """
        personal_inf = self.personal.emit_infrasound(state_embedding)
        global_inf = self.global_.emit_infrasound(state_embedding)
        return personal_weight * personal_inf + global_weight * global_inf

    def store_personal(self, embedding: torch.Tensor, text: str, importance: float = 0.8):
        """Almacena en la memoria personal (no se comparte)."""
        self.personal.store_interaction(embedding, text, importance)

    def store_global(self, embedding: torch.Tensor, text: str, importance: float = 0.7):
        """Almacena en la memoria global (se compartirá vía LSP)."""
        self.global_.store_interaction(embedding, text, importance)

    def merge_global_update(self, remote_embeddings: torch.Tensor, remote_metadata: list):
        """
        Fusiona conocimiento global recibido de otro nodo (preparación LSP Fase 2).
        Por ahora: añade memorias remotas con importancia reducida.
        En Fase 2: merge inteligente con TTL + decay.
        """
        for i, meta in enumerate(remote_metadata):
            if i >= remote_embeddings.shape[0]:
                break
            emb = remote_embeddings[i]
            text = meta.get("text", "[remote]")
            importance = meta.get("importance", 0.5) * 0.8  # descuento por distancia
            self.global_.bank.add(emb, text, importance)

    def save(self):
        self.personal.save()
        self.global_.save()

    @property
    def personal_memories(self) -> int:
        return self.personal.memory_count

    @property
    def global_memories(self) -> int:
        return self.global_.memory_count


# ─── MatriarcaEnriched — extiende Matriarca con layer genético ────────────────

class MatriarcaEnriched:
    """
    Wrapper que añade el layer genético de sectas sobre la Matriarca existente.

    Expone:
    - store_sect_legacy(): versión enriquecida (usa SectLegacyRecord)
    - query_sect_history(): busca legados para orientar spawn
    - suggest_bifurcation(): sugiere cómo bifurcar una secta
    - La Matriarca base sigue igual (emit_infrasound, store_interaction, etc.)

    Compatible con la API existente: se puede pasar como `matriarca=` en
    SectManager, LixySwarm, etc.
    """

    def __init__(self, matriarca, legacy_bank: Optional[SectLegacyBank] = None):
        """
        Args:
            matriarca: instancia de Matriarca (o MatriarcaDual)
            legacy_bank: banco de legados (se crea automático si None)
        """
        self._matriarca = matriarca
        self.legacy_bank = legacy_bank or SectLegacyBank(
            legacy_path=str(Path(matriarca.cfg.memory_path).parent / "sect_legacy.json"),
            embd_dim=matriarca.cfg.embd_dim,
            device=matriarca.device,
        )

    # ─── Delegación transparente hacia la Matriarca base ──────────────────────

    def __getattr__(self, name):
        """Delega todo lo no definido aquí a la Matriarca base."""
        return getattr(self._matriarca, name)

    # ─── API genética nueva ───────────────────────────────────────────────────

    def store_sect_legacy(
        self,
        sect,
        reason: str = "low_fitness",
        swarm_diversity: float = 0.5,
        embedding: Optional[torch.Tensor] = None,
    ) -> SectLegacyRecord:
        """
        Almacena el ADN de una secta al morir.
        Reemplaza _store_sect_legacy() del SectManager.

        Args:
            sect: SectRecord con su historial completo
            reason: causa de muerte
            swarm_diversity: diversidad del enjambre al morir
            embedding: vector real del estado final (si disponible)

        Returns:
            SectLegacyRecord guardado
        """
        record = SectLegacyRecord(
            sect_id=sect.sect_id,
            role_type=sect.role_type,
            parent_sect_id=getattr(sect, "parent_sect_id", None),
            children_sect_ids=list(getattr(sect, "children_sect_ids", [])),
            born_at=getattr(sect, "born_at", time.time() - sect.age),
            died_at=time.time(),
            death_reason=reason,
            fitness_history=list(getattr(sect, "fitness_history", [])),
            peak_fitness=max(getattr(sect, "fitness_history", []) or [sect.avg_fitness]),
            final_fitness=sect.avg_fitness,
            n_agents_peak=getattr(sect, "n_agents_peak", sect.n_agents),
            n_agents_final=sect.n_agents,
            swarm_diversity_at_death=swarm_diversity,
        )

        # Guardar en el banco genético
        self.legacy_bank.add(record, embedding)

        # También guardar en la Matriarca base para orientación general
        emb_for_base = embedding if embedding is not None else self.legacy_bank._encode_record(record)
        self._matriarca.bank.add(
            emb_for_base,
            record.to_summary(),
            importance=max(record.final_fitness, 0.15),
        )

        return record

    def query_sect_history(
        self,
        role_type: str,
        current_embedding: Optional[torch.Tensor] = None,
        top_k: int = 5,
    ) -> List[SectLegacyRecord]:
        """
        Consulta el historial de sectas del mismo rol.
        Útil para orientar el spawn de una nueva secta.

        Returns:
            Lista de SectLegacyRecord ordenados por relevancia.
        """
        results = self.legacy_bank.query(
            role_type=role_type,
            embedding=current_embedding,
            top_k=top_k,
        )
        return [r for r, _ in results]

    def suggest_bifurcation(
        self,
        sect,
        swarm_diversity: float = 0.5,
    ) -> Dict:
        """
        Dado el estado de una secta, sugiere si y cómo bifurcarla.

        Lógica:
        - Si el fitness es alto pero hay baja diversidad → bifurcar
        - Los hijos heredan el rol con sufijo: Refinador → Refinador-Lógico + Refinador-Creativo
        - Comprueba historial para evitar bifurcar roles que ya fallaron

        Returns:
            {
                "should_bifurcate": bool,
                "child_roles": [str, str],
                "reason": str,
                "confidence": float,
            }
        """
        # Umbral: bifurcar si fitness > 0.6 y diversidad baja
        should = sect.avg_fitness > 0.6 and swarm_diversity < 0.4

        # Buscar si ya hubo hijos de este rol en el pasado (evitar repetir)
        past = self.query_sect_history(sect.role_type, top_k=3)
        past_children_roles = set()
        for rec in past:
            if rec.death_reason == "bifurcation":
                for child_id in rec.children_sect_ids:
                    # Solo tenemos IDs, inferir rol del ID
                    if "-" in child_id:
                        past_children_roles.add(child_id.split("-")[0])

        # Derivar roles hijos
        role = sect.role_type
        child_roles = [f"{role}-A", f"{role}-B"]

        # Especializaciones conocidas
        specializations = {
            "refinador": ["refinador-logico", "refinador-creativo"],
            "explorador": ["explorador-profundo", "explorador-amplio"],
        }
        if role in specializations:
            child_roles = specializations[role]

        reason = ""
        if should:
            reason = f"fitness={sect.avg_fitness:.2f} + baja_diversidad={swarm_diversity:.2f}"
        else:
            reason = f"fitness={sect.avg_fitness:.2f} insuficiente o diversidad={swarm_diversity:.2f} ok"

        confidence = min(1.0, sect.avg_fitness * (1.0 - swarm_diversity))

        return {
            "should_bifurcate": should,
            "child_roles": child_roles,
            "reason": reason,
            "confidence": confidence,
            "past_similar_deaths": len(past),
        }

    def save(self):
        """Guarda Matriarca base + banco genético."""
        self._matriarca.save()
        self.legacy_bank.save()

    @property
    def memory_count(self) -> int:
        return self._matriarca.memory_count

    @property
    def legacy_count(self) -> int:
        return self.legacy_bank.size
