"""
Lixy-0.1 — La Matriarca (Capa Elefante)
=========================================
Memoria transgeneracional. No procesa — orienta.

La Matriarca es un modelo pequeño (~10M params) que:
1. Acumula patrones de interacciones pasadas
2. Emite "infrasónidos" — vectores de orientación que guían al enjambre
3. Antes de reiniciarse, destila su memoria a una nueva Matriarca

Inspiración: Las matriarcas elefante acumulan décadas de sabiduría
y guían a la manada con señales de infrasónido inaudibles a distancia.
"""

import math
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class MatriarchConfig:
    # Dimensiones
    memory_dim: int = 256           # Dimensión del vector de memoria comprimida
    infrasound_dim: int = 256       # Dimensión del "infrasónido" (orienta al enjambre)
    n_memory_slots: int = 512       # Cuántos patrones puede almacenar
    n_heads: int = 8                # Cabezas de atención para recuperación
    n_layers: int = 4               # Capas de transformación ligera

    # Persistencia
    memory_path: str = "checkpoints/matriarch_memory.pt"
    log_path: str = "checkpoints/matriarch_log.jsonl"

    # Transferencia transgeneracional
    distill_top_k: int = 128        # Top-K slots a preservar al reiniciarse


# ─── Modelo de Atención Sparse ─────────────────────────────────────────────────

class SparseAttention(nn.Module):
    """
    Atención sparse — la Matriarca no atiende todo, solo lo más relevante.
    Implementa top-k attention para eficiencia.
    """
    def __init__(self, dim: int, n_heads: int, top_k: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.top_k = top_k
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        query: [B, 1, dim] — la consulta actual
        keys:  [B, N, dim] — los slots de memoria
        values:[B, N, dim] — los valores de memoria
        """
        B, Nq, D = query.shape
        _, Nk, _ = keys.shape

        Q = self.q_proj(query).view(B, Nq, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(keys).view(B, Nk, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(values).view(B, Nk, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) * self.scale  # [B, H, Nq, Nk]

        # Sparse: solo top-k scores
        top_k = min(self.top_k, Nk)
        topk_scores, topk_idx = torch.topk(scores, top_k, dim=-1)
        sparse_scores = torch.full_like(scores, float('-inf'))
        sparse_scores.scatter_(-1, topk_idx, topk_scores)

        attn = F.softmax(sparse_scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, Nq, D)
        return self.out_proj(out)


# ─── Matriarca ─────────────────────────────────────────────────────────────────

class Matriarch(nn.Module):
    """
    La Matriarca — memoria transgeneracional del enjambre.

    Recibe un vector de experiencia (resumen de una interacción)
    y emite un "infrasónido" — vector de orientación para todos los agentes.
    """

    def __init__(self, cfg: MatriarchConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.memory_dim

        # ─── Slots de memoria (no son parámetros trainables — son estado)
        self.register_buffer(
            'memory_keys',
            torch.zeros(cfg.n_memory_slots, D)
        )
        self.register_buffer(
            'memory_values',
            torch.zeros(cfg.n_memory_slots, D)
        )
        self.register_buffer(
            'memory_ages',      # Cuándo se escribió cada slot (timestamp)
            torch.zeros(cfg.n_memory_slots)
        )
        self.register_buffer(
            'memory_importance',  # Qué tan importante es cada slot
            torch.zeros(cfg.n_memory_slots)
        )
        self.register_buffer(
            'write_ptr',
            torch.tensor(0, dtype=torch.long)
        )
        self.register_buffer(
            'n_written',
            torch.tensor(0, dtype=torch.long)
        )

        # ─── Encoder: experiencia → vector comprimido
        self.encoder = nn.Sequential(
            nn.Linear(D, D * 2),
            nn.GELU(),
            nn.Linear(D * 2, D),
            nn.LayerNorm(D),
        )

        # ─── Atención sparse para recuperar memorias relevantes
        self.attention = SparseAttention(D, cfg.n_heads, top_k=64)

        # ─── Generador de infrasónidos
        self.infrasound_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(D, D),
                nn.GELU(),
                nn.LayerNorm(D),
            )
            for _ in range(cfg.n_layers)
        ])

        # ─── Proyección final al espacio de infrasónidos
        self.infrasound_proj = nn.Linear(D, cfg.infrasound_dim)

        # ─── Importancia: predice qué tan valioso es escribir esta memoria
        self.importance_head = nn.Linear(D, 1)

    def encode_experience(self, experience: torch.Tensor) -> torch.Tensor:
        """Comprime una experiencia en un vector de memoria."""
        return self.encoder(experience)

    def write_memory(self, encoded: torch.Tensor, importance: float = 1.0):
        """
        Escribe un nuevo recuerdo en el slot más antiguo/menos importante.
        encoded: [D] — vector de experiencia codificado
        """
        slot = self.write_ptr.item()
        self.memory_keys[slot] = encoded.detach()
        self.memory_values[slot] = encoded.detach()
        self.memory_ages[slot] = time.time()
        self.memory_importance[slot] = importance

        # Avanzar puntero (circular)
        self.write_ptr = (self.write_ptr + 1) % self.cfg.n_memory_slots
        self.n_written = torch.clamp(self.n_written + 1, max=self.cfg.n_memory_slots)

    def emit_infrasound(self, query: torch.Tensor) -> torch.Tensor:
        """
        Emite un infrasónido basado en la consulta actual y la memoria.

        query: [B, D] — el estado actual del enjambre / consulta
        returns: [B, infrasound_dim] — vector de orientación
        """
        B = query.shape[0]
        n = self.n_written.item()

        if n == 0:
            # Sin memoria todavía — infrasónido neutro
            return torch.zeros(B, self.cfg.infrasound_dim, device=query.device)

        # Recuperar memorias activas
        active_keys = self.memory_keys[:n].unsqueeze(0).expand(B, -1, -1)
        active_values = self.memory_values[:n].unsqueeze(0).expand(B, -1, -1)

        # Atención sparse sobre memorias
        q = query.unsqueeze(1)  # [B, 1, D]
        retrieved = self.attention(q, active_keys, active_values)  # [B, 1, D]
        h = retrieved.squeeze(1) + query  # Residual

        # Transformar en infrasónido
        for layer in self.infrasound_layers:
            h = layer(h) + h

        return self.infrasound_proj(h)

    def forward(self, experience: torch.Tensor) -> dict:
        """
        Procesa una experiencia: la codifica, estima importancia,
        la escribe en memoria, y emite infrasónido.

        experience: [B, D] — resumen de la interacción actual
        """
        # Codificar
        encoded = self.encode_experience(experience)  # [B, D]

        # Estimar importancia
        importance = torch.sigmoid(self.importance_head(encoded)).squeeze(-1)  # [B]

        # Escribir en memoria (para el primer item del batch)
        imp_val = importance[0].item()
        self.write_memory(encoded[0], imp_val)

        # Emitir infrasónido
        infrasound = self.emit_infrasound(encoded)  # [B, infrasound_dim]

        return {
            "infrasound": infrasound,
            "encoded": encoded,
            "importance": importance,
        }

    def distill_to_new_generation(self) -> "Matriarch":
        """
        Transferencia transgeneracional — antes de reiniciarse,
        crea una nueva Matriarca con los recuerdos más importantes.
        """
        n = self.n_written.item()
        if n == 0:
            return Matriarch(self.cfg)

        # Seleccionar top-K slots por importancia
        k = min(self.cfg.distill_top_k, n)
        top_slots = torch.topk(self.memory_importance[:n], k).indices

        # Nueva Matriarca
        new_matriarch = Matriarch(self.cfg)

        # Transferir memorias más importantes
        for i, slot in enumerate(top_slots):
            new_matriarch.memory_keys[i] = self.memory_keys[slot]
            new_matriarch.memory_values[i] = self.memory_values[slot]
            new_matriarch.memory_ages[i] = self.memory_ages[slot]
            new_matriarch.memory_importance[i] = self.memory_importance[slot]

        new_matriarch.write_ptr = torch.tensor(k % self.cfg.n_memory_slots)
        new_matriarch.n_written = torch.tensor(k)

        print(f"🐘 Transferencia transgeneracional: {k} memorias preservadas de {n} totales")
        return new_matriarch

    def save(self, path: Optional[str] = None):
        """Guarda el estado completo de la Matriarca."""
        path = path or self.cfg.memory_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "config": asdict(self.cfg),
            "n_written": self.n_written.item(),
            "write_ptr": self.write_ptr.item(),
        }, path)
        print(f"🐘 Matriarca guardada: {path} ({self.n_written.item()} memorias)")

    @classmethod
    def load(cls, path: str) -> "Matriarch":
        """Carga una Matriarca desde disco."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        cfg = MatriarchConfig(**ckpt["config"])
        matriarch = cls(cfg)
        matriarch.load_state_dict(ckpt["state_dict"])
        print(f"🐘 Matriarca cargada: {path} ({ckpt['n_written']} memorias)")
        return matriarch

    def get_stats(self) -> dict:
        """Estadísticas del estado actual de la Matriarca."""
        n = self.n_written.item()
        return {
            "n_memories": n,
            "n_slots": self.cfg.n_memory_slots,
            "utilization": n / self.cfg.n_memory_slots,
            "avg_importance": self.memory_importance[:n].mean().item() if n > 0 else 0.0,
            "max_importance": self.memory_importance[:n].max().item() if n > 0 else 0.0,
        }


# ─── Test rápido ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🐘 Probando la Matriarca...")

    cfg = MatriarchConfig(
        memory_dim=256,
        infrasound_dim=256,
        n_memory_slots=16,  # pequeño para test
    )

    matriarch = Matriarch(cfg)
    n_params = sum(p.numel() for p in matriarch.parameters())
    print(f"   Parámetros: {n_params/1e6:.2f}M")

    # Simular algunas experiencias
    print("\n   Escribiendo experiencias...")
    for i in range(8):
        exp = torch.randn(1, cfg.memory_dim)
        result = matriarch(exp)
        print(f"   Exp {i+1}: importancia={result['importance'].item():.3f}, "
              f"infrasónido norm={result['infrasound'].norm().item():.3f}")

    # Estadísticas
    stats = matriarch.get_stats()
    print(f"\n   Stats: {stats}")

    # Probar infrasónido con query
    query = torch.randn(2, cfg.memory_dim)  # batch=2
    infrasound = matriarch.emit_infrasound(query)
    print(f"\n   Infrasónido: shape={infrasound.shape}, norm={infrasound.norm(dim=-1)}")

    # Transferencia transgeneracional
    print("\n   Probando transferencia transgeneracional...")
    new_matriarch = matriarch.distill_to_new_generation()
    new_stats = new_matriarch.get_stats()
    print(f"   Nueva Matriarca: {new_stats}")

    # Guardar/cargar
    matriarch.save("/tmp/test_matriarch.pt")
    loaded = Matriarch.load("/tmp/test_matriarch.pt")
    print(f"\n✅ Matriarca funcionando correctamente!")
    print(f"   Memorias guardadas: {loaded.get_stats()['n_memories']}")
