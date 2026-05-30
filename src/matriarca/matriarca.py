"""
Lixy-0.1 — Matriarca (Capa Elefante)
=====================================
Módulo de memoria transgeneracional. No genera texto — orienta al enjambre.

Principio: Como la elefanta matriarca del grupo, este módulo acumula
sabiduría de todas las interacciones y emite "infrasónidos" — vectores
de orientación global que todos los agentes reciben antes de procesar.

Transferencia transgeneracional: al reiniciarse, destila su conocimiento
a una nueva instancia (knowledge distillation comprimida).
"""

import json
import time
import math
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Configuración ────────────────────────────────────────────────────────────

@dataclass
class MatriarcaConfig:
    embd_dim: int = 512           # dimensión del embedding de memoria
    infrasound_dim: int = 256     # dimensión de los infrasónidos emitidos
    max_memories: int = 4096      # máx memorias almacenadas
    n_heads: int = 8              # cabezas de atención sparsa
    n_layers: int = 4             # capas del transformer de memoria
    dropout: float = 0.1
    memory_path: str = "checkpoints/matriarca_memory.json"
    checkpoint_path: str = "checkpoints/matriarca.pt"


# ─── Modelo ───────────────────────────────────────────────────────────────────

class MemoryAttention(nn.Module):
    """Atención sparsa sobre el banco de memorias."""

    def __init__(self, cfg: MatriarcaConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.embd_dim // cfg.n_heads

        self.q = nn.Linear(cfg.embd_dim, cfg.embd_dim, bias=False)
        self.k = nn.Linear(cfg.embd_dim, cfg.embd_dim, bias=False)
        self.v = nn.Linear(cfg.embd_dim, cfg.embd_dim, bias=False)
        self.proj = nn.Linear(cfg.embd_dim, cfg.embd_dim, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, query: torch.Tensor, memory_bank: torch.Tensor) -> torch.Tensor:
        """
        query:       (B, embd_dim) — el input actual
        memory_bank: (N, embd_dim) — todas las memorias
        returns:     (B, embd_dim) — contexto de memoria relevante
        """
        B, D = query.shape
        N = memory_bank.shape[0]

        q = self.q(query).view(B, self.n_heads, self.head_dim)
        k = self.k(memory_bank).view(N, self.n_heads, self.head_dim)
        v = self.v(memory_bank).view(N, self.n_heads, self.head_dim)

        # Atención: (B, n_heads, N)
        scale = math.sqrt(self.head_dim)
        attn = torch.einsum("bhd,nhd->bhn", q, k) / scale
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        # Contexto: (B, n_heads, head_dim)
        out = torch.einsum("bhn,nhd->bhd", attn, v)
        out = out.reshape(B, D)
        return self.proj(out)


class MatriarcaModel(nn.Module):
    """
    Modelo de la Matriarca — solo lectura/orientación, no genera texto.
    Toma el estado actual + banco de memorias → emite infrasónidos.
    """

    def __init__(self, cfg: MatriarcaConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder de input → embedding interno
        self.input_encoder = nn.Sequential(
            nn.Linear(cfg.embd_dim, cfg.embd_dim),
            nn.GELU(),
            nn.Linear(cfg.embd_dim, cfg.embd_dim),
            nn.LayerNorm(cfg.embd_dim),
        )

        # Atención sobre memorias
        self.memory_attn = nn.ModuleList([
            MemoryAttention(cfg) for _ in range(cfg.n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(cfg.embd_dim) for _ in range(cfg.n_layers)
        ])

        # Proyección final → infrasónidos
        self.infrasound_proj = nn.Sequential(
            nn.Linear(cfg.embd_dim, cfg.embd_dim),
            nn.GELU(),
            nn.Linear(cfg.embd_dim, cfg.infrasound_dim),
            nn.Tanh(),  # normalizado en [-1, 1]
        )

        # Encoder de memorias nuevas (para almacenar interacciones)
        self.memory_encoder = nn.Sequential(
            nn.Linear(cfg.embd_dim, cfg.embd_dim),
            nn.GELU(),
            nn.Linear(cfg.embd_dim, cfg.embd_dim),
            nn.LayerNorm(cfg.embd_dim),
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Matriarca inicializada: {n_params/1e6:.1f}M params")

    def forward(
        self,
        current_state: torch.Tensor,     # (B, embd_dim) — estado del enjambre
        memory_bank: torch.Tensor,        # (N, embd_dim) — banco de memorias
    ) -> torch.Tensor:
        """Emite infrasónidos de orientación."""
        x = self.input_encoder(current_state)  # (B, embd_dim)

        for attn, norm in zip(self.memory_attn, self.norms):
            ctx = attn(x, memory_bank)
            x = norm(x + ctx)

        infrasound = self.infrasound_proj(x)  # (B, infrasound_dim)
        return infrasound

    def encode_memory(self, state: torch.Tensor) -> torch.Tensor:
        """Codifica una interacción para almacenarla en el banco."""
        return self.memory_encoder(state)


# ─── Banco de Memorias (persistente) ──────────────────────────────────────────

class MemoryBank:
    """
    Banco de memorias persistente en disco.
    Almacena embeddings + metadatos de interacciones pasadas.
    """

    def __init__(self, cfg: MatriarcaConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.memory_path = Path(cfg.memory_path)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)

        # Embeddings en memoria (N, embd_dim)
        self.embeddings: Optional[torch.Tensor] = None
        # Metadatos
        self.metadata: list[dict] = []

        self._load()

    def _load(self):
        """Carga memorias desde disco si existen."""
        meta_path = self.memory_path
        emb_path = self.memory_path.with_suffix(".pt")

        if meta_path.exists() and emb_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)
            self.embeddings = torch.load(emb_path, map_location=self.device, weights_only=True)
            print(f"  → Matriarca: {len(self.metadata)} memorias cargadas")
        else:
            # Banco vacío — inicializar con embedding cero
            self.embeddings = torch.zeros(1, self.cfg.embd_dim, device=self.device)
            self.metadata = [{"text": "[memoria_inicial]", "timestamp": time.time(), "importance": 0.0}]
            print("  → Matriarca: banco de memorias vacío inicializado")

    def save(self):
        """Persiste memorias a disco."""
        with open(self.memory_path, "w") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        torch.save(self.embeddings, self.memory_path.with_suffix(".pt"))

    def add(self, embedding: torch.Tensor, text: str, importance: float = 1.0):
        """Añade una nueva memoria al banco."""
        importance = max(0.0, min(1.0, float(importance)))  # FIX: clamp [0, 1]
        emb = embedding.detach().cpu().unsqueeze(0)  # (1, embd_dim)

        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = torch.cat([self.embeddings.cpu(), emb], dim=0)

        self.metadata.append({
            "text": text[:200],  # truncar a 200 chars
            "timestamp": time.time(),
            "importance": importance,
            "access_count": 0,       # veces que fue recuperada
            "last_access": None,     # timestamp del último acceso
        })

        # Mantener solo las max_memories más importantes
        if len(self.metadata) > self.cfg.max_memories:
            self._prune()

        self.save()

    def retrieve(self, query: torch.Tensor, top_k: int = 8) -> tuple:
        """
        Busca las memorias más relevantes por similitud coseno con la query.
        Actualiza access_count y last_access de las memorias recuperadas.

        query:  (embd_dim,) o (B, embd_dim)
        returns: (top_k_embeddings, top_k_indices, top_k_scores)
        """
        bank = self.get_embeddings(self.device)  # (N, embd_dim)
        N = bank.shape[0]
        k = min(top_k, N)

        squeeze = query.dim() == 1
        if squeeze:
            query = query.unsqueeze(0)  # (1, embd_dim)
        query = query.to(self.device)

        # Similitud coseno
        q_norm = F.normalize(query, dim=-1)              # (B, embd_dim)
        b_norm = F.normalize(bank.float(), dim=-1)       # (N, embd_dim)
        scores = torch.mm(q_norm.float(), b_norm.t())    # (B, N)

        # Ponderar por importancia almacenada
        importance_weights = torch.tensor(
            [m["importance"] for m in self.metadata],
            device=self.device, dtype=torch.float32
        )  # (N,)
        scores = scores * importance_weights.unsqueeze(0)

        # Top-k por el primer query (o promedio del batch)
        agg_scores = scores.mean(dim=0)  # (N,)
        top_vals, top_idx = torch.topk(agg_scores, k)

        # Actualizar metadatos de acceso
        now = time.time()
        for idx in top_idx.tolist():
            self.metadata[idx]["access_count"] = self.metadata[idx].get("access_count", 0) + 1
            self.metadata[idx]["last_access"] = now

        top_embeddings = bank[top_idx]  # (k, embd_dim)
        return top_embeddings, top_idx, top_vals

    def update_importance(self, indices: torch.Tensor, delta: float):
        """
        Ajusta la importancia de memorias específicas.
        delta > 0: subir importancia (memoria fue útil)
        delta < 0: bajar importancia (memoria fue inútil)
        """
        for idx in indices.tolist():
            if 0 <= idx < len(self.metadata):
                old = self.metadata[idx]["importance"]
                self.metadata[idx]["importance"] = max(0.0, min(1.0, old + delta))

    def compress(self, compression_ratio: float = 0.5) -> int:
        """
        Compresión generacional: cuando el banco está al ~90% de capacidad,
        agrupa las memorias menos importantes en 'memorias sintéticas'.

        Las memorias de baja importancia se agrupan por similitud (clustering simple)
        y se reemplazan por su centroide (memoria sintética) con importancia promedio.

        Retorna el número de memorias eliminadas.
        """
        if len(self.metadata) < self.cfg.max_memories * 0.9:
            return 0  # no necesario todavía

        bank = self.get_embeddings(self.device)
        scores = [m["importance"] for m in self.metadata]

        # Identificar memorias a comprimir (importancia < mediana)
        import statistics
        median_imp = statistics.median(scores)
        low_idx = [i for i, s in enumerate(scores) if s < median_imp]

        if len(low_idx) < 2:
            return 0

        # Comprimir en grupos de `group_size` usando similitud coseno
        group_size = max(2, int(1 / compression_ratio))
        low_embeddings = bank[low_idx]  # (M, embd_dim)
        n_groups = max(1, len(low_idx) // group_size)

        # Clustering simple: dividir en n_groups por orden de similitud
        # (no KMeans para evitar dependencia — agrupación secuencial)
        synthetic_embs = []
        synthetic_meta = []
        step = max(1, len(low_idx) // n_groups)

        for g in range(n_groups):
            start = g * step
            end = min(start + step, len(low_idx))
            group_embs = low_embeddings[start:end]  # (k, embd_dim)
            group_idxs = low_idx[start:end]

            # Centroide = memoria sintética
            centroid = group_embs.mean(dim=0)
            avg_importance = sum(scores[i] for i in group_idxs) / len(group_idxs)
            texts = [self.metadata[i]["text"][:50] for i in group_idxs]

            synthetic_embs.append(centroid.cpu())
            synthetic_meta.append({
                "text": f"[sintética] {' | '.join(texts)}",
                "timestamp": time.time(),
                "importance": avg_importance,
                "access_count": 0,
                "last_access": None,
                "synthetic": True,
                "compressed_from": len(group_idxs),
            })

        # Eliminar memorias de baja importancia y agregar sintéticas
        keep_idx = [i for i in range(len(self.metadata)) if i not in set(low_idx)]
        removed = len(low_idx)

        if keep_idx:
            self.embeddings = bank[keep_idx].cpu()
            self.metadata = [self.metadata[i] for i in keep_idx]
        else:
            self.embeddings = torch.zeros(1, self.cfg.embd_dim)
            self.metadata = []

        # Agregar sintéticas
        for emb, meta in zip(synthetic_embs, synthetic_meta):
            self.embeddings = torch.cat([self.embeddings.cpu(), emb.unsqueeze(0)], dim=0)
            self.metadata.append(meta)

        self.save()
        return removed - len(synthetic_embs)

    def _prune(self):
        """Elimina memorias de menor importancia cuando el banco está lleno."""
        scores = [max(0.0, m["importance"]) for m in self.metadata]  # FIX: nunca negativos
        # Conservar las top-max_memories
        keep_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.cfg.max_memories]
        keep_idx = sorted(keep_idx)

        self.embeddings = self.embeddings[keep_idx]
        self.metadata = [self.metadata[i] for i in keep_idx]

    @property
    def size(self) -> int:
        return len(self.metadata)

    def get_embeddings(self, device: str = None) -> torch.Tensor:
        d = device or self.device
        return self.embeddings.to(d)


# ─── Matriarca completa (modelo + banco) ──────────────────────────────────────

class Matriarca:
    """
    Interfaz principal de la Matriarca.
    Combina el modelo de atención con el banco de memorias persistente.
    """

    def __init__(self, cfg: MatriarcaConfig = None, device: str = "cuda"):
        self.cfg = cfg or MatriarcaConfig()
        self.device = device

        self.model = MatriarcaModel(self.cfg).to(device)
        self.bank = MemoryBank(self.cfg, device=device)

        # Cargar checkpoint si existe
        ckpt_path = Path(self.cfg.checkpoint_path)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            self.model.load_state_dict(ckpt["model"])
            print(f"  → Matriarca checkpoint cargado")

    @torch.no_grad()
    def emit_infrasound(
        self,
        state_embedding: torch.Tensor,
        use_retrieval: bool = True,
        top_k: int = 32,
        update_importance: bool = True,    # activado por defecto en runtime
        importance_delta: float = 0.03,    # +3% por memoria usada (era 5%, más suave)
    ) -> torch.Tensor:
        """
        Dado el estado actual del enjambre, emite infrasónidos de orientación.

        Con use_retrieval=True: en lugar de usar el banco completo (O(N) mem),
        recupera solo las top_k memorias más relevantes al estado actual.
        Más eficiente y semánticamente más preciso.

        state_embedding: (embd_dim,) o (B, embd_dim)
        returns: (infrasound_dim,) o (B, infrasound_dim)
        """
        self.model.eval()
        squeeze = state_embedding.dim() == 1
        if squeeze:
            state_embedding = state_embedding.unsqueeze(0)

        state_embedding = state_embedding.to(self.device)

        # Retrieval activo: usar solo memorias relevantes
        if use_retrieval and self.bank.size > top_k:
            relevant_bank, top_idx, top_scores = self.bank.retrieve(
                state_embedding.mean(dim=0), top_k=top_k
            )
            relevant_bank = relevant_bank.to(self.device)

            # Actualizar importancia de memorias usadas (refuerzo positivo)
            if update_importance:
                self.bank.update_importance(top_idx, delta=importance_delta)
        else:
            relevant_bank = self.bank.get_embeddings(self.device)
            top_idx = None

        infrasound = self.model(state_embedding, relevant_bank)

        if squeeze:
            infrasound = infrasound.squeeze(0)
        return infrasound

    def store_interaction(
        self,
        state_embedding: torch.Tensor,
        text: str,
        importance: float = 1.0,
        auto_compress: bool = True,
    ):
        """
        Almacena una interacción en el banco de memorias.
        Con auto_compress=True, dispara compresión generacional si el banco
        está al 90% de capacidad.
        """
        self.model.eval()
        with torch.no_grad():
            emb = self.model.encode_memory(state_embedding.to(self.device))
        self.bank.add(emb, text, importance)

        # Compresión generacional automática
        if auto_compress and self.bank.size >= self.cfg.max_memories * 0.9:
            n_removed = self.bank.compress(compression_ratio=0.5)
            if n_removed > 0:
                print(f"  🗃️ Matriarca: compresión generacional — {n_removed} memorias → sintéticas")

    def penalize_unused(
        self,
        top_k_used: torch.Tensor,
        all_indices: range,
        penalty: float = -0.02,
    ):
        """
        Baja importancia de memorias que NO fueron recuperadas (inútiles).
        Llamar periódicamente para limpiar el banco.
        """
        used_set = set(top_k_used.tolist()) if top_k_used is not None else set()
        unused_idx = torch.tensor(
            [i for i in all_indices if i not in used_set],
            dtype=torch.long
        )
        self.bank.update_importance(unused_idx, delta=penalty)

    def save(self):
        """Guarda el modelo y el banco."""
        ckpt_path = Path(self.cfg.checkpoint_path)
        torch.save({"model": self.model.state_dict(), "config": asdict(self.cfg)}, ckpt_path)
        self.bank.save()
        print(f"  ✓ Matriarca guardada ({self.bank.size} memorias)")

    def distill_to_new(self, new_matriarca: "Matriarca", n_steps: int = 100):
        """
        Transferencia transgeneracional: destila el conocimiento a una nueva Matriarca.
        La nueva hereda la sabiduría acumulada de la anterior.
        """
        print(f"🐘 Destilación transgeneracional ({n_steps} pasos)...")
        opt = torch.optim.Adam(new_matriarca.model.parameters(), lr=1e-4)
        bank = self.bank.get_embeddings(self.device)

        self.model.eval()
        new_matriarca.model.train()

        for step in range(n_steps):
            # Muestra aleatoria del banco
            idx = torch.randint(0, bank.shape[0], (min(32, bank.shape[0]),))
            sample = bank[idx]

            # Teacher (vieja matriarca) → Student (nueva matriarca)
            with torch.no_grad():
                teacher_out = self.model(sample, bank)
            student_out = new_matriarca.model(sample, bank)

            # KL divergence loss
            loss = F.mse_loss(student_out, teacher_out)
            loss.backward()
            opt.step(); opt.zero_grad()

            if step % 20 == 0:
                print(f"  paso {step}: distillation_loss={loss.item():.6f}")

        # Transferir el banco de memorias completo
        new_matriarca.bank.embeddings = self.bank.embeddings.clone()
        new_matriarca.bank.metadata = self.bank.metadata.copy()
        new_matriarca.bank.save()

        print("  ✓ Conocimiento transferido a nueva Matriarca")
        return new_matriarca

    @property
    def memory_count(self) -> int:
        return self.bank.size


# ─── Test rápido ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🐘 Test de la Matriarca...")

    cfg = MatriarcaConfig(embd_dim=512, infrasound_dim=256, max_memories=100)
    matriarca = Matriarca(cfg, device="cuda" if torch.cuda.is_available() else "cpu")

    # Simular estado del enjambre
    state = torch.randn(512)

    # Emitir infrasónidos
    infrasound = matriarca.emit_infrasound(state)
    print(f"  Infrasónidos emitidos: shape={infrasound.shape}, norm={infrasound.norm():.4f}")

    # Almacenar interacción
    matriarca.store_interaction(state, "Emmanuel preguntó por el LLM", importance=0.9)
    matriarca.store_interaction(state * 0.5, "Emmanuel pidió un video", importance=0.7)
    print(f"  Memorias almacenadas: {matriarca.memory_count}")

    # Guardar
    matriarca.save()
    print("  ✓ Test completado")
