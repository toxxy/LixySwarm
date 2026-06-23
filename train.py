"""
Lixy-0.1 — Script de Entrenamiento Principal
============================================
Entrena el AgentBase (o el LixySwarm completo) desde cero.
Optimizado para RTX 5090 (32GB VRAM, Blackwell, bf16).

Uso:
  # Entrenamiento solo con corpus personal (debug rápido):
  python3 train.py --mode personal --steps 100
  
  # Pre-training con FineWeb (requiere descarga previa):
  python3 train.py --mode pretrain --steps 5000

  # Fine-tuning con corpus personal sobre base preentrenada:
  python3 train.py --mode finetune --checkpoint checkpoints/best.pt
"""

import os
import sys
import time
import math
import argparse
import json
from pathlib import Path
from contextlib import nullcontext
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Agregar src al path
SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from src.agents.agent_base import AgentBase, AgentConfig

# ─── Configuración ────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Datos
    data_dir: str = "data"
    mode: str = "personal"          # personal | pretrain | bilingual | finetune
    english_ratio: float = 0.7       # bilingual: FineWeb share
    
    # Modelo
    block_size: int = 512           # context length (512 para empezar, puede subir a 1024)
    
    # Entrenamiento
    batch_size: int = 8             # mini-batch size
    grad_accum_steps: int = 4       # gradient accumulation → batch efectivo = 32
    max_steps: int = 1000
    seed: int = 42
    
    # Learning rate schedule (cosine decay con warmup)
    learning_rate: float = 6e-4
    warmup_steps: int = 100
    min_lr: float = 6e-5            # mínimo LR al final del cosine decay
    
    # Optimizador
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Evaluación y guardado
    eval_interval: int = 100
    eval_steps: int = 20
    save_interval: int = 500
    checkpoint_dir: str = "checkpoints"
    
    # RTX 5090 config
    dtype: str = "bfloat16"         # bf16 nativo en Blackwell
    compile: bool = True            # torch.compile (PyTorch 2.x)
    device: str = "cuda"
    
    # Log
    log_interval: int = 10
    wandb: bool = False             # activar para monitoreo visual


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TokenDataset(Dataset):
    """Dataset de tokens binarios (uint16, formato nanoGPT)."""
    
    def __init__(
        self,
        data_path: Path,
        block_size: int,
        *,
        random_samples: bool = False,
        seed: int = 42,
        sample_offset: int = 0,
    ):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.random_samples = bool(random_samples)
        self.seed = int(seed)
        self.sample_offset = max(0, int(sample_offset))
        if len(self.data) <= self.block_size + 1:
            raise ValueError("token dataset is shorter than one training block")
        print(f"  Dataset: {data_path.name} — {len(self.data):,} tokens")
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        if self.random_samples:
            logical_index = self.sample_offset + int(idx)
            rng = np.random.default_rng(
                self.seed ^ (logical_index * 2654435761 & 0xFFFFFFFF)
            )
            idx = int(rng.integers(0, len(self.data) - self.block_size - 1))
        chunk = torch.from_numpy(self.data[idx:idx+self.block_size+1].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class BilingualTokenDataset(Dataset):
    """Deterministic FineWeb/Wikipedia-ES sampling without personal text."""

    def __init__(
        self,
        english_path: Path,
        spanish_path: Path,
        block_size: int,
        *,
        english_ratio: float = 0.7,
        seed: int = 42,
        sample_offset: int = 0,
    ):
        if not 0.0 <= float(english_ratio) <= 1.0:
            raise ValueError("english_ratio must be in [0, 1]")
        self.english = np.memmap(english_path, dtype=np.uint16, mode="r")
        self.spanish = np.memmap(spanish_path, dtype=np.uint16, mode="r")
        self.block_size = int(block_size)
        self.english_ratio = float(english_ratio)
        self.seed = int(seed)
        self.sample_offset = max(0, int(sample_offset))
        self.english_len = len(self.english) - self.block_size
        self.spanish_len = len(self.spanish) - self.block_size
        if min(self.english_len, self.spanish_len) <= 1:
            raise ValueError("bilingual source is shorter than one block")
        print(
            f"  Bilingual: FineWeb {len(self.english)/1e6:.0f}M + "
            f"Wikipedia-ES {len(self.spanish)/1e6:.0f}M tokens "
            f"({self.english_ratio:.0%}/{1-self.english_ratio:.0%})"
        )

    def __len__(self):
        return max(self.english_len, self.spanish_len)

    def __getitem__(self, idx):
        logical_index = self.sample_offset + int(idx)
        rng = np.random.default_rng(
            self.seed ^ (logical_index * 2654435761 & 0xFFFFFFFF)
        )
        if rng.random() < self.english_ratio:
            data, length = self.english, self.english_len
        else:
            data, length = self.spanish, self.spanish_len
        start = int(rng.integers(0, length))
        chunk = torch.from_numpy(
            data[start:start + self.block_size + 1].astype(np.int64)
        )
        return chunk[:-1], chunk[1:]


def get_dataloaders(cfg: TrainConfig, *, sample_offset: int = 0):
    data_dir = Path(cfg.data_dir)
    
    if cfg.mode == "personal":
        train_path = data_dir / "finetune" / "personal_tokens.bin"
        val_path = train_path  # con tan pocos tokens, usamos el mismo para val
    elif cfg.mode == "pretrain":
        train_path = data_dir / "pretrain" / "fineweb_train.bin"
        val_path = data_dir / "pretrain" / "fineweb_val.bin"
    elif cfg.mode == "bilingual":
        train_path = data_dir / "pretrain" / "fineweb_train.bin"
        spanish_path = data_dir / "spanish" / "wiki_es_tokens.bin"
        val_path = data_dir / "pretrain" / "fineweb_val.bin"
    elif cfg.mode == "finetune":
        train_path = data_dir / "finetune" / "personal_tokens.bin"
        val_path = train_path
    else:
        raise ValueError(f"modo desconocido: {cfg.mode}")
    
    if not train_path.exists():
        preparation = "download" if cfg.mode in {"pretrain", "bilingual"} else "personal"
        raise FileNotFoundError(
            f"No encontré el dataset: {train_path}\n"
            f"Corre primero: python3 src/data/prepare_pretrain.py --{preparation}"
        )
    
    if cfg.mode == "bilingual":
        if not spanish_path.exists():
            raise FileNotFoundError(
                f"No encontré el dataset: {spanish_path}"
            )
        train_ds = BilingualTokenDataset(
            train_path,
            spanish_path,
            cfg.block_size,
            english_ratio=cfg.english_ratio,
            seed=cfg.seed,
            sample_offset=sample_offset,
        )
    else:
        train_ds = TokenDataset(
            train_path,
            cfg.block_size,
            random_samples=True,
            seed=cfg.seed,
            sample_offset=sample_offset,
        )
    val_ds = TokenDataset(val_path, cfg.block_size)
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True
    )
    
    return train_loader, val_loader


# ─── LR Schedule ──────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: TrainConfig) -> float:
    """Cosine decay con linear warmup — mismo que nanoGPT."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / cfg.warmup_steps
    if step > cfg.max_steps:
        return cfg.min_lr
    # Cosine decay
    decay_ratio = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ─── Evaluación ───────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model, val_loader, cfg: TrainConfig, ctx):
    """Estima el loss en el conjunto de validación."""
    model.eval()
    losses = []
    
    val_iter = iter(val_loader)
    for _ in range(cfg.eval_steps):
        try:
            x, y = next(val_iter)
        except StopIteration:
            break
        
        x = x.to(cfg.device)
        y = y.to(cfg.device)
        
        with ctx:
            _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
    
    model.train()
    return sum(losses) / len(losses) if losses else float('inf')


# ─── Entrenamiento Principal ───────────────────────────────────────────────────

def _portable_model_state(model) -> dict:
    source = getattr(model, "_orig_mod", model)
    return source.state_dict()


def _save_checkpoint_atomic(value: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train(
    cfg: TrainConfig,
    checkpoint_path: str = None,
    *,
    resume: bool = False,
):
    print("🐜 Lixy-0.1 — Entrenamiento")
    print(f"   Modo: {cfg.mode}")
    print(f"   Device: {cfg.device}")
    print(f"   dtype: {cfg.dtype}")
    print(f"   Steps: {cfg.max_steps}")
    print(f"   Seed: {cfg.seed}")
    print()

    if cfg.max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if not 0.0 <= float(cfg.english_ratio) <= 1.0:
        raise ValueError("english_ratio must be in [0, 1]")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    # Device & dtype
    device = cfg.device
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map[cfg.dtype]
    
    # Context para autocast
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch_dtype) if device == 'cuda' else nullcontext()
    
    # ─── Modelo ───
    print("🧠 Inicializando modelo...")
    model_cfg = AgentConfig(
        block_size=cfg.block_size,
        agent_id=0,  # Agente léxico/base para pre-training
    )
    model = AgentBase(model_cfg)
    model = model.to(device)
    
    loaded_checkpoint = None
    start_step = 0
    best_val_loss = float('inf')
    # Cargar checkpoint si se especificó. --checkpoint is a warm start;
    # --resume also restores optimizer/global progress from this exact run.
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"  → Cargando checkpoint: {checkpoint_path}")
        loaded_checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        state = {
            key.removeprefix("_orig_mod."): value
            for key, value in loaded_checkpoint['model'].items()
        }
        model.load_state_dict(state)
        if resume:
            start_step = int(loaded_checkpoint.get("step", 0))
            best_val_loss = float(
                loaded_checkpoint.get("val_loss", float('inf'))
            )
        print(
            f"  → Checkpoint cargado (step {loaded_checkpoint.get('step', '?')}, "
            f"{'resume' if resume else 'warm start'})"
        )
    elif checkpoint_path:
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")
    if start_step > cfg.max_steps:
        raise ValueError("resume step is beyond the requested target step")
    
    # torch.compile (PyTorch 2.x — ~2x speedup en RTX 5090)
    if cfg.compile and device == 'cuda':
        print("  → Compilando con torch.compile()...")
        model = torch.compile(model)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  → Parámetros: {n_params/1e6:.1f}M")
    
    # ─── Optimizador ───
    # Separar parámetros que sí/no usan weight decay
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        fused=(device == "cuda"),
    )
    if resume and loaded_checkpoint and "optimizer" in loaded_checkpoint:
        optimizer.load_state_dict(loaded_checkpoint["optimizer"])
        print("  → Estado del optimizador restaurado")
    if resume and loaded_checkpoint:
        if "rng_state" in loaded_checkpoint:
            torch.set_rng_state(loaded_checkpoint["rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state_all" in loaded_checkpoint:
            torch.cuda.set_rng_state_all([
                value.cpu() for value in loaded_checkpoint["cuda_rng_state_all"]
            ])
        print("  → Estado RNG restaurado")
    
    # ─── Datos ───
    print("📊 Cargando datos...")
    samples_seen = start_step * cfg.batch_size * cfg.grad_accum_steps
    train_loader, val_loader = get_dataloaders(
        cfg, sample_offset=samples_seen
    )
    
    # ─── Training loop ───
    print()
    print("🚀 Iniciando entrenamiento...")
    print("=" * 60)
    
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    model.train()
    train_iter = iter(train_loader)
    
    t0 = time.time()
    for step in range(start_step, cfg.max_steps + 1):
        # Eval periódica
        if step % cfg.eval_interval == 0:
            val_loss = estimate_loss(model, val_loader, cfg, ctx)
            elapsed = time.time() - t0
            print(f"  Step {step:5d} | val_loss: {val_loss:.4f} | elapsed: {elapsed:.0f}s")
            
            # Guardar mejor checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt = {
                    "model": _portable_model_state(model),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "val_loss": val_loss,
                    "tokens_seen": (
                        step * cfg.batch_size * cfg.grad_accum_steps
                        * cfg.block_size
                    ),
                    "rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available() else []
                    ),
                    "config": asdict(cfg),
                    "model_config": asdict(model_cfg),
                }
                _save_checkpoint_atomic(ckpt, checkpoint_dir / "best.pt")
                print(f"  ✓ Mejor checkpoint guardado (val_loss={val_loss:.4f})")
        
        if step == cfg.max_steps:
            break
        
        # LR schedule
        lr = get_lr(step, cfg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Gradient accumulation
        loss_accum = 0.0
        optimizer.zero_grad()
        
        for micro_step in range(cfg.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            
            x = x.to(device)
            y = y.to(device)
            
            with ctx:
                _, loss, _ = model(x, targets=y)
                loss = loss / cfg.grad_accum_steps
            
            loss.backward()
            loss_accum += loss.item()
        
        # Gradient clipping
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        
        optimizer.step()
        
        # Log
        if step % cfg.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            tokens_per_sec = (
                cfg.batch_size * cfg.grad_accum_steps * cfg.block_size
                * (step - start_step + 1) / dt
                if dt > 0 else 0
            )
            print(
                f"  step {step:5d} | loss: {loss_accum:.4f} | "
                f"lr: {lr:.2e} | "
                f"tok/s: {tokens_per_sec:,.0f}"
            )
    
    # Guardar checkpoint final
    final_ckpt = {
        "model": _portable_model_state(model),
        "optimizer": optimizer.state_dict(),
        "step": cfg.max_steps,
        "val_loss": best_val_loss,
        "tokens_seen": (
            cfg.max_steps * cfg.batch_size * cfg.grad_accum_steps
            * cfg.block_size
        ),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else []
        ),
        "config": asdict(cfg),
        "model_config": asdict(model_cfg),
    }
    _save_checkpoint_atomic(final_ckpt, checkpoint_dir / "final.pt")
    print()
    print(f"✅ Entrenamiento completo!")
    print(f"   Mejor val_loss: {best_val_loss:.4f}")
    print(f"   Checkpoint: {checkpoint_dir / 'final.pt'}")


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar Lixy-0.1")
    parser.add_argument(
        "--mode",
        choices=["personal", "pretrain", "bilingual", "finetune"],
        default="personal",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-4)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--checkpoint", type=str,
        help="Warm start de pesos; reinicia optimizador y step",
    )
    checkpoint_group.add_argument(
        "--resume", type=str,
        help="Reanudar pesos, optimizador y step de la misma corrida",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directorio aislado para best.pt y final.pt",
    )
    parser.add_argument("--no-compile", action="store_true", help="Desactivar torch.compile")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--english-ratio",
        type=float,
        default=0.7,
        help="FineWeb fraction in bilingual mode",
    )
    args = parser.parse_args()
    
    cfg = TrainConfig(
        mode=args.mode,
        max_steps=args.steps,
        batch_size=args.batch,
        learning_rate=args.lr,
        compile=not args.no_compile,
        block_size=args.block_size,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        english_ratio=args.english_ratio,
    )
    
    train(
        cfg,
        checkpoint_path=args.resume or args.checkpoint,
        resume=bool(args.resume),
    )
