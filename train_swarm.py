"""
Lixy-0.1 — Training del LixySwarm Completo 🐜🐘🐬
==================================================
Entrena el enjambre completo: todos los agentes + Matriarca activa,
feromonas circulando, infrasónidos orientando en cada forward pass.

Parte de los pesos del fine-tuning individual (loss 0.14) — no de cero.
La Matriarca actualiza su banco de memorias cada `matriarca_update_interval` pasos.

Uso:
  # Test rápido (50 pasos):
  python3 train_swarm.py --steps 50

  # Training completo:
  python3 train_swarm.py --steps 500

  # Desde checkpoint previo del swarm:
  python3 train_swarm.py --steps 500 --checkpoint checkpoints/swarm_best.pt
"""

import sys
import math
import time
import json
import argparse
from pathlib import Path
from contextlib import nullcontext
from dataclasses import dataclass, asdict, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from src.agents.agent_base import AgentBase, AgentConfig
from src.swarm.orchestrator import LixySwarm, SwarmConfig
from src.matriarca.matriarca import Matriarca, MatriarcaConfig


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class SwarmTrainConfig:
    # Datos
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"

    # Base: pesos del fine-tuning individual
    agent_checkpoint: str = "checkpoints/finetune_best.pt"

    # Dataset
    mixed: bool = False           # True = 90% FineWeb + 10% personal
    fw_ratio: float = 0.9         # proporción FineWeb en modo mixto
    spanish: bool = False         # True = corpus español (wiki_es_tokens.bin)
    triple: bool = False          # True = 70% FineWeb + 20% Wiki-ES + 10% Personal

    # Training
    max_steps: int = 500
    batch_size: int = 4           # más pequeño — el swarm es 414M params
    grad_accum_steps: int = 4     # reducido de 8 — batch efectivo = 16 (era 32, causaba OOM con FineWeb)
    block_size: int = 512

    # LR — partir bajo para no destruir el fine-tuning
    learning_rate: float = 1e-4   # 6x menor que el fine-tuning original
    warmup_steps: int = 50
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Gradient checkpointing — reduce VRAM a costa de ~20% velocidad
    gradient_checkpointing: bool = True

    # Matriarca — actualización periódica durante training
    matriarca_update_interval: int = 20   # actualizar banco cada N pasos
    matriarca_lr: float = 5e-5           # LR propio para la Matriarca

    # Evaluación
    eval_interval: int = 50
    eval_steps: int = 10
    save_interval: int = 100
    log_interval: int = 10

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"
    compile: bool = False    # desactivar por defecto — swarm es complejo

    # Red P2P / colaboración remota
    network: bool = False
    network_checkpoint_dir: str = ""
    network_feromon_port: int = 7337
    network_gossip_port: int = 7338
    network_remote_weight: float = 0.25
    network_broadcast_interval: int = 1


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TokenDataset(Dataset):
    def __init__(self, data_path: Path, block_size: int):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        print(f"  Dataset: {data_path.name} — {len(self.data):,} tokens")

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = torch.from_numpy(self.data[idx:idx + self.block_size + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


class MixedDataset(Dataset):
    """
    Dataset mixto: mezcla FineWeb (fluencia) + corpus personal (personalidad).
    Cada muestra es elegida de FineWeb con prob `fw_ratio` o personal con `1-fw_ratio`.
    Por defecto 90% FineWeb, 10% personal.
    """
    def __init__(self, fw_path: Path, personal_path: Path, block_size: int, fw_ratio: float = 0.9, seed: int = 42):
        self.fw_path = str(fw_path)
        self.personal_path = str(personal_path)
        self.block_size = block_size
        self.fw_ratio = fw_ratio
        self.seed = seed

        # Cargar solo para obtener tamaños
        fw = np.memmap(fw_path, dtype=np.uint16, mode='r')
        personal = np.memmap(personal_path, dtype=np.uint16, mode='r')
        self.fw_len = len(fw) - block_size
        self.personal_len = len(personal) - block_size
        self._len = self.fw_len

        print(f"  MixedDataset: FineWeb {len(fw):,} + Personal {len(personal):,} tokens")
        print(f"  Ratio: {fw_ratio*100:.0f}% FineWeb / {(1-fw_ratio)*100:.0f}% Personal")

        # NO guardar mmaps como atributos — no son picklables para DataLoader workers
        # Se abren en __getitem__ con lazy init
        self._fw = None
        self._personal = None

    def _get_fw(self):
        if self._fw is None:
            self._fw = np.memmap(self.fw_path, dtype=np.uint16, mode='r')
        return self._fw

    def _get_personal(self):
        if self._personal is None:
            self._personal = np.memmap(self.personal_path, dtype=np.uint16, mode='r')
        return self._personal

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # RNG por-sample: mezcla idx con seed para variedad
        rng = np.random.default_rng(self.seed ^ (idx * 2654435761 & 0xFFFFFFFF))
        if rng.random() < self.fw_ratio:
            # FineWeb: posición aleatoria (no secuencial) para evitar correlación
            i = int(rng.integers(0, self.fw_len))
            chunk = torch.from_numpy(self._get_fw()[i:i + self.block_size + 1].astype(np.int64))
        else:
            i = int(rng.integers(0, self.personal_len))
            chunk = torch.from_numpy(self._get_personal()[i:i + self.block_size + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


class TripleDataset(Dataset):
    """
    Dataset triple: FineWeb (fluencia genérica) + Wiki ES (español) + Personal (personalidad).
    Ratios configurables, lazy-load para no saturar RAM.
    Default: 70% FineWeb, 20% Wiki, 10% Personal.
    """
    def __init__(self, fw_path: Path, wiki_path: Path, personal_path: Path,
                 block_size: int, fw_ratio: float = 0.70, wiki_ratio: float = 0.20, seed: int = 42):
        self.fw_path       = str(fw_path)
        self.wiki_path     = str(wiki_path)
        self.personal_path = str(personal_path)
        self.block_size    = block_size
        self.fw_ratio      = fw_ratio
        self.wiki_ratio    = wiki_ratio
        self.personal_ratio = 1.0 - fw_ratio - wiki_ratio
        self.seed = seed

        fw       = np.memmap(fw_path,       dtype=np.uint16, mode='r')
        wiki     = np.memmap(wiki_path,     dtype=np.uint16, mode='r')
        personal = np.memmap(personal_path, dtype=np.uint16, mode='r')
        self.fw_len       = len(fw)       - block_size
        self.wiki_len     = len(wiki)     - block_size
        self.personal_len = len(personal) - block_size
        self._len = self.fw_len  # tamaño efectivo basado en FineWeb

        print(f"  TripleDataset: FineWeb {len(fw)/1e9:.1f}B + Wiki-ES {len(wiki)/1e6:.0f}M + Personal {len(personal)/1e3:.0f}k tokens")
        print(f"  Ratio: {fw_ratio*100:.0f}% FW / {wiki_ratio*100:.0f}% Wiki / {self.personal_ratio*100:.0f}% Personal")

        self._fw = self._wiki = self._personal = None

    def _get(self, name):
        if name == 'fw':
            if self._fw is None: self._fw = np.memmap(self.fw_path, dtype=np.uint16, mode='r')
            return self._fw
        elif name == 'wiki':
            if self._wiki is None: self._wiki = np.memmap(self.wiki_path, dtype=np.uint16, mode='r')
            return self._wiki
        else:
            if self._personal is None: self._personal = np.memmap(self.personal_path, dtype=np.uint16, mode='r')
            return self._personal

    def __len__(self): return self._len

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed ^ (idx * 2654435761 & 0xFFFFFFFF))
        r = rng.random()
        if r < self.fw_ratio:
            i = int(rng.integers(0, self.fw_len))
            data, l = self._get('fw'), self.fw_len
        elif r < self.fw_ratio + self.wiki_ratio:
            i = int(rng.integers(0, self.wiki_len))
            data, l = self._get('wiki'), self.wiki_len
        else:
            i = int(rng.integers(0, self.personal_len))
            data, l = self._get('personal'), self.personal_len
        chunk = torch.from_numpy(data[i:i + self.block_size + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


# ─── Construcción del Swarm desde checkpoint ──────────────────────────────────

def build_swarm(cfg: SwarmTrainConfig) -> LixySwarm:
    """
    Construye el LixySwarm usando la config del checkpoint del agente entrenado.
    Así todos los AgentConfig (block_size, n_embd, etc.) coinciden exactamente.
    """
    agent_ckpt_path = Path(cfg.agent_checkpoint)
    if not agent_ckpt_path.exists():
        # Fallback a best.pt
        agent_ckpt_path = Path(cfg.checkpoint_dir) / "best.pt"
    if not agent_ckpt_path.exists():
        raise FileNotFoundError(f"No encontré checkpoint de agente en {cfg.agent_checkpoint}")

    ckpt = torch.load(agent_ckpt_path, map_location="cpu", weights_only=True)
    mc = ckpt.get("model_config", {})

    # Construir AgentConfig desde el checkpoint
    base_agent_cfg = AgentConfig(**{k: v for k, v in mc.items() if hasattr(AgentConfig, k)}) if mc else AgentConfig()
    base_agent_cfg.dropout = 0.1   # reactivar dropout para training

    matriarca_cfg = MatriarcaConfig(
        memory_path=str(Path(cfg.checkpoint_dir) / "matriarca_memory.json"),
        checkpoint_path=str(Path(cfg.checkpoint_dir) / "matriarca.pt"),
    )

    swarm_cfg = SwarmConfig(
        n_agents=3,
        feromon_dim=base_agent_cfg.feromon_dim,
        swarm_rounds=2,
        agent_configs=[
            AgentConfig(
                block_size=base_agent_cfg.block_size,
                vocab_size=base_agent_cfg.vocab_size,
                n_layer=base_agent_cfg.n_layer,
                n_head=base_agent_cfg.n_head,
                n_embd=base_agent_cfg.n_embd,
                dropout=0.1,
                bias=base_agent_cfg.bias,
                feromon_dim=base_agent_cfg.feromon_dim,
                identity_dim=base_agent_cfg.identity_dim,
                agent_id=i,
                n_agents=3,
            )
            for i in range(3)
        ],
        matriarca_config=matriarca_cfg,
    )

    swarm = LixySwarm(
        swarm_cfg,
        load_matriarca=True,
        agent_checkpoint=str(agent_ckpt_path),
    )

    print(f"  ✓ Swarm construido desde {agent_ckpt_path.name}")
    print(f"  ✓ block_size={base_agent_cfg.block_size}, n_embd={base_agent_cfg.n_embd}")
    return swarm, swarm_cfg, base_agent_cfg


# ─── LR Schedule ─────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: SwarmTrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step > cfg.max_steps:
        return cfg.min_lr
    decay_ratio = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ─── Evaluación ───────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(swarm: LixySwarm, val_loader, cfg: SwarmTrainConfig, ctx):
    swarm.eval()
    losses = []
    val_iter = iter(val_loader)
    for _ in range(cfg.eval_steps):
        try:
            x, y = next(val_iter)
        except StopIteration:
            break
        x, y = x.to(cfg.device), y.to(cfg.device)
        with ctx:
            _, loss, _ = swarm(x, targets=y, store_memory=False)
        if loss is not None:
            losses.append(loss.item())
    swarm.train()
    return sum(losses) / len(losses) if losses else float("inf")


# ─── Training Principal ───────────────────────────────────────────────────────

def train(cfg: SwarmTrainConfig, swarm_checkpoint: str = None):
    print("🐜🐘🐬 LixySwarm — Training Conjunto")
    print(f"   Device: {cfg.device}")
    print(f"   Steps: {cfg.max_steps}")
    print(f"   Batch efectivo: {cfg.batch_size * cfg.grad_accum_steps}")
    print(f"   LR: {cfg.learning_rate} (6x menor que fine-tuning, protege pesos)")
    print()

    device = cfg.device
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Limpiar VRAM antes de empezar
    if device == "cuda":
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  GPU libre: {torch.cuda.mem_get_info()[0]/1e9:.1f}GB")

    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    # ─── Construir swarm ───
    print("🧠 Construyendo LixySwarm...")
    swarm, swarm_cfg, agent_cfg = build_swarm(cfg)

    # Gradient checkpointing — reduce VRAM ~40% a costa de ~20% velocidad
    if cfg.gradient_checkpointing:
        for agent in swarm.agents:
            agent.use_gradient_checkpointing = True
        print(f"  ✓ Gradient checkpointing activado en {len(swarm.agents)} agentes")

    # Cargar checkpoint del swarm si existe (continuar training)
    start_step = 0
    best_val_loss = float("inf")
    if swarm_checkpoint and Path(swarm_checkpoint).exists():
        print(f"  → Continuando desde {swarm_checkpoint}")
        ckpt = torch.load(swarm_checkpoint, map_location=device, weights_only=False)
        # strict=False: claves del Delfín (v3) no están en checkpoints v2
        # Phase A: fusion layer cambió de forma (384→128 input) — filtrar size mismatches
        current_state = swarm.state_dict()
        filtered_model = {}
        skipped_shape = []
        for k, v in ckpt["model"].items():
            if k in current_state and current_state[k].shape == v.shape:
                filtered_model[k] = v
            elif k in current_state:
                skipped_shape.append(k)  # size mismatch — skip, keeps random init
        missing, unexpected = swarm.load_state_dict(filtered_model, strict=False)
        if skipped_shape:
            print(f"    ⚠ Shape mismatch (Phase A reinit): {len(skipped_shape)} keys")
            for sk in skipped_shape[:5]:
                print(f"      {sk}: ckpt={ckpt['model'][sk].shape} vs model={current_state[sk].shape}")
        dolphin_keys_missing = [k for k in missing if k.startswith("dolphin.")]
        other_missing = [k for k in missing if not k.startswith("dolphin.")]
        print(f"  ✓ Swarm cargado (strict=False)")
        print(f"    Delfín keys nuevas (init random): {len(dolphin_keys_missing)}")
        if other_missing:
            print(f"    ⚠ Otras keys faltantes: {other_missing[:3]}")
        if unexpected:
            print(f"    ⚠ Keys obsoletas (v2, ignoradas): {len(unexpected)} (EcholocationHead)")
        start_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"  ✓ Continuando desde step={start_step}, val_loss={best_val_loss:.4f}")

    swarm = swarm.to(device)
    swarm.train()

    network = None
    network_stats = {"remote_mixes": 0, "broadcasts": 0}
    if cfg.network:
        try:
            from src.network.swarm_network import SwarmNetwork

            network_checkpoint_dir = cfg.network_checkpoint_dir or cfg.checkpoint_dir
            network = SwarmNetwork.create(
                swarm=swarm,
                mode="auto",
                feromon_port=cfg.network_feromon_port,
                gossip_port=cfg.network_gossip_port,
                checkpoint_dir=network_checkpoint_dir,
            )

            @network.on_peer_connected
            def _on_training_peer(peer):
                print(
                    f"  🌐 Peer training conectado: {peer.host} "
                    f"UDP:{peer.feromon_port} TCP:{peer.gossip_port} id={peer.node_id[:16]}"
                )

            network.start()
            time.sleep(3)

            def _merge_remote_feromon(local_feromon):
                if network.collect_feromons():
                    network_stats["remote_mixes"] += 1
                return network.merge_remote_feromons(
                    local_feromon,
                    remote_weight=cfg.network_remote_weight,
                )

            swarm.remote_feromon_provider = _merge_remote_feromon
            print(f"  🌐 Red P2P training activa — {network.status()}")
        except Exception as e:
            print(f"  ⚠ Red P2P training no disponible: {e}")
            network = None

    # ─── Optimizador ───
    # Separar parámetros: agentes, pool/mixer (swarm mechanics), Matriarca
    agent_params = []
    swarm_params = []  # echolocation, feromon_pool, infrasound_mixer, confidence_heads, state_to_matriarca
    matriarca_params = list(swarm.matriarca.model.parameters()) if swarm.matriarca else []

    for name, param in swarm.named_parameters():
        if any(name.startswith(f"agents.{i}") for i in range(swarm_cfg.n_agents)):
            agent_params.append(param)
        else:
            swarm_params.append(param)

    # Weight decay solo para matrices 2D+
    def split_wd(params):
        decay = [p for p in params if p.dim() >= 2]
        nodecay = [p for p in params if p.dim() < 2]
        return decay, nodecay

    # Grupos por agente individual (para LR diferencial)
    param_groups = []
    for i in range(swarm_cfg.n_agents):
        agent_p = [p for n, p in swarm.named_parameters() if n.startswith(f"agents.{i}.")]
        decay, nodecay = split_wd(agent_p)
        param_groups.append({"params": decay,   "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
        param_groups.append({"params": nodecay, "lr": cfg.learning_rate, "weight_decay": 0.0})

    swarm_decay, swarm_nodecay = split_wd(swarm_params)
    mat_decay,   mat_nodecay   = split_wd(matriarca_params)

    param_groups += [
        {"params": swarm_decay,   "lr": cfg.learning_rate * 3, "weight_decay": cfg.weight_decay},
        {"params": swarm_nodecay, "lr": cfg.learning_rate * 3, "weight_decay": 0.0},
        {"params": mat_decay,     "lr": cfg.matriarca_lr,      "weight_decay": cfg.weight_decay},
        {"params": mat_nodecay,   "lr": cfg.matriarca_lr,      "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups, betas=(cfg.beta1, cfg.beta2))

    n_agent_p = sum(p.numel() for p in agent_params)
    n_swarm_p = sum(p.numel() for p in swarm_params)
    n_mat_p = sum(p.numel() for p in matriarca_params)
    print(f"  Parámetros entrenables:")
    print(f"    Agentes:    {n_agent_p/1e6:.1f}M (LR={cfg.learning_rate:.0e})")
    print(f"    Swarm mech: {n_swarm_p/1e6:.1f}M (LR={cfg.learning_rate*3:.0e})")
    print(f"    Matriarca:  {n_mat_p/1e6:.1f}M (LR={cfg.matriarca_lr:.0e})")

    # ─── Datos ───
    print("\n📊 Cargando datos...")
    personal_path = Path(cfg.data_dir) / "finetune" / "personal_tokens.bin"
    fw_train_path = Path(cfg.data_dir) / "pretrain" / "fineweb_train.bin"
    fw_val_path   = Path(cfg.data_dir) / "pretrain" / "fineweb_val.bin"

    if not personal_path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {personal_path}")

    if cfg.mixed:
        if not fw_train_path.exists():
            raise FileNotFoundError(f"FineWeb no encontrado: {fw_train_path}")
        print(f"  Modo: MIXTO ({cfg.fw_ratio*100:.0f}% FineWeb + {(1-cfg.fw_ratio)*100:.0f}% Personal)")
        dataset = MixedDataset(fw_train_path, personal_path, cfg.block_size, fw_ratio=cfg.fw_ratio)
        val_dataset = TokenDataset(fw_val_path, cfg.block_size) if fw_val_path.exists() else TokenDataset(personal_path, cfg.block_size)
    elif cfg.triple:
        wiki_path = Path(cfg.data_dir) / "spanish" / "wiki_es_tokens.bin"
        fw_val_path2 = fw_val_path if fw_val_path.exists() else personal_path
        if not fw_train_path.exists() or not wiki_path.exists():
            raise FileNotFoundError(f"Triple dataset requiere FineWeb ({fw_train_path}) y Wiki-ES ({wiki_path})")
        print(f"  Modo: TRIPLE (70% FineWeb + 20% Wiki-ES + 10% Personal)")
        dataset = TripleDataset(fw_train_path, wiki_path, personal_path, cfg.block_size)
        val_dataset = TokenDataset(fw_val_path2, cfg.block_size)
    elif cfg.spanish:
        spanish_path = Path(cfg.data_dir) / "spanish" / "wiki_es_tokens.bin"
        if not spanish_path.exists():
            raise FileNotFoundError(f"Corpus español no encontrado: {spanish_path}")
        print(f"  Modo: ESPAÑOL (Wikipedia es, {spanish_path.stat().st_size/1e9:.1f}GB)")
        dataset = TokenDataset(spanish_path, cfg.block_size)
        val_dataset = dataset
    else:
        print("  Modo: solo Personal")
        dataset = TokenDataset(personal_path, cfg.block_size)
        val_dataset = dataset

    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False,  # shuffle interno via __getitem__ RNG
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True
    )

    # ─── Training loop ───
    # target_step es ABSOLUTO — independiente de cuántas veces se reanude
    target_step = start_step + cfg.max_steps
    print(f"\n🚀 Iniciando training conjunto (desde step {start_step}, target step {target_step})...")
    print("=" * 65)

    swarm.train()
    data_iter = iter(loader)
    t0 = time.time()
    loss_history = []
    matriarca_update_count = 0

    for step in range(start_step, target_step + 1):
        # ─── Break al llegar al límite ───
        if step == target_step:
            # Eval final antes de salir
            val_loss = estimate_loss(swarm, val_loader, cfg, ctx)
            elapsed = time.time() - t0
            mat_mems = swarm.matriarca.memory_count if swarm.matriarca else 0
            print(f"  Step {step:5d} | val_loss: {val_loss:.4f} | 🐘 memorias: {mat_mems} | elapsed: {elapsed:.0f}s")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_checkpoint(swarm, optimizer, step, val_loss, cfg, agent_cfg, swarm_cfg, "swarm_best.pt")
                print(f"  ✓ Mejor checkpoint guardado (val_loss={val_loss:.4f})")
            break

        # ─── Evaluación periódica (no en el step inicial de resume) ───
        is_resume_step = (step == start_step and start_step > 0)
        if step % cfg.eval_interval == 0 and not is_resume_step:
            val_loss = estimate_loss(swarm, val_loader, cfg, ctx)
            elapsed = time.time() - t0
            mat_mems = swarm.matriarca.memory_count if swarm.matriarca else 0
            print(
                f"  Step {step:5d} | val_loss: {val_loss:.4f} | "
                f"🐘 memorias: {mat_mems} | elapsed: {elapsed:.0f}s"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_checkpoint(swarm, optimizer, step, val_loss, cfg, agent_cfg, swarm_cfg, "swarm_best.pt")
                print(f"  ✓ Mejor checkpoint guardado (val_loss={val_loss:.4f})")
        # ─── LR schedule + diferencial por agente según fitness ───
        lr = get_lr(step - start_step, cfg)
        fitnesses = swarm._last_fitnesses
        for i, pg in enumerate(optimizer.param_groups):
            if i < 2 * swarm_cfg.n_agents:  # grupos de agentes (decay + nodecay por agente)
                agent_idx = i // 2
                if fitnesses and agent_idx < len(fitnesses):
                    # LR diferencial: fitness alto → aprende más, bajo → se diversifica
                    lr_factor = 0.7 + 0.6 * fitnesses[agent_idx].fitness
                    pg["lr"] = lr * lr_factor
                else:
                    pg["lr"] = lr
            elif i < 2 * swarm_cfg.n_agents + 4:  # swarm mech
                pg["lr"] = lr * 3
            else:  # matriarca
                pg["lr"] = cfg.matriarca_lr * (0.5 + 0.5 * lr / cfg.learning_rate)

        # ─── Gradient accumulation ───
        optimizer.zero_grad()
        loss_accum = 0.0

        # Decidir si actualizar banco de memorias en este paso
        do_store = (step % cfg.matriarca_update_interval == 0) and step > start_step

        for micro_step in range(cfg.grad_accum_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)

            # Solo guardar memoria en el primer micro_step del intervalo
            store_this_step = do_store and micro_step == 0

            with ctx:
                _, loss, _ = swarm(
                    x, targets=y,
                    context_text=f"training_step_{step}" if store_this_step else "",
                    store_memory=store_this_step,
                )
                loss = loss / cfg.grad_accum_steps

                # Diversity regularization: penaliza agentes demasiado similares
                if swarm._last_fitnesses and micro_step == 0:
                    SIMILARITY_THRESHOLD = 0.7
                    DIVERSITY_WEIGHT = 0.01
                    feromons = [af.feromon_divergence for af in swarm._last_fitnesses]
                    # Si divergencia promedio cae por debajo del umbral, penalizar
                    avg_div = sum(feromons) / len(feromons) if feromons else 1.0
                    if avg_div < (1.0 - SIMILARITY_THRESHOLD):
                        diversity_penalty = DIVERSITY_WEIGHT * (1.0 - SIMILARITY_THRESHOLD - avg_div)
                        loss = loss + diversity_penalty

            loss.backward()
            loss_accum += loss.item()

        if do_store:
            matriarca_update_count += 1

        # ─── Gradient clip + step ───
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(swarm.parameters()) + (list(swarm.matriarca.model.parameters()) if swarm.matriarca else []),
                cfg.grad_clip
            )
        optimizer.step()
        loss_history.append(loss_accum)

        if network is not None and cfg.network_broadcast_interval > 0:
            should_broadcast = ((step - start_step + 1) % cfg.network_broadcast_interval) == 0
            local_feromon = getattr(swarm, "_last_feromon", None)
            if should_broadcast and local_feromon is not None:
                with torch.no_grad():
                    feromon_out = local_feromon.detach()
                    if feromon_out.dim() > 1:
                        feromon_out = feromon_out.mean(dim=0)
                    avg_fitness = 0.5
                    if swarm._last_fitnesses:
                        avg_fitness = sum(af.fitness for af in swarm._last_fitnesses) / len(swarm._last_fitnesses)
                    network.broadcast_feromon(feromon_out, fitness=float(avg_fitness))
                    network_stats["broadcasts"] += 1

        # ─── Tracker de especialización ───
        if swarm._last_fitnesses:
            swarm.specialization.update(swarm._last_fitnesses, step)

        # ─── Log ───
        if step % cfg.log_interval == 0 and step > start_step:
            t1 = time.time()
            tokens_per_sec = (
                cfg.batch_size * cfg.grad_accum_steps * cfg.block_size * (step - start_step) / (t1 - t0)
                if step > start_step else 0
            )
            recent_loss = sum(loss_history[-10:]) / len(loss_history[-10:])
            print(
                f"  step {step:5d} | loss: {loss_accum:.4f} | "
                f"avg10: {recent_loss:.4f} | "
                f"lr: {lr:.1e} | "
                f"tok/s: {tokens_per_sec:,.0f}"
            )
            if network is not None:
                print(
                    f"     🌐 {network.stats.summary()} | "
                    f"remote_mix={network_stats['remote_mixes']} | "
                    f"broadcasts={network_stats['broadcasts']}"
                )

        # ─── Checkpoint periódico + reporte especialización ───
        if step % cfg.save_interval == 0 and step > start_step:
            # FIX: guardar con val_loss real, no con train loss
            _save_checkpoint(swarm, optimizer, step, best_val_loss, cfg, agent_cfg, swarm_cfg, "swarm_latest.pt")
            swarm.specialization.save(step)
            # Guardar memorias de especialización en Matriarca
            if swarm.matriarca and swarm._last_fitnesses:
                for af in swarm._last_fitnesses:
                    label = swarm.specialization._infer_label(str(af.agent_id))
                    swarm.matriarca.store_interaction(
                        torch.randn(swarm.matriarca.cfg.embd_dim) * 0.1,  # embedding sintético
                        text=f"[hormiga_{af.agent_id}] label={label} fitness={af.fitness:.3f} div={af.feromon_divergence:.3f}",
                        importance=af.fitness,
                        auto_compress=False,
                    )
            # Reporte de especialización cada save_interval
            if step % (cfg.save_interval * 2) == 0:
                print(swarm.specialization.report(step))
            # Penalizar memorias no usadas cada 5×save_interval
            if swarm.matriarca and step % (cfg.save_interval * 5) == 0:
                swarm.matriarca.penalize_unused(
                    top_k_used=torch.tensor([]),
                    all_indices=range(swarm.matriarca.bank.size),
                    penalty=-0.01,
                )

            # ─── Tick del ciclo de vida del enjambre ──────────────────────────
            # Hormigas nacen/mueren + delfines escalan según fitness y diversidad
            if hasattr(swarm, 'ant_lifecycle') and swarm.ant_lifecycle:
                current_div = avg_div if 'avg_div' in dir() else 0.6
                lifecycle_events = swarm.tick_lifecycle(
                    step=step,
                    swarm_diversity=current_div,
                    n_nodes=1,  # single-node local; se actualiza cuando LSP activo
                )
                if lifecycle_events:
                    for ev in lifecycle_events:
                        etype = ev.get('type', '?')
                        if etype == 'death':
                            print(f"  💜 Hormiga {ev.get('ant_id','?')} murió [{ev.get('reason','?')}] fitness={ev.get('fitness',0):.3f} | legado → Matriarca")
                        elif etype == 'birth':
                            inherited = '❤️ heredado' if ev.get('inherited') else '🌱 nuevo'
                            print(f"  💚 Hormiga {ev.get('ant_id','?')} nació [{inherited}] padre={ev.get('parent_id','?')}")
                        elif etype in ('spawn', 'retire'):
                            print(f"  🐬 Delfín {ev.get('dolphin_idx','?')} {etype}d (pool={swarm.dolphin.n_dolphins})")

    # ─── Guardar final ───
    _save_checkpoint(swarm, optimizer, target_step, best_val_loss, cfg, agent_cfg, swarm_cfg, "swarm_final.pt")
    swarm.specialization.save(target_step)

    elapsed = time.time() - t0
    mat_mems = swarm.matriarca.memory_count if swarm.matriarca else 0

    # Reporte final de especialización
    print(swarm.specialization.report(target_step))

    # ─── Loop evolutivo post-training: actualizar Matriarca desde el log ───
    # El log actual se pasa via argumento o se detecta por el path de este proceso
    import os, sys as _sys
    # Usar el log_file del argumento si se pasó, o buscar el más reciente
    current_log = getattr(cfg, '_current_log_path', None)
    if current_log is None:
        # Buscar el swarm log más reciente en /tmp
        import glob
        candidates = sorted(glob.glob("/tmp/swarm_*.log"), key=os.path.getmtime, reverse=True)
        current_log = candidates[0] if candidates else None
    if current_log and os.path.exists(current_log):
        try:
            _sys.path.insert(0, str(Path(__file__).parent))
            from train_matriarca import train_from_swarm_log, MatriarcaTrainConfig
            mat_cfg = MatriarcaTrainConfig(checkpoint_dir=cfg.checkpoint_dir)
            print(f"\n🐘 Auto-actualizando Matriarca desde: {current_log}")
            train_from_swarm_log(current_log, cfg=mat_cfg, distill=False)
        except Exception as e:
            print(f"  ⚠ Loop evolutivo post-training falló (no crítico): {e}")
    else:
        print(f"  ⚠ No se encontró log de training para loop evolutivo")

    # ─── Reporte de convergencia ───
    report = _convergence_report(loss_history, best_val_loss, elapsed, mat_mems, matriarca_update_count, cfg)

    print()
    print("✅ Training conjunto completo!")
    print(f"   Mejor val_loss: {best_val_loss:.4f}")
    print(f"   Loss inicial:   {loss_history[0]:.4f}")
    print(f"   Loss final:     {loss_history[-1]:.4f}")
    print(f"   Reducción:      {(1 - loss_history[-1]/loss_history[0])*100:.1f}%")
    print(f"   🐘 Memorias Matriarca: {mat_mems} ({matriarca_update_count} updates)")
    print(f"   Tiempo: {elapsed:.0f}s")
    print(f"   Reporte: {report}")
    if network is not None:
        print(f"   🌐 Red final: {network.stats.summary()}")
        network.stop()

    return best_val_loss, loss_history


def _save_checkpoint(swarm, optimizer, step, val_loss, cfg, agent_cfg, swarm_cfg, name):
    ckpt_path = Path(cfg.checkpoint_dir) / name
    torch.save({
        "model": swarm.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "val_loss": val_loss,
        "agent_config": asdict(agent_cfg),
        "swarm_config": {
            "n_agents": swarm_cfg.n_agents,
            "feromon_dim": swarm_cfg.feromon_dim,
            "swarm_rounds": swarm_cfg.swarm_rounds,
        },
        "train_config": asdict(cfg),
    }, ckpt_path)
    # Guardar Matriarca por separado
    if swarm.matriarca:
        swarm.matriarca.save()


def _convergence_report(loss_history, best_val_loss, elapsed, mat_mems, mat_updates, cfg):
    report = {
        "training": {
            "steps": cfg.max_steps,
            "batch_size_effective": cfg.batch_size * cfg.grad_accum_steps,
            "learning_rate": cfg.learning_rate,
            "elapsed_s": round(elapsed, 1),
        },
        "convergence": {
            "loss_inicial": round(loss_history[0], 4) if loss_history else None,
            "loss_final": round(loss_history[-1], 4) if loss_history else None,
            "best_val_loss": round(best_val_loss, 4),
            # NOTA: best_val_loss = total_loss/(n_agents*rounds) medido durante training
            # Para perplexity real: ppl = exp(best_val_loss) solo si los agentes
            # convergen — ver benchmark.py para medición independiente
            "reduccion_pct": round((1 - loss_history[-1] / loss_history[0]) * 100, 1) if loss_history and loss_history[0] > 0 else 0,
            "loss_history_10step": [round(l, 4) for l in loss_history[::10]],
        },
        "matriarca": {
            "memorias_finales": mat_mems,
            "updates_durante_training": mat_updates,
            "update_interval": cfg.matriarca_update_interval,
        },
    }
    report_path = Path(cfg.checkpoint_dir) / f"swarm_convergence_{int(time.time())}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report_path.name


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training del LixySwarm completo 🐜🐘🐬")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5, help="LR mínimo al final del cosine schedule")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup steps")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--eval-steps", type=int, default=10, help="Batches para estimar val_loss")
    parser.add_argument("--eval-interval", type=int, default=50, help="Intervalo de evaluación")
    parser.add_argument("--checkpoint", type=str, default=None, help="Continuar desde swarm checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directorio de checkpoints de salida")
    parser.add_argument("--agent-checkpoint", type=str, default="checkpoints/finetune_best.pt", help="Checkpoint base de agente")
    parser.add_argument("--mixed", action="store_true", help="90%% FineWeb + 10%% personal")
    parser.add_argument("--spanish", action="store_true", help="Corpus Wikipedia español")
    parser.add_argument("--triple", action="store_true", help="70%% FW + 20%% Wiki-ES + 10%% Personal")
    parser.add_argument("--fw-ratio", type=float, default=0.9)
    parser.add_argument("--log-path", type=str, default=None, help="Path del log para loop evolutivo post-training")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--network", action="store_true", help="Mezclar/publicar feromonas vía LSP v2")
    parser.add_argument("--network-checkpoint-dir", type=str, default="", help="Directorio para identidad/peers LSP")
    parser.add_argument("--network-feromon-port", type=int, default=7337)
    parser.add_argument("--network-gossip-port", type=int, default=7338)
    parser.add_argument("--network-remote-weight", type=float, default=0.25)
    parser.add_argument("--network-broadcast-interval", type=int, default=1)
    args = parser.parse_args()

    cfg = SwarmTrainConfig(
        checkpoint_dir=args.checkpoint_dir,
        agent_checkpoint=args.agent_checkpoint,
        max_steps=args.steps,
        batch_size=args.batch,
        learning_rate=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup,
        grad_accum_steps=args.grad_accum,
        eval_steps=args.eval_steps,
        eval_interval=args.eval_interval,
        compile=not args.no_compile and args.steps > 100,
        block_size=args.block_size,
        mixed=args.mixed,
        fw_ratio=args.fw_ratio,
        spanish=args.spanish,
        triple=args.triple,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        network=args.network,
        network_checkpoint_dir=args.network_checkpoint_dir,
        network_feromon_port=args.network_feromon_port,
        network_gossip_port=args.network_gossip_port,
        network_remote_weight=args.network_remote_weight,
        network_broadcast_interval=args.network_broadcast_interval,
    )
    # Guardar el path del log en cfg para que el loop evolutivo lo use
    cfg._current_log_path = args.log_path

    train(cfg, swarm_checkpoint=args.checkpoint)
