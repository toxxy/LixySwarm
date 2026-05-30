"""
Lixy-0.1 — Training Dedicado de la Matriarca 🐘
================================================
Entrena los pesos de la Matriarca de forma específica para que:
1. Genere mejores "infrasónidos" basados en el historial del enjambre
2. Aprenda a comprimir y orientar desde el corpus personal
3. Ejecute el loop evolutivo post-training (distilación transgeneracional)

Uso:
  # Training básico (usa corpus personal + checkpoints existentes):
  python3 train_matriarca.py

  # Con más pasos:
  python3 train_matriarca.py --steps 500

  # Solo loop evolutivo (sin re-entrenar):
  python3 train_matriarca.py --evolve-only

  # Con distilación a nueva generación:
  python3 train_matriarca.py --distill
"""

import sys
import math
import time
import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from src.matriarca.matriarca import Matriarca, MatriarcaConfig, MatriarcaModel, MemoryBank
from src.agents.agent_base import AgentBase, AgentConfig


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class MatriarcaTrainConfig:
    # Datos
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"

    # Training
    max_steps: int = 300
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 30

    # Modelo
    embd_dim: int = 512
    infrasound_dim: int = 256
    max_memories: int = 4096
    n_heads: int = 8
    n_layers: int = 4

    # Loop evolutivo
    n_distill_steps: int = 100     # pasos de destilación transgeneracional
    store_swarm_memories: bool = True  # guardar memorias del enjambre

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"

    log_interval: int = 20
    eval_interval: int = 50


# ─── Dataset de infrasónidos ──────────────────────────────────────────────────

class InfrasoundDataset(torch.utils.data.Dataset):
    """
    Dataset para entrenar la Matriarca.

    Estrategia: usar embeddings del AgentBase (frozen) sobre el corpus personal
    como "estados del enjambre" que la Matriarca debe aprender a orientar.

    Cada muestra = (estado_enjambre, objetivo_de_orientación)
    donde el objetivo es el embedding del siguiente chunk del corpus.
    """

    def __init__(
        self,
        agent: AgentBase,
        data_path: Path,
        embd_dim: int,
        block_size: int = 128,
        n_samples: int = 2000,
        device: str = "cpu",
    ):
        self.device = device
        self.embd_dim = embd_dim
        self.block_size = block_size

        print(f"  📊 Preparando dataset de infrasónidos...")
        print(f"     Corpus: {data_path}")

        raw = np.memmap(data_path, dtype=np.uint16, mode='r')
        total_tokens = len(raw)
        print(f"     Tokens: {total_tokens:,}")

        agent = agent.to(device)
        agent.eval()

        # Proyector: n_embd → matriarca embd_dim
        projector = nn.Linear(agent.config.n_embd, embd_dim, bias=False).to(device)
        nn.init.orthogonal_(projector.weight)

        states = []
        targets = []

        step = max(1, total_tokens // n_samples)
        actual_samples = 0

        with torch.no_grad():
            for i in range(0, total_tokens - block_size * 2, step):
                if actual_samples >= n_samples:
                    break

                # Chunk actual → estado
                chunk = torch.from_numpy(
                    raw[i:i + block_size].astype(np.int64)
                ).unsqueeze(0).to(device)

                # Chunk siguiente → objetivo
                chunk_next = torch.from_numpy(
                    raw[i + block_size:i + block_size * 2].astype(np.int64)
                ).unsqueeze(0).to(device)

                # Pasar por AgentBase frozen para obtener representación
                logits, _, feromon = agent(chunk)

                # Estado = feromona del agente (ya es feromon_dim)
                # Si feromon_dim != embd_dim, proyectamos
                state = feromon[0]  # [feromon_dim]

                # Objetivo = embedding del siguiente chunk
                logits_next, _, feromon_next = agent(chunk_next)
                target = feromon_next[0]  # [feromon_dim]

                states.append(state.cpu())
                targets.append(target.cpu())
                actual_samples += 1

        self.states = torch.stack(states)    # [N, feromon_dim]
        self.targets = torch.stack(targets)  # [N, feromon_dim]
        print(f"     {actual_samples} muestras de infrasónidos generadas")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.targets[idx]


# ─── Loss de infrasónidos ─────────────────────────────────────────────────────

def infrasound_loss(
    matriarca: MatriarcaModel,
    state: torch.Tensor,       # [B, feromon_dim]
    target: torch.Tensor,      # [B, feromon_dim]
    memory_bank: torch.Tensor, # [N, embd_dim]
    feromon_dim: int,
    embd_dim: int,
    projector: nn.Linear,
    device: str,
) -> torch.Tensor:
    """
    Loss para entrenar la Matriarca:
    - Contrastive: el infrasónido debe ser más similar al target que a negativos
    - Reconstruction: el infrasónido debe predecir el estado siguiente
    """
    B = state.shape[0]

    # Proyectar estado (feromon_dim) → embd_dim de la Matriarca
    state_proj = projector(state.to(device))   # [B, embd_dim]
    target_proj = projector(target.to(device)) # [B, embd_dim]

    # Matriarca emite infrasónidos
    infrasound = matriarca(state_proj, memory_bank)  # [B, infrasound_dim]

    # Proyectar target al espacio de infrasónidos para comparar
    target_infra = F.normalize(target_proj[:, :matriarca.cfg.infrasound_dim], dim=-1)
    pred_infra = F.normalize(infrasound, dim=-1)

    # MSE loss (reconstrucción)
    loss_mse = F.mse_loss(pred_infra, target_infra)

    # Contrastive loss (cosine similarity entre positivos debe ser > negativos)
    sim_pos = (pred_infra * target_infra).sum(dim=-1)  # [B]

    # Negativos = otras muestras del batch
    sim_neg = torch.mm(pred_infra, target_infra.t())  # [B, B]
    sim_neg = sim_neg - torch.eye(B, device=device) * 1e9  # excluir diagonal
    sim_neg = sim_neg.max(dim=-1).values  # [B]

    loss_contrastive = F.relu(0.3 - sim_pos + sim_neg).mean()

    return loss_mse + 0.5 * loss_contrastive


# ─── LR Schedule ─────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: MatriarcaTrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    decay_ratio = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return 1e-5 + coeff * (cfg.learning_rate - 1e-5)


# ─── Training Principal ───────────────────────────────────────────────────────

def train_matriarca(cfg: MatriarcaTrainConfig, evolve_only: bool = False, distill: bool = False):
    print("🐘 Matriarca — Training Dedicado")
    print(f"   Device: {cfg.device}")
    print(f"   Steps: {cfg.max_steps}")
    print()

    device = cfg.device
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ─── 1. Cargar AgentBase (frozen) ───
    print("🧠 Cargando AgentBase (frozen)...")
    agent_ckpt_path = checkpoint_dir / "finetune_best.pt"
    if not agent_ckpt_path.exists():
        agent_ckpt_path = checkpoint_dir / "best.pt"
    if not agent_ckpt_path.exists():
        raise FileNotFoundError(f"No encontré checkpoint del agente en {checkpoint_dir}")

    ckpt = torch.load(agent_ckpt_path, map_location=device, weights_only=True)
    model_cfg_dict = ckpt.get("model_config", {})
    agent_cfg = AgentConfig(**{k: v for k, v in model_cfg_dict.items() if hasattr(AgentConfig, k)}) if model_cfg_dict else AgentConfig()
    agent = AgentBase(agent_cfg)
    # Cargar pesos — puede estar compilado
    state_dict = ckpt["model"]
    # Limpiar prefijo _orig_mod si fue compilado
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    agent.load_state_dict(state_dict)
    agent.eval()
    for p in agent.parameters():
        p.requires_grad = False
    agent = agent.to(device)
    print(f"  ✓ AgentBase cargado desde {agent_ckpt_path.name}")

    # ─── 2. Inicializar Matriarca ───
    print("🐘 Inicializando Matriarca...")
    matriarca_cfg = MatriarcaConfig(
        embd_dim=cfg.embd_dim,
        infrasound_dim=cfg.infrasound_dim,
        max_memories=cfg.max_memories,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        memory_path=str(checkpoint_dir / "matriarca_memory.json"),
        checkpoint_path=str(checkpoint_dir / "matriarca.pt"),
    )
    matriarca = Matriarca(matriarca_cfg, device=device)

    if evolve_only:
        print("  → Modo evolve-only: saltando training, directo al loop evolutivo")
        evolutionary_loop(matriarca, cfg, distill=distill)
        return

    # ─── 3. Dataset ───
    print("\n📊 Preparando datos...")
    data_path = Path(cfg.data_dir) / "finetune" / "personal_tokens.bin"
    if not data_path.exists():
        data_path = list(Path(cfg.data_dir).rglob("*.bin"))
        if data_path:
            data_path = data_path[0]
            print(f"  → Usando {data_path}")
        else:
            raise FileNotFoundError(f"No encontré tokens en {cfg.data_dir}")

    dataset = InfrasoundDataset(
        agent=agent,
        data_path=data_path,
        embd_dim=cfg.embd_dim,
        block_size=128,
        n_samples=min(2000, len(np.memmap(data_path, dtype=np.uint16, mode='r')) // 128),
        device=device,
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, drop_last=True
    )

    # ─── 4. Proyector feromon → embd ───
    feromon_dim = agent_cfg.feromon_dim
    projector = nn.Linear(feromon_dim, cfg.embd_dim, bias=False).to(device)
    nn.init.orthogonal_(projector.weight)

    # ─── 5. Optimizador ───
    trainable_params = list(matriarca.model.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    # ─── 6. Training loop ───
    print("\n🚀 Training Matriarca...")
    print("=" * 60)

    matriarca.model.train()
    projector.train()
    data_iter = iter(loader)
    t0 = time.time()
    best_loss = float("inf")
    loss_history = []

    for step in range(cfg.max_steps):
        # LR schedule
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        try:
            state, target = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            state, target = next(data_iter)

        state = state.to(device)
        target = target.to(device)

        memory_bank = matriarca.bank.get_embeddings(device)

        loss = infrasound_loss(
            matriarca.model, state, target,
            memory_bank, feromon_dim, cfg.embd_dim,
            projector, device
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if step % cfg.log_interval == 0:
            elapsed = time.time() - t0
            print(f"  step {step:4d} | loss: {loss_val:.4f} | lr: {lr:.2e} | {elapsed:.0f}s")

        if step % cfg.eval_interval == 0 and step > 0:
            avg_loss = sum(loss_history[-cfg.eval_interval:]) / len(loss_history[-cfg.eval_interval:])
            if avg_loss < best_loss:
                best_loss = avg_loss
                matriarca.save()
                torch.save(projector.state_dict(), checkpoint_dir / "matriarca_projector.pt")
                print(f"  ✓ Checkpoint Matriarca guardado (avg_loss={avg_loss:.4f})")

    # Guardar final
    matriarca.save()
    torch.save(projector.state_dict(), checkpoint_dir / "matriarca_projector.pt")

    elapsed = time.time() - t0
    print(f"\n✅ Training Matriarca completo en {elapsed:.0f}s")
    print(f"   Mejor loss: {best_loss:.4f}")
    print(f"   Memorias en banco: {matriarca.memory_count}")

    # ─── 7. Loop evolutivo ───
    evolutionary_loop(matriarca, cfg, distill=distill)


# ─── Loop Evolutivo Post-Training ─────────────────────────────────────────────

def evolutionary_loop(matriarca: Matriarca, cfg: MatriarcaTrainConfig, distill: bool = False):
    """
    Loop evolutivo post-training:
    1. Actualiza banco de memorias con knowledge destilado del training
    2. Opcionalmente destila a nueva generación
    3. Guarda checkpoint separado con timestamp
    """
    print("\n🧬 Loop Evolutivo Post-Training")
    print("=" * 60)

    checkpoint_dir = Path(cfg.checkpoint_dir)
    device = cfg.device

    # ─── 1. Sintetizar memorias de "sabiduría aprendida" ───
    print("  📝 Sintetizando memorias del training...")
    matriarca.model.eval()

    with torch.no_grad():
        bank = matriarca.bank.get_embeddings(device)
        n = bank.shape[0]

        # Crear memoria de "resumen del training"
        if n > 1:
            # Calcular centroide del banco → representa la "esencia" acumulada
            centroid = bank.mean(dim=0)  # [embd_dim]
            # Emitir infrasónidos sobre el centroide = auto-reflexión
            infra = matriarca.model(centroid.unsqueeze(0), bank)  # [1, infrasound_dim]

            # Codificar como nueva memoria de "síntesis"
            synthesis_emb = matriarca.model.encode_memory(centroid.unsqueeze(0)).squeeze(0)
            matriarca.bank.add(
                synthesis_emb,
                text=f"[síntesis_training] centroide de {n} memorias, training_step={cfg.max_steps}",
                importance=1.0,
            )
            print(f"  ✓ Memoria de síntesis agregada ({n} → {matriarca.memory_count} memorias)")

    # ─── 2. Checkpoint con timestamp ───
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    versioned_ckpt = checkpoint_dir / f"matriarca_v{timestamp}.pt"
    matriarca.save()
    import shutil
    shutil.copy(checkpoint_dir / "matriarca.pt", versioned_ckpt)
    print(f"  ✓ Checkpoint versionado: {versioned_ckpt.name}")

    # ─── 3. Guardar reporte evolutivo ───
    report = {
        "timestamp": timestamp,
        "memory_count": matriarca.memory_count,
        "training_steps": cfg.max_steps,
        "memories_sample": matriarca.bank.metadata[-5:],  # últimas 5 memorias
    }
    report_path = checkpoint_dir / f"evolution_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Reporte evolutivo: {report_path.name}")

    # ─── 4. Destilación transgeneracional (opcional) ───
    if distill:
        print(f"\n🐘→🐘 Destilación transgeneracional ({cfg.n_distill_steps} pasos)...")
        new_cfg = MatriarcaConfig(
            embd_dim=matriarca.cfg.embd_dim,
            infrasound_dim=matriarca.cfg.infrasound_dim,
            max_memories=matriarca.cfg.max_memories,
            n_heads=matriarca.cfg.n_heads,
            n_layers=matriarca.cfg.n_layers,
            memory_path=str(checkpoint_dir / "matriarca_next_memory.json"),
            checkpoint_path=str(checkpoint_dir / "matriarca_next.pt"),
        )
        from src.matriarca.matriarca import Matriarca as M
        new_matriarca = M(new_cfg, device=cfg.device)
        matriarca.distill_to_new(new_matriarca, n_steps=cfg.n_distill_steps)
        new_matriarca.save()
        print(f"  ✓ Nueva Matriarca guardada: matriarca_next.pt")
        print(f"  ✓ Memorias transferidas: {new_matriarca.memory_count}")

    print("\n🧬 Loop evolutivo completo")
    print(f"   Memorias totales: {matriarca.memory_count}")
    print(f"   Checkpoint: checkpoints/matriarca.pt")
    if distill:
        print(f"   Nueva generación: checkpoints/matriarca_next.pt")


# ─── Quick validation ─────────────────────────────────────────────────────────

# ─── From Swarm Log ───────────────────────────────────────────────────────────

def train_from_swarm_log(log_path: str, cfg=None, distill: bool = False):
    """
    Lee el log de un training del swarm y actualiza la Matriarca con las
    interacciones registradas.

    Mejoras v2 sobre la versión original:
    - Usa el modelo real de la Matriarca para codificar embeddings (no sintéticos)
    - Lee convergence JSON si existe para metadata enriquecida
    - Distingue entre pasos de alta/baja calidad y ajusta importancia en consecuencia
    - Agrega memorias de "hitos" (mejor val_loss, cambios de especialización)
    - Agrupa pasos en chunks para reducir ruido y tamaño del banco
    """
    import re, glob
    from pathlib import Path as _Path

    print(f"🐘 Actualizando Matriarca desde log del swarm: {log_path}")
    log_file = _Path(log_path)
    if not log_file.exists():
        print(f"  ⚠ Log no encontrado: {log_path}")
        return

    if cfg is None:
        cfg = MatriarcaTrainConfig()
    checkpoint_dir = _Path(cfg.checkpoint_dir)
    matriarca_cfg = MatriarcaConfig(
        memory_path=str(checkpoint_dir / "matriarca_memory.json"),
        checkpoint_path=str(checkpoint_dir / "matriarca.pt"),
    )
    matriarca = Matriarca(matriarca_cfg, device=cfg.device)
    embd_dim = matriarca_cfg.embd_dim

    # ─── 1. Parsear log ────────────────────────────────────────────────────────
    step_pattern = re.compile(r"step\s+(\d+)\s*\|\s*loss:\s*([\d.]+)\s*\|\s*avg10:\s*([\d.]+)")
    val_pattern  = re.compile(r"Step\s+(\d+)\s*\|\s*val_loss:\s*([\d.]+)")
    spec_pattern = re.compile(r"Agente (\d+) \[([^\]]+)\].*fitness=([\d.]+).*div=([\d.]+)")

    train_steps, val_steps, spec_events = [], [], []
    with open(log_file) as f:
        for line in f:
            m = step_pattern.search(line)
            if m:
                train_steps.append({
                    "step": int(m.group(1)),
                    "loss": float(m.group(2)),
                    "avg10": float(m.group(3)),
                })
            m = val_pattern.search(line)
            if m:
                val_steps.append({"step": int(m.group(1)), "loss": float(m.group(2))})
            m = spec_pattern.search(line)
            if m:
                spec_events.append({
                    "agent": int(m.group(1)),
                    "label": m.group(2).strip(),
                    "fitness": float(m.group(3)),
                    "div": float(m.group(4)),
                })

    if not train_steps:
        print("  ⚠ No se encontraron pasos de training en el log")
        return

    print(f"  📊 {len(train_steps)} pasos train | {len(val_steps)} val | {len(spec_events)} eventos spec")

    # ─── 2. Leer convergence JSON si existe ────────────────────────────────────
    convergence_data = {}
    json_candidates = sorted(
        glob.glob(str(checkpoint_dir / "swarm_convergence_*.json")),
        key=lambda x: _Path(x).stat().st_mtime, reverse=True
    )
    if json_candidates:
        try:
            import json as _json
            with open(json_candidates[0]) as f:
                convergence_data = _json.load(f)
            print(f"  📄 Convergence JSON: {_Path(json_candidates[0]).name}")
        except Exception:
            pass

    best_val = convergence_data.get("convergence", {}).get("best_val_loss",
               min((v["loss"] for v in val_steps), default=5.0))
    final_loss = train_steps[-1]["loss"] if train_steps else 5.0
    all_losses = [s["loss"] for s in train_steps]
    loss_min, loss_max = min(all_losses), max(all_losses)
    loss_range = max(loss_max - loss_min, 1e-6)

    # ─── 3. Agrupar pasos en chunks de 50 → una memoria por chunk ─────────────
    # Reduce ruido y tamaño del banco. Cada chunk = "episodio de aprendizaje"
    chunk_size = 50
    chunks = [train_steps[i:i+chunk_size] for i in range(0, len(train_steps), chunk_size)]

    memories_added = 0
    for chunk in chunks:
        avg_loss = sum(s["loss"] for s in chunk) / len(chunk)
        min_loss = min(s["loss"] for s in chunk)
        step_center = chunk[len(chunk)//2]["step"]
        step_norm = step_center / max(train_steps[-1]["step"], 1)

        # Importancia: inversamente proporcional a la loss promedio
        # + bonus por chunks recientes (más reciente = más relevante)
        recency_bonus = 0.1 * step_norm   # últimos pasos +10% importancia
        loss_factor = max(0.0, 1.0 - (avg_loss - loss_min) / loss_range)
        importance = max(0.1, min(0.95, 0.6 * loss_factor + 0.3 * recency_bonus + 0.1))

        # Embedding semántico usando el modelo real de la Matriarca
        # Codificar el vector de estado del chunk en el espacio de la Matriarca
        with torch.no_grad():
            # Vector de features del chunk → proyectar al embd_dim
            feature_vec = torch.zeros(embd_dim, device=cfg.device)
            # Dimensiones semánticas del training
            feature_vec[0] = step_norm * 2 - 1                            # posición temporal [-1,1]
            feature_vec[1] = (avg_loss - loss_min) / loss_range * 2 - 1  # calidad [-1,1]
            feature_vec[2] = (min_loss - loss_min) / loss_range * 2 - 1  # mejor momento [-1,1]
            feature_vec[3] = recency_bonus * 10 - 0.5                     # recencia [-0.5,0.5]
            feature_vec[4] = importance * 2 - 1                           # auto-importancia
            # Ruido estructurado para diversidad (reproducible por step)
            torch.manual_seed(step_center)
            feature_vec[5:min(20, embd_dim)] = torch.randn(min(15, embd_dim-5)) * 0.05
            # Codificar con el modelo real
            encoded = matriarca.model.encode_memory(feature_vec.unsqueeze(0)).squeeze(0)

        text = (
            f"[chunk] steps={chunk[0]['step']}-{chunk[-1]['step']} "
            f"avg_loss={avg_loss:.4f} min_loss={min_loss:.4f} "
            f"n={len(chunk)}"
        )
        matriarca.bank.add(encoded, text, importance)
        memories_added += 1

    print(f"  ✓ {memories_added} memorias de chunks agregadas")

    # ─── 4. Memorias de "hitos" especiales ────────────────────────────────────
    # Mejor val_loss del run
    if val_steps:
        best_val_step = min(val_steps, key=lambda v: v["loss"])
        with torch.no_grad():
            hito_vec = torch.zeros(embd_dim, device=cfg.device)
            hito_vec[0] = best_val_step["step"] / max(train_steps[-1]["step"], 1) * 2 - 1
            hito_vec[1] = 0.9   # señal de "hito positivo"
            hito_vec[2] = 1.0   # señal de val
            hito_vec[5] = (5.0 - best_val_step["loss"]) / 5.0  # calidad normalizada
            encoded_hito = matriarca.model.encode_memory(hito_vec.unsqueeze(0)).squeeze(0)
        matriarca.bank.add(
            encoded_hito,
            text=f"[hito_val] step={best_val_step['step']} best_val_loss={best_val_step['loss']:.4f}",
            importance=0.98,
        )
        print(f"  ✓ Hito val_loss añadido: {best_val_step['loss']:.4f} @ step {best_val_step['step']}")

    # Eventos de especialización únicos
    seen_labels = set()
    for ev in spec_events:
        key = (ev["agent"], ev["label"])
        if key not in seen_labels:
            seen_labels.add(key)
            with torch.no_grad():
                spec_vec = torch.zeros(embd_dim, device=cfg.device)
                spec_vec[0] = ev["agent"] / 3.0
                spec_vec[1] = ev["fitness"]
                spec_vec[2] = ev["div"]
                spec_vec[6] = 1.0  # señal de especialización
                encoded_spec = matriarca.model.encode_memory(spec_vec.unsqueeze(0)).squeeze(0)
            matriarca.bank.add(
                encoded_spec,
                text=f"[spec] agente={ev['agent']} label={ev['label']} fitness={ev['fitness']:.3f}",
                importance=0.75,
            )
    if seen_labels:
        print(f"  ✓ {len(seen_labels)} eventos de especialización añadidos")

    print(f"  ✓ Banco total: {matriarca.memory_count} memorias")

    # ─── 5. Penalizar memorias de alta pérdida ─────────────────────────────────
    n_penalized = 0
    for i, meta in enumerate(matriarca.bank.metadata):
        loss_m = re.search(r"avg_loss=([\d.]+)", meta.get("text", ""))
        if loss_m:
            chunk_loss = float(loss_m.group(1))
            rel_loss = (chunk_loss - loss_min) / loss_range
            if rel_loss > 0.75:  # peor 25% de chunks
                matriarca.bank.metadata[i]["importance"] = max(
                    0.05, meta["importance"] - 0.15
                )
                n_penalized += 1
    if n_penalized:
        print(f"  ✓ {n_penalized} memorias penalizadas (chunks de alta pérdida)")

    # ─── 6. Compresión si necesario ────────────────────────────────────────────
    if matriarca.bank.size >= matriarca_cfg.max_memories * 0.85:
        n_removed = matriarca.bank.compress()
        if n_removed > 0:
            print(f"  🗃️ Compresión: {n_removed} memorias → sintéticas")

    matriarca.save()
    print(f"  ✅ Matriarca actualizada: {matriarca.memory_count} memorias")
    evolutionary_loop(matriarca, cfg, distill=distill)



def matriarca_eval(matriarca: Matriarca = None, verbose: bool = True) -> dict:
    """
    Eval completo del banco de memorias de la Matriarca.

    Métricas:
    - Distribución de importancias (media, mediana, % activas vs degradadas)
    - Diversidad semántica del banco (cosine similarity promedio inter-memorias)
    - Distribución por tipo de memoria (runtime, training, spec, sintética)
    - Acceso reciente (% accedidas en últimas 24h)
    """
    from pathlib import Path as _Path
    import statistics, re, time as _time

    if matriarca is None:
        cfg = MatriarcaConfig(
            memory_path="checkpoints/matriarca_memory.json",
            checkpoint_path="checkpoints/matriarca.pt",
        )
        matriarca = Matriarca(cfg, device="cpu")

    bank = matriarca.bank
    n = bank.size
    if n == 0:
        return {"error": "banco vacío"}

    importances = [m["importance"] for m in bank.metadata]
    access_counts = [m.get("access_count", 0) for m in bank.metadata]
    last_accesses = [m.get("last_access") for m in bank.metadata]
    now = _time.time()
    recently_accessed = sum(1 for la in last_accesses if la and (now - la) < 86400)  # 24h

    # Tipos de memoria
    type_counts = {"runtime": 0, "training": 0, "spec": 0, "hito": 0, "sintética": 0, "otro": 0}
    for m in bank.metadata:
        text = m.get("text", "")
        if "[runtime]" in text or "[gen]" in text or "[sesión" in text:
            type_counts["runtime"] += 1
        elif "[chunk]" in text or "[swarm_log]" in text or "[síntesis" in text:
            type_counts["training"] += 1
        elif "[spec]" in text:
            type_counts["spec"] += 1
        elif "[hito" in text:
            type_counts["hito"] += 1
        elif m.get("synthetic") or "[sintética]" in text:
            type_counts["sintética"] += 1
        else:
            type_counts["otro"] += 1

    # Diversidad semántica (muestra de 100 memorias máx para eficiencia)
    sample_size = min(100, n)
    idx = torch.randperm(n)[:sample_size]
    embs = bank.get_embeddings("cpu")[idx].float()
    embs_norm = torch.nn.functional.normalize(embs, dim=-1)
    sim_matrix = torch.mm(embs_norm, embs_norm.t())
    # Excluir diagonal
    mask = ~torch.eye(sample_size, dtype=torch.bool)
    avg_similarity = sim_matrix[mask].mean().item()
    diversity = 1.0 - avg_similarity   # más alto = más diverso

    metrics = {
        "total_memories": n,
        "importance": {
            "mean": round(statistics.mean(importances), 4),
            "median": round(statistics.median(importances), 4),
            "stdev": round(statistics.stdev(importances) if n > 1 else 0, 4),
            "pct_active": round(sum(1 for i in importances if i > 0.3) / n, 3),
            "pct_degraded": round(sum(1 for i in importances if i < 0.1) / n, 3),
        },
        "access": {
            "total_accesses": sum(access_counts),
            "avg_per_memory": round(sum(access_counts) / max(n, 1), 2),
            "recently_accessed_24h": recently_accessed,
            "pct_never_accessed": round(sum(1 for a in access_counts if a == 0) / n, 3),
        },
        "diversity": {
            "avg_cosine_similarity": round(avg_similarity, 4),
            "semantic_diversity": round(diversity, 4),
        },
        "types": {k: v for k, v in type_counts.items() if v > 0},
    }

    if verbose:
        print(f"\n🐘 Matriarca Eval ({n} memorias)")
        print(f"  Importancia: media={metrics['importance']['mean']:.3f} "
              f"mediana={metrics['importance']['median']:.3f} "
              f"activas={metrics['importance']['pct_active']:.0%} "
              f"degradadas={metrics['importance']['pct_degraded']:.0%}")
        print(f"  Diversidad:  cosine_sim={avg_similarity:.3f} "
              f"diversity_score={diversity:.3f}")
        print(f"  Acceso:      {recently_accessed} recientes (24h) "
              f"| {metrics['access']['pct_never_accessed']:.0%} nunca accedidas")
        print(f"  Tipos:       { {k:v for k,v in type_counts.items() if v>0} }")
        print()

    return metrics


def validate_fixes():
    """Valida que los bugs de importancia negativa están corregidos."""
    print("🔧 Validando fixes de importancia negativa...")

    cfg = MatriarcaConfig(embd_dim=64, infrasound_dim=32, max_memories=10)
    cfg.memory_path = "/tmp/test_matriarca_memory.json"
    cfg.checkpoint_path = "/tmp/test_matriarca.pt"
    device = "cpu"

    matriarca = Matriarca(cfg, device=device)

    # Simular importancias problemáticas (antes daban negativas)
    state = torch.randn(64)
    test_cases = [
        ("texto normal", 0.9),
        ("importancia cero", 0.0),
        ("importancia negativa SIN fix daría bug", -0.5),  # debe clampear a 0
        ("importancia > 1 sin fix daría bug", 1.5),        # debe clampear a 1
        ("loss muy alto", -2.3),                           # simulando loss > 10
    ]

    for text, imp in test_cases:
        matriarca.store_interaction(state, text, importance=imp)
        stored_imp = matriarca.bank.metadata[-1]["importance"]
        status = "✓" if 0.0 <= stored_imp <= 1.0 else "✗ BUG!"
        print(f"  {status} imp={imp:.1f} → stored={stored_imp:.4f} | {text}")

    # Forzar prune para validar que no hay crash con negativos
    for i in range(15):  # superar max_memories=10
        matriarca.store_interaction(state * (i * 0.1), f"memoria {i}", importance=float(i) / 10)

    assert matriarca.memory_count <= cfg.max_memories, "Prune falló!"
    print(f"  ✓ Prune correcto: {matriarca.memory_count} memorias (max={cfg.max_memories})")
    print("  ✅ Todos los fixes validados\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training dedicado de la Matriarca 🐘")
    parser.add_argument("--steps", type=int, default=300, help="Pasos de training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--evolve-only", action="store_true", help="Solo loop evolutivo")
    parser.add_argument("--distill", action="store_true", help="Destilación transgeneracional")
    parser.add_argument("--validate", action="store_true", help="Solo validar fixes")
    parser.add_argument("--eval", action="store_true", help="Eval del banco de memorias")
    parser.add_argument("--from-swarm-log", type=str, default=None, help="Actualizar Matriarca desde log del swarm")
    args = parser.parse_args()

    if args.validate:
        validate_fixes()
        sys.exit(0)

    cfg = MatriarcaTrainConfig(
        max_steps=args.steps,
        learning_rate=args.lr,
        batch_size=args.batch,
    )

    # Siempre validar fixes primero
    validate_fixes()

    if args.eval:
        matriarca_eval(verbose=True)
    elif args.from_swarm_log:
        train_from_swarm_log(args.from_swarm_log, cfg=cfg, distill=args.distill)
    else:
        train_matriarca(cfg, evolve_only=args.evolve_only, distill=args.distill)
