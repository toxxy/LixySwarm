#!/usr/bin/env python3
"""
benchmark.py — Evaluación formal de LixySwarm
==============================================
Métricas:
  1. Perplexity en FineWeb val set
  2. Perplexity en corpus personal (si disponible)
  3. Repetición: rep@5, rep@10 en textos generados
  4. Diversidad léxica (TTR, RTTR)
  5. Comparativa con GPT-2 small (via HuggingFace si disponible)

Output: experiments/benchmark_TIMESTAMP.json + resumen stdout

Uso:
  python3 benchmark.py                         # checkpoint auto-detectado
  python3 benchmark.py --checkpoint ckpt.pt   # checkpoint específico
  python3 benchmark.py --no-gpt2              # sin comparativa GPT-2
  python3 benchmark.py --batches 50           # menos batches (más rápido)
  python3 benchmark.py --health-only          # solo sensores del organismo
"""

import sys, os, time, json, math, re, argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from src.swarm.orchestrator import LixySwarm, SwarmConfig
from src.agents.agent_base import AgentConfig
from src.matriarca.matriarca import MatriarcaConfig
from src.utils.sampling import sample_token
from src.utils.tokenizer import get_gpt2_encoding

CHECKPOINT_DIR = Path("checkpoints")
DATA_DIR = Path("data/pretrain")
EXPERIMENTS_DIR = Path("experiments")


# ─── Prompts de evaluación ────────────────────────────────────────────────────

EVAL_PROMPTS = [
    # EN — técnica
    ("en_tech",   "In machine learning, a neural network is"),
    ("en_tech",   "The transformer architecture was introduced in"),
    ("en_tech",   "Python is a programming language designed for"),
    # EN — factual
    ("en_fact",   "The capital of France is Paris, which is"),
    ("en_fact",   "The speed of light in a vacuum is approximately"),
    # EN — creativa
    ("en_creat",  "Once upon a time, in a world powered by"),
    ("en_creat",  "The future of artificial intelligence will"),
    # ES — general
    ("es_gen",    "La inteligencia artificial es una rama de"),
    ("es_gen",    "En el futuro, los modelos de lenguaje podrán"),
    ("es_gen",    "Hola, me llamo Lixy y soy un modelo de"),
]


# ─── Dataset helper ───────────────────────────────────────────────────────────

class TokenDataset:
    def __init__(self, path: Path, block_size: int = 512):
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def get_batch(self, batch_size: int, device: str) -> tuple:
        n = len(self)
        if n == 0:
            return None, None
        idx = np.random.randint(0, n, size=(batch_size,))
        x = torch.stack([torch.from_numpy(self.data[i:i+self.block_size].astype(np.int64))
                         for i in idx]).to(device)
        y = torch.stack([torch.from_numpy(self.data[i+1:i+1+self.block_size].astype(np.int64))
                         for i in idx]).to(device)
        return x, y


# ─── Métricas de texto ────────────────────────────────────────────────────────

def rep_n(text: str, n: int) -> float:
    """
    Repetición de n-gramas: fracción de n-gramas que son duplicados.
    rep@1 = fracción de tokens repetidos, rep@5 = de 5-gramas, etc.
    0 = sin repetición, 1 = todo repetido.
    """
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    unique = len(set(grams))
    return round(1.0 - unique / len(grams), 4)


def ttr(text: str) -> float:
    """Type-Token Ratio — diversidad léxica."""
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / len(tokens), 4)


def rttr(text: str) -> float:
    """Root Type-Token Ratio — insensible a longitud."""
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / math.sqrt(len(tokens)), 4)


def text_stats(text: str) -> dict:
    """Estadísticas completas de un texto generado."""
    tokens = text.split()
    return {
        "n_tokens": len(tokens),
        "rep_1":  rep_n(text, 1),
        "rep_5":  rep_n(text, 5),
        "rep_10": rep_n(text, 10),
        "ttr":    ttr(text),
        "rttr":   rttr(text),
    }


def _safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ─── Perplexity ───────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(
    model,
    dataset: TokenDataset,
    n_batches: int = 100,
    batch_size: int = 4,
    device: str = "cuda",
    is_swarm: bool = True,
    ctx=None,
) -> dict:
    """
    Calcula perplexidad media en un dataset.
    Retorna mean_loss, perplexity, std_loss.
    """
    if ctx is None:
        from contextlib import nullcontext
        ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    model.eval()
    losses = []
    np.random.seed(42)

    for i in range(n_batches):
        x, y = dataset.get_batch(batch_size, device)
        if x is None:
            break
        with ctx:
            if is_swarm:
                _, loss, _ = model(x, targets=y, store_memory=False)
            else:
                # HuggingFace model
                out = model(x, labels=y)
                loss = out.loss
        if loss is not None:
            losses.append(loss.item())

        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{n_batches}] loss={sum(losses)/len(losses):.4f}", end="\r")

    print()
    if not losses:
        return {"mean_loss": None, "perplexity": None, "std_loss": None}

    mean_loss = sum(losses) / len(losses)
    return {
        "mean_loss": round(mean_loss, 4),
        "perplexity": round(math.exp(mean_loss), 2),
        "std_loss": round(float(np.std(losses)), 4),
        "n_batches": len(losses),
    }


# ─── Generación de muestras ───────────────────────────────────────────────────

@torch.no_grad()
def generate_sample(
    swarm: LixySwarm,
    enc,
    prompt: str,
    max_tokens: int = 100,
    device: str = "cuda",
    temperature: float = 0.8,
    top_p: float = 0.92,
    rep_penalty: float = 1.3,
) -> str:
    """Genera texto con el swarm y retorna solo la respuesta."""
    from contextlib import nullcontext
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    block_size = swarm.config.agent_configs[0].block_size

    with ctx:
        # Warm-up
        _, _, feromon = swarm(x, context_text=prompt[:80], store_memory=False)
        feromon = feromon.detach()

        # Generación
        generated = list(tokens)
        for _ in range(max_tokens):
            x_cond = torch.tensor(generated[-block_size:], dtype=torch.long, device=device).unsqueeze(0)
            logits, _, _ = swarm(x_cond, store_memory=False)
            next_tok = sample_token(
                logits, generated_ids=x_cond,
                temperature=temperature, top_k=50, top_p=top_p,
                repetition_penalty=rep_penalty, recent_penalty=rep_penalty * 2.5,
                recent_window=8,
            )
            tok_id = next_tok.item()
            if tok_id == enc._special_tokens.get('<|endoftext|>', -1):
                break
            generated.append(tok_id)

    # Decode safely
    response_tokens = generated[len(tokens):]
    result = []
    for t in response_tokens:
        try:
            result.append(enc.decode([t]))
        except Exception:
            result.append('?')
    return ''.join(result)


# ─── GPT-2 baseline ───────────────────────────────────────────────────────────

def load_gpt2_baseline(device: str = "cuda"):
    """Carga GPT-2 small de HuggingFace como baseline."""
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        print("  Cargando GPT-2 small (HuggingFace)...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  GPT-2: {n_params/1e6:.0f}M params")
        return model, tokenizer
    except ImportError:
        print("  transformers no instalado — saltando comparativa GPT-2")
        return None, None
    except Exception as e:
        print(f"  GPT-2 no disponible: {e}")
        return None, None


@torch.no_grad()
def gpt2_perplexity(model, tokenizer, dataset_path: Path, n_batches: int, device: str) -> dict:
    """Perplexity de GPT-2 en el mismo val set."""
    from contextlib import nullcontext
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    data = np.memmap(dataset_path, dtype=np.uint16, mode='r')
    block_size = 512
    losses = []
    np.random.seed(42)

    for i in range(n_batches):
        idx = np.random.randint(0, len(data) - block_size)
        chunk = data[idx:idx+block_size].astype(np.int64)
        x = torch.tensor(chunk[:-1]).unsqueeze(0).to(device)
        y = torch.tensor(chunk[1:]).unsqueeze(0).to(device)
        with ctx:
            out = model(input_ids=x, labels=y)
        losses.append(out.loss.item())
        if (i + 1) % 20 == 0:
            print(f"    GPT-2 [{i+1}/{n_batches}] loss={sum(losses)/len(losses):.4f}", end="\r")

    print()
    mean_loss = sum(losses) / len(losses)
    return {
        "mean_loss": round(mean_loss, 4),
        "perplexity": round(math.exp(mean_loss), 2),
        "std_loss": round(float(np.std(losses)), 4),
        "n_batches": len(losses),
    }


# ─── Carga del Swarm ──────────────────────────────────────────────────────────

def load_swarm_for_bench(checkpoint_path: str, device: str) -> tuple:
    """Carga LixySwarm para benchmarking."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ac = ckpt.get("agent_config", {})
    agent_cfg = AgentConfig(**{k: v for k, v in ac.items() if hasattr(AgentConfig, k)}) if ac else AgentConfig()
    agent_cfg.dropout = 0.0

    swarm_cfg = SwarmConfig(
        n_agents=3,
        feromon_dim=agent_cfg.feromon_dim,
        swarm_rounds=2,
        agent_configs=[
            AgentConfig(
                block_size=agent_cfg.block_size,
                vocab_size=agent_cfg.vocab_size,
                n_layer=agent_cfg.n_layer,
                n_head=agent_cfg.n_head,
                n_embd=agent_cfg.n_embd,
                dropout=0.0,
                bias=agent_cfg.bias,
                feromon_dim=agent_cfg.feromon_dim,
                identity_dim=agent_cfg.identity_dim,
                agent_id=i,
                n_agents=3,
            )
            for i in range(3)
        ],
        matriarca_config=MatriarcaConfig(
            memory_path=str(CHECKPOINT_DIR / "matriarca_memory.json"),
            checkpoint_path=str(CHECKPOINT_DIR / "matriarca.pt"),
        ),
    )

    swarm = LixySwarm(swarm_cfg, load_matriarca=True, agent_checkpoint=None)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    load_result = swarm.load_state_dict(state, strict=False)
    swarm.eval()
    swarm = swarm.to(device)

    meta = {
        "step": ckpt.get("step", "?"),
        "val_loss": ckpt.get("val_loss", "?"),
        "n_params": sum(p.numel() for p in swarm.parameters()),
        "checkpoint": Path(checkpoint_path).name,
        "checkpoint_path": str(checkpoint_path),
        "agent_config": ac,
        "swarm_config": ckpt.get("swarm_config", {}),
        "train_config": ckpt.get("train_config", {}),
        "load_compatibility": {
            "missing_count": len(load_result.missing_keys),
            "unexpected_count": len(load_result.unexpected_keys),
            "missing_keys_sample": list(load_result.missing_keys[:20]),
            "unexpected_keys_sample": list(load_result.unexpected_keys[:20]),
        },
    }
    return swarm, agent_cfg, meta


# ─── Organism health ──────────────────────────────────────────────────────────

def infer_training_val_dataset(train_config: dict) -> dict:
    """Infiere el dataset de validación usado por train_swarm.py."""
    data_dir = Path(train_config.get("data_dir", "data"))
    personal_path = data_dir / "finetune" / "personal_tokens.bin"
    fineweb_val = data_dir / "pretrain" / "fineweb_val.bin"

    if train_config.get("mixed"):
        mode = "mixed"
        path = fineweb_val if fineweb_val.exists() else personal_path
    elif train_config.get("triple"):
        mode = "triple"
        path = fineweb_val if fineweb_val.exists() else personal_path
    elif train_config.get("spanish"):
        mode = "spanish"
        path = data_dir / "spanish" / "wiki_es_tokens.bin"
    else:
        mode = "personal"
        path = personal_path

    return {
        "mode": mode,
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def evaluate_training_val_loss(
    swarm: LixySwarm,
    agent_cfg: AgentConfig,
    meta: dict,
    n_batches: int,
    device: str,
    ctx,
) -> dict:
    """Recalcula la pérdida en el dominio de validación real del checkpoint."""
    val_info = infer_training_val_dataset(meta.get("train_config", {}))
    result = {
        "dataset": val_info,
        "saved_val_loss": _safe_float(meta.get("val_loss")),
        "recomputed": None,
        "loss_delta": None,
        "loss_ratio": None,
        "status": "missing_dataset",
    }
    if not val_info["exists"]:
        return result

    dataset = TokenDataset(Path(val_info["path"]), agent_cfg.block_size)
    batch_size = int(meta.get("train_config", {}).get("batch_size", 4) or 4)
    print(f"\n🫀 Health — validación real del checkpoint ({val_info['mode']}, {n_batches} batches)...")
    recomputed = compute_perplexity(
        swarm,
        dataset,
        n_batches=n_batches,
        batch_size=batch_size,
        device=device,
        is_swarm=True,
        ctx=ctx,
    )
    result["recomputed"] = recomputed

    saved = result["saved_val_loss"]
    current = _safe_float(recomputed.get("mean_loss"))
    if saved is not None and current is not None:
        result["loss_delta"] = round(current - saved, 4)
        result["loss_ratio"] = round(current / max(saved, 1e-9), 3)
        if result["loss_ratio"] <= 1.15:
            result["status"] = "stable"
        elif result["loss_ratio"] <= 1.6:
            result["status"] = "watch"
        else:
            result["status"] = "attention"
    elif current is not None:
        result["status"] = "measured"

    print(
        f"  saved={saved} | actual={recomputed.get('mean_loss')} | "
        f"ratio={result.get('loss_ratio')} | status={result['status']}"
    )
    return result


def evaluate_matriarca_health(swarm: LixySwarm) -> dict | None:
    """Evalúa Matriarca sin alterar el runtime del modelo."""
    try:
        sys.path.insert(0, str(SRC_DIR))
        from train_matriarca import matriarca_eval
        return matriarca_eval(matriarca=swarm.matriarca, verbose=False)
    except Exception as e:
        print(f"  ⚠ matriarca_eval: {e}")
        return None


def build_organism_health(results: dict, meta: dict, training_val_eval: dict | None) -> dict:
    """Construye señales de salud del organismo sin modificar arquitectura."""
    compatibility = meta.get("load_compatibility", {})
    missing = compatibility.get("missing_count", 0)
    unexpected = compatibility.get("unexpected_count", 0)
    compatibility_status = "ok" if missing == 0 and unexpected == 0 else "attention"

    fineweb = results.get("fineweb_val") or {}
    training_dataset = (training_val_eval or {}).get("dataset", {})
    benchmark_domain = {
        "fineweb_path": str(DATA_DIR / "pretrain" / "fineweb_val.bin"),
        "training_val_path": training_dataset.get("path"),
        "same_as_training_val": training_dataset.get("path") == str(DATA_DIR / "pretrain" / "fineweb_val.bin"),
        "status": "aligned" if training_dataset.get("path") == str(DATA_DIR / "pretrain" / "fineweb_val.bin") else "domain_shift",
    }

    matriarca = results.get("matriarca_eval")
    mat_signal = None
    if matriarca:
        active_pct = matriarca.get("importance", {}).get("pct_active")
        diversity = matriarca.get("diversity", {}).get("semantic_diversity")
        mat_signal = {
            "memory_count": matriarca.get("total_memories"),
            "active_pct": active_pct,
            "diversity": diversity,
            "types": matriarca.get("types", {}),
            "status": "watch" if active_pct is not None and active_pct < 0.10 else "ok",
        }

    generation = results.get("generation_summary")
    gen_signal = None
    if generation:
        gen_signal = {
            "avg_rep_5": generation.get("avg_rep_5"),
            "avg_rep_10": generation.get("avg_rep_10"),
            "avg_ttr": generation.get("avg_ttr"),
            "lang_correct": generation.get("lang_correct"),
            "status": (
                "attention"
                if generation.get("avg_rep_5", 0) >= 0.30
                else "watch"
                if generation.get("lang_correct", 1.0) < 0.60
                else "ok"
            ),
        }

    recommendations = []
    if compatibility_status != "ok":
        recommendations.append("Reentrenar o migrar checkpoint: hay capas sin match exacto.")
    if training_val_eval and training_val_eval.get("status") in {"watch", "attention"}:
        recommendations.append("Hacer ciclo corto de adaptación y repetir health contra el dominio real del checkpoint.")
    if benchmark_domain["status"] == "domain_shift" and fineweb:
        recommendations.append("Separar métricas por dominio: FineWeb mide generalización, no salud del checkpoint personal.")
    if mat_signal and mat_signal["status"] == "watch":
        recommendations.append("Revisar activación de Matriarca: muchas memorias, pocas activas.")
    if gen_signal and gen_signal["status"] == "watch":
        recommendations.append("Medir bilingüismo por dominio; el organismo está priorizando español/personalidad.")

    statuses = [
        compatibility_status,
        (training_val_eval or {}).get("status"),
        mat_signal.get("status") if mat_signal else None,
        gen_signal.get("status") if gen_signal else None,
    ]
    if "attention" in statuses:
        overall = "attention"
    elif "watch" in statuses:
        overall = "watch"
    else:
        overall = "stable"

    return {
        "overall": overall,
        "signals": {
            "checkpoint_compatibility": {
                **compatibility,
                "status": compatibility_status,
            },
            "training_val_alignment": training_val_eval,
            "benchmark_domain": benchmark_domain,
            "matriarca": mat_signal,
            "generation": gen_signal,
        },
        "recommendations": recommendations,
        "note": "Observability only: no cambia pesos, arquitectura ni memoria.",
    }


def print_organism_health(health: dict):
    def pct(value):
        return f"{value:.0%}" if isinstance(value, (int, float)) else "?"

    def num(value, digits=3):
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "?"

    print(f"\n🫀 ORGANISM HEALTH — {health['overall'].upper()}")
    signals = health.get("signals", {})
    compat = signals.get("checkpoint_compatibility", {})
    print(
        f"  Checkpoint: {compat.get('status')} | "
        f"missing={compat.get('missing_count')} unexpected={compat.get('unexpected_count')}"
    )

    train_val = signals.get("training_val_alignment") or {}
    dataset = train_val.get("dataset", {})
    if train_val:
        print(
            f"  Dominio entrenado: {dataset.get('mode')} | "
            f"saved={train_val.get('saved_val_loss')} actual={train_val.get('recomputed', {}).get('mean_loss')} | "
            f"ratio={train_val.get('loss_ratio')} status={train_val.get('status')}"
        )

    domain = signals.get("benchmark_domain", {})
    if domain.get("status") == "domain_shift":
        print("  FineWeb: domain_shift frente al dominio real del checkpoint")

    matriarca = signals.get("matriarca")
    if matriarca:
        print(
            f"  Matriarca: mem={matriarca.get('memory_count')} "
            f"activas={pct(matriarca.get('active_pct'))} "
            f"div={num(matriarca.get('diversity'))} status={matriarca.get('status')}"
        )

    generation = signals.get("generation")
    if generation:
        print(
            f"  Generación: rep@5={pct(generation.get('avg_rep_5'))} "
            f"lang={pct(generation.get('lang_correct'))} status={generation.get('status')}"
        )

    if health.get("recommendations"):
        print("  Recomendaciones:")
        for rec in health["recommendations"]:
            print(f"    - {rec}")


def save_results(results: dict, timestamp: str) -> tuple[Path, Path]:
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    out_path = EXPERIMENTS_DIR / f"benchmark_{timestamp}.json"
    ckpt_out = CHECKPOINT_DIR / f"benchmark_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    import shutil
    shutil.copy(out_path, ckpt_out)
    return out_path, ckpt_out


# ─── Main benchmark ───────────────────────────────────────────────────────────

def run_benchmark(args) -> dict:
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    enc = get_gpt2_encoding()
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else None
    if ctx is None:
        from contextlib import nullcontext
        ctx = nullcontext()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_batches = args.batches

    results = {
        "timestamp": timestamp,
        "device": device,
        "n_batches": n_batches,
        "health_batches": args.health_batches,
    }

    # ─── 1. Cargar swarm ──────────────────────────────────────────────────────
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        for name in ["swarm_best.pt", "swarm_latest.pt", "swarm_final.pt"]:
            p = CHECKPOINT_DIR / name
            if p.exists():
                ckpt_path = str(p)
                break

    print(f"\n🐜🐘🐬 Benchmark — {timestamp}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Device: {device} | Batches: {n_batches}")
    print()

    print("📦 Cargando LixySwarm...")
    swarm, agent_cfg, meta = load_swarm_for_bench(ckpt_path, device)
    results["model"] = meta
    print(f"  step={meta['step']} | val_loss={meta['val_loss']:.4f} | params={meta['n_params']/1e6:.0f}M")
    compat = meta.get("load_compatibility", {})
    print(
        f"  compatibilidad: missing={compat.get('missing_count', 0)} | "
        f"unexpected={compat.get('unexpected_count', 0)}"
    )

    if args.health_only:
        print(f"\n🐘 Diagnóstico Matriarca...")
        mat_metrics = evaluate_matriarca_health(swarm)
        results["matriarca_eval"] = mat_metrics
        if mat_metrics:
            m = mat_metrics
            print(f"  Memorias: {m['total_memories']} | "
                  f"activas: {m['importance']['pct_active']:.0%} | "
                  f"diversidad: {m['diversity']['semantic_diversity']:.3f}")
            print(f"  Tipos: {m['types']}")

        training_val_eval = evaluate_training_val_loss(
            swarm, agent_cfg, meta, args.health_batches, device, ctx
        )
        results["training_val_eval"] = training_val_eval
        results["organism_health"] = build_organism_health(results, meta, training_val_eval)
        print_organism_health(results["organism_health"])
        out_path, ckpt_out = save_results(results, timestamp)
        print(f"\n💾 Resultados: {out_path}")
        print(f"💾 Copia en:   {ckpt_out}")
        return results

    # ─── 2. Perplexity FineWeb val ────────────────────────────────────────────
    fw_val = DATA_DIR / "fineweb_val.bin"
    if fw_val.exists():
        print(f"\n📊 Perplexity — FineWeb val ({n_batches} batches × 4)...")
        fw_dataset = TokenDataset(fw_val, agent_cfg.block_size)
        fw_ppl = compute_perplexity(swarm, fw_dataset, n_batches, batch_size=4,
                                    device=device, is_swarm=True, ctx=ctx)
        results["fineweb_val"] = fw_ppl
        print(f"  LixySwarm: loss={fw_ppl['mean_loss']} | perplexity={fw_ppl['perplexity']} ± {fw_ppl['std_loss']}")
    else:
        print(f"  ⚠ {fw_val} no encontrado")
        results["fineweb_val"] = None

    # ─── 3. Perplexity corpus personal ───────────────────────────────────────
    for personal_name in ["es_train.bin", "personal_train.bin", "wiki_personal.bin"]:
        personal_path = DATA_DIR / personal_name
        if personal_path.exists():
            print(f"\n📊 Perplexity — corpus personal ({personal_name}, {min(n_batches, 50)} batches)...")
            personal_dataset = TokenDataset(personal_path, agent_cfg.block_size)
            if len(personal_dataset) > 0:
                personal_ppl = compute_perplexity(swarm, personal_dataset, min(n_batches, 50),
                                                   batch_size=2, device=device, is_swarm=True, ctx=ctx)
                results["personal_corpus"] = {"file": personal_name, **personal_ppl}
                print(f"  loss={personal_ppl['mean_loss']} | perplexity={personal_ppl['perplexity']}")
                break

    # ─── 4. GPT-2 baseline ───────────────────────────────────────────────────
    if not args.no_gpt2 and fw_val.exists():
        print(f"\n🤖 GPT-2 small baseline ({min(n_batches, 50)} batches)...")
        gpt2_model, gpt2_tok = load_gpt2_baseline(device)
        if gpt2_model is not None:
            gpt2_ppl = gpt2_perplexity(gpt2_model, gpt2_tok, fw_val,
                                        min(n_batches, 50), device)
            results["gpt2_baseline"] = gpt2_ppl
            print(f"  GPT-2: loss={gpt2_ppl['mean_loss']} | perplexity={gpt2_ppl['perplexity']}")
            del gpt2_model  # liberar VRAM
            torch.cuda.empty_cache()

    # ─── 4b. Comparativa de checkpoints ────────────────────────────────────
    if not args.no_compare and fw_val.exists():
        print(f"\n🔄 Comparativa de checkpoints...")
        checkpoints_to_compare = []
        for name in ["swarm_best.pt", "swarm_latest.pt", "swarm_final.pt"]:
            p = CHECKPOINT_DIR / name
            if p.exists() and str(p) != str(ckpt_path):
                checkpoints_to_compare.append(str(p))

        ckpt_comparisons = {}
        for ckpt_cmp in checkpoints_to_compare[:2]:  # max 2 comparativas
            try:
                swarm_cmp, _, meta_cmp = load_swarm_for_bench(ckpt_cmp, device)
                fw_cmp = compute_perplexity(swarm_cmp, TokenDataset(fw_val, agent_cfg.block_size),
                                            min(n_batches, 30), batch_size=4, device=device,
                                            is_swarm=True, ctx=ctx)
                ckpt_comparisons[Path(ckpt_cmp).name] = {
                    "step": meta_cmp["step"],
                    "val_loss": meta_cmp["val_loss"],
                    **fw_cmp,
                }
                print(f"  {Path(ckpt_cmp).name}: step={meta_cmp['step']} "
                      f"ppl={fw_cmp['perplexity']}")
                del swarm_cmp
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ⚠ {Path(ckpt_cmp).name}: {e}")
        results["checkpoint_comparison"] = ckpt_comparisons

    # ─── 4c. Diagnóstico de la Matriarca ───────────────────────────────────
    print(f"\n🐘 Diagnóstico Matriarca...")
    mat_metrics = evaluate_matriarca_health(swarm)
    results["matriarca_eval"] = mat_metrics
    if mat_metrics:
        m = mat_metrics
        print(f"  Memorias: {m['total_memories']} | "
              f"activas: {m['importance']['pct_active']:.0%} | "
              f"diversidad: {m['diversity']['semantic_diversity']:.3f}")
        print(f"  Tipos: {m['types']}")

    # ─── 5. Muestras generadas + métricas de texto ───────────────────────────
    print(f"\n✍️  Generando {len(EVAL_PROMPTS)} muestras...")
    samples = []
    all_stats = []

    for tag, prompt in EVAL_PROMPTS:
        response = generate_sample(swarm, enc, prompt, max_tokens=100, device=device)
        stats = text_stats(response)
        # Detección simple de idioma: contar palabras españolas vs inglesas comunes
        resp_lower = response.lower()
        es_words = sum(1 for w in ['el','la','es','que','de','en','y','un','una','con','por','para','los','las']
                       if f' {w} ' in resp_lower or resp_lower.startswith(w+' '))
        en_words = sum(1 for w in ['the','is','of','and','to','a','in','that','for','it','was','with']
                       if f' {w} ' in resp_lower or resp_lower.startswith(w+' '))
        detected_lang = 'es' if es_words > en_words else ('en' if en_words > es_words else 'unknown')
        stats['detected_lang'] = detected_lang

        samples.append({
            "tag": tag,
            "prompt": prompt,
            "response": response[:200],
            **stats,
        })
        all_stats.append(stats)
        quality = "✅" if stats["rep_5"] < 0.3 else "⚠️"
        lang_ok = "✅" if (tag.startswith('es') == (detected_lang == 'es')) else "⚠️"
        print(f"  {quality} [{tag}] rep@5={stats['rep_5']:.0%} ttr={stats['ttr']:.2f} "
              f"lang={detected_lang}{lang_ok} | "
              f"{repr(response[:55])}")

    results["samples"] = samples

    # Agregados
    results["generation_summary"] = {
        "avg_rep_1":  round(sum(s["rep_1"]  for s in all_stats) / len(all_stats), 4),
        "avg_rep_5":  round(sum(s["rep_5"]  for s in all_stats) / len(all_stats), 4),
        "avg_rep_10": round(sum(s["rep_10"] for s in all_stats) / len(all_stats), 4),
        "avg_ttr":    round(sum(s["ttr"]    for s in all_stats) / len(all_stats), 4),
        "avg_rttr":   round(sum(s["rttr"]   for s in all_stats) / len(all_stats), 4),
        "avg_tokens": round(sum(s["n_tokens"] for s in all_stats) / len(all_stats), 1),
        "pct_low_rep": round(sum(1 for s in all_stats if s["rep_5"] < 0.3) / len(all_stats), 3),
        "lang_correct": round(sum(
            1 for s, (tag, _) in zip(all_stats, EVAL_PROMPTS)
            if (tag.startswith("es") == (s.get("detected_lang") == "es"))
        ) / len(all_stats), 3),
    }

    # ─── 5b. Health del organismo ───────────────────────────────────────────
    training_val_eval = evaluate_training_val_loss(
        swarm, agent_cfg, meta, args.health_batches, device, ctx
    )
    results["training_val_eval"] = training_val_eval
    results["organism_health"] = build_organism_health(results, meta, training_val_eval)
    print_organism_health(results["organism_health"])

    # ─── 6. Resumen comparativo ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"📊 RESUMEN — LixySwarm step={meta['step']}")
    print(f"{'='*60}")

    if results.get("fineweb_val"):
        ppl = results["fineweb_val"]["perplexity"]
        print(f"  Perplexity FineWeb:  {ppl:8.1f}")
        if results.get("gpt2_baseline"):
            g2ppl = results["gpt2_baseline"]["perplexity"]
            ratio = ppl / g2ppl
            print(f"  Perplexity GPT-2:    {g2ppl:8.1f}  (LixySwarm es {ratio:.1f}× peor/mejor)")

    if results.get("personal_corpus"):
        print(f"  Perplexity personal: {results['personal_corpus']['perplexity']:8.1f}")

    if results.get("checkpoint_comparison"):
        print(f"  Comparativa checkpoints:")
        current_ppl = results.get("fineweb_val", {}).get("perplexity", "?")
        print(f"    {Path(ckpt_path).name}: step={meta['step']} ppl={current_ppl} ← actual")
        for cname, cdata in results["checkpoint_comparison"].items():
            print(f"    {cname}: step={cdata['step']} ppl={cdata.get('perplexity','?')}")

    if results.get("matriarca_eval"):
        me = results["matriarca_eval"]
        print(f"  Matriarca:           {me['total_memories']} mem | "
              f"activas={me['importance']['pct_active']:.0%} | "
              f"diversidad={me['diversity']['semantic_diversity']:.3f}")

    gs = results["generation_summary"]
    print(f"  Rep@5 promedio:      {gs['avg_rep_5']:8.1%}  ({'✅ bajo' if gs['avg_rep_5'] < 0.3 else '⚠️ alto'})")
    print(f"  Rep@10 promedio:     {gs['avg_rep_10']:8.1%}")
    print(f"  TTR promedio:        {gs['avg_ttr']:8.3f}")
    print(f"  RTTR promedio:       {gs['avg_rttr']:8.3f}")
    print(f"  Tokens promedio:     {gs['avg_tokens']:8.1f}")
    print(f"  Muestras sin bucles: {gs['pct_low_rep']:8.0%}")
    print(f"  Idioma correcto:     {gs['lang_correct']:8.0%}")

    # ─── 7. Guardar resultados ────────────────────────────────────────────────
    out_path, ckpt_out = save_results(results, timestamp)
    print(f"\n💾 Resultados: {out_path}")
    print(f"💾 Copia en:   {ckpt_out}")

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark LixySwarm 🐜🐘🐬")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint .pt")
    parser.add_argument("--batches", type=int, default=100, help="Batches para perplexity")
    parser.add_argument("--no-gpt2", action="store_true", help="Skip GPT-2 baseline")
    parser.add_argument("--no-compare", action="store_true", help="Skip comparativa de checkpoints")
    parser.add_argument("--health-only", action="store_true", help="Solo sensores del organismo, sin benchmark completo")
    parser.add_argument("--health-batches", type=int, default=10, help="Batches para health del dominio entrenado")
    parser.add_argument("--cpu", action="store_true", help="Forzar CPU")
    args = parser.parse_args()

    t0 = time.time()
    results = run_benchmark(args)
    elapsed = time.time() - t0
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
