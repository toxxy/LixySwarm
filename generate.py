"""
Lixy-0.1 — Script de Inferencia / Generación
=============================================
Carga el LixySwarm completo (Hormiga + Elefante + Delfín) y genera texto.
La Matriarca orienta al enjambre con sus infrasónidos en cada paso.

Uso:
  # Modo enjambre completo (recomendado):
  python3 generate.py --prompt "Hola, soy Lixy y"

  # Solo AgentBase (modo legacy):
  python3 generate.py --prompt "Hola" --legacy

  # Test de diagnóstico (verifica que Matriarca influye):
  python3 generate.py --diagnose

  # Interactivo:
  python3 generate.py --interactive
"""

import sys
import argparse
from pathlib import Path
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from src.agents.agent_base import AgentBase, AgentConfig
from src.swarm.orchestrator import LixySwarm, SwarmConfig
from src.swarm.runtime_session import RuntimeSession, interactive_session
from src.matriarca.matriarca import Matriarca, MatriarcaConfig
from src.utils.sampling import sample_token, SAMPLING_DEFAULTS
from src.utils.tokenizer import get_gpt2_encoding

CHECKPOINT_DIR = Path("checkpoints")


# ─── Carga del Enjambre Completo ─────────────────────────────────────────────

def load_swarm(
    agent_checkpoint: str = None,
    matriarca_checkpoint: str = None,
    device: str = "cuda",
) -> LixySwarm:
    """
    Carga el LixySwarm completo con agentes entrenados y Matriarca activa.
    """
    # Resolver paths de checkpoint
    if agent_checkpoint is None:
        for name in ["finetune_best.pt", "finetune_final.pt", "best.pt", "final.pt"]:
            p = CHECKPOINT_DIR / name
            if p.exists():
                agent_checkpoint = str(p)
                break

    if matriarca_checkpoint is None:
        p = CHECKPOINT_DIR / "matriarca.pt"
        if p.exists():
            matriarca_checkpoint = str(p)

    # Leer config del checkpoint del agente
    agent_cfg = AgentConfig()
    if agent_checkpoint and Path(agent_checkpoint).exists():
        ckpt = torch.load(agent_checkpoint, map_location="cpu", weights_only=False)
        # Swarm checkpoints use 'agent_config', legacy checkpoints use 'model_config'
        mc = ckpt.get("agent_config") or ckpt.get("model_config", {})
        if mc:
            valid_fields = {k: v for k, v in mc.items() if hasattr(AgentConfig, k)}
            agent_cfg = AgentConfig(**valid_fields)

    # Config del enjambre
    matriarca_cfg = MatriarcaConfig(
        memory_path=str(CHECKPOINT_DIR / "matriarca_memory.json"),
        checkpoint_path=matriarca_checkpoint or str(CHECKPOINT_DIR / "matriarca.pt"),
    )

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
        matriarca_config=matriarca_cfg,
    )

    swarm = LixySwarm(
        swarm_cfg,
        load_matriarca=True,
        agent_checkpoint=agent_checkpoint,
    )
    swarm.eval()
    swarm = swarm.to(device)

    print(f"  🐜 Agentes: {swarm_cfg.n_agents} × {agent_cfg.n_layer}L/{agent_cfg.n_head}H/{agent_cfg.n_embd}d")
    print(f"  🐘 Matriarca: {swarm.matriarca.memory_count if swarm.matriarca else 0} memorias")
    print(f"  🐬 Ecolocalización: activa")
    return swarm


# ─── Legacy: solo AgentBase ───────────────────────────────────────────────────

def load_model_legacy(checkpoint_path: str, device: str = "cuda"):
    """Carga solo el AgentBase (modo legacy)."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No encontré el checkpoint: {ckpt_path}")

    print(f"📦 Cargando (legacy): {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    mc = ckpt["model_config"]
    cfg = AgentConfig(**{k: v for k, v in mc.items() if hasattr(AgentConfig, k)})
    cfg.dropout = 0.0

    model = AgentBase(cfg)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print(f"   Step: {ckpt.get('step', '?')} | Val loss: {ckpt.get('val_loss', '?')}")
    return model, cfg


# ─── Generación con Enjambre ──────────────────────────────────────────────────

@torch.no_grad()
def generate_swarm(
    swarm: LixySwarm,
    enc,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.92,
    repetition_penalty: float = 1.3,
    device: str = "cuda",
    store_memory: bool = False,
) -> tuple[str, dict]:
    """
    Genera texto usando el LixySwarm completo.

    Estrategia estabilizada:
    - Paso 1 (warm-up): Delfín + Matriarca calculan feromon guiado desde el prompt completo.
    - Pasos siguientes: solo los agentes base usan ese feromon FIJO — sin re-calcular
      Delfín/Matriarca en cada token. Esto evita la acumulación de ruido que causa
      loops de repetición.

    Retorna (texto_generado, stats_de_infrasónidos).
    """
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    block_size = swarm.config.agent_configs[0].block_size

    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    # Stats para diagnóstico
    infrasound_norms = []
    feromon_norms = []
    cached_feromon = None

    with torch.no_grad(), ctx:
        for step_i in range(max_new_tokens):
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:]

            if cached_feromon is None:
                # ── Paso 1: warm-up completo con Delfín + Matriarca ──
                logits, _, feromon = swarm(
                    x_cond,
                    context_text=prompt[:100] if store_memory else "",
                    store_memory=store_memory,
                )
                cached_feromon = feromon.detach()  # fijar feromona guiada

                # Stats
                if swarm.matriarca is not None:
                    infra = swarm._get_infrasound(cached_feromon)
                    if infra is not None:
                        infrasound_norms.append(infra.norm().item())
                feromon_norms.append(cached_feromon.norm().item())
            else:
                # ── Pasos siguientes: solo agentes base con feromon fijo ──
                # Agrega logits de los 3 agentes ponderados por confianza
                all_logits = []
                all_conf = []
                n_embd = swarm.config.agent_configs[0].n_embd
                for agent, conf_head in zip(swarm.agents, swarm.confidence_heads):
                    ag_logits, _, _ = agent(x_cond, feromon_in=cached_feromon)
                    # conf_head espera [B, n_embd] — mismo hack que el orchestrator
                    rep = ag_logits.mean(dim=1)  # [B, vocab]
                    rep_proj = rep[:, :n_embd] if rep.shape[-1] >= n_embd else F.pad(rep, (0, n_embd - rep.shape[-1]))
                    conf = conf_head(rep_proj)  # [B, 1]
                    all_logits.append(ag_logits)
                    all_conf.append(conf)

                weights = F.softmax(torch.cat(all_conf, dim=-1), dim=-1)  # [B, n_agents]
                logits = sum(w.unsqueeze(-1).unsqueeze(-1) * l
                             for w, l in zip(weights.unbind(dim=-1), all_logits))

                # Refrescar feromona cada 32 tokens — ahora incluye Matriarca
                if step_i % 32 == 0:
                    _, _, new_feromon = swarm(
                        x_cond,
                        context_text=prompt[:100] if store_memory else "",
                        store_memory=False,
                    )
                    # Blend: 70% nuevo, 30% acumulado para estabilidad
                    cached_feromon = (0.7 * new_feromon.detach() + 0.3 * cached_feromon)
                    cached_feromon = F.normalize(cached_feromon, dim=-1)

            # Sampling — rep_penalty + top_k + top_p
            next_token = sample_token(
                logits,
                generated_ids=x,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                recent_penalty=repetition_penalty * 2.5,
                recent_window=8,
            )
            x = torch.cat((x, next_token), dim=1)

            if next_token.item() == enc._special_tokens.get("<|endoftext|>", -1):
                break

    stats = {
        "infrasound_norm": sum(infrasound_norms) / len(infrasound_norms) if infrasound_norms else 0.0,
        "feromon_norm": sum(feromon_norms) / len(feromon_norms) if feromon_norms else 0.0,
        "tokens_generated": x.size(1) - len(tokens),
    }

    # ── Feedback post-generación a la Matriarca 🐘 ──────────────────────────
    # Ahora que tenemos el output completo, almacenamos con contexto rico
    if store_memory and swarm.matriarca is not None and prompt:
        full_output_text = enc.decode(x[0].tolist())
        response_text = full_output_text[len(prompt):]
        mem_text = f"[gen] Q: {prompt[:80]} A: {response_text[:120]}"

        # Usar feromon final como embedding de la interacción
        embd_dim = swarm.config.matriarca_config.embd_dim
        state = cached_feromon.mean(dim=0) if cached_feromon.dim() > 1 else cached_feromon
        if state.shape[-1] != embd_dim:
            state = F.interpolate(
                state.float().unsqueeze(0).unsqueeze(0),
                size=embd_dim, mode='linear', align_corners=False,
            ).squeeze()
        with torch.no_grad():
            swarm.matriarca.store_interaction(
                state.to(device),
                text=mem_text,
                importance=min(1.0, 0.3 + 0.7 * min(1.0, stats['tokens_generated'] / 200)),
            )

    return enc.decode(x[0].tolist()), stats


# ─── Legacy generate ─────────────────────────────────────────────────────────

@torch.no_grad()
def generate_legacy(model, enc, prompt, max_new_tokens=200, temperature=0.8, top_k=50, device="cuda", block_size=512):
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    with ctx:
        for _ in range(max_new_tokens):
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
            logits, _, _ = model(x_cond)
            next_token = sample_token(
                logits, generated_ids=x,
                temperature=temperature, top_k=top_k, top_p=0.92, repetition_penalty=1.3,
            )
            x = torch.cat((x, next_token), dim=1)
            if next_token.item() == enc._special_tokens.get("<|endoftext|>", -1):
                break
    return enc.decode(x[0].tolist())


# ─── Test de Diagnóstico ──────────────────────────────────────────────────────

def diagnose(device: str = "cuda"):
    """
    Test end-to-end que verifica que la Matriarca realmente influye en el enjambre.
    Compara logits CON y SIN infrasónidos para medir el efecto real.
    """
    print("\n🔬 Diagnóstico End-to-End: Matriarca → Enjambre")
    print("=" * 60)

    enc = get_gpt2_encoding()

    print("\n📦 Cargando enjambre...")
    swarm = load_swarm(device=device)
    swarm.eval()

    prompt = "Hola, soy Lixy y"
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    print(f"\n📝 Prompt: '{prompt}' ({len(tokens)} tokens)\n")

    with torch.no_grad():
        # ─── Test 1: Flujo CON Matriarca ───
        print("Test 1: Flujo CON Matriarca activa")
        logits_con, _, feromon_con = swarm(x, context_text=prompt, store_memory=False)
        logits_con_last = logits_con[:, -1, :]

        # Obtener infrasónidos directamente para medir
        infrasound = swarm._get_infrasound(feromon_con)
        if infrasound is not None:
            print(f"  🐘 Infrasónidos emitidos:")
            print(f"     norm  = {infrasound.norm().item():.4f}")
            print(f"     mean  = {infrasound.mean().item():.4f}")
            print(f"     std   = {infrasound.std().item():.4f}")
            print(f"     shape = {infrasound.shape}")
        else:
            print("  ⚠️  Matriarca no emitió infrasónidos")

        print(f"  🐜 Feromona del enjambre:")
        print(f"     norm  = {feromon_con.norm().item():.4f}")
        print(f"     shape = {feromon_con.shape}")

        # ─── Test 2: Flujo SIN Matriarca (desactivada temporalmente) ───
        print("\nTest 2: Flujo SIN Matriarca (desactivada)")
        matriarca_backup = swarm.matriarca
        swarm.matriarca = None

        logits_sin, _, feromon_sin = swarm(x)
        logits_sin_last = logits_sin[:, -1, :]
        swarm.matriarca = matriarca_backup

        print(f"  🐜 Feromona sin Matriarca:")
        print(f"     norm  = {feromon_sin.norm().item():.4f}")

        # ─── Comparación ───
        print("\n📊 Comparación de logits (último token):")
        diff = (logits_con_last - logits_sin_last).abs()
        print(f"  Diferencia max    = {diff.max().item():.6f}")
        print(f"  Diferencia mean   = {diff.mean().item():.6f}")
        print(f"  Diferencia L2     = {diff.norm().item():.6f}")

        # Top-5 tokens distintos
        top5_con = torch.topk(logits_con_last[0], 5)
        top5_sin = torch.topk(logits_sin_last[0], 5)
        print(f"\n  Top-5 con Matriarca: {[enc.decode([t.item()]) for t in top5_con.indices]}")
        print(f"  Top-5 sin Matriarca: {[enc.decode([t.item()]) for t in top5_sin.indices]}")

        matriarca_influye = diff.mean().item() > 1e-6
        print(f"\n  {'✅ CONFIRMADO' if matriarca_influye else '⚠️ ATENCIÓN'}: Matriarca {'SÍ influye' if matriarca_influye else 'NO influye'} en las decisiones del enjambre")
        print(f"  Efecto medio por logit: {diff.mean().item():.6f}")

        # ─── Test 3: Generación completa ───
        print("\nTest 3: Generación completa con enjambre")
        swarm.matriarca = matriarca_backup
        output, stats = generate_swarm(
            swarm, enc, prompt,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50,
            device=device,
            store_memory=True,
        )
        print(f"  Input:  '{prompt}'")
        print(f"  Output: '{output}'")
        print(f"  Tokens generados: {stats['tokens_generated']}")
        print(f"  Infrasound norm: {stats['infrasound_norm']:.4f}")
        print(f"  Feromon norm: {stats['feromon_norm']:.4f}")

        if swarm.matriarca:
            print(f"  🐘 Memorias post-generación: {swarm.matriarca.memory_count}")

    print("\n✅ Diagnóstico completo")
    return matriarca_influye


# ─── Modo Interactivo ─────────────────────────────────────────────────────────

def interactive_mode(swarm, enc, device):
    # Usar RuntimeSession para estado persistente entre turnos
    interactive_session(swarm, enc, device=device)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generar texto con Lixy-0.1 🐜🐘🐬")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint del agente")
    parser.add_argument("--prompt", default="Hola amor, ")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--legacy", action="store_true", help="Usar solo AgentBase")
    parser.add_argument("--diagnose", action="store_true", help="Test diagnóstico end-to-end")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}\n")

    enc = get_gpt2_encoding()

    # ─── Modo diagnóstico ───
    if args.diagnose:
        diagnose(device=device)
        return

    # ─── Modo legacy (solo AgentBase) ───
    if args.legacy:
        ckpt = args.checkpoint or str(CHECKPOINT_DIR / "finetune_best.pt")
        model, cfg = load_model_legacy(ckpt, device)
        if args.interactive:
            # Interactive legacy
            print("\n🐜 Lixy-0.1 — Modo Legacy\n")
            while True:
                try:
                    prompt = input("Tú: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not prompt or prompt.lower() == "salir":
                    break
                out = generate_legacy(model, enc, prompt, args.max_tokens, args.temperature, args.top_k, device, cfg.block_size)
                print("Lixy:", out[len(prompt):], "\n")
        else:
            out = generate_legacy(model, enc, args.prompt, args.max_tokens, args.temperature, args.top_k, device, cfg.block_size)
            print(f"📝 '{args.prompt}'\n{'-'*40}\n{out}\n{'-'*40}")
        return

    # ─── Modo enjambre completo (default) ───
    print("🐜🐘🐬 Cargando LixySwarm completo...")
    swarm = load_swarm(agent_checkpoint=args.checkpoint, device=device)

    if args.interactive:
        interactive_mode(swarm, enc, device)
        return
    else:
        print(f"\n📝 Prompt: '{args.prompt}'")
        print("-" * 40)
        output, stats = generate_swarm(
            swarm, enc, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
            store_memory=True,
        )
        print(output)
        print("-" * 40)
        print(f"🐘 infrasound_norm={stats['infrasound_norm']:.4f} | 🐜 feromon_norm={stats['feromon_norm']:.4f} | tokens={stats['tokens_generated']}")


if __name__ == "__main__":
    main()
