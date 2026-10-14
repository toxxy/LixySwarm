"""Reproducible public byte-language swarm for the first Internet learning gate.

This small reference proves the learning/transport loop. It is not the final
conversational model or a claim of general language quality.
"""
from __future__ import annotations
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

from src.agents.agent_base import AgentConfig
from src.agents.dolphin_agent import DolphinConfig
from src.swarm.orchestrator import LixySwarm, SwarmConfig

REFERENCE_CONFIG = {"family": "lixyswarm-byte-reference-v1", "vocab_size": 256,
                    "n_agents": 3, "n_embd": 32, "n_head": 4, "n_layer": 1,
                    "block_size": 64, "feromon_dim": 8, "swarm_rounds": 2}


def make_reference_model(config, device="cpu"):
    if config != REFERENCE_CONFIG:
        raise ValueError("Unsupported public reference model configuration")
    agents = [AgentConfig(vocab_size=256, block_size=64, n_layer=1, n_head=4,
                          n_embd=32, identity_dim=8, feromon_dim=8, agent_id=i, n_agents=3)
              for i in range(3)]
    config = SwarmConfig(n_agents=3, feromon_dim=8, echolocation_dim=8, swarm_rounds=2,
                         agent_configs=agents,
                         dolphin_config=DolphinConfig(vocab_size=256, n_embd=32,
                            feromon_dim=8, identity_dim=8, echo_dim=8, sleep_dim=8, dropout=0))
    return LixySwarm(config, load_matriarca=False, load_runtime_state=False).to(device)


def load_reference_checkpoint(path: Path, device="cpu"):
    if path.stat().st_size > 32 * 1024**2:
        raise ValueError("Reference checkpoint exceeds its size limit")
    with zipfile.ZipFile(path) as archive:
        if sum(info.file_size for info in archive.infolist()) > 64 * 1024**2:
            raise ValueError("Reference checkpoint exceeds its unpacked size limit")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != 1 or checkpoint.get("config") != REFERENCE_CONFIG:
        raise ValueError("Unsupported reference checkpoint")
    model = make_reference_model(checkpoint["config"], device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def reference_corpus():
    facts = [
        ("Una hormiga", "es un equipo que participa en la red de LixySwarm."),
        ("Un relay", "transporta conexiones cifradas y no decide qué modelo aprende la red."),
        ("El conocimiento global", "se conserva en versiones verificadas y replicadas entre participantes."),
        ("La memoria personal", "permanece protegida en el equipo de su usuario."),
        ("Una máquina potente", "puede realizar más trabajo y alojar agentes mayores."),
        ("Pausar el aporte", "detiene el cálculo autorizado y conserva la conectividad."),
        ("Una mejora candidata", "se evalúa antes de convertirse en una versión aceptada."),
        ("Los datos compartidos", "son públicos o sintéticos y tienen procedencia conocida."),
    ]
    training = "".join(f"Definición: {subject} {meaning}\n" for _ in range(20) for subject, meaning in facts)
    evaluation = "".join(f"En LixySwarm, {subject.lower()} {meaning}\n" for subject, meaning in facts)
    return np.frombuffer(training.encode("utf-8"), dtype=np.uint8).copy(), np.frombuffer(evaluation.encode("utf-8"), dtype=np.uint8).copy()


def evaluate_reference(model, tokens):
    was_training = model.training
    model.eval()
    total, predicted = 0.0, 0
    device = next(model.parameters()).device
    try:
        with torch.no_grad():
            for start in range(0, len(tokens) - 1, 64):
                batch = np.asarray(tokens[start:start + 65], dtype=np.int64)
                if len(batch) < 2:
                    continue
                x = torch.from_numpy(batch[:-1].copy()).unsqueeze(0).to(device)
                y = torch.from_numpy(batch[1:].copy()).unsqueeze(0).to(device)
                logits, _, _ = model(x, targets=y, store_memory=False, use_memory=False,
                                     update_runtime_state=False, update_memory_importance=False)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
                total += float(loss) * y.numel(); predicted += y.numel()
        return total / predicted
    finally:
        model.train(was_training)


def reference_logits(model, tokens):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.tensor(list(tokens[-64:]), dtype=torch.long, device=device).unsqueeze(0)
        logits, _, _ = model(x, store_memory=False, use_memory=False,
                             update_runtime_state=False, update_memory_importance=False)
    return logits[:, -1].detach().cpu().tolist()[0]
