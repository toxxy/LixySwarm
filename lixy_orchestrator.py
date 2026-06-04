"""
Lixy-0.1 — Orquestador Principal 🐜🐘🐬
=========================================
Punto de entrada unificado para interactuar con el LixySwarm v3.

Funciones:
- Cargar el enjambre entrenado (swarm_best.pt) con Matriarca activa
- Mantener contexto persistente entre sesiones (sueño unihemisférico)
- CLI interactiva y API programática
- Reportar estado de especialización de agentes

Uso:
  # CLI interactiva:
  python3 lixy_orchestrator.py

  # Una sola respuesta:
  python3 lixy_orchestrator.py --prompt "Hola, soy Emmanuel"

  # Con checkpoint específico:
  python3 lixy_orchestrator.py --checkpoint checkpoints/swarm_best.pt

  # API programática:
  from lixy_orchestrator import LixyOrchestrator
  lixy = LixyOrchestrator()
  response = lixy.chat("Hola amor")
"""

import sys
import argparse
import json
import os
import time
from pathlib import Path
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from src.swarm.orchestrator import LixySwarm, SwarmConfig
from src.agents.agent_base import AgentConfig
from src.matriarca.matriarca import MatriarcaConfig
from src.swarm.runtime_session import RuntimeSession
from src.network import SwarmNetwork
from src.utils.tokenizer import get_gpt2_encoding

CHECKPOINT_DIR = Path("checkpoints")


def _find_best_checkpoint() -> str:
    """
    Auto-detecta el mejor checkpoint disponible.
    Prioridad: swarm_best.pt > swarm_final.pt > swarm_latest.pt
    """
    for name in ["swarm_best.pt", "swarm_final.pt", "swarm_latest.pt"]:
        p = CHECKPOINT_DIR / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"No hay checkpoints del enjambre en {CHECKPOINT_DIR}")


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class OrchestratorConfig:
    checkpoint: str = "auto"       # "auto" = detectar mejor checkpoint
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    temperature: float = 0.7
    top_k: int = 50
    max_tokens: int = 200
    session_file: str = "checkpoints/lixy_session.json"
    eval_matriarca: bool = False   # imprimir diagnóstico de la Matriarca al cargar
    # Red P2P (LSP v2, zero-config — auto-bootstrap)
    network: bool = True                          # siempre ON, P2P automático
    feromon_port: int = 7337
    gossip_port: int = 7338
    network_remote_weight: float = 0.3


# ─── Orquestador ──────────────────────────────────────────────────────────────

class LixyOrchestrator:
    """
    Interfaz principal para interactuar con el LixySwarm v3.

    Mantiene estado persistente entre llamadas:
    - Sueño unihemisférico del Delfín (contexto conversacional)
    - Banco de memorias de la Matriarca (contexto de largo plazo)
    - Historial de la sesión
    """

    def __init__(self, cfg: OrchestratorConfig = None):
        self.cfg = cfg or OrchestratorConfig()
        self.enc = get_gpt2_encoding()
        self.ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.cfg.device == "cuda" else nullcontext()
        )
        self.swarm: LixySwarm = None
        self.agent_cfg: AgentConfig = None
        self.session_history = []
        self.net: SwarmNetwork = None
        self._runtime_session: RuntimeSession = None
        self._load()

    def _load(self):
        """Carga el enjambre desde checkpoint."""
        ckpt_path = Path(self.cfg.checkpoint if self.cfg.checkpoint != "auto" else _find_best_checkpoint())

        print(f"🐜🐘🐬 LixyOrchestrator arrancando...")
        print(f"  Checkpoint: {ckpt_path.name}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ac = ckpt.get("agent_config", {})
        self.agent_cfg = AgentConfig(**{k: v for k, v in ac.items() if hasattr(AgentConfig, k)}) if ac else AgentConfig()
        self.agent_cfg.dropout = 0.0

        swarm_cfg = SwarmConfig(
            n_agents=3,
            feromon_dim=self.agent_cfg.feromon_dim,
            swarm_rounds=2,
            agent_configs=[
                AgentConfig(
                    block_size=self.agent_cfg.block_size,
                    vocab_size=self.agent_cfg.vocab_size,
                    n_layer=self.agent_cfg.n_layer,
                    n_head=self.agent_cfg.n_head,
                    n_embd=self.agent_cfg.n_embd,
                    dropout=0.0,
                    bias=self.agent_cfg.bias,
                    feromon_dim=self.agent_cfg.feromon_dim,
                    identity_dim=self.agent_cfg.identity_dim,
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

        self.swarm = LixySwarm(swarm_cfg, load_matriarca=True, agent_checkpoint=None)
        state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
        missing, unexpected = self.swarm.load_state_dict(state, strict=False)
        dolphin_new = [k for k in missing if k.startswith("dolphin.")]
        other_missing = [k for k in missing if not k.startswith("dolphin.")]
        self.swarm.eval()
        self.swarm = self.swarm.to(self.cfg.device)

        print(f"  Step: {ckpt.get('step', '?')} | val_loss: {ckpt.get('val_loss', '?')}")
        print(f"  Parámetros: {sum(p.numel() for p in self.swarm.parameters())/1e6:.0f}M")
        print(f"  🐘 Matriarca: {self.swarm.matriarca.memory_count} memorias")
        if dolphin_new:
            print(f"  🐬 Delfín: {len(dolphin_new)} pesos nuevos (init random)")
        if other_missing:
            print(f"  ⚠ Keys faltantes (no Delfín): {other_missing[:3]}")

        # Cargar estado de sesión persistente
        self._load_session()
        print(f"  ✓ Listo ({len(self.session_history)} mensajes en historial)")

        # Diagnóstico de la Matriarca al arrancar (si está activado)
        if self.cfg.eval_matriarca and self.swarm.matriarca:
            try:
                import sys as _sys
                _sys.path.insert(0, str(Path(__file__).parent))
                from train_matriarca import matriarca_eval
                matriarca_eval(matriarca=self.swarm.matriarca, verbose=True)
            except Exception as e:
                print(f"  ⚠ matriarca_eval falló (no crítico): {e}")

        # Inicializar RuntimeSession — estado cross-turn con Matriarca activa
        self._runtime_session = RuntimeSession(
            self.swarm,
            self.enc,
            device=self.cfg.device,
            max_new_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            top_k=self.cfg.top_k,
            session_file=str(CHECKPOINT_DIR / "runtime_session.json"),
            verbose=False,
        )
        print(f"  🧠 RuntimeSession iniciada (feromon warm-start, roles dinámicos, historial persistente)")

        # Arrancar red P2P si se pidió (zero-config — auto-bootstrap)
        if self.cfg.network:
            self.net = SwarmNetwork.create(
                swarm=self.swarm,
                mode="auto",
                feromon_port=self.cfg.feromon_port,
                gossip_port=self.cfg.gossip_port,
            )

            @self.net.on_peer_connected
            def _peer_up(peer):
                print(f"\n🌐 ¡Peer conectado al enjambre! {peer}")

            self.net.start()
            # Auto-bootstrap en background vía peers.json + seeds + peer exchange

    def _load_session(self):
        """Carga historial de sesión y estado del Delfín si existe."""
        session_path = Path(self.cfg.session_file)
        if session_path.exists():
            try:
                with open(session_path) as f:
                    data = json.load(f)
                self.session_history = data.get("history", [])
                sleep_data = data.get("dolphin_sleep")
                if sleep_data and self.swarm:
                    sleep_state = self.swarm.dolphin.primary.dolphin.sleep_state
                    # Restaurar awake_state
                    awake = torch.tensor(sleep_data["awake_state"])
                    sleep_state.awake_state = awake.to(self.cfg.device)
                    # Restaurar context_buffer completo
                    ctx_buf = sleep_data.get("context_buffer", [])
                    from collections import deque
                    sleep_state.context_buffer = deque(
                        [torch.tensor(c).to(self.cfg.device) for c in ctx_buf],
                        maxlen=sleep_state.cfg.sleep_buffer_size,
                    )
                    print(f"  🌙 Sueño del Delfín restaurado (norm={awake.norm():.3f}, {len(ctx_buf)} contextos)")
            except Exception as e:
                print(f"  ⚠ No se pudo cargar sesión: {e}")

    def _save_session(self):
        """Guarda historial y estado del Delfín para la próxima sesión."""
        session_path = Path(self.cfg.session_file)
        sleep_state = self.swarm.dolphin.primary.dolphin.sleep_state
        sleep_norm = sleep_state.get_state().norm().item()
        # Guardar tanto awake_state como context_buffer
        context_buf = [c.cpu().tolist() for c in list(sleep_state.context_buffer)]
        data = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "history": self.session_history[-50:],
            "dolphin_sleep": {
                "awake_state": sleep_state.get_state().cpu().tolist(),
                "context_buffer": context_buf,
                "norm": sleep_norm,
            },
            "matriarca_memories": self.swarm.matriarca.memory_count if self.swarm.matriarca else 0,
        }
        with open(session_path, "w") as f:
            json.dump(data, f, ensure_ascii=False)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        top_k: int = None,
        store_memory: bool = True,
    ) -> str:
        """
        Genera una respuesta dado un prompt.
        Delega a RuntimeSession para:
        - Feromon warm-start cross-turn
        - Feedback post-generación a la Matriarca (con output completo)
        - Refresh de feromona con Matriarca activa cada 32 tokens
        - Merge de feromonas remotas desde peers P2P
        """
        # Inyectar feromonas remotas ANTES del turn (si hay peers activos)
        # Esto hace el loop bidireccional: transmitimos Y escuchamos
        if self.net and self.net.is_distributed:
            merged = self.net.merge_remote_feromons(
                local_feromon=self._runtime_session._cached_feromon
                    if self._runtime_session._cached_feromon is not None
                    else torch.zeros(self.swarm.config.feromon_dim,
                                     device=self.cfg.device),
                remote_weight=self.cfg.network_remote_weight,
            )
            if merged is not None:
                self._runtime_session.inject_remote_feromon(
                    merged, blend_weight=self.cfg.network_remote_weight
                )

        # Propagar overrides temporales a la RuntimeSession
        response = self._runtime_session.turn(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        # Broadcast feromona a la red P2P post-generación
        if self.net and self.net.is_distributed:
            cached = self._runtime_session._cached_feromon
            if cached is not None:
                self.net.broadcast_feromon(cached.squeeze(0), agent_id=0)

        return response

    def chat(self, message: str, store: bool = True) -> str:
        """
        Procesa un mensaje y retorna respuesta.
        Registra en historial y actualiza estado persistente.
        RuntimeSession maneja internamente el ciclo Matriarca.
        """
        response = self.generate(message, store_memory=store)

        if store:
            self.session_history.append({
                "role": "user",
                "text": message,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            self.session_history.append({
                "role": "assistant",
                "text": response,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            self._save_session()

        # ─── Tick del ciclo de vida (cada 10 turns para no frenar inferencia) ───
        if hasattr(self.swarm, 'ant_lifecycle') and self.swarm.ant_lifecycle:
            self._gen_count = getattr(self, '_gen_count', 0) + 1
            if self._gen_count % 10 == 0:
                n_nodes = (self.net.peer_count + 1) if (self.net and self.net.is_distributed) else 1
                # Diversidad actual del enjambre
                spec = self.swarm.specialization
                if spec and spec.current:
                    divs = [v.get('feromon_divergence', 0.6) for v in spec.current.values()]
                    swarm_div = sum(divs) / max(len(divs), 1)
                else:
                    swarm_div = 0.6
                events = self.swarm.tick_lifecycle(
                    step=self._gen_count,
                    swarm_diversity=swarm_div,
                    n_nodes=n_nodes,
                )
                if events and self.cfg.verbose:
                    for ev in events:
                        print(f"  🐜 lifecycle [{ev.get('type')}]: {ev}")

        return response

    def close(self):
        """
        Cierra el orquestador correctamente:
        - Llama end_session() en la RuntimeSession (penaliza memorias, guarda resumen)
        - Guarda historial de sesión y estado del Delfín
        - Detiene la red P2P si está activa
        """
        if self._runtime_session:
            self._runtime_session.end_session(save_matriarca=True)
        self._save_session()
        if self.net:
            self.net.stop()

    def status(self) -> dict:
        """Retorna estado actual del orquestador."""
        sleep_norm = self.swarm.dolphin.primary.dolphin.sleep_state.get_state().norm().item()
        net_status = self.net.status() if self.net else {"mode": "local", "peers": 0}
        rs = self._runtime_session

        return {
            "matriarca_memories": self.swarm.matriarca.memory_count if self.swarm.matriarca else 0,
            "dolphin_sleep_norm": round(sleep_norm, 4),
            "session_messages": len(self.session_history),
            "specialization_labels": {
                k: self.swarm.specialization._infer_label(k)
                for k in self.swarm.specialization.current
            } if self.swarm.specialization else {},
            "network": net_status,
            "runtime_session": {
                "turns": rs.stats.total_turns if rs else 0,
                "tokens": rs.stats.total_tokens if rs else 0,
                "memories_accessed": len(rs._accessed_memory_indices) if rs else 0,
                "feromon_warm": rs._cached_feromon is not None if rs else False,
            },
        }

    def print_status(self):
        """Imprime estado legible."""
        s = self.status()
        net = s["network"]
        print(f"\n{'='*50}")
        print(f"🐜🐘🐬 LixySwarm Status")
        print(f"  🐘 Matriarca: {s['matriarca_memories']} memorias")
        print(f"  🐬 Sueño Delfín: norm={s['dolphin_sleep_norm']:.4f}")
        print(f"  💬 Historial: {s['session_messages']} mensajes")
        print(f"  🐜 Especialización: {s['specialization_labels']}")
        print(f"  🧠 RuntimeSession: {s['runtime_session']['turns']} turnos | {s['runtime_session']['tokens']} tokens | feromon={'activo' if s['runtime_session']['feromon_warm'] else 'frío'}")
        if net['mode'] == 'local':
            print(f"  🌐 Red: local (single-node)")
        else:
            peers = net.get('peers', 0)
            node_id = net.get('node_id', '?')[:8]
            print(f"  🌐 Red: {net['mode']} | node={node_id}... | peers={peers}")
            if peers > 0:
                for p in net.get('peers_list', []):
                    print(f"     └─ {p['node_id'][:8]}@{p['host']}")
        print(f"{'='*50}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def interactive_cli(lixy: LixyOrchestrator):
    """Modo interactivo CLI."""
    print()
    print("🐜🐘🐬 Lixy-0.1 — CLI Interactiva")
    print("  Comandos: /status, /reset, /salir")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nTú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Hasta luego!")
            lixy.close()
            break

        if not user_input:
            continue

        if user_input == "/salir":
            print("👋 Hasta luego!")
            lixy.close()
            break
        elif user_input == "/status":
            lixy.print_status()
            continue
        elif user_input == "/reset":
            lixy.session_history = []
            lixy.swarm.dolphin.primary.dolphin.sleep_state.awake_state = torch.zeros(
                lixy.swarm.dolphin.primary.dolphin.sleep_state.cfg.sleep_dim,
                device=lixy.cfg.device
            )
            lixy._runtime_session.reset_feromon()   # reset feromon context too
            print("  ✓ Sesión reiniciada")
            continue

        print("Lixy: ", end="", flush=True)
        response = lixy.chat(user_input)
        print(response)

        # Mostrar stats del sueño y RuntimeSession
        sleep_norm = lixy.swarm.dolphin.primary.dolphin.sleep_state.get_state().norm().item()
        rs = lixy._runtime_session
        last_turn = rs.history[-1] if rs.history else None
        infra_str = f" | 🐘 infra={last_turn.infrasound_norm:.3f}" if last_turn else ""
        print(f"  [🐬 sleep={sleep_norm:.3f} | 🐘 {lixy.swarm.matriarca.memory_count} mem | 🧠 turno #{rs.stats.total_turns}{infra_str}]")


def main():
    parser = argparse.ArgumentParser(description="LixyOrchestrator 🐜🐘🐬")
    parser.add_argument("--checkpoint", default="checkpoints/swarm_best.pt")
    parser.add_argument("--prompt", type=str, default=None, help="Una sola respuesta")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--status", action="store_true", help="Solo mostrar status")
    parser.add_argument("--evolve", action="store_true", help="Correr loop evolutivo de la Matriarca y salir")
    parser.add_argument("--network", action="store_true", help="Activar red P2P LAN (mDNS)")
    parser.add_argument("--feromon-port", type=int, default=7337)
    parser.add_argument("--gossip-port", type=int, default=7338)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--eval-matriarca", action="store_true", help="Mostrar diagnóstico de la Matriarca al cargar")
    args = parser.parse_args()

    cfg = OrchestratorConfig(
        checkpoint=args.checkpoint,
        device="cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        network=args.network,
        feromon_port=args.feromon_port,
        gossip_port=args.gossip_port,
        eval_matriarca=args.eval_matriarca,
    )

    lixy = LixyOrchestrator(cfg)
    lixy.print_status()

    if args.status:
        return

    if args.evolve:
        # Loop evolutivo standalone: actualiza Matriarca desde los logs más recientes
        import glob, os
        logs = sorted(glob.glob("/tmp/swarm_*.log"), key=os.path.getmtime, reverse=True)
        if logs:
            log_path = logs[0]
            print(f"\n🐘 Loop evolutivo desde: {log_path}")
            from train_matriarca import train_from_swarm_log, MatriarcaTrainConfig
            mat_cfg = MatriarcaTrainConfig()
            before = lixy.swarm.matriarca.memory_count
            train_from_swarm_log(log_path, cfg=mat_cfg)
            after = lixy.swarm.matriarca.memory_count
            print(f"\n✅ Memorias: {before} → {after} (+{after-before})")
        else:
            print("⚠ No hay logs de training en /tmp/swarm_*.log")
        return

    if args.prompt:
        print(f"\n📝 Prompt: '{args.prompt}'")
        print("-" * 40)
        response = lixy.chat(args.prompt)
        print(f"Lixy: {response}")
        print("-" * 40)
    else:
        interactive_cli(lixy)


if __name__ == "__main__":
    main()
