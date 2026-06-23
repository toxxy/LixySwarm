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
  python3 lixy_orchestrator.py --prompt "Hello"

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
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from contextlib import nullcontext
from dataclasses import dataclass

log = logging.getLogger("lixy.orchestrator")

import torch
import torch.nn.functional as F

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from src.swarm.orchestrator import LixySwarm, SwarmConfig
from src.agents.agent_base import AgentConfig
from src.matriarca.matriarca import MatriarcaConfig
from src.swarm.runtime_session import RuntimeSession
from src.network import (
    ArtifactStore,
    GradientAggregator,
    GradientCandidate,
    SwarmNetwork,
    TrainingWorker,
    digest_file,
    validate_gradient_artifact,
)
from src.contribution import (
    ContributionPolicy,
    ResourceGovernor,
    ResourceRequirements,
)
from src.release import ReleaseRegistry, TrustPolicy
from src.utils.tokenizer import get_gpt2_encoding

CHECKPOINT_DIR = Path("checkpoints")
MAX_REMOTE_PROMPT_BYTES = 16 * 1024
MAX_REMOTE_OUTPUT_BYTES = 128 * 1024
MAX_REMOTE_TOKENS = 512


def _validate_remote_inference_payload(payload: dict) -> dict:
    """Validate the stable inference.generate.v1 network schema."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    allowed = {"prompt", "max_tokens", "temperature", "top_k"}
    if set(payload) - allowed:
        raise ValueError("unsupported inference field")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if len(prompt.encode("utf-8")) > MAX_REMOTE_PROMPT_BYTES:
        raise ValueError("prompt exceeds 16 KiB")
    max_tokens = int(payload.get("max_tokens", 200))
    temperature = float(payload.get("temperature", 0.7))
    top_k = int(payload.get("top_k", 50))
    if not 1 <= max_tokens <= MAX_REMOTE_TOKENS:
        raise ValueError("max_tokens is out of range")
    if not 0.05 <= temperature <= 2.0:
        raise ValueError("temperature is out of range")
    if not 1 <= top_k <= 1000:
        raise ValueError("top_k is out of range")
    return {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
    }


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
    checkpoint_weights_only: bool = False
    release_id: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    temperature: float = 0.7
    top_k: int = 50
    max_tokens: int = 200
    session_file: str = "checkpoints/lixy_session.json"
    eval_matriarca: bool = False   # imprimir diagnóstico de la Matriarca al cargar
    verbose: bool = False
    # Red P2P (LSP v3, zero-config once public DNS seeds are configured)
    network: bool = True                          # siempre ON, P2P automático
    feromon_port: int = 7337
    gossip_port: int = 7338
    target_outbound: int = 8
    allow_private_peers: bool = False
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
        self.governor: ResourceGovernor = None
        self.artifact_store: ArtifactStore = None
        self.training_worker: TrainingWorker = None
        self.model_artifact_id: str = None
        self._runtime_session: RuntimeSession = None
        self._inference_lock = threading.RLock()
        self._load()

    def _load(self):
        """Carga el enjambre desde checkpoint."""
        ckpt_path = Path(self.cfg.checkpoint if self.cfg.checkpoint != "auto" else _find_best_checkpoint())

        print(f"🐜🐘🐬 LixyOrchestrator arrancando...")
        print(f"  Checkpoint: {ckpt_path.name}")

        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=self.cfg.checkpoint_weights_only,
        )
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
            contribution_home = Path(
                os.environ.get("LIXYSWARM_HOME", "~/.lixyswarm")
            ).expanduser()
            policy = ContributionPolicy.load(
                contribution_home / "contribution.json"
            )
            self.governor = ResourceGovernor(
                policy,
                storage_path=contribution_home,
            )
            self.artifact_store = ArtifactStore(
                contribution_home / "artifacts",
                max_total_bytes=max(1, int(policy.max_disk_gb * 1024 ** 3)),
            )
            self.model_artifact_id, _ = digest_file(ckpt_path)
            available_work = {"artifact", "inference"}
            if policy.mode in {"balanced", "maximum"}:
                available_work.add("training")
            contribution_profile = self.governor.advertised_profile(
                available_work=available_work
            )
            if self.model_artifact_id:
                contribution_profile["models"] = [self.model_artifact_id]
            if self.cfg.release_id:
                contribution_profile["releases"] = [self.cfg.release_id]
            self.net = SwarmNetwork.create(
                swarm=self.swarm,
                mode="lan" if self.cfg.allow_private_peers else "auto",
                feromon_port=self.cfg.feromon_port,
                gossip_port=self.cfg.gossip_port,
                checkpoint_dir=str(contribution_home),
                protocol="v3",
                target_outbound=self.cfg.target_outbound,
                allow_private_peers=self.cfg.allow_private_peers,
                contribution_profile=contribution_profile,
            )

            @self.net.on_peer_connected
            def _peer_up(peer):
                log.info(f"Peer connected: {peer.node_id[:12]}...@{peer.host}")

            self.net.start()
            if self.net._lsp_v3_node is not None:
                self.net.enable_work(self.governor, max_workers=1)
                self.net.enable_artifacts(self.artifact_store)
                trust_path = contribution_home / "release_trust.json"
                if trust_path.is_file():
                    release_policy = TrustPolicy.load(trust_path)
                    self.net.enable_release_distribution(
                        ReleaseRegistry(contribution_home / "releases"),
                        release_policy,
                        self.artifact_store,
                        auto_activate=release_policy.auto_activate,
                    )
                self.net.register_work_handler(
                    "inference.generate.v1",
                    "inference",
                    self._handle_remote_inference,
                )
                if policy.mode in {"balanced", "maximum"}:
                    self.training_worker = TrainingWorker(
                        self.net.work_coordinator,
                        self.net.artifact_service,
                        self.swarm,
                        model_artifact_id=self.model_artifact_id,
                        device=self.cfg.device,
                        execution_lock=self._inference_lock,
                    )
            # Auto-bootstrap en background vía peers.json + seeds + peer exchange

    def _handle_remote_inference(self, payload: dict, _work) -> dict:
        """Run an isolated, non-persistent inference for a signed peer."""
        request = _validate_remote_inference_payload(payload)
        with self._inference_lock:
            session = RuntimeSession(
                self.swarm,
                self.enc,
                device=self.cfg.device,
                max_new_tokens=request["max_tokens"],
                temperature=request["temperature"],
                top_k=request["top_k"],
                session_file=None,
                verbose=False,
            )
            response = session.turn(
                request["prompt"],
                max_new_tokens=request["max_tokens"],
                temperature=request["temperature"],
                top_k=request["top_k"],
                store_memory=False,
                record_history=False,
                update_runtime_state=False,
                use_memory=False,
            )
        if len(response.encode("utf-8")) > MAX_REMOTE_OUTPUT_BYTES:
            raise ValueError("generated output exceeds 128 KiB")
        return {"text": response}

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
        with self._inference_lock:
            # Inyectar feromonas remotas ANTES del turn (si hay peers activos)
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

            response = self._runtime_session.turn(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                store_memory=store_memory,
                record_history=store_memory,
                update_runtime_state=store_memory,
                use_memory=store_memory,
            )

            # Broadcast only state from an explicitly stored local turn.
            if store_memory and self.net and self.net.is_distributed:
                cached = self._runtime_session._cached_feromon
                if cached is not None:
                    self.net.broadcast_feromon(cached.squeeze(0), agent_id=0)

        return response

    def generate_distributed(
        self,
        prompt: str,
        *,
        max_tokens: int = None,
        temperature: float = None,
        top_k: int = None,
        peer_id: str = None,
        timeout_s: float = 120.0,
    ) -> str:
        """Request inference from a consenting peer; never silently falls back."""
        if self.net is None or self.net.work_coordinator is None:
            raise RuntimeError("Distributed inference is not available")
        payload = _validate_remote_inference_payload({
            "prompt": prompt,
            "max_tokens": max_tokens if max_tokens is not None else self.cfg.max_tokens,
            "temperature": temperature if temperature is not None else self.cfg.temperature,
            "top_k": top_k if top_k is not None else self.cfg.top_k,
        })
        result = self.net.submit_work(
            "inference.generate.v1",
            payload,
            ResourceRequirements(
                kind="inference", cpu_slots=1, ram_gb=0.5
            ),
            peer_id=peer_id,
            timeout_s=timeout_s,
        )
        if result.status != "ok":
            raise RuntimeError(result.error or "distributed inference failed")
        text = result.output.get("text")
        if not isinstance(text, str):
            raise RuntimeError("peer returned an invalid inference result")
        return text

    def publish_artifact(
        self,
        path: str,
        *,
        kind: str = "other",
        media_type: str = "application/octet-stream",
    ):
        """Explicitly import a local file; source names are never advertised."""
        if self.artifact_store is None:
            raise RuntimeError("Artifact service is not available")
        return self.artifact_store.import_file(
            path, kind=kind, media_type=media_type
        )

    def fetch_artifact(
        self,
        artifact_id: str,
        *,
        peer_id: str,
        timeout_s: float = 120.0,
    ) -> Path:
        if self.net is None:
            raise RuntimeError("Artifact service is not available")
        return self.net.fetch_artifact(
            artifact_id, peer_id=peer_id, timeout_s=timeout_s
        )

    def compute_gradient_distributed(
        self,
        dataset_artifact_id: str,
        *,
        start_token: int = 0,
        token_count: int = 512,
        peer_id: str = None,
        timeout_s: float = 300.0,
        ram_gb: float = 2.0,
        disk_gb: float = 2.0,
    ) -> dict:
        """Compute and retrieve a gradient candidate; never apply it automatically."""
        if self.net is None or self.net.work_coordinator is None:
            raise RuntimeError("Distributed training is not available")
        if not self.model_artifact_id:
            raise RuntimeError("This node has no versioned training model")
        if self.artifact_store is None or not self.artifact_store.has(dataset_artifact_id):
            raise ValueError("Dataset artifact must be published locally first")
        requirements = ResourceRequirements(
            kind="training", cpu_slots=1,
            ram_gb=float(ram_gb), disk_gb=float(disk_gb),
        )
        selected_peer = peer_id or self.net.work_coordinator.select_peer(
            requirements, required_model_id=self.model_artifact_id
        )
        if not selected_peer:
            raise RuntimeError("No connected peer has the required training model")
        candidate, details = self._request_gradient_candidate(
            selected_peer,
            dataset_artifact_id=dataset_artifact_id,
            start_token=int(start_token),
            token_count=int(token_count),
            requirements=requirements,
            timeout_s=timeout_s,
        )
        return {
            "gradient_artifact_id": candidate.artifact_id,
            "path": str(candidate.path),
            **details,
            "applied": False,
        }

    def compute_gradient_quorum(
        self,
        dataset_artifact_id: str,
        *,
        start_token: int = 0,
        token_count: int = 512,
        quorum: int = 3,
        timeout_s: float = 300.0,
        ram_gb: float = 2.0,
        disk_gb: float = 2.0,
    ) -> dict:
        """Request distinct peers and produce an unapplied coordinate median."""
        if self.net is None or self.net.work_coordinator is None:
            raise RuntimeError("Distributed training is not available")
        if not self.model_artifact_id:
            raise RuntimeError("This node has no versioned training model")
        if self.artifact_store is None or not self.artifact_store.has(dataset_artifact_id):
            raise ValueError("Dataset artifact must be published locally first")
        quorum = int(quorum)
        if not 3 <= quorum <= 31:
            raise ValueError("quorum must be between 3 and 31")
        requirements = ResourceRequirements(
            kind="training", cpu_slots=1,
            ram_gb=float(ram_gb), disk_gb=float(disk_gb),
        )
        peers = self.net.work_coordinator.select_peers(
            requirements,
            required_model_id=self.model_artifact_id,
            limit=quorum,
        )
        if len(peers) != quorum:
            raise RuntimeError(f"Gradient quorum requires {quorum} eligible peers")
        candidates = []
        details = []
        with ThreadPoolExecutor(
            max_workers=quorum, thread_name_prefix="lixy-gradient-quorum"
        ) as executor:
            futures = {
                executor.submit(
                    self._request_gradient_candidate,
                    peer,
                    dataset_artifact_id=dataset_artifact_id,
                    start_token=int(start_token),
                    token_count=int(token_count),
                    requirements=requirements,
                    timeout_s=timeout_s,
                ): peer
                for peer in peers
            }
            for future in as_completed(futures):
                candidate, detail = future.result()
                candidates.append(candidate)
                details.append(detail)
        manifest, aggregation = GradientAggregator(self.artifact_store).aggregate(
            candidates,
            self.swarm,
            model_artifact_id=self.model_artifact_id,
            dataset_artifact_id=dataset_artifact_id,
            start_token=int(start_token),
            token_count=int(token_count),
        )
        return {
            "gradient_artifact_id": manifest.artifact_id,
            "path": str(self.artifact_store._object_path(manifest.artifact_id)),
            "aggregation": aggregation,
            "candidate_metrics": details,
            "applied": False,
        }

    def _request_gradient_candidate(
        self,
        selected_peer: str,
        *,
        dataset_artifact_id: str,
        start_token: int,
        token_count: int,
        requirements: ResourceRequirements,
        timeout_s: float,
    ) -> tuple[GradientCandidate, dict]:
        result = self.net.submit_work(
            "training.gradient.v1",
            {
                "model_artifact_id": self.model_artifact_id,
                "dataset_artifact_id": dataset_artifact_id,
                "start_token": start_token,
                "token_count": token_count,
            },
            requirements,
            peer_id=selected_peer,
            timeout_s=timeout_s,
        )
        if result.status != "ok":
            raise RuntimeError(result.error or "distributed training failed")
        gradient = result.output.get("gradient", {})
        gradient_id = gradient.get("artifact_id")
        if not isinstance(gradient_id, str) or len(gradient_id) != 64:
            raise RuntimeError("peer returned an invalid gradient manifest")
        if result.output.get("model_artifact_id") != self.model_artifact_id:
            raise RuntimeError("peer returned a gradient for another model")
        if result.output.get("dataset_artifact_id") != dataset_artifact_id:
            raise RuntimeError("peer returned a gradient for another dataset")
        path = self.net.fetch_artifact(
            gradient_id,
            peer_id=selected_peer,
            timeout_s=timeout_s,
        )
        validation = validate_gradient_artifact(
            path,
            self.swarm,
            expected_model_id=self.model_artifact_id,
            expected_dataset_id=dataset_artifact_id,
            expected_start_token=start_token,
            expected_token_count=token_count,
        )
        return GradientCandidate(
            artifact_id=gradient_id,
            peer_id=selected_peer,
            path=path,
            receipt=result.receipt,
        ), {
            "loss": result.output.get("loss"),
            "gradient_norm": result.output.get("gradient_norm"),
            "validation": validation,
        }

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
        if self.net:
            self.net.stop()
        if self._runtime_session:
            self._runtime_session.end_session(save_matriarca=True)
        self._save_session()

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
