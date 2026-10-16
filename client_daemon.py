"""Persistent lightweight Internet participant and public model provider.

Identity and runtime files are machine-local, outside Dropbox. Public artifacts
are verified and served without loading models, personal memory or tensors.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from src.network.bootstrap import get_seed_endpoints
from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.telemetry import atomic_json, build_status, sign_status, verify_status
from src.network.client_state import ClientSettings, NodeStateLock, state_directory
from src.network.public_knowledge import KnowledgeSettings, PublicKnowledgeService
from src.network.public_memory import PublicMemoryService
from src.network.reference_compute import ReferenceComputeService
from src.network.learning_coordinator import LearningCoordinator
from src.network.language_compute import LanguageComputeService
from src.network.chat_service import ChatBroker,LocalChatAPI
from src.network.language_families import LanguageFamilies
from src.network.language_training import LanguageTraining
from src.network.language_preparation import LanguagePreparation
from src.network.language_cache_repair import LanguageCacheRepair
from src.network.language_learning import LanguageLearningCoordinator
from src.network.local_language_work import LocalLanguageWork
from src.network.own_swarm import OwnSwarmService
from src.network.local_own_work import LocalOwnWork
from src.network.own_replication import OwnStateReplication
from src.network.own_learning import OwnLearningCoordinator
from src.network.own_network import OwnNetworkCoordinator
from src.network.own_preparation import OwnPreparation
from src.network.language_follow import LanguageVersionFollower
from src.version import VERSION

log = logging.getLogger("lixy.client")


def hardware_resources() -> dict:
    ram_gb = 0.0
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1024**3
    except ImportError:
        pass
    try:
        ram_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3
    except (AttributeError, ValueError, OSError):
        pass
    vram_gb = 0.0
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        vram_gb = max((float(line.strip()) / 1024 for line in result.stdout.splitlines()), default=0.0)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return {"cpu_cores": os.cpu_count() or 1, "ram_gb": round(ram_gb, 2),
            "gpu_vram_gb": round(vram_gb, 2), "has_gpu": vram_gb > 0}


def node_command(directory: Path, *, host="0.0.0.0", port=7338, allow_private=False, executable=None):
    command = [str(executable or sys.executable)]
    command += ["--node"] if getattr(sys, "frozen", False) else [str(Path(__file__).resolve())]
    command += ["--state-dir", str(directory.resolve()), "--host", host, "--port", str(port)]
    return command + (["--allow-private"] if allow_private else [])


def install_user_service(directory: Path, *, host="0.0.0.0", port=7338, executable=None, restart=False):
    """Linux reference installation; keep the service config machine-local."""
    if sys.platform != "linux":
        raise RuntimeError("The reference user service currently requires Linux")
    def quote(value):
        return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%') + '"'
    unit = f"""[Unit]
Description=LixySwarm Internet participant
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart={' '.join(quote(value) for value in node_command(directory, host=host, port=port, executable=executable))}
Restart=always
RestartSec=5
TimeoutStopSec=20
UMask=0077
NoNewPrivileges=true
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
"""
    config = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    path = config / "systemd" / "user" / "lixyswarm-node.service"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.lixyswarm-node-', delete=False) as stream:
        staging = Path(stream.name)
        stream.write(unit)
    try:
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)
    for command in (["systemctl", "--user", "daemon-reload"],
                    ["systemctl", "--user", "enable", "--now", "lixyswarm-node.service"]):
        subprocess.run(command, check=True)
    if restart:
        subprocess.run(["systemctl", "--user", "restart", "lixyswarm-node.service"], check=True)
    print(f"Installed {path}")


def run(directory: Path, *, host="0.0.0.0", port=7338, interval=15, allow_private=False):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    with NodeStateLock(directory):
        identity = LSPIdentity.load(str(directory / "identity.key"))
        if identity is None:
            identity = LSPIdentity.generate()
            identity.save(str(directory / "identity.key"))
        sequence = 0
        status_path = directory / "status.json"
        if status_path.exists():
            sequence = verify_status(json.loads(status_path.read_text()), sender_id=identity.node_id_hex)["sequence"]
        resources = hardware_resources()
        node = LSPNodeV3(
            identity, host=host, port=port, seeds=get_seed_endpoints(),
            address_book_path=directory / "peers_v3.json", target_outbound=4,
            capabilities={"peer_exchange": True, "pheromone": False, "global_memory": False,
                          "training": False, "inference": False, "worker_ready": False},
            resource_profile={**resources, "mode": "network-only"}, allow_private=allow_private,
        )
        knowledge = PublicKnowledgeService(node, directory, resources)
        compute = ReferenceComputeService(knowledge,directory,resources)
        language = LanguageComputeService(knowledge,directory,resources)
        families=LanguageFamilies(knowledge,directory)
        language_training=LanguageTraining(knowledge,families,directory,resources)
        own_swarm=OwnSwarmService(knowledge,directory,resources)
        own_work=LocalOwnWork(own_swarm)
        own_replication=OwnStateReplication(own_swarm)
        own_learning=OwnLearningCoordinator(own_swarm,own_work)
        own_network=OwnNetworkCoordinator(own_swarm,own_work)
        own_preparation=OwnPreparation(own_swarm)
        preparation=LanguagePreparation(node,knowledge,families,directory,resources)
        cache_repair=LanguageCacheRepair(node,knowledge,families,directory)
        public_memory=PublicMemoryService(knowledge,directory)
        chat=ChatBroker(node,knowledge.work,language,public_memory)
        learning = LearningCoordinator(knowledge,compute,directory)
        language_learning=LanguageLearningCoordinator(knowledge,language_training,directory)
        owner_work=LocalLanguageWork(knowledge,language_training,language_learning)
        chat_api=LocalChatAPI(chat,directory,preparation,owner_work,cache_repair,own_work,own_preparation);chat_api.start()
        language_follow=LanguageVersionFollower(knowledge,families,directory)
        stop = threading.Event()
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_: stop.set())
        started = time.monotonic()
        refresh = threading.Event()
        node.on_peer_connected(lambda *_: refresh.set())
        node.on_peer_lost(lambda *_: refresh.set())

        last_sent = 0.0

        def publish(state, settings, error=None):
            nonlocal sequence
            nonlocal last_sent
            sequence += 1
            status = build_status(node, sequence=sequence, role="participant", state=state,
                                  uptime_s=time.monotonic() - started, resources=resources)
            computation=compute.snapshot()
            conversation=language.snapshot()
            training=language_training.snapshot()
            own=own_swarm.snapshot()
            preparing=preparation.snapshot()
            own_preparing=own_preparation.snapshot()
            status["jobs"]={"active":computation["active"]+conversation["active"]+training["active"]+own['active']+int(preparing['state']=='testing')+int(own_preparing['state']=='validating'),"completed":computation["completed"]+conversation["completed"]+training["completed"]+own['completed']}
            envelope = sign_status(identity, status)
            # Persist sequence before sending, so restart cannot reuse it.
            atomic_json(status_path, envelope)
            peers = node.peers()
            public_knowledge = knowledge.snapshot()
            atomic_json(directory / "diagnostics.json", {
                "node_id": identity.node_id_hex, "timestamp_ms": status["timestamp_ms"],
                "port": node.port, "peers": peers,
                "pid": os.getpid(), "runtime_version": VERSION, "controls_version": 1,
                "settings": asdict(settings), "worker_ready": computation["ready"] or conversation["ready"] or training["ready"] or own['ready'], "control_error": error,
                "knowledge": public_knowledge,
                "compute":computation,
                "language":conversation,
                "language_families":families.snapshot(),
                "language_training":training,
                "own_swarm":own,
                "own_learning":own_learning.snapshot(),
                "own_network":own_network.snapshot(),
                "own_preparation":own_preparation.snapshot(),
                "language_preparation":preparing,
                "language_cache_repair":cache_repair.snapshot(),
                "chat":chat.snapshot(),
                "learning":learning.snapshot(),
                "language_learning":language_learning.snapshot(),
                "language_follow":language_follow.snapshot(),
                'public_memory':public_memory.snapshot(),
            })
            eligible = time.monotonic() - last_sent >= 2
            if eligible:
                for peer in peers:
                    if peer.get("capabilities", {}).get("status_collector_v1"):
                        node.send_node_status(peer["node_id"], envelope)
                last_sent = time.monotonic()
            log.info("state=%s peers=%s sequence=%s", state, len(peers), sequence)
            return eligible

        try:
            running = False
            next_publish = 0.0
            last_settings = None
            last_knowledge = None
            settings = ClientSettings()
            log.info("Started persistent node id=%s mode=network-only", identity.node_id_hex[:16])
            while not stop.is_set():
                error = None
                try:
                    settings = ClientSettings.load(directory)
                except (OSError, ValueError, TypeError) as exc:
                    error = str(exc)
                    settings = ClientSettings(connected=False, paused=True)
                changed = asdict(settings) != last_settings
                if changed:
                    mode = settings.policy.mode
                    node.resource_profile = {
                        **resources, "mode": "moderate" if mode == "balanced" else mode,
                        "cpu_percent": settings.policy.cpu_percent, "gpu_percent": settings.policy.gpu_percent,
                        "ram_budget_gb": settings.policy.max_ram_gb, "consent_version": settings.policy.version,
                        "worker_ready": False,
                    }
                if settings.connected and not running:
                    node.start()
                    running = True
                if not settings.connected and running:
                    publish("disconnected", settings, error)
                    node.stop()
                    running = False
                try:
                    sharing = KnowledgeSettings.load(directory)
                except (OSError, ValueError, TypeError):
                    sharing = KnowledgeSettings()
                knowledge.update(settings, sharing)
                public_memory.update(settings)
                compute.update(settings)
                language.update(settings)
                language_training.update(settings)
                own_swarm.update(settings)
                own_learning.update()
                own_network.update()
                own_preparation.update()
                cache_repair.update()
                preparation.update()
                chat.update(settings)
                learning.update()
                language_learning.update()
                language_follow.update()
                current_state = "disconnected" if not running else "paused" if settings.paused else "connected" if node.peer_count else "connecting"
                public_state = knowledge.snapshot()
                available={"artifact"} if public_state["sharing"] else set()
                if compute.ready or language.ready: available.add("inference")
                if compute.ready and compute.engine.training_requesters or language_training.ready or own_swarm.ready: available.add("training")
                node.resource_profile.update(knowledge.governor.advertised_profile(available_work=available))
                if "cuda" not in compute.engine.devices and not language.ready and not language_training.ready:
                    node.resource_profile.update(has_gpu=False,gpu_percent=0,gpu_vram_gb=0)
                node.resource_profile["work"] = {"connectivity":running, "artifact":public_state["sharing"],
                    "inference":compute.ready or language.ready, "training":compute.ready and bool(compute.engine.training_requesters) or language_training.ready or own_swarm.ready, "memory":False}
                node.resource_profile["worker_ready"]=compute.ready or language.ready or language_training.ready or own_swarm.ready
                node.resource_profile["model_artifact_id"]=public_state["model_id"] if compute.ready else language.engine.adapter_id if language.ready else ""
                language_models=language.approved_models()
                node.resource_profile["models"]=list(dict.fromkeys(([public_state["model_id"]] if compute.ready else [])+language_models+([language_training.snapshot()['model_id']] if language_training.ready else [])+([own_swarm.engine.model_id] if own_swarm.ready else [])))
                node.capabilities["inference"]=compute.ready or language.ready
                node.capabilities["language_inference_v1"]=language.ready
                node.capabilities["language_inference_models"]=language.approved_models(maximum_descriptor=3)
                node.capabilities["language_inference_models_v3"]=language.approved_models(maximum_descriptor=4)
                node.capabilities["language_inference_models_v4"]=language_models
                memory_state=public_memory.snapshot()
                node.capabilities['public_memory_v1']=memory_state['sharing']
                node.capabilities['public_memory_head']=memory_state['release_id'] if memory_state['sharing'] else ''
                node.resource_profile["language_families_v1"]=families.snapshot()["available"]
                node.capabilities["training"]=compute.ready and bool(compute.engine.training_requesters) or language_training.ready or own_swarm.ready
                node.capabilities['own_swarm_v1']=own_swarm.ready and own_swarm.engine.version==1
                node.capabilities['own_swarm_v2']=own_swarm.ready and own_swarm.engine.version>=2
                node.capabilities['own_swarm_v3']=own_swarm.ready and own_swarm.engine.version==3
                node.capabilities['own_swarm_state_id']=own_swarm.snapshot()['state_id'] if own_swarm.ready else ''
                node.capabilities["language_training_v1"]=language_training.ready and language_training.engine.base_name=='qwen3-0.6b'
                node.capabilities["language_training_v2"]=language_training.ready
                node.capabilities["language_candidate_v1"]=language_training.ready and language_training.engine.candidates_enabled
                node.capabilities["worker_ready"]=compute.ready or language.ready or language_training.ready or own_swarm.ready
                node.resource_profile["public_releases"] = [public_state["release_id"]] if public_state["sharing"] and public_state["release_id"] else []
                knowledge_change = (public_state["state"], public_state["release_id"],
                                    public_state["versions_done"], public_state["sharing"])
                if running and current_state == "connected" and public_state["state"] == "downloading":
                    current_state = "preparing"
                preparing=preparation.snapshot()['state']
                if running and not settings.paused and preparing not in {'idle','waiting','stopped','complete'}:
                    current_state='working' if preparing=='testing' else 'preparing'
                own_preparing=own_preparation.snapshot()['state']
                if running and not settings.paused and own_preparing not in {'idle','waiting','stopped','complete'}:
                    current_state='working' if own_preparing=='validating' else 'preparing'
                if running and not settings.paused and cache_repair.snapshot()['state'] in {'starting','receiving'}:
                    current_state='preparing'
                if running and not settings.paused and language_follow.settings.enabled and language_follow.snapshot()['state']=='acquiring':
                    current_state='preparing'
                if running and not settings.paused and (compute.snapshot()["active"] or language.snapshot()["active"] or language_training.snapshot()['active'] or own_swarm.snapshot()['active']):
                    current_state="working"
                if knowledge_change != last_knowledge:
                    refresh.set(); last_knowledge = knowledge_change
                if compute.changed.is_set():
                    compute.changed.clear(); refresh.set()
                if language.changed.is_set():
                    language.changed.clear();refresh.set()
                if language_training.changed.is_set():
                    language_training.changed.clear();refresh.set()
                if own_swarm.changed.is_set():
                    own_swarm.changed.clear();refresh.set()
                if own_learning.changed.is_set():
                    own_learning.changed.clear();refresh.set()
                if own_network.changed.is_set():
                    own_network.changed.clear();refresh.set()
                if own_preparation.changed.is_set():
                    own_preparation.changed.clear();refresh.set()
                if learning.changed.is_set():
                    learning.changed.clear(); refresh.set()
                if language_learning.changed.is_set():
                    language_learning.changed.clear();refresh.set()
                if language_follow.changed.is_set():
                    language_follow.changed.clear();refresh.set()
                if changed or refresh.is_set() or time.monotonic() >= next_publish:
                    refresh.clear()
                    eligible = publish(current_state, settings, error)
                    next_publish = time.monotonic() + interval if eligible else max(time.monotonic() + 0.1, last_sent + 2.1)
                    last_settings = asdict(settings)
                stop.wait(0.5)
            publish("disconnected", settings)
        finally:
            own_preparation.close()
            own_network.close()
            own_learning.close()
            cache_repair.close()
            preparation.close()
            chat_api.close()
            learning.close()
            language_learning.close()
            language_follow.close()
            public_memory.close()
            compute.ready=False
            language.ready=False
            language_training.ready=False
            own_swarm.ready=False
            knowledge.close()
            node.stop()
            log.info("Node stopped cleanly")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=state_directory())
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7338)
    parser.add_argument("--install-user-service", action="store_true")
    parser.add_argument("--allow-private", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--import-public-run", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    handlers = None
    if sys.stderr is None:
        args.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        handlers = [logging.FileHandler(args.state_dir / "client.log")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", handlers=handlers)
    if args.import_public_run:
        from src.network.public_knowledge import import_public_run
        with NodeStateLock(args.state_dir):
            release = import_public_run(args.import_public_run, args.state_dir)
            print(f"Imported public release {release.release_id}")
    elif args.install_user_service:
        install_user_service(args.state_dir, host=args.host, port=args.port)
    else:
        run(args.state_dir, host=args.host, port=args.port, allow_private=args.allow_private)


if __name__ == "__main__":
    main()
