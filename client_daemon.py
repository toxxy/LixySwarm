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
import time
from dataclasses import asdict
from pathlib import Path

from src.network.bootstrap import get_seed_endpoints
from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.telemetry import atomic_json, build_status, sign_status, verify_status
from src.network.client_state import ClientSettings, NodeStateLock, state_directory
from src.network.public_knowledge import KnowledgeSettings, PublicKnowledgeService

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


def node_command(directory: Path, *, host="0.0.0.0", port=7338, allow_private=False):
    command = [sys.executable]
    command += ["--node"] if getattr(sys, "frozen", False) else [str(Path(__file__).resolve())]
    command += ["--state-dir", str(directory.resolve()), "--host", host, "--port", str(port)]
    return command + (["--allow-private"] if allow_private else [])


def install_user_service(directory: Path, *, host="0.0.0.0", port=7338):
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
ExecStart={' '.join(quote(value) for value in node_command(directory, host=host, port=port))}
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
    path.write_text(unit)
    for command in (["systemctl", "--user", "daemon-reload"],
                    ["systemctl", "--user", "enable", "--now", "lixyswarm-node.service"]):
        subprocess.run(command, check=True)
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
            envelope = sign_status(identity, status)
            # Persist sequence before sending, so restart cannot reuse it.
            atomic_json(status_path, envelope)
            peers = node.peers()
            public_knowledge = knowledge.snapshot()
            atomic_json(directory / "diagnostics.json", {
                "node_id": identity.node_id_hex, "timestamp_ms": status["timestamp_ms"],
                "port": node.port, "peers": peers,
                "pid": os.getpid(), "runtime_version": "0.4.0a1", "controls_version": 1,
                "settings": asdict(settings), "worker_ready": False, "control_error": error,
                "knowledge": public_knowledge,
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
                current_state = "disconnected" if not running else "paused" if settings.paused else "connected" if node.peer_count else "connecting"
                public_state = knowledge.snapshot()
                knowledge_change = (public_state["state"], public_state["release_id"],
                                    public_state["versions_done"], public_state["sharing"])
                if running and current_state == "connected" and public_state["state"] == "downloading":
                    current_state = "preparing"
                if knowledge_change != last_knowledge:
                    refresh.set(); last_knowledge = knowledge_change
                if changed or refresh.is_set() or time.monotonic() >= next_publish:
                    refresh.clear()
                    eligible = publish(current_state, settings, error)
                    next_publish = time.monotonic() + interval if eligible else max(time.monotonic() + 0.1, last_sent + 2.1)
                    last_settings = asdict(settings)
                stop.wait(0.5)
            publish("disconnected", settings)
        finally:
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
