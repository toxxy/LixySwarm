"""Persistent lightweight Internet participant. Calculation workers come later.

Identity and runtime files are machine-local, outside Dropbox. No models,
personal memories or training files are read by this service.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from src.network.bootstrap import get_seed_endpoints
from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.telemetry import atomic_json, build_status, sign_status, verify_status

log = logging.getLogger("lixy.client")


def hardware_resources() -> dict:
    ram_gb = 0.0
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


def state_directory() -> Path:
    root = Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state")))
    return Path(os.environ.get("LIXYSWARM_STATE_DIR", str(root / "lixyswarm")))


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
ExecStart={quote(sys.executable)} {quote(Path(__file__).resolve())} --state-dir {quote(directory.resolve())} --host {quote(host)} --port {port}
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


def run(directory: Path, *, host="0.0.0.0", port=7338, interval=15):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    with (directory / "node.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A node already uses this state directory") from exc
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
            resource_profile={**resources, "mode": "network-only"},
        )
        stop = threading.Event()
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_: stop.set())
        started = time.monotonic()

        def publish(state):
            nonlocal sequence
            sequence += 1
            status = build_status(node, sequence=sequence, role="participant", state=state,
                                  uptime_s=time.monotonic() - started, resources=resources)
            envelope = sign_status(identity, status)
            # Persist sequence before sending, so restart cannot reuse it.
            atomic_json(status_path, envelope)
            peers = node.peers()
            atomic_json(directory / "diagnostics.json", {
                "node_id": identity.node_id_hex, "timestamp_ms": status["timestamp_ms"],
                "port": node.port, "peers": peers,
            })
            for peer in peers:
                if peer.get("capabilities", {}).get("status_collector_v1"):
                    node.send_node_status(peer["node_id"], envelope)
            log.info("state=%s peers=%s sequence=%s", state, len(peers), sequence)

        try:
            node.start()
            log.info("Started persistent node id=%s TCP=%s mode=network-only", identity.node_id_hex[:16], node.port)
            while not stop.is_set():
                publish("connected" if node.peer_count else "connecting")
                stop.wait(interval)
            publish("disconnected")
        finally:
            node.stop()
            log.info("Node stopped cleanly")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=state_directory())
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7338)
    parser.add_argument("--install-user-service", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.install_user_service:
        install_user_service(args.state_dir, host=args.host, port=args.port)
    else:
        run(args.state_dir, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
