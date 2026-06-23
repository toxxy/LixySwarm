"""Single-entry CLI for joining and contributing to LixySwarm."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

from src.contribution import ContributionPolicy, ResourceGovernor
from src.network import ArtifactStore, SwarmNetwork


def _home() -> Path:
    return Path(os.environ.get("LIXYSWARM_HOME", "~/.lixyswarm")).expanduser()


def _policy_path(home: Path) -> Path:
    return home / "contribution.json"


def _confirm_compute(mode: str) -> bool:
    if mode not in {"balanced", "maximum"}:
        return True
    print(
        f"Mode '{mode}' allows signed LixySwarm inference/training jobs to use "
        "this computer within the displayed limits."
    )
    try:
        answer = input("Enable compute contribution? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def initialize(args) -> int:
    home = Path(args.home).expanduser()
    policy = ContributionPolicy.for_mode(args.mode)
    if not args.yes and not _confirm_compute(args.mode):
        print("No policy was changed.")
        return 1
    policy.save(_policy_path(home))
    governor = ResourceGovernor(policy, storage_path=home)
    print(json.dumps(governor.advertised_profile(), indent=2, sort_keys=True))
    print(f"Contribution policy saved to {_policy_path(home)}")
    return 0


def show_status(args) -> int:
    home = Path(args.home).expanduser()
    policy = ContributionPolicy.load(_policy_path(home))
    governor = ResourceGovernor(policy, storage_path=home)
    peers_path = home / "peers_v3.json"
    known_peers = 0
    if peers_path.exists():
        try:
            known_peers = len(json.loads(peers_path.read_text()).get("peers", []))
        except Exception:
            pass
    print(json.dumps({
        "home": str(home),
        "profile": governor.advertised_profile(),
        "governor": governor.status(),
        "known_peers": known_peers,
        "bootstrap_configured": bool(os.environ.get("LIXYSWARM_BOOTSTRAP_SEEDS")),
    }, indent=2, sort_keys=True))
    return 0


def start_node(args) -> int:
    home = Path(args.home).expanduser()
    home.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        return start_model_node(args, home)
    policy = ContributionPolicy.load(_policy_path(home))
    governor = ResourceGovernor(policy, storage_path=home)
    profile = governor.advertised_profile(available_work={"artifact"})

    network = SwarmNetwork.create(
        mode="lan" if args.allow_private_peers else "auto",
        gossip_port=args.port,
        checkpoint_dir=str(home),
        protocol="v3",
        target_outbound=args.outbound_peers,
        allow_private_peers=args.allow_private_peers,
        contribution_profile=profile,
    )
    network.start()
    if network._lsp_v3_node is not None:
        network.enable_work(governor, max_workers=max(1, policy.max_cpu_jobs or 1))
        artifact_store = ArtifactStore(
            home / "artifacts",
            max_total_bytes=max(1, int(policy.max_disk_gb * 1024 ** 3)),
        )
        network.enable_artifacts(artifact_store)

    stopping = False

    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    if policy.mode == "relay" and not _policy_path(home).exists():
        print("No contribution policy found; running connectivity-only relay mode.")
        print("Run 'lixyswarm init --mode balanced' to contribute compute.")
    if not network.seeds and not (home / "peers_v3.json").exists():
        print("No bootstrap seed is configured and no saved peers exist.")
        print("Set LIXYSWARM_BOOTSTRAP_SEEDS until official DNS seeds ship.")

    print(
        f"LixySwarm node started: protocol=v3 mode={policy.mode} "
        f"port={network._lsp_v3_node.port if network._lsp_v3_node else args.port}"
    )
    try:
        while not stopping:
            if args.status_interval > 0:
                print(json.dumps({
                    "peers": network.peer_count,
                    "known": (
                        network._lsp_v3_node.address_book.count
                        if network._lsp_v3_node else 0
                    ),
                    "resources": governor.status(),
                }, sort_keys=True))
            time.sleep(max(1.0, args.status_interval))
    finally:
        network.stop()
    return 0


def start_model_node(args, home: Path) -> int:
    """Start the complete model runtime and contribute registered compute."""
    from lixy_orchestrator import LixyOrchestrator, OrchestratorConfig

    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_file():
        print(f"Checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2
    os.environ["LIXYSWARM_HOME"] = str(home)
    orchestrator = LixyOrchestrator(OrchestratorConfig(
        checkpoint=str(checkpoint),
        device="cpu" if args.cpu else "cuda" if _cuda_available() else "cpu",
        network=True,
        gossip_port=args.port,
        target_outbound=args.outbound_peers,
        allow_private_peers=args.allow_private_peers,
    ))
    stopping = False

    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(json.dumps({
        "protocol": "v3",
        "mode": orchestrator.governor.policy.mode if orchestrator.governor else "local",
        "model_artifact_id": orchestrator.model_artifact_id,
        "inference_handler": True,
        "training_handler": orchestrator.training_worker is not None,
    }, sort_keys=True))
    try:
        while not stopping:
            if args.status_interval > 0:
                print(json.dumps(orchestrator.status(), sort_keys=True, default=str))
            time.sleep(max(1.0, args.status_interval))
    finally:
        orchestrator.close()
    return 0


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def add_artifact(args) -> int:
    home = Path(args.home).expanduser()
    policy = ContributionPolicy.load(_policy_path(home))
    store = ArtifactStore(
        home / "artifacts",
        max_total_bytes=max(1, int(policy.max_disk_gb * 1024 ** 3)),
    )
    manifest = store.import_file(
        args.path, kind=args.kind, media_type=args.media_type
    )
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return 0


def list_artifacts(args) -> int:
    home = Path(args.home).expanduser()
    policy = ContributionPolicy.load(_policy_path(home))
    store = ArtifactStore(
        home / "artifacts",
        max_total_bytes=max(1, int(policy.max_disk_gb * 1024 ** 3)),
    )
    manifests = []
    for path in sorted(store.manifests.glob("*.json")):
        try:
            manifests.append(store.manifest(path.stem).to_dict())
        except Exception:
            continue
    print(json.dumps(manifests, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lixyswarm")
    parser.add_argument("--home", default=str(_home()))
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="configure contribution consent and limits")
    init_parser.add_argument(
        "--mode", choices=["disabled", "relay", "balanced", "maximum"],
        default="balanced",
    )
    init_parser.add_argument("--yes", action="store_true", help="accept the selected policy non-interactively")
    init_parser.set_defaults(handler=initialize)

    start_parser = subparsers.add_parser("start", help="join the persistent LSP v3 network")
    start_parser.add_argument("--port", type=int, default=7338)
    start_parser.add_argument("--outbound-peers", type=int, default=8)
    start_parser.add_argument("--allow-private-peers", action="store_true")
    start_parser.add_argument("--status-interval", type=float, default=30.0)
    start_parser.add_argument(
        "--checkpoint",
        help="load a trusted local checkpoint and contribute inference/training",
    )
    start_parser.add_argument("--cpu", action="store_true")
    start_parser.set_defaults(handler=start_node)

    status_parser = subparsers.add_parser("status", help="show local policy and peer cache status")
    status_parser.set_defaults(handler=show_status)

    artifact_add = subparsers.add_parser(
        "artifact-add", help="explicitly add a content-addressed artifact"
    )
    artifact_add.add_argument("path")
    artifact_add.add_argument(
        "--kind",
        choices=["model", "dataset", "gradient", "evaluation", "other"],
        default="other",
    )
    artifact_add.add_argument(
        "--media-type", default="application/octet-stream"
    )
    artifact_add.set_defaults(handler=add_artifact)

    artifact_list = subparsers.add_parser(
        "artifact-list", help="list local artifact manifests without source paths"
    )
    artifact_list.set_defaults(handler=list_artifacts)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
