"""Run on two machines to verify public LSP data and receipt-backed work.

Uses synthetic messages and a tiny CPU handler, without loading a model or any
personal memory. Run both nodes with the same unique --test-id. Reports contain
interfaces and transport types, not operator IP addresses or private keys.
"""

import argparse
import hashlib
import ipaddress
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contribution import ContributionPolicy, ResourceGovernor, ResourceRequirements
from src.network.bootstrap import get_seed_endpoints
from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.work_protocol import ResultReceipt, WorkCoordinator


def route(host):
    try:
        public = ipaddress.ip_address(host).is_global
        entry = json.loads(subprocess.check_output(["ip", "-j", "route", "get", host], text=True))[0]
        interface = entry.get("dev", "")
        return {"public": public, "interface": interface,
                "vpn": interface.startswith(("tailscale", "tun", "wg"))}
    except (ValueError, OSError, subprocess.CalledProcessError, IndexError):
        return {"public": False, "route_unknown": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--test-id", required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    if not 5 <= args.duration <= 300:
        parser.error("duration must be between 5 and 300 seconds")
    state = args.state or Path(tempfile.mkdtemp(prefix="lixy-internet-probe-"))
    state.mkdir(parents=True, exist_ok=True)
    governor = ResourceGovernor(ContributionPolicy.for_mode("balanced"), storage_path=state)
    node = LSPNodeV3(
        LSPIdentity.generate(), host="0.0.0.0", port=args.port,
        seeds=get_seed_endpoints(), address_book_path=state / "peers.json",
        target_outbound=3, allow_private=False,
        resource_profile=governor.advertised_profile(available_work={"inference"}),
    )
    coordinator = WorkCoordinator(node, governor)
    observed = {}

    def calculate(payload, _work):
        if payload != {"test_id": args.test_id, "value": 41}:
            raise ValueError("Only this probe's synthetic job is accepted")
        return {"value": 42, "digest": hashlib.sha256(args.test_id.encode()).hexdigest()}

    coordinator.register_handler("inference.probe.v1", "inference", calculate)

    @node.on_gossip_delta_received
    def capture(value, sender):
        if value.get("test_id") == args.test_id and value.get("label") != args.label:
            observed[sender] = value.get("label")

    def report(event, **data):
        print(json.dumps({"event": event, "label": args.label, **data}), flush=True)

    jobs = []
    tried = set()
    try:
        node.start()
        report("started", node_id=node.identity.node_id_hex[:16], port=node.port)
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            node.broadcast_global_delta({"kind": "internet_probe", "test_id": args.test_id, "label": args.label})
            for peer_id in list(observed):
                if peer_id in tried or peer_id not in node.connected_node_ids:
                    continue
                tried.add(peer_id)
                try:
                    result = coordinator.submit(
                        "inference.probe.v1", {"test_id": args.test_id, "value": 41},
                        ResourceRequirements(kind="inference", cpu_slots=1, ram_gb=0.001),
                        peer_id=peer_id, timeout_s=5,
                    )
                    verified = bool(result.receipt) and ResultReceipt.from_dict(result.receipt).verify(
                        result, expected_worker_id=peer_id, expected_requester_id=node.identity.node_id_hex,
                    )
                    correct = result.output == {"value": 42, "digest": hashlib.sha256(args.test_id.encode()).hexdigest()}
                    jobs.append({"peer_label": observed[peer_id], "status": result.status,
                                 "receipt_verified": verified, "output_correct": correct})
                    report("work", **jobs[-1])
                except Exception as error:
                    jobs.append({"peer_label": observed[peer_id], "error": type(error).__name__})
                    report("work", **jobs[-1])
            time.sleep(2)
        peers = node.peers()
        paths = [{"transport": peer["transport"], "encrypted": peer["encrypted"],
                  "seed": peer.get("capabilities", {}).get("seed", False),
                  "route": route(peer.get("relay_host") or peer["host"])} for peer in peers]
        success = bool(observed) and any(job.get("receipt_verified") and job.get("output_correct") for job in jobs)
        result = {"label": args.label, "connected_peers": len(peers), "other_nodes": list(observed.values()),
                  "paths": paths, "work": jobs, "success": success,
                  "public_routes_only": bool(paths) and all(p["route"].get("public") and not p["route"].get("vpn") for p in paths)}
        (state / "report.json").write_text(json.dumps(result, indent=2))
        report("summary", **{key: value for key, value in result.items() if key != "label"})
        return 0 if success and result["public_routes_only"] else 1
    finally:
        coordinator.close()
        node.stop()


if __name__ == "__main__":
    raise SystemExit(main())
