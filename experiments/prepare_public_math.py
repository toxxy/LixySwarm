"""Stage a pinned public curriculum or acquire it over encrypted public LSP.

Uses current physical-node storage/network consent. A lightweight visitor is
only an administrative downloader, never an additional physical worker.
"""
import argparse
import hashlib
import ipaddress
import json
from pathlib import Path
import signal
import threading
import time
from types import SimpleNamespace

from src.contribution import ContributionPolicy, ResourceGovernor
from src.learning.public_math import (ARTIFACTS, build_snapshots, canonical, curriculum_id,
                                      gate_definition, gate_id, load_snapshot)
from src.network.artifact_store import ArtifactError, ArtifactStore
from src.network.bootstrap import get_seed_endpoints
from src.network.client_state import ClientSettings, NodeStateLock
from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.public_knowledge import KnowledgeSettings, BandwidthBudget, managed_public_bytes
from src.network.public_math_dataset import replicate_dataset, transfer_proof, verify_transfer_proofs
from src.network.report_privacy import public_route
from src.network.telemetry import atomic_json
from src.network.work_protocol import WorkCoordinator


def outside_checkout(path):
    return not any(path.resolve().is_relative_to(root) for root in [Path(__file__).resolve().parents[1], Path.home()/'Dropbox'])


def fingerprints(directory):
    names = ['controls.json', 'public-knowledge.json', 'language-engine.json', 'language-training.json',
             'language-follow-controls.json', 'language-learning-controls.json', 'language-learning/ledger.json']
    return {name: hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in names if (directory/name).is_file()}


def run(args):
    node_directory = args.node_directory.resolve();directory = args.directory.resolve()
    if not outside_checkout(node_directory) or not outside_checkout(directory) or directory.exists():
        raise ValueError('Use existing physical node controls and fresh state outside Dropbox')
    before = fingerprints(node_directory)
    diagnostic = json.loads((node_directory/'diagnostics.json').read_text())
    store_root = node_directory/'public/language/artifacts'
    if args.recovery_copy:
        if args.action != 'fetch':raise ValueError('Recovery copies are acquisition tests only')
        store_root = node_directory/'public/math-recovery'/directory.name/'artifacts'
        if store_root.exists():raise ValueError('Recovery store must be initially empty')
    store = ArtifactStore(store_root, max_artifact_bytes=4*1024**3, max_total_bytes=8*1024**3)
    cancel = threading.Event()
    for sig in [signal.SIGTERM, signal.SIGINT]:signal.signal(sig, lambda *_: cancel.set())
    def allowed():
        try:
            settings = ClientSettings.load(node_directory);sharing = KnowledgeSettings.load(node_directory)
            return (not cancel.is_set() and settings.connected and not settings.paused and sharing.enabled
                    and settings.policy.mode != 'disabled' and settings.policy.max_bandwidth_mbps > 0
                    and managed_public_bytes(node_directory) <= settings.policy.max_disk_gb*1024**3)
        except (OSError, ValueError, TypeError):return False
    def preflight(missing):
        if not allowed():raise ArtifactError('Public dataset storage or connectivity consent is disabled')
        if managed_public_bytes(node_directory)+2*missing+8*1024**2 > ClientSettings.load(node_directory).policy.max_disk_gb*1024**3:
            raise ValueError('Dataset preparation exceeds current physical storage consent')
    preflight(sum(size for identifier, size in ARTIFACTS.values() if not store.has(identifier)))
    directory.mkdir(mode=0o700, parents=True)
    report = {'success': False, 'action': args.action, 'physical_node_id': diagnostic['node_id'],
              'curriculum_id': curriculum_id(), 'gate_id': gate_id(), 'gate_definition': gate_definition(),
              'started_at': time.time(), 'private_inputs': False, 'model_promoted': False,
              'native_learning_window_changed': False, 'visitor_is_extra_physical_worker': False}
    atomic_json(directory/'report.json', report)
    if args.action == 'stage':
        if args.source_directory is None or not outside_checkout(args.source_directory):
            raise ValueError('Stage requires exact upstream files outside Dropbox')
        for split, snapshot in build_snapshots(args.source_directory).items():
            preflight(ARTIFACTS[split][1])
            path = directory/(split+'.json');path.write_bytes(canonical(snapshot));load_snapshot(path, split)
            value = store.import_file(path, kind='dataset', media_type='application/json')
            if not allowed():raise ArtifactError('Dataset storage permission withdrawn')
            assert (value.artifact_id, value.size) == ARTIFACTS[split]
        report.update(success=True, source_files_verified=True, source_origin='Pinned upstream public HTTPS acquisition',
                      artifacts=[{'split': split, 'artifact_id': identifier, 'bytes': size} for split, (identifier, size) in ARTIFACTS.items()],
                      training_started=False)
    else:
        if (not args.provider or len(args.provider) > 8 or len(set(args.provider)) != len(args.provider)
                or any(len(p) != 64 or any(c not in '0123456789abcdef' for c in p) for p in args.provider)):
            raise ValueError('Fetch requires distinct explicit provider identities')
        with NodeStateLock(directory/'visitor'):
            identity = LSPIdentity.generate()
            node = LSPNodeV3(identity, host='0.0.0.0', port=0, seeds=get_seed_endpoints(),
                             address_book_path=directory/'peers.json', allow_private=False)
            work = WorkCoordinator(node, ResourceGovernor(ContributionPolicy.for_mode('relay'), hardware=SimpleNamespace(
                cpu_cores=1, ram_gb=1, disk_gb=1, has_gpu=False, gpu_vram_gb=0), storage_path=directory))
            offers, proofs = {}, []
            send = node.send_work_offer;submit = work.submit
            def capture_offer(peer, offer):
                offers[offer['job_id']] = json.loads(json.dumps(offer))
                return send(peer, offer)
            def capture_result(operation, payload, requirements, **kwargs):
                result = submit(operation, payload, requirements, **kwargs)
                if result.status == 'ok':proofs.append(transfer_proof(offers.pop(result.job_id), result, kwargs['peer_id']))
                return result
            node.send_work_offer = capture_offer;work.submit = capture_result
            def guard():
                while not cancel.wait(.2):
                    if not allowed():cancel.set();return
            monitor = threading.Thread(target=guard, daemon=True);monitor.start()
            try:
                node.start();routes = {}
                for provider in args.provider:
                    until = time.monotonic()+45
                    while provider not in node.connected_node_ids and not cancel.is_set() and time.monotonic() < until:
                        node.connect_node(provider, timeout=10)
                    if provider not in node.connected_node_ids:continue
                    route = next(peer for peer in node.peers() if peer['node_id'] == provider)
                    if not route['encrypted'] or not ipaddress.ip_address(route.get('relay_host') or route['host']).is_global:
                        raise ValueError('Dataset transfer must use encrypted public Internet')
                    routes[provider] = public_route({k: route.get(k) for k in ['transport', 'encrypted', 'host', 'relay_host']})
                if not routes:raise RuntimeError('No public dataset provider connected')
                stats = []
                budget = BandwidthBudget(lambda: ClientSettings.load(node_directory).policy.max_bandwidth_mbps/2 if allowed() else 0)
                acquired = replicate_dataset(work, store, list(routes), receive_budget=budget, cancel_event=cancel,
                                             allowed=allowed, preflight=preflight, observer=stats.append)
                checked = verify_transfer_proofs(proofs, store, requester_id=identity.node_id_hex, providers=list(routes))
                fresh = all(not row['initially_local'] and not row['initial_partial_file'] for row in acquired['acquired'])
                if fresh and not all(checked['complete_byte_coverage'].values()):
                    raise ValueError('Fresh dataset acquisition has incomplete signed byte coverage')
                report.update(acquired, requester_id=identity.node_id_hex, routes=routes, signed_transfer_proofs=proofs,
                              transfer_verification=checked, initially_empty=fresh,
                              encoded_data_bytes=sum(row['encoded_data_bytes'] for row in stats),
                              recovery_store=args.recovery_copy)
            finally:
                cancel.set();monitor.join(timeout=2);work.close();node.stop()
    report.update(controls_and_native_progress_unchanged=fingerprints(node_directory) == before, finished_at=time.time())
    if not report['controls_and_native_progress_unchanged']:raise ValueError('Native state changed during data preparation')
    atomic_json(directory/'report.json', report)
    print(json.dumps({k:report[k] for k in ['success', 'action', 'physical_node_id', 'curriculum_id', 'controls_and_native_progress_unchanged']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['stage', 'fetch'])
    parser.add_argument('--node-directory', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--source-directory', type=Path)
    parser.add_argument('--provider', action='append')
    parser.add_argument('--recovery-copy', action='store_true')
    args = parser.parse_args()
    try:run(args)
    except BaseException as error:
        path = args.directory/'report.json'
        if path.is_file():
            report = json.loads(path.read_text())
            report.update(success=False, error_type=type(error).__name__, finished_at=time.time())
            atomic_json(path, report)
        raise


if __name__ == '__main__':main()
