"""Compare lossless artifact transport between existing consenting PCs over Internet.

Helpers have separate temporary identities and never count as additional PCs.
The server only reads the physical node's already-public language artifact store.
"""
import argparse
import ipaddress
import json
from pathlib import Path
import re
import signal
import statistics
import threading
import time
from types import SimpleNamespace

from src.contribution import ContributionPolicy, ResourceGovernor
from src.network.artifact_store import ArtifactService, ArtifactStore, digest_file
from src.network.bootstrap import get_seed_endpoints
from src.network.client_state import ClientSettings
from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.public_knowledge import BandwidthBudget, KnowledgeSettings, managed_public_bytes
from src.network.telemetry import atomic_json
from src.network.work_protocol import WorkCoordinator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['serve', 'receive'])
    parser.add_argument('--node-directory', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--artifact-id', required=True)
    parser.add_argument('--provider')
    parser.add_argument('--legacy-provider', help='Optional existing physical provider for new-to-old compatibility')
    parser.add_argument('--mbps', type=float, default=2.5)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    physical, root = args.node_directory.resolve(), args.directory.resolve()
    if (root.is_relative_to(repo) or root.is_relative_to(physical) or physical.is_relative_to(root)
            or root.exists() or not re.fullmatch('[0-9a-f]{64}', args.artifact_id)
            or not 0 < args.mbps <= 5):
        parser.error('Use fresh bounded experiment state outside the checkout and physical node state')
    if args.action == 'receive' and (not args.provider or any(
            not re.fullmatch('[0-9a-f]{64}', value) for value in [args.provider]+([args.legacy_provider] if args.legacy_provider else []))):
        parser.error('Receiving requires explicit public provider identities')
    def allowed():
        settings = ClientSettings.load(physical)
        return (settings.connected and not settings.paused and KnowledgeSettings.load(physical).enabled
                and settings.policy.mode != 'disabled' and settings.policy.max_bandwidth_mbps >= 2*args.mbps
                and managed_public_bytes(physical)+managed_public_bytes(root) <= settings.policy.max_disk_gb*1024**3)
    if not allowed():
        raise ValueError('Physical operator sharing, storage and bandwidth consent are required')
    controls_before = (physical/'controls.json').read_bytes()
    root.mkdir(mode=0o700, parents=True)
    cancel = threading.Event()
    for signum in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(signum, lambda *_: cancel.set())
    def monitor():
        while not cancel.wait(.5):
            try:
                if not allowed():
                    cancel.set(); return
            except (OSError, ValueError, TypeError):
                cancel.set(); return
    guard = threading.Thread(target=monitor, daemon=True)
    identity = LSPIdentity.generate(); identity.save(str(root/'helper.key'))
    node = LSPNodeV3(identity, host='0.0.0.0', port=0, seeds=get_seed_endpoints(), target_outbound=1,
        address_book_path=root/'peers.json', allow_private=False)
    policy = ContributionPolicy.for_mode('relay'); policy.max_bandwidth_mbps = args.mbps
    governor = ResourceGovernor(policy, hardware=SimpleNamespace(cpu_cores=1, ram_gb=1, disk_gb=1,
        has_gpu=False, gpu_vram_gb=0), storage_path=root)
    work = WorkCoordinator(node, governor, max_workers=2, max_offers_per_minute=600)
    budget = BandwidthBudget(lambda: args.mbps if allowed() else 0)
    report = {'success': False, 'started_at': time.time(), 'helper_id': identity.node_id_hex,
        'helper_is_additional_physical_pc': False, 'gpu_work_requested': False, 'mbps': args.mbps,
        'artifact_id': args.artifact_id, 'trials': [],
        'scope': 'Artifact transfer only; unchanged bytes and model semantics, not a training speedup or accepted learning claim'}
    atomic_json(root/'report.json', report)
    try:
        if args.action == 'serve':
            store = ArtifactStore(physical/'public/language/artifacts', max_artifact_bytes=4*1024**3)
            manifest = store.manifest(args.artifact_id)
            if not store.has(args.artifact_id) or manifest.size > 32*1024**2:
                raise ValueError('Only an existing small verified public artifact may be served')
            service = ArtifactService(work, store, namespace='language.artifact', register_handlers=False)
            def chunk(payload, execution, encoded):
                execution.raise_if_cancelled()
                if cancel.is_set() or not allowed() or payload.get('artifact_id') != args.artifact_id:
                    raise ValueError('Artifact experiment permission withdrawn or unrelated artifact')
                value = (service._read_encoded_chunk if encoded else service._read_chunk)(payload, execution)
                budget.wait(len(value['data'])+2048, cancel_event=cancel, deadline=execution.deadline)
                execution.raise_if_cancelled()
                return value
            def describe(payload, execution):
                if cancel.is_set() or not allowed() or payload.get('artifact_id') != args.artifact_id:
                    raise ValueError('Only the selected public artifact may be described')
                return service._describe(payload, execution)
            def cached(execution, output):
                if cancel.is_set() or not allowed():
                    return False
                if execution.operation.startswith('language.artifact.read-chunk.'):
                    budget.wait(len(output['data'])+2048, cancel_event=cancel, deadline=execution.deadline)
                return not cancel.is_set() and allowed()
            work.register_handler('language.artifact.describe.v1', 'artifact', describe, permit_cached=cached)
            for version in [1, 2]:
                work.register_handler(f'language.artifact.read-chunk.v{version}', 'artifact',
                    lambda payload, execution, encoded=version==2: chunk(payload, execution, encoded), permit_cached=cached)
            report['artifact_bytes'] = manifest.size
            node.start(); guard.start(); atomic_json(root/'ready.json', report)
            cancel.wait(600)
        else:
            node.start(); guard.start()
            providers = [args.provider]+([args.legacy_provider] if args.legacy_provider else [])
            routes = {}
            for provider in providers:
                until = time.monotonic()+60
                while provider not in node.connected_node_ids:
                    if cancel.is_set() or time.monotonic() > until:
                        raise TimeoutError('Public artifact provider discovery expired')
                    node.connect_node(provider, timeout=10); cancel.wait(.2)
                route = next(row for row in node.peers() if row['node_id'] == provider)
                if not route['encrypted'] or not ipaddress.ip_address(route.get('relay_host') or route['host']).is_global:
                    raise ValueError('Artifact experiment requires encrypted public Internet transport')
                routes[provider] = {key: route[key] for key in ['transport', 'encrypted', 'relay_host'] if key in route}
            order = [('raw', args.provider, False), ('encoded', args.provider, True),
                     ('encoded', args.provider, True), ('raw', args.provider, False)]
            if args.legacy_provider:
                order.append(('legacy-provider', args.legacy_provider, True))
            for index, (label, provider, encoded) in enumerate(order):
                store = ArtifactStore(root/'public'/f'trial-{index}', max_artifact_bytes=32*1024**2, max_total_bytes=40*1024**2)
                client = ArtifactService(work, store, namespace='language.artifact', register_handlers=False,
                    prefer_compressed=encoded, receive_budget=budget)
                stats = []; began = time.monotonic()
                downloaded = client.fetch(args.artifact_id, peer_id=provider, timeout_s=120,
                    cancel_event=cancel, observer=stats.append)
                elapsed = time.monotonic()-began
                identifier, size = digest_file(downloaded)
                if identifier != args.artifact_id or sum(row['bytes'] for row in stats) != size:
                    raise ValueError('Transferred artifact or fresh acquisition count differs')
                report['trials'].append({'mode': label, 'provider_id': provider, 'seconds': elapsed,
                    'artifact_bytes': size, 'sha256_verified': True, 'chunks': len(stats),
                    'encoded_data_bytes': sum(row['encoded_data_bytes'] for row in stats),
                    'encodings': sorted({row['encoding'] for row in stats}), 'route': routes[provider]})
                atomic_json(root/'report.json', report)
            groups = {label: [row for row in report['trials'] if row['mode'] == label] for label in ['raw', 'encoded']}
            report['comparison'] = {label: {'median_seconds': statistics.median(row['seconds'] for row in rows),
                'encoded_data_bytes': rows[0]['encoded_data_bytes']} for label, rows in groups.items()}
        if (physical/'controls.json').read_bytes() != controls_before:
            raise ValueError('Physical operator controls changed during transfer observation')
        report.update(success=True, controls_unchanged=True)
    finally:
        cancel.set()
        if guard.is_alive(): guard.join(timeout=2)
        work.close(); node.stop()
        report['finished_at'] = time.time(); atomic_json(root/'report.json', report)


if __name__ == '__main__':
    main()
