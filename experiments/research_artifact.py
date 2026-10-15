"""Restricted, bounded LSP copies of public research artifacts, never releases.

SSH may exchange the public helper IDs and compact reports, not artifact bytes.
The helpers are temporary transports on existing PCs, not additional workers.
"""
import argparse
import ipaddress
import json
from pathlib import Path
import re
import signal
import threading
import time
from types import SimpleNamespace

from src.contribution import ContributionPolicy, ResourceGovernor
from src.network.artifact_store import ArtifactService, ArtifactStore, digest_file
from src.network.bootstrap import get_seed_endpoints
from src.network.client_state import ClientSettings, NodeStateLock
from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.public_knowledge import BandwidthBudget, KnowledgeSettings, managed_public_bytes
from src.network.report_privacy import public_route
from src.network.telemetry import atomic_json
from src.network.work_protocol import WorkCoordinator
from .prepare_public_math import fingerprints, outside_checkout


RESEARCH_NAMES = ('lixyswarm-language-selector-20261015', 'lixyswarm-math-specialist-20261015',
                  'lixyswarm-specialist-answers-20261015', 'lixyswarm-research-replica-20261015',
                  'lixyswarm-outcomes-20261015', 'lixyswarm-outcome-selector-20261015')
MAX_BYTES = 16*1024**2


def retained_bytes():
    """Include saved research states in addition to the native public store."""
    total = 0; seen = set()
    for name in RESEARCH_NAMES:
        for path in (Path.home()/'.local/state'/name).rglob('*'):
            if path.is_file() and not path.is_symlink():
                stat = path.stat(); key = stat.st_dev, stat.st_ino
                if key not in seen: total += stat.st_size; seen.add(key)
    return total


def authorize(payload, execution, peer_id, artifact_id, allowed):
    execution.raise_if_cancelled()
    if (execution.origin_node_id != peer_id or payload.get('artifact_id') != artifact_id
            or not allowed()):
        raise ValueError('Only the selected peer and research artifact are authorized')


def initialize(directory):
    if directory.exists() or not outside_checkout(directory):
        raise ValueError('Research transport needs fresh state outside Dropbox')
    directory.mkdir(mode=0o700, parents=True)
    identity = LSPIdentity.generate(); identity.save(str(directory/'helper.key'))
    atomic_json(directory/'identity.json', {'helper_id': identity.node_id_hex})
    return identity.node_id_hex


def run(args):
    directory = args.directory.resolve(); physical = args.node_directory.resolve()
    if (not outside_checkout(directory) or not outside_checkout(physical)
            or directory == physical or directory.is_relative_to(physical)
            or physical.is_relative_to(directory)):
        raise ValueError('Use separate research transport and physical node state')
    # Only registered research folders enter the common quota calculation.
    if not any(directory.is_relative_to(Path.home()/'.local/state'/name) for name in RESEARCH_NAMES):
        raise ValueError('Research transport must be inside an accounted research folder')
    if args.action == 'init': return {'helper_id': initialize(directory)}
    if (not re.fullmatch('[0-9a-f]{64}', args.peer or '')
            or not re.fullmatch('[0-9a-f]{64}', args.artifact_id or '')
            or type(args.size) is not int or not 0 < args.size <= MAX_BYTES):
        raise ValueError('An explicit peer, exact artifact hash and bounded size are required')
    identity = LSPIdentity.load(str(directory/'helper.key'))
    if identity.node_id_hex != json.loads((directory/'identity.json').read_text())['helper_id']:
        raise ValueError('Transport identity changed')
    before = fingerprints(physical)
    diagnostic = json.loads((physical/'diagnostics.json').read_text()); pid = diagnostic['pid']
    cancelled = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT): signal.signal(signum, lambda *_: cancelled.set())
    settings = ClientSettings.load(physical)
    mbps = min(2.5, settings.policy.max_bandwidth_mbps/2)
    def allowed():
        try:
            now = ClientSettings.load(physical)
            return (not cancelled.is_set() and now.connected and not now.paused and KnowledgeSettings.load(physical).enabled
                    and now.policy.mode != 'disabled' and mbps > 0 and now.policy.max_bandwidth_mbps >= 2*mbps
                    and fingerprints(physical) == before
                    and json.loads((physical/'diagnostics.json').read_text())['pid'] == pid
                    and managed_public_bytes(physical)+retained_bytes() <= now.policy.max_disk_gb*1024**3)
        except (OSError, ValueError, KeyError, TypeError): return False
    if not allowed() or managed_public_bytes(physical)+retained_bytes()+2*args.size+1024**2 > settings.policy.max_disk_gb*1024**3:
        raise ValueError('Research copy exceeds current storage or network consent')
    store = ArtifactStore(directory/'artifacts', max_artifact_bytes=MAX_BYTES, max_total_bytes=MAX_BYTES)
    if store.has(args.artifact_id): raise ValueError('This test requires a fresh artifact store')
    if args.action == 'serve':
        if args.source is None or not outside_checkout(args.source) or digest_file(args.source) != (args.artifact_id, args.size):
            raise ValueError('Research source differs from the explicitly selected public artifact')
        store.import_file(args.source, kind='other')
    elif args.source is not None: raise ValueError('The receiver cannot import a local source')
    policy = ContributionPolicy.for_mode('relay'); policy.max_bandwidth_mbps = mbps
    governor = ResourceGovernor(policy, hardware=SimpleNamespace(cpu_cores=1, ram_gb=1, disk_gb=1,
        has_gpu=False, gpu_vram_gb=0), storage_path=directory)
    node = LSPNodeV3(identity, host='0.0.0.0', port=0, seeds=get_seed_endpoints(),
        address_book_path=directory/'peers.json', allow_private=False)
    work = WorkCoordinator(node, governor, max_workers=2, max_offers_per_minute=600)
    budget = BandwidthBudget(lambda: mbps if allowed() else 0)
    service = ArtifactService(work, store, namespace='language.artifact', register_handlers=False, receive_budget=budget)
    report = {'success': False, 'action': args.action, 'helper_id': identity.node_id_hex,
        'peer_id': args.peer, 'physical_node_id': diagnostic['node_id'], 'physical_pid': pid,
        'artifact_id': args.artifact_id, 'artifact_bytes': args.size, 'started_at': time.time(),
        'extra_physical_worker': False, 'model_release_published': False, 'native_learning_cycles_added': 0,
        'scope': 'Research preservation only; no release registration, public inference or model promotion'}
    def monitor():
        while not cancelled.wait(.5):
            if not allowed(): cancelled.set(); return
    guard = threading.Thread(target=monitor, daemon=True)
    try:
        with NodeStateLock(directory/'transport'):
            if args.action == 'serve':
                def handler(payload, execution, method):
                    authorize(payload, execution, args.peer, args.artifact_id, allowed)
                    value = method(payload, execution)
                    if 'data' in value: budget.wait(len(value['data'])+2048, cancel_event=cancelled, deadline=execution.deadline)
                    authorize(payload, execution, args.peer, args.artifact_id, allowed)
                    return value
                def cached(execution, output):
                    authorize(execution.payload, execution, args.peer, args.artifact_id, allowed)
                    if 'data' in output: budget.wait(len(output['data'])+2048, cancel_event=cancelled, deadline=execution.deadline)
                    return allowed()
                for operation, method in [('describe.v1',service._describe),('read-chunk.v1',service._read_chunk),
                                          ('read-chunk.v2',service._read_encoded_chunk)]:
                    work.register_handler('language.artifact.'+operation, 'artifact',
                        lambda payload, execution, method=method: handler(payload, execution, method), permit_cached=cached)
            node.start(); guard.start(); atomic_json(directory/'ready.json', report)
            if args.action == 'serve':
                if not cancelled.wait(1200): report['expired'] = True
            else:
                deadline = time.monotonic()+90
                while args.peer not in node.connected_node_ids:
                    if not allowed() or time.monotonic() > deadline: raise TimeoutError('Public research provider discovery expired')
                    node.connect_node(args.peer, timeout=10); cancelled.wait(.2)
                route = next(row for row in node.peers() if row['node_id'] == args.peer)
                if not route['encrypted'] or not ipaddress.ip_address(route.get('relay_host') or route['host']).is_global:
                    raise ValueError('Research transfer requires encrypted public Internet')
                chunks = []; began = time.monotonic()
                downloaded = service.fetch(args.artifact_id, peer_id=args.peer, timeout_s=600,
                    expected_size=args.size, cancel_event=cancelled, observer=chunks.append)
                if digest_file(downloaded) != (args.artifact_id,args.size) or sum(r['bytes'] for r in chunks) != args.size:
                    raise ValueError('Fresh research acquisition did not reconstruct its exact source')
                report.update(route=public_route({k:route[k] for k in ['host','relay_host','transport','encrypted'] if k in route}),
                    chunks=len(chunks), bytes_received=sum(r['bytes'] for r in chunks),
                    seconds=time.monotonic()-began, sha256_verified=True)
            if (fingerprints(physical) != before or json.loads((physical/'diagnostics.json').read_text())['pid'] != pid):
                raise ValueError('Research copy changed native state or its physical process')
            report.update(success=True, native_controls_unchanged=True)
    finally:
        cancelled.set()
        if guard.is_alive(): guard.join(timeout=2)
        work.close(); node.stop(); report['finished_at'] = time.time()
        atomic_json(directory/'report.json', report)
    return {k:v for k,v in report.items() if k != 'route'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init','serve','receive'])
    parser.add_argument('--node-directory',type=Path,required=True)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--peer'); parser.add_argument('--artifact-id'); parser.add_argument('--size',type=int)
    parser.add_argument('--source',type=Path)
    print(json.dumps(run(parser.parse_args()), allow_nan=False),flush=True)


if __name__ == '__main__': main()
