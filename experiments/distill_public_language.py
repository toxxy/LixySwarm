"""Collect signed public lessons over Internet; evaluate a bounded small student.

This is a research path, not an autonomous coordinator or publication command.
The lightweight parent supervises the tensor child with current local consent.
"""
import argparse
from dataclasses import asdict
import ipaddress
import json
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace

from src.contribution import ContributionPolicy, ResourceGovernor, ResourceRequirements
from src.learning.public_teaching import (VERSION, digest, public_teaching_corpus, teaching_prompt,
                                         verify_lessons, verify_student_report)
from src.network.artifact_store import ArtifactStore
from src.network.bootstrap import get_seed_endpoints
from src.network.client_state import ClientSettings
from src.network.compute_process import ComputeSupervisor
from src.network.language_install import materialize_family
from src.network.language_version_service import fetch_history
from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.report_privacy import public_route
from src.network.telemetry import atomic_json
from src.network.work_protocol import WorkCoordinator
from src.release.language_versions import LanguageVersionRegistry, WORKERS

SMALL = 'qwen3-0.6b'
LARGE = 'qwen3-1.7b'


def outside_checkout(path):
    path = path.resolve()
    return not any(path.is_relative_to(root) for root in [Path(__file__).resolve().parents[1], Path.home() / 'Dropbox'])


def approved_teacher(directory):
    root = directory / 'public/language'
    registry = LanguageVersionRegistry(root, LARGE)
    head = registry.head()
    if head['release']['sequence'] < 1:
        raise ValueError('Teaching requires a previously accepted learned large expert')
    chain = registry.history(head['release']['release_id'])
    store = ArtifactStore(root / 'artifacts', max_artifact_bytes=4 * 1024**3, max_total_bytes=8 * 1024**3)
    registry.install(chain, store)  # Recheck actual evaluations and content, not just signatures.
    return head, chain


def options_for(lessons):
    return {k: lessons[k] for k in ['teacher_model_id', 'teacher_release_id', 'teacher_worker_id', 'requester_id']}


def collect(args, cancel):
    head, trusted_chain = approved_teacher(args.node_directory)
    controls = ClientSettings.load(args.node_directory)
    if not controls.connected or controls.paused or controls.policy.max_bandwidth_mbps <= 0:
        raise ValueError('Public lesson acquisition requires current connectivity consent')
    identity = LSPIdentity.generate()
    # A visitor requests labels only. It is not an additional physical worker.
    node = LSPNodeV3(identity, host='0.0.0.0', port=0, seeds=get_seed_endpoints(),
        address_book_path=args.directory / 'peers.json', allow_private=False)
    work = WorkCoordinator(node, ResourceGovernor(ContributionPolicy.for_mode('relay'),
        hardware=SimpleNamespace(cpu_cores=1, ram_gb=1, has_gpu=False, gpu_vram_gb=0, disk_gb=1), storage_path=args.directory))
    offers = {}
    send = node.send_work_offer
    def capture(peer, offer):
        if peer == args.teacher and offer['operation'] == 'language.inference.v1':
            offers[offer['job_id']] = json.loads(json.dumps(offer))
        return send(peer, offer)
    node.send_work_offer = capture
    lessons = {'version': VERSION, 'dataset_id': digest(public_teaching_corpus()),
        'teacher_model_id': head['release']['model_artifact_id'], 'teacher_release_id': head['release']['release_id'],
        'teacher_worker_id': args.teacher, 'requester_id': identity.node_id_hex, 'batches': []}
    began = time.monotonic()
    try:
        node.start()
        while args.teacher not in node.connected_node_ids:
            if time.monotonic() - began > 60 or cancel.is_set():
                raise TimeoutError('Public teacher discovery timed out')
            node.connect_node(args.teacher, timeout=10)
        chain = fetch_history(work, LARGE, args.teacher, cancel_event=cancel)
        if chain != trusted_chain:
            raise ValueError('Remote teacher history differs from the locally accepted version')
        routes = []
        for start in range(0, 64, 4):
            current = ClientSettings.load(args.node_directory)
            if not current.connected or current.paused or current.policy.max_bandwidth_mbps <= 0:
                raise ValueError('Connectivity permission withdrawn')
            route = next(p for p in node.peers() if p['node_id'] == args.teacher)
            if not route['encrypted'] or not ipaddress.ip_address(route.get('relay_host') or route['host']).is_global:
                raise ValueError('Teacher must use encrypted public Internet transport')
            routes.append(public_route({k: route.get(k) for k in ['transport', 'host', 'relay_host', 'encrypted']}))
            payload = {'model_artifact_id': lessons['teacher_model_id'],
                       'messages': [{'role': 'user', 'content': teaching_prompt(start)}]}
            result = work.submit('language.inference.v1', payload,
                ResourceRequirements(kind='inference', gpu_required=True, ram_gb=5),
                peer_id=args.teacher, timeout_s=90, cancel_event=cancel)
            if result.status != 'ok':
                raise RuntimeError('Teacher inference failed: ' + result.error)
            lessons['batches'].append({'start': start, 'offer': offers.pop(result.job_id), 'result': result.to_dict()})
            atomic_json(args.directory / 'lessons.partial.json', lessons)
            atomic_json(args.directory / 'status.json', {'phase': 'collecting', 'batches': len(lessons['batches']), 'time': time.time()})
        checked = verify_lessons(lessons, **options_for(lessons))
        atomic_json(args.directory / 'lessons.json', lessons)
        report = {'success': True, 'scope': 'Signed public labels from the existing physical large expert; visitor is not another worker',
            'teacher_head': head, 'teacher_history': chain, 'physical_student': json.loads((args.node_directory / 'diagnostics.json').read_text())['node_id'],
            'teacher_worker_id': args.teacher, 'requester_id': identity.node_id_hex, 'routes': routes,
            'dataset_id': checked['dataset_id'], 'lessons_id': digest(lessons), 'accepted_labels': len(checked['accepted']),
            'rejected_labels': checked['rejected'], 'evaluation_sent_to_teacher': False, 'private_inputs': False,
            'public_internet': True, 'seconds': time.monotonic() - began, 'new_model_published': False}
        atomic_json(args.directory / 'collection.json', report)
        atomic_json(args.directory / 'status.json', {'success': True, 'phase': 'lessons_complete', 'accepted_labels': len(checked['accepted'])})
        print(json.dumps({'success': True, 'accepted_labels': len(checked['accepted']), 'seconds': report['seconds']}), flush=True)
    except Exception as error:
        atomic_json(args.directory / 'status.json', {'success': False, 'phase': 'collection_failed',
            'batches': len(lessons['batches']), 'error_type': type(error).__name__})
        raise
    finally:
        work.close()
        node.stop()


def train(args, cancel):
    head, _ = approved_teacher(args.node_directory)
    with args.lessons.open('rb') as stream:
        raw = stream.read(512 * 1024 + 1)
    if len(raw) > 512 * 1024:
        raise ValueError('Lessons exceed their bound')
    lessons = json.loads(raw)
    if (lessons['teacher_model_id'] != head['release']['model_artifact_id']
            or lessons['teacher_release_id'] != head['release']['release_id']
            or lessons['teacher_worker_id'] not in WORKERS):
        raise ValueError('Lessons do not use the current accepted public teacher')
    checked = verify_lessons(lessons, **options_for(lessons))
    if len(checked['accepted']) < 48:
        raise ValueError('Too few correctly taught public examples for the fixed experiment')
    controls = ClientSettings.load(args.node_directory)
    policy = asdict(controls.policy)
    governor = ResourceGovernor(controls.policy, hardware=SimpleNamespace(has_gpu=True), storage_path=args.directory)
    before = {name: (args.node_directory / name).read_bytes() for name in [
        'controls.json', 'language-engine.json', 'language-training.json', 'language-follow-controls.json',
        'language-learning-controls.json', 'language-learning/ledger.json'] if (args.node_directory / name).is_file()}
    def allowed():
        try:
            now = ClientSettings.load(args.node_directory)
            status = json.loads((args.node_directory / 'diagnostics.json').read_text())
            return (not cancel.is_set() and now.connected and not now.paused and asdict(now.policy) == policy
                and now.policy.mode in {'balanced', 'maximum'} and now.policy.max_ram_gb >= 4
                and now.policy.cpu_percent > 0 and now.policy.gpu_percent > 0 and now.policy.max_gpu_jobs > 0
                and now.policy.max_disk_gb > 0 and status['language_training']['active'] == 0
                and status['language']['active'] == 0 and not governor._system_busy())
        except (OSError, ValueError, KeyError):
            return False
    if not allowed():
        raise ValueError('Student needs idle permitted compute and four GiB RAM')
    from src.network.public_knowledge import managed_public_bytes
    if managed_public_bytes(args.node_directory) + 64 * 1024**2 > controls.policy.max_disk_gb * 1024**3:
        raise ValueError('No consented room for a bounded student candidate')
    base = materialize_family(args.node_directory, SMALL, allowed=allowed)
    deadline = time.time() + 1200
    class Execution:
        def raise_if_cancelled(self):
            if cancel.is_set():
                raise RuntimeError('Student experiment cancelled')
    execution = Execution()
    execution.deadline = deadline
    request = {'action': 'distill-public-v1', 'base_directory': str(base), 'lessons_file': str(args.lessons.resolve()),
        'output_directory': str(args.directory.resolve()), 'teacher': options_for(lessons), 'deadline': deadline,
        'cpu_threads': min(8, max(1, int(4 * controls.policy.cpu_percent / 100))),
        'gpu_fraction': min(.8, controls.policy.gpu_percent / 100 / max(1, controls.policy.max_gpu_jobs))}
    atomic_json(args.directory / 'status.json', {'phase': 'student_starting', 'success': False})
    try:
        lease, reason = governor.acquire(ResourceRequirements(kind='training', gpu_required=True,
            ram_gb=min(6, controls.policy.max_ram_gb), disk_gb=.04))
        if lease is None:
            raise ValueError('Local research compute lease denied: ' + reason)
        with lease:
            report = ComputeSupervisor().run([str(args.engine_python.absolute()), '-m', 'src.learning.language_distillation_job'],
                request, execution, allowed=allowed, ram_gb=min(6, controls.policy.max_ram_gb))
        verify_student_report(report, lessons)
        report.update(physical_student=json.loads((args.node_directory / 'diagnostics.json').read_text())['node_id'],
            controls_and_native_progress_unchanged=all((args.node_directory / name).read_bytes() == data for name, data in before.items()))
        if not report['controls_and_native_progress_unchanged']:
            raise ValueError('Native state changed during the research experiment')
        atomic_json(args.directory / 'report.json', report)
        atomic_json(args.directory / 'status.json', {'success': True, 'phase': 'student_complete', 'eligible_for_research_transfer': report['eligible_for_research_transfer']})
        print(json.dumps({'success': True, 'before': report['transfer_verdict']['baseline_passed'],
            'after': report['transfer_verdict']['candidate_passed'], 'eligible_for_research_transfer': report['eligible_for_research_transfer']}), flush=True)
    except Exception as error:
        atomic_json(args.directory / 'status.json', {'success': False, 'phase': 'student_failed', 'error_type': type(error).__name__})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['collect', 'train'])
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--node-directory', type=Path, required=True)
    parser.add_argument('--teacher')
    parser.add_argument('--lessons', type=Path)
    parser.add_argument('--engine-python', type=Path)
    args = parser.parse_args()
    if args.directory.exists() or not outside_checkout(args.directory) or not outside_checkout(args.node_directory):
        parser.error('Use fresh experiment storage and existing node state outside Dropbox')
    if (args.action == 'collect' and args.teacher not in WORKERS
            or args.action == 'train' and (not args.lessons or not args.engine_python or not args.engine_python.is_file())):
        parser.error('Select a teacher identity or local lessons and tensor interpreter')
    args.directory.mkdir(parents=True, mode=0o700)
    cancel = threading.Event()
    for signum in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(signum, lambda *_: cancel.set())
    (collect if args.action == 'collect' else train)(args, cancel)


if __name__ == '__main__':
    main()
