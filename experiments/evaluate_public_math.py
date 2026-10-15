"""Measure a predeclared accepted expert on new public data without training."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace

from src.contribution import ResourceGovernor, ResourceRequirements
from src.learning.public_math import ARTIFACTS, curriculum_id, gate_id, load_snapshot, score_answers
from src.learning.public_quality import preservation_verdict as factual_verdict
from src.learning.public_language_gate import preservation_verdict as instruction_verdict
from src.network.artifact_store import ArtifactStore
from src.network.client_state import ClientSettings, NodeStateLock
from src.network.compute_process import ComputeSupervisor
from src.network.language_install import materialize_version
from src.network.public_knowledge import managed_public_bytes
from src.network.telemetry import atomic_json
from src.release.language_versions import LanguageVersionRegistry
from .prepare_public_math import outside_checkout, fingerprints


def run(args, cancel):
    node = args.node_directory.resolve();directory = args.directory.resolve()
    if not outside_checkout(node) or not outside_checkout(directory) or directory.exists():
        raise ValueError('Use current physical controls and fresh evaluation state outside Dropbox')
    if not args.engine_python.is_absolute() or not args.engine_python.is_file():
        raise ValueError('Use an existing locally selected tensor runtime')
    repo = Path(__file__).resolve().parents[1]
    protocol = json.loads((repo/'experiments/reports/public_math_curriculum_protocol_20261015.json').read_text())
    if protocol['gate_id'] != gate_id() or protocol['curriculum_id'] != curriculum_id():
        raise ValueError('Math evaluation differs from its public predeclared protocol')
    before = fingerprints(node);controls = ClientSettings.load(node);policy = asdict(controls.policy)
    registry = LanguageVersionRegistry(node/'public/language', 'qwen3-1.7b');head = registry.head()
    if (head['release']['release_id'] != protocol['baseline_parent_release_id']
            or head['release']['model_artifact_id'] != protocol['baseline_parent_model_id']):
        raise ValueError('Math baseline parent differs from the previously declared accepted expert')
    store = ArtifactStore(node/'public/language/artifacts', max_artifact_bytes=4*1024**3, max_total_bytes=8*1024**3)
    registry.install(registry.history(head['release']['release_id']), store)
    evaluation_path = store.object_path(ARTIFACTS['evaluation'][0]);evaluation = load_snapshot(evaluation_path, 'evaluation')
    governor = ResourceGovernor(controls.policy, hardware=SimpleNamespace(has_gpu=True), storage_path=node)
    def allowed():
        try:
            now = ClientSettings.load(node);state = json.loads((node/'diagnostics.json').read_text())
            accepted = json.loads(registry.head_path.read_text())['release_id']
            return (not cancel.is_set() and now.connected and not now.paused and asdict(now.policy) == policy
                    and now.policy.mode in {'balanced', 'maximum'} and now.policy.max_ram_gb >= 6
                    and now.policy.cpu_percent > 0 and now.policy.gpu_percent > 0 and now.policy.max_gpu_jobs > 0
                    and now.policy.max_disk_gb > 0 and managed_public_bytes(node)+8*1024**2 <= now.policy.max_disk_gb*1024**3
                    and state['language']['active'] == 0 and state['language_training']['active'] == 0
                    and accepted == head['release']['release_id'] and not governor._system_busy())
        except (OSError, ValueError, TypeError, KeyError):return False
    if not allowed():raise ValueError('Math baseline needs idle permitted GPU compute and six GiB RAM')
    base = materialize_version(node, 'qwen3-1.7b', head['release']['release_id'], allowed=allowed)
    directory.mkdir(mode=0o700, parents=True)
    atomic_json(directory/'status.json', {'success': False, 'phase': 'starting', 'gate_id': gate_id()})
    atomic_json(directory/'invariants-before.json', before)
    deadline = time.time()+1800
    class Execution:
        def raise_if_cancelled(self):
            if cancel.is_set():raise RuntimeError('Public math baseline cancelled')
    execution = Execution();execution.deadline = deadline
    request = {'action': 'public-math-baseline-v1', 'base_name': 'qwen3-1.7b', 'base_directory': str(base),
               'adapter_file': str(base/'adapter.safetensors'), 'model_id': head['release']['model_artifact_id'],
               'parent_release_id': head['release']['release_id'], 'evaluation_file': str(evaluation_path),
               'output_directory': str(directory/'engine'), 'device': 'cuda', 'cpu_threads': min(4, max(1, int(4*policy['cpu_percent']/100))),
               'gpu_fraction': policy['gpu_percent']/100/max(1, policy['max_gpu_jobs']), 'deadline': deadline, 'gate_id': gate_id()}
    lease, reason = governor.acquire(ResourceRequirements(kind='inference', gpu_required=True, ram_gb=6, disk_gb=.008))
    if lease is None:raise ValueError('Math research compute lease denied: '+reason)
    try:
        with lease, NodeStateLock(node/'research/public-math'):
            atomic_json(directory/'status.json', {'success': False, 'phase': 'evaluating', 'gate_id': gate_id()})
            response = ComputeSupervisor().run([str(args.engine_python.absolute()), '-m', 'src.learning.public_math_job'],
                                               request, execution, allowed=allowed, ram_gb=6)
        path = directory/'engine/baseline.json'
        if path.is_symlink() or path.stat().st_size > 2*1024**2:raise ValueError('Math baseline report exceeds its bound')
        raw = path.read_bytes();result = json.loads(raw)
        if (hashlib.sha256(raw).hexdigest() != response['evaluation_id'] or len(raw) != response['bytes']
                or not result['success'] or result['model_id'] != head['release']['model_artifact_id']
                or result['parent_release_id'] != head['release']['release_id'] or result['gate_id'] != gate_id()
                or result['curriculum_id'] != curriculum_id() or result['training_performed'] is not False):
            raise ValueError('Math baseline engine returned different content or lineage')
        scored = score_answers(result['answers'], evaluation)
        if sum(scored.values()) != result['correct'] or result['total'] != 128:
            raise ValueError('Math baseline score differs from the exact numerical answers')
        if fingerprints(node) != before:raise ValueError('Native controls or learning progress changed')
        diagnostic = json.loads((node/'diagnostics.json').read_text())
        facts = factual_verdict(result['legacy']['quality'], result['legacy']['quality'])
        instructions = instruction_verdict(result['legacy']['instructions'], result['legacy']['instructions'])
        report = {'success': True, 'research_path': True, 'native_daemon_job': False, 'physical_node_id': diagnostic['node_id'],
                  'source_protocol_commit': '33f027f', 'baseline': result, 'engine_report_sha256': response['evaluation_id'],
                  'historical_selfcheck': {'facts': facts, 'instructions': instructions},
                  'controls_and_native_progress_unchanged': True, 'new_model_published': False,
                  'native_learning_cycles_added': 0, 'research_resource_reservation': {'ram_gb': 6, 'gpu_fraction': request['gpu_fraction']},
                  'comparative_scaling_trial': False, 'finished_at': time.time()}
        atomic_json(directory/'report.json', report)
        atomic_json(directory/'status.json', {'success': True, 'phase': 'complete', 'correct': result['correct'], 'total': result['total']})
        print(json.dumps({'success': True, 'physical_node_id': diagnostic['node_id'], 'correct': result['correct'],
                          'total': result['total'], 'nll': result['nll'], 'at_output_limit': result['at_output_limit']}), flush=True)
    except BaseException as error:
        atomic_json(directory/'status.json', {'success': False, 'phase': 'failed', 'error_type': type(error).__name__,
                    'controls_and_native_progress_unchanged': fingerprints(node) == before})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node-directory', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--engine-python', type=Path, required=True)
    args = parser.parse_args();cancel = threading.Event()
    for sig in [signal.SIGTERM, signal.SIGINT]:signal.signal(sig, lambda *_:cancel.set())
    try:run(args, cancel)
    except BaseException as error:
        if args.directory.is_dir():
            atomic_json(args.directory/'status.json', {'success': False, 'phase': 'failed', 'error_type': type(error).__name__})
        raise


if __name__ == '__main__':main()
