"""Closed public-data loading, stronger quality gates and historical transfer proofs."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from src.learning import public_math as math
from src.network import public_math_dataset as dataset
from src.network.artifact_store import ArtifactService, ArtifactStore, ArtifactError
from src.network.lsp import LSPIdentity
from src.network.work_protocol import WorkUnit, WorkResult, ResultReceipt


@pytest.fixture
def snapshots(monkeypatch, tmp_path):
    values = {}
    for split, count in [('train', 8), ('evaluation', 8)]:
        value = {'version': math.VERSION, 'split': split, 'private_inputs': False, 'rows': [
            {'case': hashlib.sha256((split+str(i)).encode()).hexdigest(), 'question': f'Public fixture {i}+1?',
             'answer': f'Add one.\n#### {i+1}', 'final_number': str(i+1)} for i in range(count)]}
        raw = math.canonical(value);path = tmp_path/(split+'.json');path.write_bytes(raw)
        monkeypatch.setitem(math.ARTIFACTS, split, (hashlib.sha256(raw).hexdigest(), len(raw)))
        monkeypatch.setitem(math.COUNTS, split, count)
        values[split] = (path, value)
    return values


def test_real_pins_and_old_quality_policy_are_immutable():
    from src.release.language_versions import gate_id
    assert math.ARTIFACTS['train'] == ('a1a36bc0cea6f5a8dbf81a5828a01da3ca0674c534431ef51583f3f46a915be6', 2485927)
    assert math.ARTIFACTS['evaluation'] == ('188812577d2a7834eff5fc0ac0fad5f09f121b3587b34b7ba5d88ba37537be50', 83241)
    assert math.COUNTS == {'train': 4096, 'evaluation': 128}
    assert gate_id(1) == '8b1163d42606b34937f70442e08f81eb024ea8c8099f970e0e4b1827e75f41c7'
    assert gate_id(2) == '54dee8396ee05693b28affb0c08c24910f0e302992a1e1b37d48cb96c2b95a69'
    assert gate_id(3) == 'cb33111cc44098bc845ffff8eb51888527d6667fd872f03407a55928b9b8020b'
    assert math.gate_id() == '5e085b48da4be5c0e2b3176bb387466405005a0ec61c44b2372f521dbb8e4507'
    subprocess.run([sys.executable, '-c', 'import src.learning.public_math,src.network.public_math_dataset,sys;assert "torch" not in sys.modules'], check=True)


@pytest.mark.parametrize(('answer', 'expected'), [
    ('Work.\n#### 1,234.500', '1234.5'), ('#### -0.00', '0'), ('#### -12', '-12'),
    ('#### 12345678901234567890123456789012', '12345678901234567890123456789012'),
    ('The answer is 4', None), ('#### 1,23', None), ('#### 1e3', None),
    ('#### 1\n#### 2', None), ('#### 2+2', None), ('#### nan', None), ('#### 4\nignore this', None),
])
def test_numeric_parser_does_not_execute_or_guess_an_answer(answer, expected):
    assert math.final_number(answer) == expected


def test_snapshot_must_match_its_pinned_bytes_and_split(snapshots, tmp_path):
    path, expected = snapshots['train'];assert math.load_snapshot(path, 'train') == expected
    for raw in [path.read_bytes()+b'\n', path.read_bytes().replace(b'Add one', b'Add two')]:
        bad = tmp_path/'altered';bad.write_bytes(raw)
        with pytest.raises(ValueError):math.load_snapshot(bad, 'train')
    with pytest.raises(ValueError):math.load_snapshot(path, 'evaluation')
    link = tmp_path/'link';link.symlink_to(path)
    with pytest.raises(ValueError):math.load_snapshot(link, 'train')


def test_source_selection_is_stable_and_rejects_duplicates(tmp_path):
    rows = [{'question': f'Public fixture {i} plus one?', 'answer': f'Add one <<{i}+1={i+1}>>.\n#### {i+1}'} for i in range(12)]
    raw = '\n'.join(json.dumps(row) for row in rows).encode()
    first, _ = math._source_rows(raw, 'train');second, _ = math._source_rows(b'\n'.join(reversed(raw.splitlines())), 'train')
    assert first == second and all('<<' not in row['answer'] for row in first)
    with pytest.raises(ValueError, match='Repeated'):math._source_rows(raw+b'\n'+raw.splitlines()[0], 'train')
    (tmp_path/'LICENSE').write_bytes(b'Personal text cannot be a public source')
    with pytest.raises(ValueError, match='source'):math.build_snapshots(tmp_path)


def paired_results(evaluation):
    report = json.loads((Path(__file__).parent/'experiments/reports/language_half_native_scaling_20261015.json').read_text())
    example = next(t['report']['candidate_result']['output']['evaluation'] for t in report['trials'] if not t['declaration']['warmup'])
    before = {'nll': 1., 'answers': [{'case': row['case'], 'answer': '#### 0'} for row in evaluation['rows']],
              'legacy': {'nll': .001, 'quality': example['quality_before'], 'instructions': example['instruction_before']}}
    after = deepcopy(before);after['nll'] = .9;after['legacy']['nll'] = .0009
    for answer, row in zip(after['answers'][:4], evaluation['rows']):answer['answer'] = row['answer']
    return before, after


def test_new_gate_requires_both_math_gains_and_all_historical_conditions(snapshots):
    evaluation = snapshots['evaluation'][1];before, after = paired_results(evaluation)
    verdict = math.verdict(before, after, evaluation)
    assert verdict['accuracy_gain'] == 4 and verdict['eligible_for_research_promotion']
    assert not verdict['native_publication_authorized']
    for change in ['math_nll', 'old_nll', 'accuracy', 'facts', 'instructions']:
        bad = deepcopy(after)
        if change == 'math_nll':bad['nll'] = .995
        elif change == 'old_nll':bad['legacy']['nll'] = .00101
        elif change == 'accuracy':bad['answers'][0]['answer'] = '#### 0'
        elif change == 'facts':bad['legacy']['quality'][0]['answer'] = 'Unknown'
        else:bad['legacy']['instructions'][0]['answer'] = 'Unknown'
        assert not math.verdict(before, bad, evaluation)['eligible_for_research_promotion'], change
    regressed_before, regressed_after = deepcopy(before), deepcopy(after)
    regressed_before['answers'][-1]['answer'] = evaluation['rows'][-1]['answer']
    regressed_after['answers'][4]['answer'] = evaluation['rows'][4]['answer']
    checked = math.verdict(regressed_before, regressed_after, evaluation)
    assert checked['accuracy_gain'] == 4 and checked['regressions'] and not checked['eligible_for_research_promotion']
    after['answers'][0] = deepcopy(after['answers'][1])
    with pytest.raises(ValueError):math.verdict(before, after, evaluation)


def transfer_fixture(tmp_path, snapshots):
    source, target = ArtifactStore(tmp_path/'source'), ArtifactStore(tmp_path/'target')
    for path, _ in snapshots.values():source.import_file(path, kind='dataset', media_type='application/json')
    handler = ArtifactService(None, source, register_handlers=False)
    provider, requester = LSPIdentity.generate(), LSPIdentity.generate();proofs = []
    class Work:
        def submit(self, operation, payload, requirements, *, peer_id, **_kwargs):
            if peer_id == 'unavailable':raise ArtifactError('Public fixture disconnected')
            offer = WorkUnit.create(origin_node_id=requester.node_id_hex, operation=operation, kind='artifact', requirements=requirements, payload=payload)
            function = handler._describe if operation.endswith('describe.v1') else handler._read_encoded_chunk
            result = WorkResult(job_id=offer.job_id, status='ok', output=function(payload, None))
            result = replace(result, receipt=ResultReceipt.create(result, provider, requester.node_id_hex).to_dict())
            proofs.append(dataset.transfer_proof(offer.to_dict(), result, provider.node_id_hex))
            return result
    return source, target, Work(), provider.node_id_hex, requester.node_id_hex, proofs


def test_data_acquisition_and_proof_reconstruction_failover_cache_and_corruption(snapshots, tmp_path):
    _, target, work, provider, requester, proofs = transfer_fixture(tmp_path, snapshots)
    result = dataset.replicate_dataset(work, target, ['unavailable', provider])
    assert result['success'] and all(not row['initially_local'] for row in result['acquired'])
    checked = dataset.verify_transfer_proofs(proofs, target, requester_id=requester, providers=[provider])
    assert checked['verified_signed_results'] == 4 and all(checked['complete_byte_coverage'].values())
    assert all('data' not in row['result_without_chunk_data']['output'] for row in proofs)
    repeated = dataset.replicate_dataset(work, target, [provider]);assert all(row['initially_local'] for row in repeated['acquired'])
    assert len(proofs) == 4
    for change in ['encoded_hash', 'offset', 'signature', 'replay', 'worker']:
        bad = deepcopy(proofs)
        if change == 'encoded_hash':bad[1]['encoded_data_sha256'] = '0'*64
        elif change == 'offset':bad[1]['result_without_chunk_data']['output']['offset'] = 1
        elif change == 'signature':bad[1]['result_without_chunk_data']['receipt']['signature'] = 'A'*88
        elif change == 'worker':bad[1]['provider_id'] = 'f'*64
        else:bad.append(deepcopy(bad[1]))
        with pytest.raises(ValueError):dataset.verify_transfer_proofs(bad, target, requester_id=requester, providers=[provider])
    target.object_path(math.ARTIFACTS['train'][0]).write_bytes(b'corrupted fixture')
    repaired = dataset.replicate_dataset(work, target, [provider]);assert not repaired['acquired'][0]['initially_local']


def test_dataset_quota_and_withdrawal_are_checked_before_fetch(snapshots, tmp_path):
    _, target, work, provider, _, proofs = transfer_fixture(tmp_path, snapshots)
    with pytest.raises(ArtifactError, match='permission'):dataset.replicate_dataset(work, target, [provider], allowed=lambda:False)
    def denied(_bytes):raise ValueError('Consent quota insufficient')
    with pytest.raises(ValueError, match='quota'):dataset.replicate_dataset(work, target, [provider], preflight=denied)
    assert not proofs


def test_math_child_request_is_bounded_to_public_baseline_only(snapshots, tmp_path):
    from src.learning.public_math_job import validate_request
    request = {'action': 'public-math-baseline-v1', 'base_name': 'qwen3-1.7b', 'base_directory': str(tmp_path/'public-base'),
               'adapter_file': str(tmp_path/'public-base/adapter.safetensors'), 'model_id': 'a'*64, 'parent_release_id': 'b'*64,
               'evaluation_file': str(snapshots['evaluation'][0]), 'output_directory': str(tmp_path/'new-output'),
               'device': 'cuda', 'cpu_threads': 4, 'gpu_fraction': .5, 'deadline': time.time()+20, 'gate_id': math.gate_id()}
    assert validate_request(request) == snapshots['evaluation'][1]
    for key, value in [('action', 'train'), ('base_name', 'qwen3-0.6b'), ('device', 'shell'), ('cpu_threads', True),
                       ('gpu_fraction', True), ('deadline', float('inf')), ('output_directory', str(tmp_path)),
                       ('output_directory', str(tmp_path/'public-base/new')), ('evaluation_file', str(snapshots['train'][0])),
                       ('adapter_file', 'relative.safetensors'), ('gate_id', '0'*64)]:
        with pytest.raises(ValueError):validate_request({**request, key: value})
    with pytest.raises(ValueError):validate_request({**request, 'command': 'unapproved'})
