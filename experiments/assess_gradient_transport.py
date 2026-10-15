"""Assess unsigned int8/float16 approximations of signed public gradients.

This experiment cannot publish, sign work, replace an accepted model, or change
the daemon's training policy. Actual public-LSP transfer is measured separately.
"""
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

from src.learning.language_scaling import verify_timed_run
from src.network.artifact_codec import CHUNK_BYTES, encode_chunk
from src.network.artifact_store import ArtifactStore, digest_file
from src.network.telemetry import atomic_json


def encoded_bytes(path):
    size = 0
    with Path(path).open('rb') as stream:
        while block := stream.read(CHUNK_BYTES):
            size += len(encode_chunk(block)['data'])
    return size


def prepare(args):
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    import torch
    from src.learning.experimental_gradient_transport import save, load, error, FORMAT, HALF_FORMAT, BLOCK
    from src.learning.language_gradient import metadata as gradient_metadata
    from src.learning.public_bases import base_hash
    from src.learning.public_conversation import public_corpus, corpus_digest
    from src.release.language_versions import LanguageVersionRegistry
    from src.network.client_state import ClientSettings
    from src.network.public_knowledge import managed_public_bytes
    torch.set_num_threads(2)
    reference = json.loads(args.reference.read_text())['trials'][0]
    verify_timed_run(reference)
    head = LanguageVersionRegistry(args.node_directory/'public/language', 'qwen3-1.7b').head()
    if reference['head'] != head:
        raise ValueError('Research requires gradients of the current accepted parent')
    controls = ClientSettings.load(args.node_directory)
    if managed_public_bytes(args.node_directory) + 128*1024**2 > controls.policy.max_disk_gb * 1024**3:
        raise ValueError('Research would exceed the existing disk consent')
    store = ArtifactStore(args.node_directory/'public/language/artifacts', max_artifact_bytes=4*1024**3, max_total_bytes=8*1024**3)
    rows = []; began = time.monotonic()
    for signed in reference['signed_results']:
        row = signed['output']; source = store.object_path(row['gradient_id'])
        if digest_file(source) != (row['gradient_id'], row['gradient_bytes']):
            raise ValueError('Public source differs from its actual signed work')
        expected = gradient_metadata(**{k:row[k] for k in ['base_hash','adapter_id','dataset_id','start','count','tokens']})
        if row['base_hash'] != base_hash('qwen3-1.7b') or row['dataset_id'] != corpus_digest(public_corpus()):
            raise ValueError('Only pinned public source data is permitted')
        with safe_open(source, framework='pt', device='cpu') as opened:
            if opened.metadata() != expected: raise ValueError('Public source has other shard metadata')
        tensors = load_file(source, device='cpu'); shapes = {k:t.shape for k,t in tensors.items()}
        packed = args.directory/f"{args.encoding}-{row['start']}.safetensors"
        reconstructed = args.directory/f"decoded-{row['start']}.safetensors"
        started = time.monotonic(); identifier = save(source, packed,encoding=args.encoding); encode_seconds = time.monotonic()-started
        started = time.monotonic(); decoded = load(packed, source_id=identifier, metadata=expected, shapes=shapes,encoding=args.encoding); decode_seconds=time.monotonic()-started
        metric = error(tensors, decoded)
        if metric['relative_l2_error'] > .03 or metric['cosine_similarity'] < .999 or metric['decoded_norm'] > 1.00001:
            raise ValueError('Research approximation exceeds the predefined numerical gate')
        # This float32 file is a local unsigned reconstruction, not the artifact
        # that the original receipt signed. Its different hash stays explicit.
        save_file(decoded, reconstructed, metadata=expected)
        rows.append({'source_result':signed,'packed_id':digest_file(packed)[0],'packed_bytes':packed.stat().st_size,
            'reconstructed_id':digest_file(reconstructed)[0],'reconstructed_bytes':reconstructed.stat().st_size,
            'original_encoded_data_bytes':encoded_bytes(source),'packed_encoded_data_bytes':encoded_bytes(packed),
            'encode_seconds':encode_seconds,'decode_seconds':decode_seconds,'error':metric})
    report = {'version':1,'encoding':args.encoding,'format':FORMAT if args.encoding=='block8' else HALF_FORMAT,
        'block_size':BLOCK if args.encoding=='block8' else None,'source_trial':reference,'gradients':rows,
        'quality_policy':{'maximum_relative_nll_increase_vs_fp32':.001,'allow_factual_regressions':False,'allow_instruction_regressions':False},
        'numerical_policy':{'maximum_relative_l2_error':.03,'minimum_cosine_similarity':.999,'maximum_decoded_norm':1.00001},
        'learning_rate':.0005,'private_inputs':False,'new_model_published':False,'live_training_policy_changed':False,
        'source_receipts_bind_reconstructed_artifacts':False,'prepare_seconds':time.monotonic()-began}
    atomic_json(args.directory/'prepared.json',report)
    print(json.dumps({'prepared':len(rows),'original_wire_bytes':sum(r['original_encoded_data_bytes'] for r in rows),
        'packed_wire_bytes':sum(r['packed_encoded_data_bytes'] for r in rows),
        'maximum_relative_l2_error':max(r['error']['relative_l2_error'] for r in rows)}),flush=True)


def evaluate(args, cancel):
    from src.contribution import ResourceGovernor,ResourceRequirements
    from src.network.client_state import ClientSettings
    from src.network.compute_process import ComputeSupervisor
    from src.learning.language_policy import BATCH_POLICY_ID
    controls = ClientSettings.load(args.node_directory); policy=asdict(controls.policy)
    before = {name:(args.node_directory/name).read_bytes() for name in ['controls.json','language-engine.json',
        'language-training.json','language-learning-controls.json','language-learning/ledger.json'] if (args.node_directory/name).is_file()}
    governor=ResourceGovernor(controls.policy,hardware=SimpleNamespace(has_gpu=True),storage_path=args.directory)
    def allowed():
        now=ClientSettings.load(args.node_directory);status=json.loads((args.node_directory/'diagnostics.json').read_text())
        checks={'not_cancelled':not cancel.is_set(),'connected':now.connected,'not_paused':not now.paused,
            'policy_unchanged':asdict(now.policy)==policy,'compute_consented':now.policy.cpu_percent>0 and now.policy.gpu_percent>0 and now.policy.max_ram_gb>=6,
            'native_compute_idle':status['language']['active']==status['language_training']['active']==0,
            'system_available':not governor._system_busy()}
        if not all(checks.values()):atomic_json(args.directory/'permission-stop.json',{'at':time.time(),'checks':checks})
        return all(checks.values())
    if not allowed(): raise ValueError('Research requires idle currently permitted GPU resources')
    prepared=json.loads((args.directory/'prepared.json').read_text());trial=prepared['source_trial'];verify_timed_run(trial)
    engine=json.loads((args.node_directory/'language-engine.json').read_text())
    if engine['adapter_id']!=trial['head']['release']['model_artifact_id']:raise ValueError('Accepted parent changed')
    deadline=time.time()+900
    class Execution:
        def raise_if_cancelled(self):
            if cancel.is_set():raise RuntimeError('Research cancelled')
    execution=Execution();execution.deadline=deadline
    base={'action':'candidate','base_name':trial['base_name'],'base_directory':engine['base_directory'],
        'adapter_file':engine['adapter_file'],'adapter_id':engine['adapter_id'],
        'dataset_id':trial['signed_results'][0]['output']['dataset_id'],'parent_release_id':trial['head']['release']['release_id'],
        'learning_rate':prepared['learning_rate'],'cpu_threads':min(8,max(1,int(8*policy['cpu_percent']/100))),
        'gpu_fraction':min(.8,policy['gpu_percent']/100),'device':'cuda','deadline':deadline,'training_policy_id':BATCH_POLICY_ID}
    store=ArtifactStore(args.node_directory/'public/language/artifacts');results={}
    if args.reference_run:
        previous=json.loads((args.reference_run/'report.json').read_text())
        if (not previous['success'] or not previous['controls_and_native_progress_unchanged']
                or previous['prepared']['source_trial']!=trial
                or previous['prepared']['learning_rate']!=prepared['learning_rate']):raise ValueError('Research reference differs')
        fp32=previous['evaluations']['fp32']
        if digest_file(args.reference_run/'fp32-candidate.safetensors')!=(fp32['adapter_id'],fp32['adapter_bytes']):
            raise ValueError('Research reference candidate content differs')
        results['fp32']=previous['evaluations']['fp32']
    for mode in ['fp32',prepared.get('encoding','block8')]:
        if mode in results:continue
        contributions=[]
        for item in prepared['gradients']:
            row=item['source_result']['output'];selected={k:row[k] for k in ['gradient_id','gradient_bytes','base_hash','adapter_id','dataset_id','start','count','tokens']}
            if mode=='fp32':selected['path']=str(store.object_path(row['gradient_id']))
            else:selected.update(path=str((args.directory/f"decoded-{row['start']}.safetensors").resolve()),gradient_id=item['reconstructed_id'],gradient_bytes=item['reconstructed_bytes'])
            contributions.append(selected)
        request={**base,'contributions':contributions,'output_file':str((args.directory/f'{mode}-candidate.safetensors').resolve())}
        atomic_json(args.directory/'status.json',{'phase':'evaluating_'+mode,'time':time.time()})
        lease,reason=governor.acquire(ResourceRequirements(kind='training',gpu_required=True,ram_gb=6,disk_gb=.05))
        if lease is None:raise ValueError('Research lease denied: '+reason)
        with lease:
            result=ComputeSupervisor().run([str(args.engine_python.absolute()),'-m','src.learning.language_candidate_job'],request,execution,allowed=allowed,ram_gb=6)
        results[mode]=result;atomic_json(args.directory/(mode+'-evaluation.json'),result)
    original,packed=results['fp32'],results[prepared.get('encoding','block8')]
    if original['baseline_nll']!=packed['baseline_nll'] or original['quality_before']!=packed['quality_before'] or original['instruction_before']!=packed['instruction_before']:
        raise ValueError('Paired research baselines differ')
    from src.learning.public_quality import preservation_verdict as facts
    from src.learning.public_language_gate import preservation_verdict as instructions
    factual=facts(original['quality_after'],packed['quality_after']);instruction=instructions(original['instruction_after'],packed['instruction_after'])
    # Reuse exact historical answer comparisons, plus a separate NLL tolerance.
    nonregression=not factual['regressions'] and not instruction['regressions']
    delta=packed['candidate_nll']/original['candidate_nll']-1
    unchanged=all((args.node_directory/name).read_bytes()==value for name,value in before.items())
    report={'success':True,'prepared':prepared,'evaluations':results,'comparison':{'relative_nll_change':delta,
        'historical_answers_do_not_regress':nonregression,'eligible_for_transport_integration_research':delta<=.001 and nonregression,
        'factual':factual,'instruction':instruction},'controls_and_native_progress_unchanged':unchanged,
        'physical_evaluator':json.loads((args.node_directory/'diagnostics.json').read_text())['node_id'],
        'new_model_published':False,'live_training_policy_changed':False,'physical_network_transfer_measured':False}
    report['reference_evaluation_reused']=bool(args.reference_run)
    if not unchanged:raise ValueError('Native policy or learning progress changed')
    atomic_json(args.directory/'report.json',report)
    atomic_json(args.directory/'status.json',{'phase':'evaluated','success':True,'comparison':report['comparison']})
    print(json.dumps(report['comparison']),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','evaluate'])
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--node-directory',type=Path,required=True)
    parser.add_argument('--reference',type=Path,default=Path(__file__).parent/'reports/language_native_batch_scaling_20261014.json')
    parser.add_argument('--engine-python',type=Path)
    parser.add_argument('--encoding',choices=['block8','half'],default='block8')
    parser.add_argument('--reference-run',type=Path)
    args=parser.parse_args()
    if any(p.resolve().is_relative_to(Path.home()/'Dropbox') for p in [args.directory,args.node_directory]):
        parser.error('Store pilot artifacts outside Dropbox')
    if args.action=='prepare':args.directory.mkdir(parents=True,exist_ok=False,mode=0o700)
    elif not args.engine_python or not args.engine_python.is_file():parser.error('Select an existing local tensor interpreter')
    cancel=threading.Event()
    for signum in [signal.SIGINT,signal.SIGTERM]:signal.signal(signum,lambda *_:cancel.set())
    prepare(args) if args.action=='prepare' else evaluate(args,cancel)


if __name__=='__main__':main()
