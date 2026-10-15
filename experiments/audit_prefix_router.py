"""Reload frozen prefix routing and its optimizer in an independent CPU child."""
import argparse
import contextlib
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import zipfile

from src.learning.language_prefix_router import definition,protocol_id,feature_id,verify_report
from src.learning.language_outcome_selector import OUTCOME_IDS,dataset
from src.learning.public_math import ARTIFACTS,canonical,load_snapshot
from src.network.artifact_store import digest_file
from src.network.telemetry import atomic_json


def verify_saved_state(root,report,targets):
    """No model/tokenizer loading, gradient operation or optimizer step here."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from src.learning.language_dense_selector import DenseExpertSelector,load_dense
    from src.learning.language_selector import tensor_digest
    p=definition();torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    commitment=report['commitment'];artifacts=report['artifacts']
    expected={*(n+'.safetensors' for n in ['dolphin',*p['memory_modes']]),'optimizer-state.pt',
        'prefix-features.safetensors','prediction-commitment.json'}
    if set(artifacts)!=expected:raise ValueError('Prefix recovery requires all seven original artifacts')
    if any(artifacts[name]!=value for name,value in commitment['learned_artifacts'].items()):
        raise ValueError('Prefix report differs from its committed learned artifact hashes')
    for name,value in artifacts.items():
        if Path(name).name!=name or digest_file(root/name)!=(value['sha256'],value['bytes']):
            raise ValueError('Prefix saved-state bytes differ from their recorded integrity')
    if json.loads((root/'prediction-commitment.json').read_text())!=commitment:
        raise ValueError('Saved prefix commitment differs from its report')
    feature_path=root/'prefix-features.safetensors'
    if feature_path.stat().st_size>4*1024**2:raise ValueError('Prefix feature artifact exceeds its bound')
    with safe_open(feature_path,framework='pt',device='cpu') as reader:
        if (reader.metadata()!={'format':'lixyswarm-public-prefix-features-v1','binding':canonical(report['feature_binding']).decode()}
                or list(reader.keys())!=['features']):raise ValueError('Prefix feature space or public input provenance changed')
    features=load_file(str(feature_path),device='cpu')['features']
    if (features.shape!=(256,p['feature_definition']['dimensions']) or features.dtype!=torch.float32
            or not torch.isfinite(features).all() or targets.shape!=(128,2) or targets.dtype!=torch.float32):
        raise ValueError('Recovered prefix features or verified targets have a different schema')
    support=report['fit']['support_indices'];permutation=torch.randperm(len(support),generator=torch.Generator().manual_seed(p['seed']+2))
    comparisons={};models={}
    for name in ['dolphin',*p['memory_modes']]:
        metadata=report['bindings'][name]
        if metadata['dolphin_state_id']!=report['fit']['dolphin_state_id']:raise ValueError('Recovered memory and Dolphin feature spaces disagree')
        net=DenseExpertSelector(features.shape[1],support_count=len(support));load_dense(net,root/(name+'.safetensors'),metadata)
        net.eval();net.requires_grad_(False);before=tensor_digest(net.state_dict())
        choices,probabilities=net.choose(features[128:],use_memory=name!='dolphin')
        expected_rows=commitment['predictions'][name]
        committed=torch.tensor([row['probabilities'] for row in expected_rows],dtype=torch.float32)
        if (choices.tolist()!=[row['choice'] for row in expected_rows]
                or not torch.allclose(probabilities,committed,atol=1e-6,rtol=1e-5)):
            raise ValueError('Independent prefix reload changed a committed route')
        with torch.no_grad():
            bank=torch.nn.functional.pad(torch.cat([net.dolphin(features[support]),targets[support]],-1),(0,6))
        if name in {'zero','dolphin'}:bank.zero_()
        elif name=='shuffled':bank[:,8:10]=bank[permutation,8:10]
        bank_error=float((bank-net.public_memory).abs().max())
        if not torch.allclose(bank,net.public_memory,atol=2e-6,rtol=1e-5):
            raise ValueError('Recovered prefix memory is not its declared public support and ablation')
        if tensor_digest(net.state_dict())!=before:raise ValueError('Recovery mutated the frozen learned state')
        comparisons[name]={'choices_identical':True,'maximum_probability_difference':float((probabilities-committed).abs().max()),
            'maximum_support_bank_difference':bank_error,'state_unchanged':True,'learned_state_id':before}
        models[name]=net
    optimizer_path=root/'optimizer-state.pt'
    if optimizer_path.stat().st_size>8*1024**2:raise ValueError('Prefix optimizer exceeds its bound')
    with zipfile.ZipFile(optimizer_path) as archive:
        if len(archive.infolist())>10000 or sum(i.file_size for i in archive.infolist())>16*1024**2:
            raise ValueError('Prefix optimizer archive exceeds its unpacked bound')
    optimizer=torch.load(optimizer_path,map_location='cpu',weights_only=True)
    if set(optimizer)!={'dolphin','memory'} or set(optimizer['memory'])!=set(p['memory_modes']):
        raise ValueError('Prefix optimizer arms do not match the fixed training schedule')
    states={'dolphin':optimizer['dolphin'],**optimizer['memory']};optimizer_counts={}
    for name,value in states.items():
        count=0
        if not value['state']:raise ValueError('Prefix optimizer has no learned moments')
        for state in value['state'].values():
            if (set(state)!={'step','exp_avg','exp_avg_sq'} or float(state['step'])!=p['phase_steps']
                    or any(not torch.isfinite(v).all() for v in state.values())):
                raise ValueError('Prefix optimizer moments or update counts are invalid')
            count+=1
        optimizer_counts[name]={'parameters_with_moments':count,'steps':p['phase_steps']}
    return {'success':True,'comparisons':comparisons,'optimizer_states':optimizer_counts,
        'independent_cpu_process':True,'language_model_loaded':False,'new_optimizer_updates':0,
        'native_cycles_added':0,'model_promoted':False,'probability_absolute_tolerance':1e-6,
        'probability_relative_tolerance':1e-5,'support_bank_absolute_tolerance':2e-6}


def child(request):
    expected={'action','source_directory','training_file','outcome_files'}
    if (set(request)!=expected or request['action']!='audit-frozen-prefix-v1'
            or not isinstance(request['outcome_files'],list) or len(request['outcome_files'])!=2
            or any(not Path(v).is_absolute() for v in [request['source_directory'],request['training_file'],*request['outcome_files']])):
        raise ValueError('Unknown local prefix recovery request')
    root=Path(request['source_directory']);raw=(root/'engine/diagnostic.json').read_bytes()
    if len(raw)>2*1024**2:raise ValueError('Prefix diagnostic exceeds its bound')
    report=json.loads(raw);outer=json.loads((root/'report.json').read_text())
    if (outer['report_id']!=hashlib.sha256(raw).hexdigest() or outer['report_bytes']!=len(raw)
            or outer['protocol_id']!=protocol_id()):raise ValueError('Prefix outer report does not bind the original learned-state result')
    snapshot=load_snapshot(request['training_file'],'train');verification=verify_report(report,snapshot)
    if verification!=report['verification'] or verification!=outer['verification']:raise ValueError('Recovered prefix outcome scores do not reconstruct')
    values=[]
    for name,identifier in zip(request['outcome_files'],OUTCOME_IDS):
        if Path(name).stat().st_size>128*1024 or digest_file(name)[0]!=identifier:raise ValueError('Prefix recovery requires exact physical calibration outcomes')
        values.append(json.loads(Path(name).read_text()))
    import torch
    targets=torch.tensor([r['target'] for r in dataset(values,snapshot)],dtype=torch.float32)
    result=verify_saved_state(root/'engine',report,targets)
    if 'transformers' in sys.modules:raise ValueError('Prefix state recovery unexpectedly loaded a language runtime')
    return {**result,'protocol_id':protocol_id(),'source_diagnostic_id':outer['report_id'],'verification':verification}


def run(args,cancel):
    from src.contribution import ResourceGovernor,ResourceRequirements
    from src.network.client_state import ClientSettings,NodeStateLock
    from src.network.compute_process import ComputeSupervisor
    from src.network.public_knowledge import managed_public_bytes
    from experiments.prepare_public_math import fingerprints,outside_checkout
    from experiments.research_artifact import retained_bytes
    from experiments.learn_prefix_router import NUMERICAL_FILES
    source=args.source_directory.resolve();output=source/'independent-recovery';node=args.node_directory.resolve()
    if output.exists() or not all(outside_checkout(v) for v in [source,node,*args.outcome_file]):
        raise ValueError('Prefix recovery requires fresh accounted state separate from code and public inputs')
    if not source.is_relative_to(Path.home()/'.local/state/lixyswarm-prefix-router-20261015'):
        raise ValueError('Prefix source must remain inside the accounted research directory')
    repository=Path(__file__).resolve().parents[1];outer=json.loads((source/'report.json').read_text())
    if any(digest_file(repository/name)[0]!=outer['numerical_source_files'][name] for name in NUMERICAL_FILES):
        raise ValueError('The reader numerical source differs from the learned-state source')
    before=fingerprints(node);settings=ClientSettings.load(node);policy=asdict(settings.policy)
    initial=json.loads((node/'diagnostics.json').read_text());pid=initial['pid'];deadline=time.time()+180
    def allowed():
        now=ClientSettings.load(node);live=json.loads((node/'diagnostics.json').read_text())
        return (not cancel.is_set() and now.connected and not now.paused and asdict(now.policy)==policy
            and fingerprints(node)==before and live['pid']==pid and time.time()*1000-live['timestamp_ms']<60000
            and managed_public_bytes(node)+retained_bytes()+1024**2<=now.policy.max_disk_gb*1024**3)
    if not allowed():raise ValueError('Prefix recovery is outside current physical permission')
    governor=ResourceGovernor(settings.policy,storage_path=node)
    lease,reason=governor.acquire(ResourceRequirements(kind='inference',ram_gb=2,disk_gb=.001))
    if lease is None:raise ValueError('CPU prefix recovery unavailable: '+reason)
    output.mkdir(mode=0o700);request={'action':'audit-frozen-prefix-v1','source_directory':str(source),
        'training_file':str(node/'public/language/artifacts/objects'/ARTIFACTS['train'][0][:2]/ARTIFACTS['train'][0][2:]),
        'outcome_files':[str(v.resolve()) for v in args.outcome_file]}
    execution=SimpleNamespace(deadline=deadline,raise_if_cancelled=lambda:None)
    try:
        with lease,NodeStateLock(output):
            result=ComputeSupervisor().run([str(args.engine_python),'-m','experiments.audit_prefix_router','--tensor-child'],
                request,execution,allowed=allowed,ram_gb=2)
        result.update(native_controls_unchanged=fingerprints(node)==before,physical_pid=pid)
        atomic_json(output/'report.json',result);return result
    except BaseException as error:
        atomic_json(output/'report.json',{'success':False,'error_type':type(error).__name__,'error':str(error)[:1024]});raise


def main():
    if sys.argv[1:]==['--tensor-child']:
        raw=sys.stdin.buffer.read(16*1024+1)
        if len(raw)>16*1024:raise ValueError('Prefix recovery request exceeds its bound')
        with contextlib.redirect_stdout(sys.stderr):result=child(json.loads(raw))
    else:
        parser=argparse.ArgumentParser(description=__doc__)
        for name in ['source-directory','node-directory','engine-python']:parser.add_argument('--'+name,type=Path,required=True)
        parser.add_argument('--outcome-file',type=Path,action='append',required=True);cancel=threading.Event()
        for signum in [signal.SIGINT,signal.SIGTERM]:signal.signal(signum,lambda *_:cancel.set())
        result=run(parser.parse_args(),cancel)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()
