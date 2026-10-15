"""Read-only fresh-process recovery of learned selectors and optimizer states."""
import argparse
import contextlib
from dataclasses import asdict
import json
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

from src.contribution import ResourceGovernor,ResourceRequirements
from src.learning.language_outcome_selector import definition,protocol_id,dataset,fold_indices,verify_report
from src.learning.public_math import ARTIFACTS,load_snapshot,prompt
from src.network.artifact_store import ArtifactStore,digest_file
from src.network.client_state import ClientSettings,NodeStateLock
from src.network.telemetry import atomic_json
from .prepare_public_math import fingerprints,outside_checkout
from .research_artifact import retained_bytes
from src.network.public_knowledge import managed_public_bytes


def run(args):
    directory=args.directory.resolve();node=args.node_directory.resolve();p=definition();cancelled=False
    if not outside_checkout(directory) or not directory.is_relative_to(Path.home()/'.local/state/lixyswarm-outcome-selector-20261015'):
        raise ValueError('Audit only the accounted external outcome experiment')
    result=json.loads((directory/'report.json').read_text());report=json.loads((directory/'engine/diagnostic.json').read_text())
    if not result['success'] or digest_file(directory/'engine/diagnostic.json')!=(result['report_id'],result['report_bytes']):
        raise ValueError('A completed, hash-verified training report is required')
    source=Path(__file__).resolve().parents[1]
    if any(digest_file(source/name)[0]!=identifier for name,identifier in result['numerical_source_files'].items()):
        raise ValueError('Audit must recover the exact trained numerical implementation')
    expected={str(i)+'/'+name for i in range(8) for name in ['dolphin.safetensors','zero.safetensors','shuffled.safetensors','public.safetensors','optimizer-state.pt']}
    if set(report['artifacts'])!=expected:raise ValueError('Selector artifact paths changed their scope')
    settings=ClientSettings.load(node);policy=asdict(settings.policy);before=fingerprints(node)
    def cancel(*_):
        nonlocal cancelled
        cancelled=True
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,cancel)
    def check():
        now=ClientSettings.load(node);diagnostic=json.loads((node/'diagnostics.json').read_text())
        if (cancelled or not now.connected or now.paused or asdict(now.policy)!=policy or fingerprints(node)!=before
                or diagnostic['pid']!=result['physical_pid'] or time.time()*1000-diagnostic['timestamp_ms']>60000
                or any(diagnostic[name].get('active',0) for name in ['compute','language','language_training'])
                or managed_public_bytes(node)+retained_bytes()>now.policy.max_disk_gb*1024**3):
            raise RuntimeError('Audit permission or physical native state changed')
    check();governor=ResourceGovernor(settings.policy,hardware=SimpleNamespace(has_gpu=True),storage_path=node)
    lease,reason=governor.acquire(ResourceRequirements(kind='inference',gpu_required=True,ram_gb=4,disk_gb=.001))
    if lease is None:raise ValueError('Audit compute unavailable: '+reason)
    store=ArtifactStore(node/'public/language/artifacts',max_artifact_bytes=4*1024**3)
    snapshot=load_snapshot(store.object_path(ARTIFACTS['train'][0]),'train')
    rows=dataset([json.loads(path.read_text()) for path in args.outcome_file],snapshot)
    verification=verify_report(report,rows)
    from src.learning.public_bases import BASES
    base=Path(json.loads((node/'language-engine.json').read_text())['base_directory'])
    for name in ['config.json','tokenizer.json','tokenizer_config.json']:
        if digest_file(base/name)[0]!=BASES['qwen3-1.7b'][name]:raise ValueError('Audit tokenizer changed')
    import torch
    from transformers import AutoTokenizer
    from src.learning.language_selector import LanguageExpertSelector,load_selector,tensor_digest
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(policy['gpu_percent']/100/max(1,policy['max_gpu_jobs']))
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
    queries=[tokenizer.encode(tokenizer.apply_chat_template([{'role':'user','content':prompt(row)}],tokenize=False,
        add_generation_prompt=True,enable_thinking=False),add_special_tokens=False) for row in snapshot['rows'][:128]]
    vocab=json.loads((base/'config.json').read_text())['vocab_size'];maximum=0.;optimizer_counts=[]
    with lease,NodeStateLock(node/'research/public-math'):
        for fold in range(8):
            _,test=fold_indices(fold);f=report['folds'][fold];binding=f['binding']
            if binding['protocol_id']!=protocol_id() or binding['expert_ids']!=p['expert_ids'] or binding['dolphin_state_id']!=f['fit']['dolphin_state_id']:
                raise ValueError('Recovered selector expert or learned feature binding changed')
            for name in ['dolphin',*p['memory_modes']]:
                check();relative=str(fold)+'/'+name+'.safetensors';path=directory/'engine'/relative
                if digest_file(path)!=(report['artifacts'][relative]['sha256'],report['artifacts'][relative]['bytes']):raise ValueError('Recovered selector bytes changed')
                model=LanguageExpertSelector(vocab,support_count=len(f['fit']['support_indices'])).cuda().eval()
                load_selector(model,path,{**binding,'variant':name});before_state=tensor_digest(model.state_dict())
                choices,probabilities=model.choose([queries[i] for i in test],use_memory=name!='dolphin')
                previous=[report['predictions'][name][i] for i in test]
                expected_probabilities=torch.tensor([r['probabilities'] for r in previous],device='cuda')
                error=float((probabilities-expected_probabilities).abs().max());maximum=max(maximum,error)
                if error>1e-6 or choices.cpu().tolist()!=[r['choice'] for r in previous] or tensor_digest(model.state_dict())!=before_state:
                    raise ValueError('Fresh-process prediction differs from its learned state')
                del model
            relative=str(fold)+'/optimizer-state.pt';path=directory/'engine'/relative
            if digest_file(path)!=(report['artifacts'][relative]['sha256'],report['artifacts'][relative]['bytes']):raise ValueError('Optimizer artifact changed')
            optimizers=torch.load(path,weights_only=True,map_location='cpu');groups={'dolphin':optimizers['dolphin'],**optimizers['memory']}
            if set(groups)!={'dolphin',*p['memory_modes']}:raise ValueError('Optimizer states are incomplete')
            counts={}
            for name,value in groups.items():
                if len(value['state'])!=(45 if name=='dolphin' else 18):raise ValueError('Optimizer active parameter coverage changed')
                for state in value['state'].values():
                    if float(state['step'])!=p['phase_steps'] or any(not torch.isfinite(state[k]).all() for k in ['exp_avg','exp_avg_sq']):
                        raise ValueError('Optimizer step or finite moments changed')
                counts[name]=len(value['state'])
            optimizer_counts.append(counts);del optimizers,groups;torch.cuda.empty_cache()
    check()
    audit={'success':True,'protocol_id':protocol_id(),'verification':verification,'selectors_reloaded':32,
        'optimizer_parameter_states_per_fold':optimizer_counts,'maximum_probability_absolute_difference':maximum,
        'native_controls_unchanged':fingerprints(node)==before,'original_evaluation_opened':False,
        'new_optimizer_steps':0,'new_answer_generation':False,'native_promotion_authorized':False,'timestamp':time.time()}
    atomic_json(directory/'independent-state-audit.json',audit);return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--node-directory',type=Path,required=True)
    parser.add_argument('--outcome-file',type=Path,action='append',required=True)
    with contextlib.redirect_stdout(sys.stderr):result=run(parser.parse_args())
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()
