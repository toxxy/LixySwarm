"""Reconstruct the completed comparison from public bytes in a fresh CPU process."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

from src.learning import language_repair_comparison as spec
from src.learning.public_math import canonical, final_number

BROKEN_REPORT_REVISION='d143a93a6c1151d476b7638e180c491624d8895c'
BROKEN_REPORT_ID='f2c1e86ff266c2fd506a98372a7fcec74ab23724f5c4569ca377d605ed0cb275'


def read(path, maximum):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError('Audit input is not a bounded local public file')
    raw=path.read_bytes()
    return json.loads(raw),hashlib.sha256(raw).hexdigest(),len(raw)


def equivalent(a,b):
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if type(a) is float and type(b) is float:
        return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8)
    return type(a) is type(b) and a==b


def audit(directory,source_directory,*,recover_saved_report=False,original_source=None):
    directory=Path(directory);p=spec.definition()
    trace,trace_id,size=read(directory/'engine/diagnostic.json',4*1024**2)
    recovered=False;observation_id=None
    if recover_saved_report:
        observation,observation_id,_=read(directory/'recovery-observation-before-fix.json',256*1024)
        failure,_,_=read(directory/'status.json',4096)
        partial,_,_=read(directory/'engine/partial-outcomes.json',4*1024**2)
        if (observation['initial_supervisor_success'] is not False or failure['phase']!='failed'
                or observation['original_child_receipt_captured'] is not False
                or observation['source_revision']!=BROKEN_REPORT_REVISION or trace_id!=BROKEN_REPORT_ID
                or (trace_id,size)!=(observation['stored_report_sha256'],observation['stored_report_bytes'])
                or hashlib.sha256(canonical(trace)).hexdigest()!=observation['canonical_report_sha256']
                or trace_id==observation['canonical_report_sha256'] or partial['records']!=trace['records']):
            raise ValueError('Saved evidence does not support this serialization failure recovery')
        revision=observation['source_revision'];source=Path(__file__).resolve().parents[1]
        if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('Recovery source revision is not pinned')
        for name,identifier in observation['numerical_source_files'].items():
            if not re.fullmatch(r'src/learning/[A-Za-z0-9_]+\.py',name):raise ValueError('Unexpected recovery source path')
            if original_source is None:
                raw=subprocess.check_output(['git','show',revision+':'+name],cwd=source)
            else:
                path=Path(original_source)/name
                if path.is_symlink() or path.stat().st_size>256*1024:raise ValueError('Original source exceeds its audit bound')
                raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=identifier:raise ValueError('Original numerical source does not reconstruct')
        admin={'success':True,'protocol_id':spec.protocol_id(),'source_revision':revision,
            'report_id':trace_id,'report_bytes':size,'model_promoted':False,'native_cycles_added':0,
            'verification':trace['verification']}
        admin_id=None;recovered=True
    else:
        admin,admin_id,_=read(directory/'report.json',256*1024)
        receipt,_,_=read(directory/'engine-receipt.json',4096)
        if receipt!={'success':True,'protocol_id':spec.protocol_id(),'report_id':trace_id,'bytes':size}:
            raise ValueError('The saved child receipt does not bind the audited file')
    snapshot=spec.load_snapshot(directory/'prospective.json')
    if (admin['success'] is not True or trace['success'] is not True
            or admin['protocol_id']!=spec.protocol_id() or trace['protocol_id']!=spec.protocol_id()
            or (trace_id,size)!=(admin['report_id'],admin['report_bytes'])
            or admin['model_promoted'] is not False or trace['model_promoted'] is not False
            or trace['training_performed'] is not False or admin['native_cycles_added']!=0
            or not recovered and admin['native_controls_unchanged'] is not True):
        raise ValueError('Audit report bindings or read-only invariants differ')
    if canonical(spec.build_snapshot(source_directory))!=(directory/'prospective.json').read_bytes():
        raise ValueError('The selected cases do not reconstruct from pinned upstream bytes')
    commitment,commitment_id,_=read(directory/'engine/prospective-commitment.json',64*1024)
    expected={'protocol_id':spec.protocol_id(),'snapshot_id':spec.SNAPSHOT_ID,
        'cases':[row['case'] for row in snapshot['rows']],'rule':p['rule'],'second_call_order':p['second_call_order']}
    if commitment!=expected or len(trace['records'])!=128:
        raise ValueError('Audit cannot reconstruct the complete prospective commitment')
    metrics={arm:{'correct':0,'new_correct':0,'regressions':0,'calls':0,
        'input_tokens':0,'output_tokens':0,'generation_seconds':0.} for arm in p['arms']}
    times={arm:[] for arm in p['arms']}; generated=0; second_count=0
    for index,(row,record) in enumerate(zip(snapshot['rows'],trace['records'])):
        parent=record['parent']; second=final_number(parent['answer']) is None
        order=['repair','specialist'] if index%2==0 else ['specialist','repair']
        if (record['case']!=row['case'] or record['second_requested'] is not second
                or record['second_order']!=order or set(record['responses'])!=({'repair','specialist'} if second else set())):
            raise ValueError('Audit found a changed case or a correctness-dependent route')
        decision,_,_=read(directory/f'engine/decision-{index:03}.json',2048)
        if decision!={'case':row['case'],'parent_answer_id':hashlib.sha256(parent['answer'].encode()).hexdigest(),
                'second_requested':second,'second_order':order}:
            raise ValueError('Audit decision differs from the returned parent response')
        calls=[parent,*record['responses'].values()]; generated+=len(calls); second_count+=second
        for call in calls:
            if (set(call)!={'answer','input_tokens','output_tokens','generation_seconds'}
                    or not isinstance(call['answer'],str) or len(call['answer'].encode())>8192
                    or type(call['input_tokens']) is not int or not 1<=call['input_tokens']<=1024
                    or type(call['output_tokens']) is not int or not 1<=call['output_tokens']<=384
                    or type(call['generation_seconds']) not in (int,float)
                    or not math.isfinite(call['generation_seconds']) or call['generation_seconds']<0):
                raise ValueError('Audit generation evidence exceeds the declared budget')
        parent_correct=final_number(parent['answer'])==row['final_number']
        for arm in metrics:
            policy_calls=[parent]
            if second and arm!='parent':policy_calls.append(record['responses']['repair' if arm=='parent-repair' else 'specialist'])
            correct=final_number(policy_calls[-1]['answer'])==row['final_number'];m=metrics[arm]
            m['correct']+=correct; m['new_correct']+=correct and not parent_correct
            m['regressions']+=parent_correct and not correct; m['calls']+=len(policy_calls)
            m['input_tokens']+=sum(c['input_tokens'] for c in policy_calls)
            m['output_tokens']+=sum(c['output_tokens'] for c in policy_calls)
            times[arm].extend(c['generation_seconds'] for c in policy_calls)
    for arm in metrics:metrics[arm]['generation_seconds']=math.fsum(times[arm])
    fallback=metrics['specialist-fallback'];repair=metrics['parent-repair']
    checks={'gain_at_least_four':fallback['new_correct']>=4,'no_regressions':fallback['regressions']==0,
        'exceeds_same_parent_repair':fallback['correct']>repair['correct'],
        'within_declared_time_budget':fallback['generation_seconds']<=repair['generation_seconds']*1.1}
    result={'total':128,'invalid_parent_answers':second_count,'policies':metrics,'component_conditions':checks,
        'component_passed':all(checks.values()),'actual_fresh_generation_calls':generated,
        'counterfactual_generation_not_counted_as_primary_policy_cost':True,'neural_routing_evaluated':False,'model_promoted':False}
    if not equivalent(result,trace['verification']) or not equivalent(result,admin['verification']):
        raise ValueError('Audit independently reconstructed different scores or costs')
    if 'torch' in sys.modules:raise ValueError('The audit must not load a tensor engine')
    return {'success':True,'protocol_id':spec.protocol_id(),'administrative_report_id':admin_id,
        'diagnostic_report_id':trace_id,'snapshot_id':spec.SNAPSHOT_ID,'commitment_id':commitment_id,
        'decision_files_verified':128,'verification':result,'tensor_engine_loaded':False,
        'generation_performed':False,'training_performed':False,'model_promoted':False,
        'initial_supervisor_success':not recovered,'saved_evidence_recovered':recovered,
        'recovery_observation_id':observation_id,'original_child_receipt_captured':not recovered,
        'timing_scope':'Sum of observed model.generate times for each policy, excluding counterfactual calls; not distributed end-to-end speed',
        'source_revision':admin['source_revision']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--source-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--recover-saved-report',action='store_true')
    parser.add_argument('--original-source',type=Path)
    args=parser.parse_args();result=audit(args.directory,args.source_directory,
        recover_saved_report=args.recover_saved_report,original_source=args.original_source)
    if args.output:
        from src.network.telemetry import atomic_json
        atomic_json(args.output,result)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()
