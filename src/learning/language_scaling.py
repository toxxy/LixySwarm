"""Compare fixed public training work; report measured gains or slowdowns."""
import math

from src.learning.public_conversation import public_corpus,corpus_digest
from src.network.work_protocol import ResultReceipt
from src.release.language_versions import verify_chain,verify_contributions


def verify_timed_run(report):
    if report.get('success') is not True or report.get('operation')!='language.gradient.v2':
        raise ValueError('Scaling requires completed versioned gradient trials')
    if report.get('eligible_for_uncontended_scaling_comparison') is False or report.get('concurrent_initial_model_copy') is True:
        raise ValueError('A trial with known initial-download contention is only a warm-up')
    head=report['head'];history=report.get('history',[head]);verify_chain(history)
    if history[-1]!=head:raise ValueError('Trial head differs from its verified history')
    results=report['signed_results']
    proof=verify_contributions(results,base_name=report['base_name'],
        adapter_id=head['release']['model_artifact_id'],dataset_id=corpus_digest(public_corpus()),require_distributed=False)
    if set(report['physical_workers'])!=set(proof['physical_workers']):
        raise ValueError('Reported physical workers differ from signed contributions')
    rows=report['contributions'];by_job={row['receipt']['job_id']:row for row in rows}
    if len(by_job)!=len(rows) or len(rows)!=len(results):raise ValueError('Missing or repeated trial timing rows')
    shards=[]
    for result in results:
        receipt=ResultReceipt.from_dict(result['receipt']);row=by_job.get(result['job_id'])
        if row is None or row['receipt']!=result['receipt'] or row['worker_id']!=receipt.worker_node_id:
            raise ValueError('Timing row has a different signed worker or job')
        if report.get('requester_id') is not None and receipt.requester_node_id!=report['requester_id']:
            raise ValueError('Timing report belongs to another requester')
        if row.get('encrypted') is not True:raise ValueError('Trial transport was not encrypted')
        for key,value in result['output'].items():
            if row.get(key)!=value:raise ValueError('Timing row differs from its signed contribution')
        seconds=row['seconds']
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<=0:
            raise ValueError('Invalid per-shard observed time')
        shards.append(tuple(result['output'][key] for key in ['start','count','tokens']))
    wall=report['contribution_wall_seconds']
    if (type(wall) not in (int,float) or not math.isfinite(wall) or wall<=0
            or any(row['seconds']>wall+.01 for row in rows)):
        raise ValueError('Invalid total observed trial time')
    return proof,sorted(shards),wall


def compare_runs(baseline,distributed):
    before,first_shards,first_seconds=verify_timed_run(baseline)
    after,second_shards,second_seconds=verify_timed_run(distributed)
    if (len(before['physical_workers'])!=1 or len(after['physical_workers'])<2
            or not set(before['physical_workers']).issubset(after['physical_workers'])):
        raise ValueError('Compare one physical worker with the same worker plus another')
    if (baseline['base_name']!=distributed['base_name'] or baseline['head']!=distributed['head']
            or first_shards!=second_shards
            or baseline.get('requester_id')!=distributed.get('requester_id')
            or baseline.get('receive_bandwidth_mbps')!=distributed.get('receive_bandwidth_mbps')):
        raise ValueError('Scaling trials must use identical model, data, ranges and token counts')
    before_slots=baseline.get('worker_slot_limits');after_slots=distributed.get('worker_slot_limits')
    if (before_slots is None)!=(after_slots is None) or before_slots is not None and any(
            after_slots.get(worker)!=limit for worker,limit in before_slots.items()):
        raise ValueError('Adding workers cannot also increase the original worker slot limit')
    ratio=first_seconds/second_seconds
    return {'scope':'One observed Internet training comparison; excludes initial model download and candidate evaluation; no general scaling or model promotion claim',
        'base_name':baseline['base_name'],'release_id':baseline['head']['release']['release_id'],
        'model_id':baseline['head']['release']['model_artifact_id'],'dataset_id':corpus_digest(public_corpus()),
        'tokens':before['tokens'],'receive_bandwidth_mbps':baseline.get('receive_bandwidth_mbps'),
        'shards':[{'start':start,'count':count,'tokens':tokens} for start,count,tokens in first_shards],
        'one_pc':{'workers':before['physical_workers'],'seconds':first_seconds,'verified_tokens_per_second':before['tokens']/first_seconds},
        'multiple_pcs':{'workers':after['physical_workers'],'seconds':second_seconds,'verified_tokens_per_second':after['tokens']/second_seconds},
        'observed_speedup':ratio,'training_throughput_improved':ratio>1,'new_model_promoted':False}
