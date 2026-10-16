"""Verify native own-cycle receipts, causal state ancestry and physical adoption evidence."""
import argparse
import json
from pathlib import Path

from src.learning.swarm_distributed import WORKERS,policy_id,state_id,verify_contributions,verdict
from src.network.artifact_store import ArtifactStore,digest_file
from src.network.own_state import verify_transition
from src.network.own_learning_settings import OwnLearningSettings
from src.network.own_swarm import OwnSwarmSettings
from src.network.work_protocol import WorkResult,ResultReceipt


def verified_result(value,worker,owner):
    result=WorkResult.from_dict(value)
    if (result.status!='ok' or not result.receipt or not ResultReceipt.from_dict(result.receipt).verify(result,
            expected_worker_id=worker,expected_requester_id=owner)):
        raise ValueError('Native cycle has an invalid physical receipt')
    return result.output


def verify_cycles(directory,protocol,*,laptop=None,allow_running=False):
    node=Path(directory);root=node/'own-learning';ledger=json.loads((root/'ledger.json').read_text())
    settings=OwnLearningSettings(**protocol['coordinator_controls']);settings.validate()
    if (ledger['contract_id']!=settings.contract() or protocol['coordinator_contract_id']!=settings.contract()
            or ledger['training_policy_id']!=policy_id() or not 0<=ledger['round']<=settings.maximum_rounds):
        raise ValueError('Native program differs from its prospective contract')
    if not allow_running and (ledger['pending'] is not None or ledger['status'] not in {'complete','no_improvement','curriculum_complete'}):
        raise ValueError('Native program is still live; preserve and observe the same process')
    owner=settings.owner_id;store=ArtifactStore(node/'public/own/artifacts',max_artifact_bytes=16*1024**2,max_total_bytes=1024**3)
    parent=settings.initial_state_id;step=protocol['initial_optimizer_step'];cursor=settings.initial_cursor
    accepted=0;rejections=0;records=[];audits={};transfers=[];objects={}
    for index in range(ledger['round']):
        row=json.loads((root/'rounds'/f'{index:04d}.json').read_text())
        expected=[{'worker_id':worker,'start':cursor+i*settings.documents_per_worker,
            'count':settings.documents_per_worker} for i,worker in enumerate(WORKERS)]
        if (row['round']!=index or row['cursor']!=cursor or row['state_id']!=parent
                or row['assignments']!=expected or not row['autonomous'] or row['finished_at']<row['started_at']):
            raise ValueError('Native cycle reset or changed its curriculum')
        contributions=list(row['contributions'].values())
        verify_contributions(contributions,model_id=row['model_id'],training_state_id=parent,requester_id=owner)
        for assignment in expected:
            worker=assignment['worker_id'];record=row['contributions'][worker]
            output=record['result']['output'];work=row['work'][worker]
            if any(output[k]!=assignment[k] for k in ['start','count']) or work['result']!=record['result']:
                raise ValueError('Native cycle result differs from the assigned physical range')
            if worker!=owner and (not work['route']['public_internet'] or not work['route']['encrypted']):
                raise ValueError('Native gradient did not use encrypted public Internet')
        result=row['candidate']['result'];output=verified_result(result,owner,owner)
        if (output['state_id']!=parent or output['model_id']!=row['model_id']
                or output['previous_optimizer_step']!=step or output['training_policy_id']!=policy_id()
                or output['accepted']!=row['accepted'] or row['accepted']!=verdict(output['before'],output['after'])):
            raise ValueError('Native cycle changed its fixed evaluator or optimizer ancestry')
        evaluation_file=store.object_path(output['evaluation_id'])
        if digest_file(evaluation_file)!=(output['evaluation_id'],output['evaluation_bytes']):
            raise ValueError('Native evaluation differs from its signed candidate receipt')
        evaluation=json.loads(evaluation_file.read_text())
        if ({v['worker_id']:v for v in evaluation['contributions']}!=row['contributions']
                or any(evaluation[k]!=output[k] for k in ['state_id','model_id','optimizer_id','accepted','before','after'])):
            raise ValueError('Native evaluator used another state, contribution set or gate result')
        if row['accepted']:
            transition=verify_transition(result,evaluation_file.read_bytes())
            if set(row['adoptions'])!=set(WORKERS):raise ValueError('Accepted cycle lacks both physical adoptions')
            parent=transition['state_id'];step+=1;accepted+=1;rejections=0;audits[parent]={}
            for worker,value in row['adoptions'].items():
                adoption=verified_result(value,worker,owner)
                if adoption['state_id']!=parent or adoption['optimizer_step']!=step:
                    raise ValueError('Physical adoption selected another state or optimizer step')
                if worker==owner:
                    audit=json.loads((node/'public/own/audits'/(parent+'.json')).read_text())
                else:
                    audit=adoption.get('audit') or (laptop or {}).get('audits',{}).get(parent)
                if (not audit or audit['success'] is not True or audit['state_id']!=parent
                        or audit['model_id']!=transition['model_id'] or audit['optimizer_id']!=transition['optimizer_id']
                        or audit['optimizer_step']!=step or audit['optimizer_states']!=175
                        or audit['cuda_initialized'] or not audit['gradient_replay_verified']
                        or not audit['saved_state_unchanged'] or audit['new_gradient_computed']):
                    raise ValueError('Physical adoption lacks its independent CPU gradient replay')
                audits[parent][worker]=audit
                for transfer in adoption.get('transfers',[]):
                    if not transfer['route']['public_internet'] or not transfer['route']['encrypted']:
                        raise ValueError('Learned state replica used a non-public route')
                    transfers.append(transfer)
            for kind in ['candidate_model','candidate_optimizer']:
                identifier=output[kind+'_id'];size=output[kind+'_bytes']
                if digest_file(store.object_path(identifier))!=(identifier,size):
                    raise ValueError('Accepted own state no longer matches its signed content')
                objects[identifier]={'bytes':size,'sha256_verified':True}
        else:
            rejections+=1
            if row['adoptions']:raise ValueError('Rejected cycle was adopted')
        if row['resulting_state_id']!=parent:raise ValueError('Native cycle recorded another selected state')
        if ledger['history'][index]!={'round':index,'accepted':row['accepted'],'state_id':parent,
                'cursor':cursor,'finished_at':row['finished_at']}:
            raise ValueError('Native ledger history does not describe its completed cycle')
        records.append(row);cursor+=2*settings.documents_per_worker
    if (ledger['cursor']!=cursor or ledger['accepted']!=accepted or ledger['state_id']!=parent
            or ledger['rejections']!=rejections or len(ledger['history'])!=len(records)):
        raise ValueError('Durable native progress does not match completed signed cycles')
    if not allow_running:
        selected=OwnSwarmSettings.load(node)
        if state_id(selected.model_id,selected.optimizer_id,selected.step)!=parent or selected.step!=step:
            raise ValueError('Completed program did not retain its last accepted own state')
        if laptop and (laptop['own_swarm']['state_id']!=parent or laptop['own_swarm']['optimizer_step']!=step):
            raise ValueError('The second physical replica did not retain the completed own state')
    return {'verified_completed_cycles':len(records),'accepted':accepted,'rejected':len(records)-accepted,
        'consecutive_rejections':rejections,'cursor':cursor,'state_id':parent,'optimizer_step':step,
        'status':ledger['status'],'pending':ledger['pending'] is not None,'records':records,
        'independent_audits':audits,'replica_transfers':transfers,'verified_learned_objects':objects,
        'all_replays_model_bitwise_equal':bool(audits) and all(a['model_tensors_bitwise_equal'] for per_state in audits.values() for a in per_state.values()),
        'training_policy_id':policy_id(),'coordinator_contract_id':settings.contract(),
        'fresh_blind_evaluation':False,'general_chat_quality_proved':False,'one_vs_two_speedup_proved':False,
        'learned_metabolism_or_open_consensus_proved':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--node-directory',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=Path('experiments/reports/own_continuous_protocol_20261015.json'))
    p.add_argument('--laptop-report',type=Path)
    p.add_argument('--allow-running',action='store_true');a=p.parse_args()
    report=verify_cycles(a.node_directory,json.loads(a.protocol.read_text()),
        laptop=json.loads(a.laptop_report.read_text()) if a.laptop_report else None,allow_running=a.allow_running)
    print(json.dumps({k:v for k,v in report.items() if k not in {'records','independent_audits','replica_transfers','verified_learned_objects'}},sort_keys=True))


if __name__=='__main__':main()
