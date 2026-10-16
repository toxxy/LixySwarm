"""Audit actual native food programs, signed gradients, gates and preserved history."""
import json
from pathlib import Path

from experiments.audit_own_network import signed_output
from src.learning.swarm_policy import definition,state_id,verify_contributions,verdict
from src.learning.own_food_catalog import descriptor,policy_identifier
from src.network.artifact_store import ArtifactStore,digest_file
from src.network.own_network_settings import OwnNetworkSettings
from src.network.own_state import verify_transition


def audit(directory,program_ids,laptop,*,initial_state_id,initial_step,parent_contract_id):
    node=Path(directory);owner=json.loads((node/'diagnostics.json').read_text())['node_id']
    store=ArtifactStore(node/'public/own/artifacts',max_artifact_bytes=16*1024**2,max_total_bytes=1024**3)
    parent=initial_state_id;step=initial_step;contract=parent_contract_id;all_programs=[];jobs=set()
    for program_id in program_ids:
        selected=json.loads((node/'own-network/admissions'/program_id/'selection.json').read_text())
        settings=OwnNetworkSettings(**selected['controls']);settings.validate()
        identifier=policy_identifier(program_id);p=definition(identifier)
        if (settings.program_id!=program_id or settings.parent_program_id!=contract
                or settings.initial_state_id!=parent or settings.initial_cursor!=0):
            raise ValueError('Own food changed its previous program or optimizer lineage')
        root=node/'own-network/programs'/settings.contract();ledger=json.loads((root/'ledger.json').read_text())
        if ledger['pending'] is not None or ledger['training_policy_id']!=identifier or ledger['contract_id']!=settings.contract():
            raise ValueError('Own food still has pending work or changed its contract')
        cursor=0;accepted=0;rejections=0;records=[];reports={};audits={};transfers=[]
        for index in range(ledger['round']):
            row=json.loads((root/'rounds'/f'{index:08d}.json').read_text())
            if row['round']!=index or row['cursor']!=cursor or row['state_id']!=parent or not row['autonomous']:
                raise ValueError('Own food window changed its durable cursor or state')
            contributions=[row['contributions'][v['shard_id']] for v in row['assignments']]
            outputs=verify_contributions(contributions,model_id=row['model_id'],training_state_id=parent,
                requester_id=owner,training_policy_id=identifier)
            end=cursor
            for assignment,contribution,output in zip(row['assignments'],contributions,outputs):
                key=assignment['shard_id'];result=contribution['result'];worker=contribution['worker_id']
                if (assignment['start']!=end or key!=str(end) or assignment['worker_id']!=worker
                        or any(output[k]!=assignment[k] for k in ['start','count'])
                        or result['job_id'] in jobs or row['work'][key]['result']!=result):
                    raise ValueError('Own food duplicated a range or used another worker receipt')
                jobs.add(result['job_id']);end+=assignment['count']
                if digest_file(store.object_path(output['gradient_id']))!=(output['gradient_id'],output['gradient_bytes']):
                    raise ValueError('Own food gradient differs from its physical signed content')
                route=row['work'][key]['route']
                if worker!=owner and (not route['encrypted'] or not route['public_internet']):
                    raise ValueError('Own food remote gradient used a non-public route')
            result=row['candidate']['result'];output=signed_output(result,owner,owner)
            if (output['training_policy_id']!=identifier or output['state_id']!=parent
                    or state_id(output['model_id'],output['optimizer_id'],step)!=parent
                    or output['previous_optimizer_step']!=step or output['candidate_optimizer_step']!=step+1
                    or output['accepted']!=row['accepted'] or row['accepted']!=verdict(output['before'],output['after'],identifier)
                    or output['prediction_bytes']!=sum(v['tokens'] for v in outputs)):
                raise ValueError('Own food candidate changed its policy or optimizer history')
            report_path=store.object_path(output['evaluation_id']);raw=report_path.read_bytes()
            if digest_file(report_path)!=(output['evaluation_id'],output['evaluation_bytes']):
                raise ValueError('Own food evaluation differs from its signed report')
            report=json.loads(raw);reports[output['evaluation_id']]=raw.decode()
            if report['contributions']!=contributions:raise ValueError('Own food evaluated another contribution set')
            if row['accepted']:
                transition=verify_transition(result,raw);parent=transition['state_id'];step+=1;accepted+=1;rejections=0
                adopted=signed_output(row['adoptions'][owner],owner,owner)
                local=json.loads((node/'public/own/audits'/(parent+'.json')).read_text());remote=laptop['audits'].get(parent)
                for checked in [local,remote]:
                    if (not checked or not checked['success'] or checked['state_id']!=parent
                            or checked['training_policy_id']!=identifier or checked['optimizer_step']!=step
                            or checked['optimizer_states']!=175 or not checked['gradient_replay_verified']
                            or not checked['saved_state_unchanged'] or checked['new_gradient_computed'] or checked['cuda_initialized']):
                        raise ValueError('Own food accepted state lacks independent physical replay')
                if adopted['state_id']!=parent or adopted['audit']!=local:raise ValueError('Own food local adoption differs from its replay')
                audits[parent]={'coordinator':local,'other_physical_pc':remote}
                for transfer in laptop['adoptions'][parent]['transfers']:
                    if not transfer['route']['encrypted'] or not transfer['route']['public_internet']:
                        raise ValueError('Own food learned replica used a non-public route')
                    transfers.append(transfer)
                for prefix in ['candidate_model','candidate_optimizer']:
                    if digest_file(store.object_path(output[prefix+'_id']))!=(output[prefix+'_id'],output[prefix+'_bytes']):
                        raise ValueError('Own food lost an accepted learned state')
            else:
                rejections+=1
                if row['adoptions']:raise ValueError('Own food rejected state was adopted')
            if row['resulting_state_id']!=parent:raise ValueError('Own food recorded another resulting state')
            cursor=end;records.append(row)
        if (cursor>p['training_units'] or ledger['cursor']!=cursor or ledger['accepted']!=accepted
                or ledger['rejections']!=rejections or ledger['state_id']!=parent):
            raise ValueError('Own food ledger contradicts its completed windows')
        all_programs.append({'program_id':program_id,'training_policy_id':identifier,'contract_id':settings.contract(),
            'descriptor':descriptor(program_id),'ledger':ledger,'records':records,'evaluation_reports_raw':reports,
            'independent_cpu_audits':audits,'public_replica_transfers':transfers})
        contract=settings.contract()
    current=json.loads((node/'own-swarm.json').read_text())
    if (state_id(current['model_id'],current['optimizer_id'],current['step'])!=parent or current['step']!=step
            or laptop['own_swarm']['state_id']!=parent):raise ValueError('Physical selected replicas disagree with audited food')
    for key in ['model_id','optimizer_id']:
        remote=laptop['selected_objects'][current[key]]
        if remote['sha256']!=current[key] or digest_file(store.object_path(current[key]))!=(current[key],remote['bytes']):
            raise ValueError('Other physical PC did not preserve exact own learned content')
    return {'success':True,'programs':all_programs,'physical_training_pcs':2,
        'rounds':sum(v['ledger']['round'] for v in all_programs),'accepted':sum(v['ledger']['accepted'] for v in all_programs),
        'rejected':sum(v['ledger']['round']-v['ledger']['accepted'] for v in all_programs),
        'final_state_id':parent,'final_optimizer_step':step,'new_gradient_jobs':len(jobs),
        'remote_replay_evidence':'Signed local adoption plus public replay/transfer reports read over administrative SSH; remote adoption receipts checked live but not retained by this coordinator version',
        'larger_physical_network_or_generalization_claimed':False}
