"""Audit the cold native own-lineage trial without counting profiles as PCs."""
import json
import sys
from pathlib import Path

from src.learning.swarm_policy import verify_contributions,network_policy_id
from src.network.own_catalog import ANCHOR,COORDINATOR,anchor_id,dependencies
from src.network.artifact_store import digest_file


def audit(report,store=None):
    cold=report['cold_app'];restart=report['restart']
    if (report['physical_training_pcs']!=2 or not cold['success'] or not cold['fresh_profile']
            or not cold['internet'] or cold['extra_physical_pc'] or cold['model_state_copied_by_admin']
            or cold['anchor_id']!=anchor_id() or cold['state_id']!=ANCHOR['state_id']
            or cold['optimizer_step']!=1033 or not cold['resumed_same_identity_and_state']
            or cold['calculation_engines_loaded'] or cold['pause_preserved_partial_bytes']<=0
            or not cold['no_selection_before_validation']):
        raise ValueError('Cold application did not preserve its consent or own lineage')
    prepared=cold['preparation'];checked=prepared['audit']
    if (not prepared['success'] or prepared['state_id']!=ANCHOR['state_id']
            or prepared['engine_manifest_sha256']!=cold['cpu_engine_manifest_sha256']
            or checked['optimizer_step']!=1033 or checked['optimizer_states']!=175
            or not checked['saved_state_unchanged'] or checked['new_gradient_computed']
            or checked['cuda_initialized'] or not checked['causality']['passed']):
        raise ValueError('Own initialization and optimizer did not pass independent CPU validation')
    if {t['artifact_id']:t['bytes'] for t in prepared['transfers']}!=dict(dependencies()):
        raise ValueError('Cold profile did not acquire every pinned own dependency')
    for key in ['model_id','optimizer_id','initialization_protocol_id']:
        if checked[key]!=ANCHOR[key]:raise ValueError('Cold validator acquired another lineage')
    for t in prepared['transfers']:
        if not t['route']['encrypted'] or not t['route']['public_internet']:
            raise ValueError('Cold learned state arrived through a non-public route')
    if (not restart['success'] or not restart['same_identity_and_learned_state']
            or restart['old_pid']==restart['new_pid'] or restart['optimizer_step']!=1033
            or restart['physical_pcs']!=2 or not restart['restarted_profile_is_not_an_extra_pc']):
        raise ValueError('Cold profile did not survive a complete native process restart')
    results=[]
    for trial in [cold['native_gradient_work'],restart['native_gradient_work']]:
        value=trial['result'];output=value['output']
        record={'worker_id':cold['physical_profile_node_id'],'result':value}
        verified=verify_contributions([record],model_id=ANCHOR['model_id'],training_state_id=ANCHOR['state_id'],
            requester_id=COORDINATOR,training_policy_id=network_policy_id())[0]
        if (not trial['route']['encrypted'] or not trial['route']['public_internet']
                or verified['count']!=64 or verified['start'] not in {3584,3648}):
            raise ValueError('Cold native contribution did not use the expected public work path')
        if store and digest_file(store.object_path(output['gradient_id']))!=(output['gradient_id'],output['gradient_bytes']):
            raise ValueError('Cold own gradient differs from its signed content')
        results.append(verified)
    if results[0]['start']==results[1]['start']:raise ValueError('Restart proof reused its earlier manual work range')
    return {'success':True,'physical_pcs':2,'cold_native_work_results':2,'own_optimizer_step':1033,
        'cold_profile_id':cold['physical_profile_node_id'],'full_process_restart_verified':True,
        'new_accepted_updates_claimed':0,'scope':'Own native onboarding and persistence; manual gradients are not promoted updates'}


if __name__=='__main__':print(json.dumps(audit(json.loads(Path(sys.argv[1]).read_text())),indent=2))
