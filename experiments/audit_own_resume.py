"""Check a preserved own bootstrap profile catching up to current signed learning."""
from desktop.own_self_test import followed_history
from src.learning.swarm_policy import verify_contributions
from src.network.own_catalog import ANCHOR,COORDINATOR,anchor_id,own_engine
from src.network.artifact_store import digest_file


def audit(value,node,store):
    cold=value['resumed_app'];prior=value['interrupted_attempt']['snapshot']
    if (not cold['success'] or cold['fresh_profile'] or not cold['resumed_unselected_profile']
            or prior['own_selection_exists'] or not cold['no_selection_before_validation']
            or cold['model_state_copied_by_admin'] or cold['calculation_engines_loaded']
            or cold['physical_profile_node_id']!=value['preserved_profile_node_id']
            or cold['state_id']!=ANCHOR['state_id'] or cold['anchor_id']!=anchor_id()
            or not cold['resumed_same_identity_and_state'] or cold['pause_preserved_partial_bytes']<=0):
        raise ValueError('Own resume changed its identity, selected state or preparation contract')
    prepared=cold['preparation'];checked=prepared['audit']
    if (not prepared['success'] or checked['optimizer_step']!=1033 or checked['optimizer_states']!=175
            or checked['new_gradient_computed'] or not checked['saved_state_unchanged']
            or not checked['causality']['passed'] or prepared['engine_manifest_sha256']!=own_engine().manifest_sha256):
        raise ValueError('Own resumed bootstrap did not independently validate the pinned optimizer')
    for transfer in prepared['transfers']:
        if not transfer['route']['public_internet'] or not transfer['route']['encrypted']:
            raise ValueError('Own resume copied public state through a non-public path')
    final=cold['final_own_swarm'];chain=followed_history(node,ANCHOR['state_id'],final['state_id'])
    if chain!=cold['followed_states'] or len(chain)!=4 or final['optimizer_step']!=1037:
        raise ValueError('Own late participant did not recover all four accepted successor states')
    work=cold['native_gradient_work'];result=work['result'];output=result['output']
    record={'worker_id':cold['physical_profile_node_id'],'result':result}
    verify_contributions([record],model_id=final['model_id'],training_state_id=final['state_id'],
        requester_id=COORDINATOR,training_policy_id=final['training_policy_id'])
    if (output['count']!=16 or output['start']!=0 or not work['route']['encrypted']
            or not work['route']['public_internet'] or digest_file(store.object_path(output['gradient_id']))!=(output['gradient_id'],output['gradient_bytes'])):
        raise ValueError('Own resumed participant did not produce the signed current-state gradient')
    return {'success':True,'physical_training_pcs':2,'preserved_identity':True,
        'bootstrap_optimizer_step':1033,'followed_accepted_states':len(chain),'current_optimizer_step':1037,
        'current_state_id':final['state_id'],'native_gradient_results':1,'new_accepted_updates_claimed':0,
        'scope':'Resumed originally empty profile, independent successor recovery and current-state native work; not a third physical PC'}
