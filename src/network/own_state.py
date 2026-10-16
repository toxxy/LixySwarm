"""Verified own-state adoption; weights and optimizer advance together."""
import json
from pathlib import Path

from src.learning.swarm_distributed import WORKERS,policy_id,state_id,verify_contributions,verdict,definition
from src.learning.public_math import digest
from .work_protocol import WorkResult,ResultReceipt
from .artifact_store import ArtifactStore,digest_file
from .own_swarm import OwnSwarmSettings
from .client_state import ClientSettings,NodeStateLock
from .telemetry import atomic_json


def verify_transition(value,evaluation_raw,*,workers=WORKERS):
    result=WorkResult.from_dict(value);receipt=ResultReceipt.from_dict(result.receipt)
    evaluator=receipt.worker_node_id
    if (result.status!='ok' or evaluator not in workers or not receipt.verify(result,
            expected_worker_id=evaluator,expected_requester_id=evaluator)):
        raise ValueError('Own transition is not signed by a known physical evaluator')
    output=result.output
    required={'accepted','after','before','candidate_model_bytes','candidate_model_id',
        'candidate_optimizer_bytes','candidate_optimizer_id','candidate_optimizer_step',
        'evaluation_bytes','evaluation_id','model_id','optimizer_id','prediction_bytes',
        'previous_optimizer_step','state_id','training_policy_id'}
    if set(output)!=required or output['training_policy_id']!=policy_id() or output['accepted'] is not True:
        raise ValueError('Own transition did not pass the selected learning policy')
    import hashlib
    if (len(evaluation_raw)!=output['evaluation_bytes'] or len(evaluation_raw)>64*1024
            or hashlib.sha256(evaluation_raw).hexdigest()!=output['evaluation_id']):
        raise ValueError('Own transition evaluation differs from its signed receipt')
    evaluation=json.loads(evaluation_raw)
    for key in ['accepted','after','before','candidate_optimizer_step','previous_optimizer_step',
                'prediction_bytes','state_id','model_id','optimizer_id','training_policy_id']:
        if evaluation[key]!=output[key]:raise ValueError('Own transition report has inconsistent state or quality')
    if not verdict(output['before'],output['after']):raise ValueError('Own transition quality gate failed')
    parent=state_id(output['model_id'],output['optimizer_id'],output['previous_optimizer_step'])
    following=state_id(output['candidate_model_id'],output['candidate_optimizer_id'],output['candidate_optimizer_step'])
    if (parent!=output['state_id'] or evaluation['candidate_state_id']!=following
            or output['candidate_optimizer_step']!=output['previous_optimizer_step']+1):
        raise ValueError('Own transition resets or mixes optimizer history')
    contributions=verify_contributions(evaluation['contributions'],model_id=output['model_id'],
        training_state_id=parent,requester_id=evaluator,workers=workers)
    if sum(v['tokens'] for v in contributions)!=output['prediction_bytes']:
        raise ValueError('Own transition prediction-byte total is incompatible')
    for name in ['candidate_model_bytes','candidate_optimizer_bytes']:
        if type(output[name]) is not int or not 0<output[name]<=definition()['maximum_artifact_bytes']:
            raise ValueError('Own transition artifact exceeds its bound')
    return {'version':1,'parent_state_id':parent,'state_id':following,'model_id':output['candidate_model_id'],
        'optimizer_id':output['candidate_optimizer_id'],'step':output['candidate_optimizer_step'],
        'evaluation_id':output['evaluation_id'],'training_policy_id':policy_id(),
        'evaluator_id':evaluator,'candidate_result':value}


def activate(directory,candidate_result,audit):
    """Owner-selected transition after local independent tensor verification.

    This pilot trusts the two known operators. It does not introduce a general
    open-network consensus rule or grant permission to unknown publishers.
    """
    directory=Path(directory).resolve()
    if (directory.is_relative_to(Path.home()/'Dropbox')
            or directory.is_relative_to(Path(__file__).resolve().parents[2])):
        raise ValueError('Own learned state must remain outside the code checkout and Dropbox')
    store=ArtifactStore(directory/'public/own/artifacts',
        max_artifact_bytes=16*1024**2,max_total_bytes=1024**3)
    output=candidate_result['output']
    evaluation=store.object_path(output['evaluation_id']).read_bytes()
    transition=verify_transition(candidate_result,evaluation)
    if (audit.get('success') is not True or audit.get('state_id')!=transition['state_id']
            or audit.get('model_id')!=transition['model_id'] or audit.get('optimizer_id')!=transition['optimizer_id']
            or audit.get('training_policy_id')!=policy_id() or audit.get('saved_state_unchanged') is not True
            or audit.get('gradient_replay_verified') is not True or audit.get('optimizer_step')!=transition['step']):
        raise ValueError('Own state must have an independent local recovery and gradient replay audit')
    for key,id_key in [('candidate_model','model_id'),('candidate_optimizer','optimizer_id')]:
        if digest_file(store.object_path(transition[id_key]))!=(transition[id_key],output[key+'_bytes']):
            raise ValueError('Own activation requires the complete verified learned state')
    with NodeStateLock(directory/'own-state-selection'):
        settings=OwnSwarmSettings.load(directory);controls=ClientSettings.load(directory)
        previous=(directory/'own-swarm.json').read_bytes()
        if not settings.enabled or not controls.connected or controls.paused:
            raise ValueError('Own state selection is not currently enabled by the operator')
        current=state_id(settings.model_id,settings.optimizer_id,settings.step)
        history=directory/'public/own/history'/(transition['state_id']+'.json')
        if current==transition['state_id']:
            if not history.is_file():raise ValueError('Own selected state has no recovery history')
            return {'success':True,'already_selected':True,'state_id':current,'optimizer_step':settings.step}
        if current!=transition['parent_state_id']:raise ValueError('Own transition does not extend the selected state')
        if history.exists() and json.loads(history.read_text())!=transition:
            raise ValueError('Own transition history already contains different provenance')
        atomic_json(history,transition)
        proof=store.import_file(history,kind='other')
        if (directory/'own-swarm.json').read_bytes()!=previous:
            raise ValueError('Own local selection changed during activation')
        settings.model_id=transition['model_id'];settings.optimizer_id=transition['optimizer_id'];settings.step=transition['step']
        settings.save(directory)
        return {'success':True,'already_selected':False,'parent_state_id':current,
            'state_id':transition['state_id'],'model_id':settings.model_id,'optimizer_id':settings.optimizer_id,
            'optimizer_step':settings.step,'transition_artifact_id':proof.artifact_id,
            'scope':'Known-operator own-state adoption; not open consensus or autonomous scheduling'}
