"""Physical-node evaluation of bounded public contributions; no signing key."""
from dataclasses import asdict
import json
import math
import shutil

from src.learning.public_quality import preservation_verdict as factual_verdict
from src.learning.public_language_gate import preservation_verdict as instruction_verdict
from src.learning.language_policy import policy_id,descriptor_policy,enabled_policy
from src.release.language_versions import verify_contributions,gate_id
from src.release import language_versions
from src.release.adapter_format import validate_adapter
from .artifact_store import digest_file
from .compute_process import ComputeProcessError
from .language_install import materialize_version
from .public_knowledge import managed_public_bytes
from .telemetry import atomic_json
from .work_protocol import WorkResult,ResultReceipt

OPERATION='language.candidate.v1'


def run_candidate(training,payload,execution,*,policy_version=1):
    expected={'base_name','release_id','adapter_id','dataset_id','contributions','learning_rate'}
    if policy_version>=2:expected.add('training_policy_id')
    if not isinstance(payload,dict) or set(payload)!=expected:raise ValueError('Invalid public language candidate payload')
    if policy_version>=2 and (not enabled_policy(training.engine,policy_version) or payload['training_policy_id']!=policy_id(policy_version)):
        raise ComputeProcessError('Public batch candidate policy is not locally enabled')
    if (not training.ready or not training.engine.candidates_enabled or not training.knowledge.allowed
            or execution.work.origin_node_id not in training.engine.requesters
            or payload['base_name']!=training.engine.base_name):
        raise ComputeProcessError('Language candidate family or requester is not locally approved')
    if type(payload['learning_rate']) not in (int,float) or not 1e-6<=payload['learning_rate']<=1e-3:
        raise ValueError('Candidate learning rate exceeds its local bound')
    requirements=execution.resource_requirements()
    if not requirements.gpu_required or requirements.ram_gb<training.required_ram():
        raise ComputeProcessError('Language candidate requires its full GPU/RAM lease')
    head=training._head()
    if head.get('descriptor') and descriptor_policy(head['descriptor'])>policy_version:
        raise ComputeProcessError('This accepted parent requires the batch candidate policy')
    if payload['release_id']!=head['release']['release_id'] or payload['adapter_id']!=head['release']['model_artifact_id']:
        raise ValueError('Language candidate refers to a stale parent')
    proof=verify_contributions(payload['contributions'],base_name=payload['base_name'],
        adapter_id=payload['adapter_id'],dataset_id=payload['dataset_id'],require_distributed=False,policy_version=policy_version)
    permission=asdict(training.settings.policy);engine=asdict(training.engine)
    def allowed():
        return (training.ready and training.knowledge.allowed and asdict(training.settings.policy)==permission
            and asdict(training.engine)==engine and training._head()['release']['release_id']==payload['release_id'])
    def check():
        execution.raise_if_cancelled()
        if not allowed():raise ComputeProcessError('Language candidate permission or parent changed')
    # Include incoming gradients, weights and the bounded report before starting.
    reserve=33*1024**2+sum(row['output']['gradient_bytes'] for row in payload['contributions'])
    if managed_public_bytes(training.directory)+reserve>permission['max_disk_gb']*1024**3:
        raise ComputeProcessError('Language candidate exceeds the local disk consent')
    base=materialize_version(training.directory,payload['base_name'],payload['release_id'],allowed=allowed)
    contributions=[]
    for row in payload['contributions']:
        check();result=WorkResult.from_dict(row);output=result.output
        worker=ResultReceipt.from_dict(result.receipt).worker_node_id
        identifier=output['gradient_id'];store=training.families.store
        if not store.has(identifier):
            training.families.artifacts.fetch(identifier,peer_id=worker,expected_size=output['gradient_bytes'],
                timeout_s=min(120,execution.remaining_s()),cancel_event=execution.cancel_event)
        path=store.object_path(identifier)
        if digest_file(path)!=(identifier,output['gradient_bytes']):raise ValueError('Candidate gradient hash or size mismatch')
        contributions.append({'path':str(path),**{k:output[k] for k in
            ['gradient_id','gradient_bytes','base_hash','adapter_id','dataset_id','start','count','tokens']}})
        if policy_version==3:contributions[-1]['quantization_relative_l2_error']=output['quantization_relative_l2_error']
    check();directory=training.directory/'language-candidates'/execution.work.job_id
    directory.mkdir(parents=True,exist_ok=False,mode=0o700);path=directory/'adapter.safetensors'
    request={'action':'candidate','base_name':payload['base_name'],'base_directory':str(base),
        'adapter_file':str(base/'adapter.safetensors'),'adapter_id':payload['adapter_id'],'dataset_id':payload['dataset_id'],
        'parent_release_id':payload['release_id'],'contributions':contributions,'learning_rate':payload['learning_rate'],
        'output_file':str(path),'cpu_threads':max(1,min(4,int(training.resources['cpu_cores']*permission['cpu_percent']/100))),
        'gpu_fraction':permission['gpu_percent']/100/max(1,permission['max_gpu_jobs']),'device':'cuda','deadline':execution.deadline}
    if policy_version>=2:request['training_policy_id']=policy_id(policy_version)
    with training.lock:training.active+=1
    training.changed.set()
    complete=False
    try:
        value=training.supervisor.run(training.engine.command,request,execution,allowed=allowed,ram_gb=requirements.ram_gb)
        check();identifier,size=digest_file(path)
        if size>32*1024**2 or value['adapter_id']!=identifier or value['adapter_bytes']!=size:
            raise ValueError('Candidate engine returned inconsistent weights')
        validate_adapter(path,json.loads((base/'config.json').read_text()),head['descriptor']['base_hash'])
        fields={'version','private_inputs','base_name','base_hash','dataset_id','parent_release_id','adapter_id',
                'baseline_nll','candidate_nll','quality_before','quality_after','instruction_before','instruction_after'}
        evaluation={key:value[key] for key in fields}
        if (type(evaluation['version']) is not int or evaluation['version']!=policy_version or evaluation['private_inputs'] is not False
                or any(evaluation[key]!=expected for key,expected in {
                    'base_name':payload['base_name'],'base_hash':head['descriptor']['base_hash'],
                    'dataset_id':payload['dataset_id'],'parent_release_id':payload['release_id']}.items())):
            raise ValueError('Candidate evaluation belongs to different public inputs')
        for key in ['quality_before','quality_after','instruction_before','instruction_after']:
            if not isinstance(evaluation[key],list) or any(not isinstance(row,dict) or set(row)!={'case','answer'}
                    or not isinstance(row['answer'],str) or len(row['answer'].encode('utf-8'))>1024 for row in evaluation[key]):
                raise ValueError('Invalid bounded candidate evaluation answers')
        before,after=evaluation['baseline_nll'],evaluation['candidate_nll']
        if any(type(n) not in (int,float) or not math.isfinite(n) or n<=0 for n in (before,after)):
            raise ValueError('Invalid candidate paired loss')
        facts=factual_verdict(evaluation['quality_before'],evaluation['quality_after'])
        instructions=instruction_verdict(evaluation['instruction_before'],evaluation['instruction_after'])
        numerical=after<before-max(1e-7,before*.001) and facts['eligible_for_promotion'] and instructions['eligible_for_promotion']
        distributed=set(proof['physical_workers'])==language_versions.WORKERS and proof['tokens']>=16
        evaluation.update(gate_id=gate_id(policy_version),contributions=payload['contributions'])
        atomic_json(directory/'evaluation.json',evaluation)
        summary={'parent_release_id':payload['release_id'],'adapter_id':identifier,'adapter_bytes':size,
            'baseline_nll':before,'candidate_nll':after,'factual_verdict':facts,'instruction_verdict':instructions,
            **proof,'numerical_gates_passed':numerical,'distributed_worker_gate_passed':distributed,
            'eligible_for_publication':numerical and distributed,'published':False}
        if policy_version>=2:summary['training_policy_id']=policy_id(policy_version)
        if not summary['eligible_for_publication']:path.unlink()
        check();atomic_json(directory/'summary.json',summary);complete=True
        with training.lock:
            training.completed+=1;atomic_json(training.directory/'language-training-status.json',{'completed':training.completed})
        return {'summary':summary,'evaluation':evaluation}
    finally:
        if not complete:shutil.rmtree(directory)
        with training.lock:training.active-=1
        training.changed.set()
