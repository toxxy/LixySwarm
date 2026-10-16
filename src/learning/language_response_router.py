"""Pinned public response-aware utility learning, separate from native releases."""
import hashlib
import json
import math
from pathlib import Path

from .language_outcome_selector import OUTCOME_IDS
from .language_repair_calibration import load_outcomes,verify_calibration
from .language_repair_comparison import needs_second_call
from .language_retention import PARENT_MODEL,PARENT_RELEASE
from .language_specialist_answers import SPECIALIST_ID
from .public_bases import BASES
from .public_math import ARTIFACTS,canonical,final_number,prompt

REPAIR_ARTIFACTS=(('bc4dc6687d2b76fd9ec9c32cb8607c3eff5984ebcf66b02c475edbdcaf11e6c2',30717),
    ('a226cf96b7aa4b8ab47c13524c4cf7ea95a5fdf3e62dbd167db5cce5cea50ab5',31409))
FUTURE_SNAPSHOT_ID='770219d8446cf0a24f0f82985e85a666639a6151da5dafb2cfdf58d6abeecfab'
FUTURE_SNAPSHOT_BYTES=79665


def feature_definition():
    return {'version':1,'space':'lixyswarm-public-response-utility-v1',
        'parent_model_id':PARENT_MODEL,'tokenizer_id':BASES['qwen3-1.7b']['tokenizer.json'],
        'input':'public question and cached/generated parent draft only; no reference answer',
        'pooling':'last hidden state of the closed assistant draft, including the chat end token',
        'raw_dimensions':2048,'projection':'centered CPU FP64 SVD on invalid TRAIN drafts only; 16 sign-canonical components',
        'length_features':['question template tokens divided by 1024','parent generated output tokens divided by 384'],
        'normalization':'TRAIN mean and population standard deviation clamped to at least 1e-4',
        'execution':'frozen-bf16-storage-fp32-linear-recompute-v1','tf32':False,
        'scope':'public-training-only'}


def feature_id():return hashlib.sha256(canonical(feature_definition())).hexdigest()


def definition():
    return {'version':1,'action':'public-response-utility-fit-v1',
        'expert_ids':[PARENT_MODEL,SPECIALIST_ID],'parent_release_id':PARENT_RELEASE,
        'training_artifact_id':ARTIFACTS['train'][0],'outcome_ids':list(OUTCOME_IDS),
        'repair_artifacts':[{'id':identifier,'bytes':size} for identifier,size in REPAIR_ARTIFACTS],
        'training_indices':list(range(128)), 'fit_only_invalid_parent_drafts':True,
        'feature_definition':feature_definition(),'feature_space_id':feature_id(),
        'seed':20261015,'projection_components':16,'input_dimensions':18,
        'phase_steps':256,'batch_size':16,'learning_rate':.002,'weight_decay':.01,
        'support_stride':4,'memory_modes':['zero','shuffled','public'],'primary_selector':'public',
        'target_order':['repair_correct','specialist_correct','repair_tokens/384','specialist_tokens/384'],
        'loss':'mean Brier error for two correctness heads plus 0.25 times mean squared normalized-token error',
        'cost_penalty':.25,'specialist_advantage_margin':.1,'maximum_specialist_calls_per_128_cases':8,
        'decision':'keep valid parent; otherwise specialist only if predicted correctness gain minus positive token-cost penalty exceeds margin and cap; else repair',
        'memory_comparison':'same initial state, architecture, optimizer steps and query order; only support labels differ',
        'dolphin_budget':'256 steps; each memory arm freezes Dolphin and adds 256 steps',
        'maximum_input_tokens':1024,'maximum_messages_bytes':4096,'maximum_new_tokens_per_call':384,
        'ram_gb':10,'disk_reservation_gb':.032,'maximum_seconds':1800,
        'future_evaluation':{'snapshot_id':FUTURE_SNAPSHOT_ID,'snapshot_bytes':FUTURE_SNAPSHOT_BYTES,
            'selection':'pinned sorted upstream TEST[256:384]; original TEST[0:128] and consumed TEST[128:256] excluded',
            'maximum_calls_per_case':2,'minimum_new_correct_over_parent':4,'maximum_regressions':0,
            'must_exceed_same_parent_repair':True,'must_exceed_zero_and_shuffled':True,
            'maximum_time_ratio_to_parent_repair':1.1,
            'cost_scope':'shared parent plus selected secondary complete-call time plus encoder and own router time; exclude counterfactual arms'},
        'fresh_evaluation_performed_by_fit':False,'pretraining_contamination_excluded':False,
        'language_experts_updated':False,'native_publication_authorized':False,
        'original_global_quality_gates_unchanged':True,
        'scope':'Public TRAIN-only learned utility; prospective component evaluation and original global gates remain required'}


def protocol_id():return hashlib.sha256(canonical(definition())).hexdigest()


def training_rows(snapshot,outcome_paths,repair_paths):
    if len(repair_paths)!=2:raise ValueError('Both exact physical repair artifacts are required')
    values=load_outcomes([str(v) for v in outcome_paths],snapshot);rows={}
    for shard,(path,(identifier,size)) in enumerate(zip(repair_paths,REPAIR_ARTIFACTS)):
        path=Path(path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size!=size:
            raise ValueError('TRAIN repair artifact has another size or file type')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=identifier:raise ValueError('TRAIN repair artifact differs from its physical pin')
        value=json.loads(raw)
        if value['shard']!=shard or canonical(value)!=raw:raise ValueError('TRAIN repair encoding or shard changed')
        verify_calibration(value,snapshot,values)
        prior=values[shard]['experts']
        for offset,record in enumerate(value['records']):
            index=record['index'];row=snapshot['rows'][index];draft=prior['accepted-parent']['answers'][offset]['answer']
            if not needs_second_call(draft):continue
            specialist=prior['research-specialist']['answers'][offset]['answer'];repair=record['repair']
            rows[index]={'index':index,'case':row['case'],'question':row['question'],'draft':draft,
                'draft_output_tokens':prior['accepted-parent']['output_tokens'][offset],
                'target':[float(final_number(repair['answer'])==row['final_number']),
                    float(final_number(specialist)==row['final_number']),repair['output_tokens']/384,
                    prior['research-specialist']['output_tokens'][offset]/384]}
    if len(rows)!=65:raise ValueError('The pinned public invalid-draft TRAIN population changed')
    return [rows[i] for i in sorted(rows)]


def response_messages(row):
    return [{'role':'user','content':prompt({'question':row['question']})},
            {'role':'assistant','content':row['draft']}]


def select_secondary(prediction,specialist_calls):
    if (not isinstance(prediction,list) or len(prediction)!=4
            or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in prediction)
            or type(specialist_calls) is not int or not 0<=specialist_calls<=8):
        raise ValueError('Invalid learned confidence, token estimate or fixed specialist budget')
    p=definition();gain=prediction[1]-prediction[0]-p['cost_penalty']*max(0,prediction[3]-prediction[2])
    return 'specialist' if gain>p['specialist_advantage_margin'] and specialist_calls<8 else 'repair'
