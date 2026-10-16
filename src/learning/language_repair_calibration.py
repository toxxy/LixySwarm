"""Public TRAIN labels for choosing between parent repair and a specialist.

The first parent/specialist answers are the two pinned physical calibration
artifacts. Only missing-format parent answers receive a new parent-repair call.
No evaluation snapshot, optimizer, route choice, or release is involved.
"""
import hashlib
import json
import math
from pathlib import Path

from .language_outcomes import verify_outcomes, partition
from .language_outcome_selector import OUTCOME_IDS
from .language_retention import PARENT_MODEL, PARENT_RELEASE
from .language_specialist_answers import SPECIALIST_ID
from .language_repair_comparison import REPAIR_INSTRUCTION, repair_messages, needs_second_call
from .public_math import ARTIFACTS, canonical, final_number


def definition():
    return {'version':1, 'action':'public-parent-repair-calibration-v1',
        'parent_model_id':PARENT_MODEL, 'parent_release_id':PARENT_RELEASE,
        'specialist_id':SPECIALIST_ID, 'cached_outcome_ids':list(OUTCOME_IDS),
        'training_artifact_id':ARTIFACTS['train'][0], 'training_indices':list(range(128)),
        'shards':2, 'partition':'original physical even/odd TRAIN calibration shards',
        'rule':'keep a cached parent with valid final format; otherwise collect one new parent repair',
        'repair_instruction':REPAIR_INSTRUCTION,
        'maximum_input_tokens_per_call':1024, 'maximum_new_tokens_per_call':384,
        'maximum_messages_bytes_per_call':4096, 'maximum_calls_per_case':1,
        'compute_dtype':'float32', 'streamed_base':True, 'seed':20261015,
        'ram_gb':10, 'disk_reservation_gb':.016, 'maximum_seconds':1800,
        'evaluation_data_used':False, 'optimizer_updates':False,
        'model_promoted':False, 'native_learning_cycles_added':0,
        'cost_scope':'historical parent/specialist plus new repair costs are TRAIN labels; not a latency comparison',
        'future_test_policy':'TRAIN labels may teach a selector; consumed TEST[128:256] cannot become its fresh test',
        'scope':'Public response-aware calibration only; no useful-router, quality-gain or scaling claim'}


def protocol_id():
    return hashlib.sha256(canonical(definition())).hexdigest()


def load_outcomes(paths, snapshot):
    if not isinstance(paths,list) or len(paths)!=2:
        raise ValueError('Both pinned physical calibration artifacts are required')
    values=[]
    for shard,(path,identifier) in enumerate(zip(paths,OUTCOME_IDS)):
        path=Path(path)
        if path.is_symlink() or not path.is_file() or not 1<=path.stat().st_size<=128*1024:
            raise ValueError('Cached TRAIN artifact exceeds its regular-file bound')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=identifier:
            raise ValueError('Cached physical TRAIN answers differ from their pin')
        value=json.loads(raw)
        if canonical(value)!=raw or value.get('shard')!=shard:
            raise ValueError('Cached TRAIN artifact has another encoding or physical shard')
        verify_outcomes(value,snapshot);values.append(value)
    return values


def selected_rows(snapshot, values, shard):
    if type(shard) is not int or shard not in (0,1) or len(values)!=2:
        raise ValueError('Only a fixed physical TRAIN shard may collect repair labels')
    for i,value in enumerate(values):
        if (hashlib.sha256(canonical(value)).hexdigest()!=OUTCOME_IDS[i] or value.get('shard')!=i):
            raise ValueError('Repair labels require the unchanged two physical artifacts')
        verify_outcomes(value,snapshot)
    selected,indices=partition(snapshot,shard)
    parent=values[shard]['experts']['accepted-parent']
    return [(index,row,parent['answers'][offset]['answer'])
            for offset,(index,row) in enumerate(zip(indices,selected['rows']))]


def commitment(snapshot, values, shard):
    return {'protocol_id':protocol_id(), 'shard':shard, 'cached_outcome_id':OUTCOME_IDS[shard],
        'cases':[{'index':index,'case':row['case'],
                  'parent_answer_id':hashlib.sha256(draft.encode()).hexdigest(),
                  'repair_requested':needs_second_call(draft)}
                 for index,row,draft in selected_rows(snapshot,values,shard)]}


def verify_calibration(value, snapshot, values):
    if (not isinstance(value,dict) or set(value)!={'version','protocol_id','shard','commitment','records',
            'evaluation_data_used','optimizer_updates'} or type(value['version']) is not int or value['version']!=1
            or value['protocol_id']!=protocol_id() or value['evaluation_data_used'] is not False
            or value['optimizer_updates'] is not False):
        raise ValueError('Repair calibration has incompatible provenance or exceeds TRAIN scope')
    shard=value['shard'];expected=commitment(snapshot,values,shard)
    if value['commitment']!=expected or not isinstance(value['records'],list) or len(value['records'])!=64:
        raise ValueError('Repair calibration changed its committed cases or first answers')
    selected=selected_rows(snapshot,values,shard)
    summary={'cases':64, 'new_generation_calls':0, 'parent_correct':0, 'repair_fallback_correct':0,
        'specialist_fallback_correct':0, 'specialist_only_correct_after_invalid_parent':0,
        'repair_only_correct_after_invalid_parent':0, 'new_repair_output_tokens':0,
        'new_repair_generation_seconds':0.0, 'new_repair_input_tokens':0}
    for offset,((index,row,draft),record) in enumerate(zip(selected,value['records'])):
        if (not isinstance(record,dict) or set(record)!={'index','case','parent_answer_id','repair_requested','repair'}
                or type(record['index']) is not int or type(record['repair_requested']) is not bool
                or {k:record[k] for k in ['index','case','parent_answer_id','repair_requested']}!=expected['cases'][offset]):
            raise ValueError('Repair records do not match the committed ordered TRAIN cases')
        repair=record['repair'];needed=needs_second_call(draft)
        if needed:
            if (not isinstance(repair,dict) or set(repair)!={'answer','output_tokens','generation_seconds','input_tokens'}
                    or not isinstance(repair['answer'],str) or not 1<=len(repair['answer'].encode())<=8192
                    or type(repair['output_tokens']) is not int or not 1<=repair['output_tokens']<=384
                    or type(repair['input_tokens']) is not int or not 1<=repair['input_tokens']<=1024
                    or type(repair['generation_seconds']) not in (int,float)
                    or not math.isfinite(repair['generation_seconds']) or repair['generation_seconds']<=0):
                raise ValueError('Missing, unbounded or nonfinite TRAIN repair result')
            summary['new_generation_calls']+=1
            for source,target in [('output_tokens','new_repair_output_tokens'),('input_tokens','new_repair_input_tokens'),
                    ('generation_seconds','new_repair_generation_seconds')]:summary[target]+=repair[source]
        elif repair is not None:
            raise ValueError('A valid parent must not receive another generation')
        parent_ok=final_number(draft)==row['final_number']
        repair_ok=final_number(repair['answer'] if needed else draft)==row['final_number']
        specialist=values[shard]['experts']['research-specialist']['answers'][offset]['answer']
        specialist_ok=final_number(specialist if needed else draft)==row['final_number']
        summary['parent_correct']+=parent_ok
        summary['repair_fallback_correct']+=repair_ok
        summary['specialist_fallback_correct']+=specialist_ok
        summary['specialist_only_correct_after_invalid_parent']+=needed and specialist_ok and not repair_ok
        summary['repair_only_correct_after_invalid_parent']+=needed and repair_ok and not specialist_ok
    return {**summary, 'public_training_only':True, 'model_promoted':False,
        'cost_scope':definition()['cost_scope'], 'neural_routing_evaluated':False}
