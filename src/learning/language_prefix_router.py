"""Prospective frozen-prefix routing component; no original EVAL or promotion."""
import hashlib
import math
import re

from .language_outcome_selector import OUTCOME_IDS
from .language_retention import PARENT_MODEL, PARENT_RELEASE
from .language_specialist_answers import SPECIALIST_ID
from .public_bases import BASES, base_hash
from .public_math import ARTIFACTS, canonical, final_number


def feature_definition():
    return {'version':1,'base_hash':base_hash('qwen3-1.7b'),'adapter_id':PARENT_MODEL,
        'tokenizer_id':BASES['qwen3-1.7b']['tokenizer.json'],'dimensions':2048,
        'input':'user query plus fixed assistant generation prefix only; no reference or generated answer',
        'pooling':'last token of Qwen3Model.last_hidden_state; per-query layer normalization in the router',
        'execution':'frozen-bf16-storage-fp32-linear-recompute-v1','tf32':False,'scope':'public-training-only'}


def feature_id():return hashlib.sha256(canonical(feature_definition())).hexdigest()


def definition():
    return {'version':1,'action':'public-frozen-prefix-router-v1','parent_model_id':PARENT_MODEL,
        'parent_release_id':PARENT_RELEASE,'expert_ids':[PARENT_MODEL,SPECIALIST_ID],
        'feature_definition':feature_definition(),'feature_space_id':feature_id(),
        'training_artifact_id':ARTIFACTS['train'][0],'calibration_outcome_ids':list(OUTCOME_IDS),
        'fit_indices':list(range(128)),'prediction_indices':list(range(128,256)),
        'specialist_update_partition':'TRAIN[1024:4096]; disjoint from fit and prediction',
        'target':'exclusive correct expert one-hot; [0.5,0.5] if both or neither correct',
        'seed':20261015,'phase_steps':256,'batch_size':16,'learning_rate':.002,'weight_decay':0.,
        'support_stride':4,'ping_hidden':16,'echo_dim':8,'memory_modes':['zero','shuffled','public'],
        'primary_selector':'public','primary_selected_before_new_outcomes':True,
        'freeze_order':'fit 128; save and reload every state; commit all 128 new predictions; then generate fresh answers',
        'dolphin_budget':'256 steps; each memory arm shares that state and adds 256 steps',
        'memory_comparison':'public versus zero/shuffled have identical architecture, update budget and query order',
        'component_screen':{'minimum_new_correct':4,'maximum_regressions':0,
            'public_must_exceed_zero_and_shuffled_correct_counts':True,'original_publication_gates_still_required':True},
        'maximum_new_tokens':384,'maximum_seconds':3600,'ram_gb':10,'disk_reservation_gb':.064,
        'original_evaluation_opened':False,'original_quality_gates_changed':False,
        'pretraining_contamination_excluded':False,
        'prediction_partition_scope':'Outside this selector fit; public TRAIN partly used by earlier research, not a project-wide untouched test',
        'native_promotion_authorized':False,'language_experts_updated':False,
        'scope':'New TRAIN queries outside selector fit; component comparison only, not original held-out quality, native chat or speedup'}


def protocol_id():return hashlib.sha256(canonical(definition())).hexdigest()


def verify_predictions(commitment):
    p=definition();names={'dolphin',*p['memory_modes']}
    if (set(commitment)!={'protocol_id','feature_space_id','expert_ids','learned_artifacts','predictions'}
            or commitment['protocol_id']!=protocol_id() or commitment['feature_space_id']!=feature_id()
            or commitment['expert_ids']!=p['expert_ids'] or set(commitment['predictions'])!=names
            or set(commitment['learned_artifacts'])!={n+'.safetensors' for n in names}|{'optimizer-state.pt'}):
        raise ValueError('Frozen prefix prediction commitment has incompatible provenance')
    for name,value in commitment['learned_artifacts'].items():
        maximum=8*1024**2 if name=='optimizer-state.pt' else 2*1024**2
        if (not isinstance(value,dict) or set(value)!={'sha256','bytes'}
                or not isinstance(value['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',value['sha256'])
                or type(value['bytes']) is not int or not 1<=value['bytes']<=maximum):
            raise ValueError('Prefix commitment contains an invalid learned artifact')
    for rows in commitment['predictions'].values():
        if len(rows)!=128:raise ValueError('Frozen prefix predictions are incomplete')
        for index,row in zip(p['prediction_indices'],rows):
            if not isinstance(row,dict) or set(row)!={'index','choice','probabilities'}:
                raise ValueError('Unknown prefix prediction schema')
            prob=row['probabilities']
            if (row['index']!=index or type(row['choice']) is not int or not isinstance(prob,list) or len(prob)!=2
                    or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in prob)
                    or not math.isclose(sum(prob),1,abs_tol=1e-6,rel_tol=0)
                    or row['choice']!=int(prob[1]>prob[0])):
                raise ValueError('Prefix choice is not its committed learned prediction')


def verify_report(report,snapshot):
    p=definition();commitment=report['commitment'];verify_predictions(commitment)
    if (report['protocol_id']!=protocol_id() or report['commitment_id']!=hashlib.sha256(canonical(commitment)).hexdigest()
            or report['original_evaluation_opened'] is not False or report['native_promotion_authorized'] is not False
            or report['language_experts_updated'] is not False or report['committed_before_fresh_generation'] is not True
            or snapshot.get('split')!='train' or set(report['outcomes'])!={'accepted-parent','research-specialist'}):
        raise ValueError('Frozen prefix report exceeds its prospectively fixed scope')
    selected=[snapshot['rows'][i] for i in p['prediction_indices']];correct=[];tokens=[]
    fit=report['fit'];support=list(range(0,128,p['support_stride']))
    if (fit['support_indices']!=support or fit['query_indices']!=[i for i in range(128) if i not in support]
            or fit['prediction_rows_used_for_fit'] is not False or fit['phase1']['steps']!=p['phase_steps']
            or set(fit['phase2'])!=set(p['memory_modes'])
            or any(v['steps']!=p['phase_steps'] for v in fit['phase2'].values())
            or fit['memory_scope']!='public-training-only'
            or report['generation_expert_order']!=[[r['choice'],1-r['choice']] for r in commitment['predictions'][p['primary_selector']]]
            or report['feature_binding']!={'feature_space_id':feature_id(),'training_artifact_id':p['training_artifact_id'],
                'case_ids':[snapshot['rows'][i]['case'] for i in p['fit_indices']+p['prediction_indices']], 'answer_tokens_used':False}):
        raise ValueError('Prefix fit, feature provenance or actual primary execution changed')
    for name in ['accepted-parent','research-specialist']:
        results=report['outcomes'][name]
        if len(results)!=128:raise ValueError('Fresh TRAIN outcomes are incomplete')
        scored=[];counts=[]
        for row,result in zip(selected,results):
            if (set(result)!={'case','answer','output_tokens','generation_seconds'} or result['case']!=row['case']
                    or not isinstance(result['answer'],str) or len(result['answer'].encode())>8192
                    or type(result['output_tokens']) is not int or not 1<=result['output_tokens']<=p['maximum_new_tokens']
                    or type(result['generation_seconds']) not in (int,float)
                    or not math.isfinite(result['generation_seconds']) or result['generation_seconds']<=0):
                raise ValueError('Fresh outcome text, identity or cost is invalid')
            scored.append(final_number(result['answer'])==row['final_number']);counts.append(result['output_tokens'])
        correct.append(scored);tokens.append(counts)
    variants={}
    for name,rows in commitment['predictions'].items():
        choices=[r['choice'] for r in rows]
        variants[name]={'correct':sum(correct[c][i] for i,c in enumerate(choices)),
            'newly_correct':sum(not correct[0][i] and correct[c][i] for i,c in enumerate(choices)),
            'regressions':sum(correct[0][i] and not correct[c][i] for i,c in enumerate(choices)),
            'specialist_choices':sum(choices),'selected_output_tokens':sum(tokens[c][i] for i,c in enumerate(choices))}
    primary=variants[p['primary_selector']]
    screen=(primary['newly_correct']>=p['component_screen']['minimum_new_correct']
        and primary['regressions']<=p['component_screen']['maximum_regressions']
        and all(primary['correct']>variants[n]['correct'] for n in ['zero','shuffled']))
    return {'examples':128,'parent_correct':sum(correct[0]),'specialist_correct':sum(correct[1]),
        'oracle_correct_diagnostic_only':sum(a or b for a,b in zip(*correct)), 'variants':variants,
        'primary_actual_generations':128,'other_arms_are_counterfactual':True,
        'component_screen_passed':screen,
        'original_evaluation_opened':False,'native_promotion_authorized':False}
