"""Fixed cross-fitting of learned expert utility from public TRAIN answers.

No original EVAL example is opened. This measures a selector component, never
authorizes an expert, routed chat, release, or relaxation of a quality gate.
"""
import hashlib
import math

from .language_outcomes import verify_outcomes,partition,score_training_answers,protocol_id as outcomes_id
from .language_retention import PARENT_MODEL
from .language_specialist_answers import SPECIALIST_ID
from .public_bases import BASES,base_hash
from .public_math import ARTIFACTS,canonical

OUTCOME_IDS=('a6016b3aaaa19b8bd9a8cfb0aa3207fe27a2eafce9fd9802228a786514dd6a73',
    'f74817ffa03108933d63249bd25238c3627c26909a4f6b9d77353f81256bb1b4')


def definition():
    return {'version':1,'action':'public-language-outcome-selector-v1','outcome_protocol_id':outcomes_id(),
        'outcome_artifact_ids':list(OUTCOME_IDS),'training_artifact_id':ARTIFACTS['train'][0],
        'expert_ids':[PARENT_MODEL,SPECIALIST_ID],'base_hash':base_hash('qwen3-1.7b'),
        'tokenizer_id':BASES['qwen3-1.7b']['tokenizer.json'],'training_indices':list(range(128)),
        'folds':8,'fold_partition':'index modulo 8; predict each TRAIN calibration query once outside its fit',
        'target':'One-hot for the exclusively correct expert; [0.5,0.5] when both or neither answer is correct',
        'objective':'Cross-entropy of verified frozen-expert strict numeric TRAIN outcomes',
        'seed':20261015,'dolphin_n_embd':32,'phase_steps':256,'batch_size':16,'learning_rate':.002,
        'weight_decay':0.,'support_stride':4,'memory_modes':['zero','shuffled','public'],
        'primary_selector':'public','memory_feature_space':'lixyswarm-public-language-adapter-selection-v2',
        'maximum_seconds':1200,'ram_gb':4,'disk_reservation_gb':1.4,
        'original_evaluation_opened':False,'language_experts_updated':False,'answer_generation_performed':False,
        'original_quality_gates_changed':False,'native_promotion_authorized':False,
        'scope':'Out-of-fit predictions on previously collected TRAIN calibration; not original held-out quality or routed language execution'}


def protocol_id():return hashlib.sha256(canonical(definition())).hexdigest()


def dataset(outcomes,snapshot):
    if len(outcomes)!=2:raise ValueError('Both exact physical outcome shards are required')
    rows={}
    for shard,value in enumerate(outcomes):
        if hashlib.sha256(canonical(value)).hexdigest()!=OUTCOME_IDS[shard]:
            raise ValueError('Outcome selector requires the exact independently verified physical artifacts')
        verify_outcomes(value,snapshot);selected,indices=partition(snapshot,shard)
        parent=score_training_answers(value['experts']['accepted-parent']['answers'],selected)
        specialist=score_training_answers(value['experts']['research-specialist']['answers'],selected)
        for i,row in zip(indices,selected['rows']):
            a,b=parent[row['case']],specialist[row['case']]
            rows[i]={'index':i,'case':row['case'],'correct':[a,b],
                'target':[float(a),float(b)] if a!=b else [.5,.5],
                'output_tokens':[value['experts'][name]['output_tokens'][indices.index(i)] for name in ['accepted-parent','research-specialist']]}
    if sorted(rows)!=list(range(128)):raise ValueError('Outcome calibration coverage changed')
    return [rows[i] for i in range(128)]


def fold_indices(fold):
    if type(fold) is not int or not 0<=fold<8:raise ValueError('Unknown fixed outcome fold')
    return [i for i in range(128) if i%8!=fold],[i for i in range(128) if i%8==fold]


def verify_report(report,rows):
    if (report['protocol_id']!=protocol_id() or report['original_evaluation_opened'] is not False
            or report['native_promotion_authorized'] is not False or report['language_experts_updated'] is not False
            or len(report['folds'])!=8 or set(report['predictions'])!={'zero','shuffled','public','dolphin'}):
        raise ValueError('Outcome selector report exceeds its component scope')
    p=definition()
    for fold,value in enumerate(report['folds']):
        train,test=fold_indices(fold);fit=value['fit'];support=list(range(0,len(train),p['support_stride']))
        if (value['training_indices']!=train or value['prediction_indices']!=test
                or fit['support_indices']!=support or fit['query_indices']!=[i for i in range(len(train)) if i not in support]
                or fit['phase1']['steps']!=p['phase_steps'] or set(fit['phase2'])!=set(p['memory_modes'])
                or any(v['steps']!=p['phase_steps'] for v in fit['phase2'].values())
                or fit['memory_scope']!='public-training-only' or fit['held_out_used_for_training_or_selection'] is not False):
            raise ValueError('Outcome fold mixed prediction queries or changed its equal training budget')
    summary={}
    for name,predictions in report['predictions'].items():
        if len(predictions)!=128:raise ValueError('Outcome predictions are incomplete')
        choices=[]
        for row,prediction in zip(rows,predictions):
            probability=prediction['probabilities'];choice=prediction['choice']
            if (prediction['index']!=row['index'] or len(probability)!=2
                    or any(type(n) not in (int,float) or not math.isfinite(n) or not 0<=n<=1 for n in probability)
                    or not math.isclose(sum(probability),1.,abs_tol=1e-6) or type(choice) is not int or choice!=int(probability[1]>probability[0])):
                raise ValueError('Outcome choice is not its frozen learned prediction')
            choices.append(choice)
        summary[name]={'correct':sum(row['correct'][c] for row,c in zip(rows,choices)),
            'newly_correct':sum(not row['correct'][0] and row['correct'][c] for row,c in zip(rows,choices)),
            'regressions':sum(row['correct'][0] and not row['correct'][c] for row,c in zip(rows,choices)),
            'specialist_choices':sum(choices),'output_tokens_if_selected':sum(row['output_tokens'][c] for row,c in zip(rows,choices))}
    return {'out_of_fit_predictions_verified':True,'equal_memory_training_budget_verified':True,'examples':128,
        'parent_correct':sum(row['correct'][0] for row in rows),'specialist_correct':sum(row['correct'][1] for row in rows),
        'variants':summary,'original_evaluation_opened':False,'native_promotion_authorized':False}
