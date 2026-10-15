"""Prospective TRAIN-only selector experiment; not a public release policy."""
import hashlib
import json
import math

from .language_retention import PARENT_MODEL, PARENT_RELEASE
from .public_bases import base_hash, BASES
from .public_conversation import corpus_digest, public_corpus
from .public_math import ARTIFACTS, gate_id


def definition():
    return dict(version=1, action='public-language-selector-v1', base_name='qwen3-1.7b',
        base_hash=base_hash('qwen3-1.7b'), tokenizer_id=BASES['qwen3-1.7b']['tokenizer.json'],
        parent_model_id=PARENT_MODEL, parent_release_id=PARENT_RELEASE,
        experts=['accepted-parent-adapter', 'same-upstream-base-adapter-disabled'],
        compute_dtype='float32', streamed_base=True, ram_gb=10, disk_reservation_gb=.2,
        maximum_seconds=1800, corpus_id=corpus_digest(public_corpus()), math_training_id=ARTIFACTS['train'][0],
        math_evaluation_id=ARTIFACTS['evaluation'][0], historical_training_examples=200,
        math_training_examples=192, math_evaluation_examples=128, maximum_sequence_tokens=1024, maximum_answer_tokens=512,
        seed=20261015, dolphin_n_embd=32, echo_dim=8, feromon_dim=8, memory_dim=16,
        phase_steps=256, batch_size=16, learning_rate=.002, weight_decay=0., teacher_temperature=.1,
        support_stride=4, memory_modes=['zero', 'shuffled', 'public'],
        objective='cross-entropy of soft expert targets from frozen TRAIN answer NLL',
        phase1='Train five Dolphin pings and expert head on all 392 public TRAIN rows',
        phase2='Freeze Dolphin; equal updates for Matriarca/readout using non-support TRAIN queries',
        memory='Every fourth TRAIN row: frozen Dolphin features and soft expert outcomes; no query self-entry',
        public_scope='public-training-only', inference='One hard selected expert, ties retain accepted parent',
        held_out_used_for_training_or_selection=False, evaluation='All fixed historical and math held-out NLL; no generation',
        quality_gate_id=gate_id(), new_model_publication_authorized=False,
        native_learning_cycles_added=0, private_inputs=False,
        scope='Mechanism and held-out loss comparison; full answer gates and native Internet admission remain required')


def protocol_id():
    return hashlib.sha256(json.dumps(definition(),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def verify_loss_report(report):
    """Independent scalar reconstruction; does not authorize model promotion."""
    p=definition();historical=len(public_corpus()['evaluation']);count=historical+p['math_evaluation_examples']
    losses=report['heldout_expert_nll'];tokens=report['heldout_answer_tokens'];rows=report['evaluations'];fit=report['fit']
    names={'accepted-parent','upstream-base','oracle-only-diagnostic','dolphin',*p['memory_modes']}
    if (report['protocol_id']!=protocol_id() or report['held_out_data_used_for_training_or_selection'] is not False
            or report['eligible_for_native_promotion'] is not False or report['generation_performed'] is not False
            or set(rows)!=names or len(losses)!=count or len(tokens)!=count
            or any(len(v)!=2 or any(type(n) not in (float,int) or not math.isfinite(n) or n<0 for n in v) for v in losses)
            or any(type(n) not in (float,int) or not math.isfinite(n) or not 1<=n<=512 or int(n)!=n for n in tokens)):
        raise ValueError('Selector held-out evidence changed its bounded scope')
    total_train=p['historical_training_examples']+p['math_training_examples']
    support=list(range(0,total_train,p['support_stride']))
    if (fit['phase1']['steps']!=p['phase_steps'] or set(fit['phase2'])!=set(p['memory_modes'])
            or any(r['steps']!=p['phase_steps'] for r in fit['phase2'].values())
            or fit['support_indices']!=support or fit['query_indices']!=[i for i in range(total_train) if i not in support]
            or fit['memory_scope']!='public-training-only' or fit['held_out_used_for_training_or_selection'] is not False):
        raise ValueError('Selector TRAIN partition or equal budget changed')
    for name,row in rows.items():
        if name=='accepted-parent':choices=[0]*count
        elif name=='upstream-base':choices=[1]*count
        elif name=='oracle-only-diagnostic':choices=[int(b<a) for a,b in losses]
        else:
            choices=row['choices'];probabilities=row['probabilities']
            if (len(choices)!=count or any(type(v) is not int or v not in {0,1} for v in choices)
                    or len(probabilities)!=count or any(len(v)!=2 or any(type(n) not in (float,int) or not math.isfinite(n) or not 0<=n<=1 for n in v)
                        or not math.isclose(sum(v),1.,abs_tol=1e-6) or int(v[1]>v[0])!=c for v,c in zip(probabilities,choices))
                    or row['single_selected_expert_executed_per_request'] is not True or row['prefix_only_routing_verified'] is not True):
                raise ValueError('Selector choices differ from its learned probabilities')
        for group,start,end in [('historical',0,historical),('math',historical,count)]:
            expected=math.fsum(losses[i][choices[i]]*tokens[i] for i in range(start,end))/sum(tokens[start:end])
            value=row[group]
            if (not math.isclose(value['nll'],expected,abs_tol=1e-12,rel_tol=1e-12)
                    or value['examples']!=end-start or value['tokens']!=sum(tokens[start:end])
                    or value['upstream_base_choices']!=sum(choices[start:end])):
                raise ValueError('Selector aggregate NLL does not reconstruct from frozen expert losses')
    return {'losses_reconstructed':True,'examples':count,'equal_training_budget_verified':True,
            'full_generation_gates_evaluated':False,'native_promotion_authorized':False}
