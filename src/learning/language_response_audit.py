"""Recover learned utility and optimizer on CPU, without a language model."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import zipfile

from .language_response_router import definition,protocol_id,feature_id,training_rows,select_secondary
from .public_math import canonical,load_snapshot
from .weights_io import digest_file


def verify_saved_state(root,report,rows):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from .language_response_selector import fit_projection,project,ResponseSelector,load
    from .language_selector import tensor_digest
    p=definition();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    if (report['protocol_id']!=protocol_id() or report['feature_space_id']!=feature_id()
            or report['fresh_evaluation_performed'] is not False or report['language_experts_updated'] is not False
            or report['model_promoted'] is not False or report['training_indices']!=[v['index'] for v in rows]
            or report['case_ids']!=[v['case'] for v in rows]):raise ValueError('Response fit provenance or TRAIN coverage changed')
    names={'dolphin','zero','shuffled','public'}
    expected={*(name+'.safetensors' for name in names),'train-features.safetensors','projection.safetensors','optimizer-state.pt'}
    if set(report['artifacts'])!=expected or sum(v['bytes'] for v in report['artifacts'].values())>8*1024**2:
        raise ValueError('Response recovery requires the complete bounded state')
    for name,item in report['artifacts'].items():
        if Path(name).name!=name or digest_file(root/name)!=(item['sha256'],item['bytes']):raise ValueError('Response learned-state bytes changed')
    common={'protocol_id':protocol_id(),'feature_space_id':feature_id(),'scope':'public-training-only'}
    for name,format_name in [('train-features.safetensors','lixyswarm-public-response-features-v1'),
            ('projection.safetensors','lixyswarm-public-response-projection-v1')]:
        with safe_open(root/name,framework='pt',device='cpu') as reader:
            if reader.metadata()!={'format':format_name,'binding':canonical(common).decode()}:
                raise ValueError('Response feature provenance or scope changed')
    raw=load_file(str(root/'train-features.safetensors'),device='cpu')
    if (set(raw)!={'hidden','lengths','targets'} or raw['hidden'].shape!=(65,2048) or raw['lengths'].shape!=(65,2)
            or raw['targets'].shape!=(65,4) or any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in raw.values())
            or not torch.equal(raw['targets'],torch.tensor([v['target'] for v in rows],dtype=torch.float32))):
        raise ValueError('Recovered response features or physical TRAIN targets changed')
    projection=load_file(str(root/'projection.safetensors'),device='cpu');reconstructed=fit_projection(raw['hidden'],raw['lengths'])
    if (set(projection)!=set(reconstructed) or any(projection[k].shape!=v.shape or projection[k].dtype!=v.dtype
            or not torch.isfinite(projection[k]).all() or not torch.allclose(projection[k],v,atol=2e-6,rtol=1e-5)
            for k,v in reconstructed.items())):raise ValueError('Learned projection is not reconstructed from its declared TRAIN responses')
    features=project(raw['hidden'],raw['lengths'],projection)
    fit=report['fit'];support=list(range(0,65,p['support_stride']));queries=[i for i in range(65) if i not in support]
    if (fit['support_indices']!=support or fit['query_indices']!=queries or fit['phase1']['steps']!=256
            or set(fit['phase2'])!=set(p['memory_modes']) or any(v['steps']!=256 for v in fit['phase2'].values())
            or fit['projection_fit_used_test'] is not False or fit['memory_scope']!='public-training-only'):
        raise ValueError('Response memory changed its equal training budget or support/query separation')
    permutation=torch.randperm(len(support),generator=torch.Generator().manual_seed(p['seed']+2))
    comparisons={};models={}
    for name in ['dolphin',*p['memory_modes']]:
        metadata=report['bindings'][name]
        if (metadata['projection_id']!=report['artifacts']['projection.safetensors']['sha256']
                or metadata['dolphin_state_id']!=fit['dolphin_state_id']):raise ValueError('Response memory and its learned space disagree')
        net=ResponseSelector(len(support));load(net,root/(name+'.safetensors'),metadata);net.eval();net.requires_grad_(False)
        before=tensor_digest(net.state_dict())
        with torch.no_grad():
            prediction=net(features,use_memory=name!='dolphin')
            bank=torch.nn.functional.pad(torch.cat([net.dolphin(features[support]),raw['targets'][support]],-1),(0,4))
        original=torch.tensor(report['train_predictions'][name],dtype=torch.float32)
        if original.shape!=(65,4) or not torch.allclose(prediction,original,atol=1e-6,rtol=1e-5):
            raise ValueError('Independent response reload changed its learned predictions')
        if name in {'zero','dolphin'}:bank.zero_()
        elif name=='shuffled':bank[:,8:12]=bank[permutation,8:12]
        if not torch.allclose(bank,net.public_memory,atol=2e-6,rtol=1e-5):raise ValueError('Response public bank or its declared ablation changed')
        if tensor_digest(net.state_dict())!=before:raise ValueError('Response recovery mutated its learned state')
        spent=0;choices=[]
        for values in prediction.tolist():
            choice=select_secondary(values,spent);spent+=choice=='specialist';choices.append(choice)
        comparisons[name]={'maximum_prediction_difference':float((prediction-original).abs().max()),
            'maximum_support_difference':float((bank-net.public_memory).abs().max()),'state_unchanged':True,
            'training_specialist_choices_under_cap':spent,'learned_state_id':before}
        models[name]=net
    optimizer_path=root/'optimizer-state.pt'
    with zipfile.ZipFile(optimizer_path) as archive:
        if len(archive.infolist())>10000 or sum(v.file_size for v in archive.infolist())>16*1024**2:
            raise ValueError('Response optimizer unpacked state exceeds its bound')
    saved=torch.load(optimizer_path,map_location='cpu',weights_only=True)
    if set(saved)!={'dolphin','memory'} or set(saved['memory'])!=set(p['memory_modes']):raise ValueError('Response optimizer schedule changed')
    states={'dolphin':saved['dolphin'],**saved['memory']};counts={}
    for name,optimizer in states.items():
        parameters=[v for k,v in models[name].named_parameters() if (k.startswith(('dolphin.','utility_head.'))
            if name=='dolphin' else k.startswith(('memory.','memory_vote.')) and not k.startswith('memory.memory_encoder.'))]
        identifiers=[identifier for group in optimizer['param_groups'] for identifier in group['params']]
        if len(identifiers)!=len(parameters) or set(identifiers)!=set(optimizer['state']):raise ValueError('Response optimizer parameter coverage changed')
        for identifier,parameter in zip(identifiers,parameters):
            state=optimizer['state'][identifier]
            if (set(state)!={'step','exp_avg','exp_avg_sq'} or float(state['step'])!=256
                    or state['exp_avg'].shape!=parameter.shape or state['exp_avg_sq'].shape!=parameter.shape
                    or any(not torch.isfinite(v).all() for v in state.values())):
                raise ValueError('Response optimizer moments or step count changed')
        counts[name]={'parameters_with_moments':len(parameters),'steps':256}
    return {'success':True,'protocol_id':protocol_id(),'comparisons':comparisons,'optimizer_states':counts,
        'projection_reconstructed_from_train':True,'language_model_loaded':False,'new_optimizer_updates':0,
        'new_generation_performed':False,'fresh_evaluation_performed':False,'useful_routing_proven':False,
        'model_promoted':False,'cuda_initialized':torch.cuda.is_initialized()}


def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--training-file',type=Path,required=True)
    parser.add_argument('--outcome-file',type=Path,action='append',required=True)
    parser.add_argument('--repair-file',type=Path,action='append',required=True);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    rows=training_rows(load_snapshot(args.training_file,'train'),args.outcome_file,args.repair_file)
    report=json.loads((args.directory/'diagnostic.json').read_text())
    with contextlib.redirect_stdout(sys.stderr):result=verify_saved_state(args.directory,report,rows)
    if result['cuda_initialized']:raise RuntimeError('Independent response recovery must remain on CPU')
    if args.output:
        with args.output.open('x') as output:json.dump(result,output,sort_keys=True)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':main()
