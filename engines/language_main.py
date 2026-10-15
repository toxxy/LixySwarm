"""Offline language worker without a node identity, personal store or publisher."""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

if not getattr(sys,'frozen',False):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'


def self_test():
    import torch
    from transformers.models.qwen3 import Qwen3Config,Qwen3ForCausalLM
    from src.learning.language_adapter import install_adapter,save_adapter,load_adapter
    from src.learning.language_gradient import save_gradient,load_gradient
    from src.learning.language_gradient_job import run as gradient_job
    from src.learning.language_candidate_job import apply_update
    from src.learning.language_policy import BATCH_POLICY_ID,retention_plan
    torch.set_num_threads(2);torch.manual_seed(20261014)
    config=Qwen3Config(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                      num_attention_heads=2,num_key_value_heads=1,head_dim=16)
    model=Qwen3ForCausalLM(config).eval();install_adapter(model)
    inputs=torch.tensor([[1,2,3,4]])
    with torch.no_grad():result=model(input_ids=inputs).logits
    assert result.shape==(1,4,128) and torch.isfinite(result).all()
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'adapter.safetensors';save_adapter(model,path,'a'*64);load_adapter(model,path,'a'*64)
        model(input_ids=inputs).logits.float().square().mean().backward()
        parameters=[parameter for parameter in model.parameters() if parameter.requires_grad]
        torch.nn.utils.clip_grad_norm_(parameters,1,error_if_nonfinite=True)
        path=Path(directory)/'gradient.safetensors'
        fields={'base_hash':'a'*64,'adapter_id':'b'*64,'dataset_id':'c'*64,'start':0,'count':1,'tokens':3}
        save_gradient(model,path,**fields);gradient=load_gradient(model,path,**fields)
        assert any(torch.count_nonzero(tensor) for tensor in gradient.values()) and callable(gradient_job)
        before={name:parameter.detach().clone() for name,parameter in model.named_parameters()}
        apply_update(model,[{'path':str(path),**fields}],.001)
        assert any(not torch.equal(before[name],parameter) for name,parameter in model.named_parameters() if parameter.requires_grad)
        assert all(torch.equal(before[name],parameter) for name,parameter in model.named_parameters() if not parameter.requires_grad)
    forbidden=['src.network.lsp','desktop.personal_store','src.matriarca']
    assert retention_plan(0,[1]*16)==[(0,.5),(1,.5)]
    assert not any(name in sys.modules for name in forbidden)
    return {'success':True,'frozen':bool(getattr(sys,'frozen',False)),'torch':torch.__version__,
            'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
            'qwen3_logits':128,'adapter_fixture_verified':True,'personal_state_loaded':False,
            'gradient_fixture_verified':True,'signing_identity_loaded':False,
            'candidate_fixture_verified':True,'batch_policy_id':BATCH_POLICY_ID,'batch_retention_fixture_verified':True,
            'scope':'Synthetic architecture/gradient/update fixture; not public-model learning'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.self_test:
        result=self_test()
        if args.report:args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result));return
    raw=sys.stdin.buffer.read(16*1024+1)
    if len(raw)>16*1024:raise ValueError('Language request exceeds its bound')
    request=json.loads(raw)
    action=request.get('action','inference')
    if action=='inference':
        from src.learning.language_job import run
        run(request)
    elif action in {'gradient','gradient-batch'}:
        from src.learning.language_gradient_job import run
        print(json.dumps(run(request),allow_nan=False),flush=True)
    elif action=='candidate':
        from src.learning.language_candidate_job import run
        print(json.dumps(run(request),ensure_ascii=False,allow_nan=False),flush=True)
    else:raise ValueError('Unknown locally approved language operation')


if __name__=='__main__':main()
