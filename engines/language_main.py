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
    torch.set_num_threads(2);torch.manual_seed(20261014)
    config=Qwen3Config(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                      num_attention_heads=2,num_key_value_heads=1,head_dim=16)
    model=Qwen3ForCausalLM(config).eval();install_adapter(model)
    inputs=torch.tensor([[1,2,3,4]])
    with torch.no_grad():result=model(input_ids=inputs).logits
    assert result.shape==(1,4,128) and torch.isfinite(result).all()
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'adapter.safetensors';save_adapter(model,path,'a'*64);load_adapter(model,path,'a'*64)
    forbidden=['src.network.lsp','desktop.personal_store','src.matriarca']
    assert not any(name in sys.modules for name in forbidden)
    return {'success':True,'frozen':bool(getattr(sys,'frozen',False)),'torch':torch.__version__,
            'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
            'qwen3_logits':128,'adapter_fixture_verified':True,'personal_state_loaded':False,
            'signing_identity_loaded':False,'scope':'Synthetic architecture fixture; not public-model inference'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.self_test:
        result=self_test()
        if args.report:args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result));return
    from src.learning.language_job import main as job
    job()


if __name__=='__main__':main()
