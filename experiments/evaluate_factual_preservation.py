"""Compare base and adapter against a fixed public paraphrase preservation gate."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base-directory',type=Path,required=True)
    parser.add_argument('--adapter',type=Path,required=True);parser.add_argument('--base-name',choices=['qwen3-0.6b','qwen3-1.7b'],required=True)
    parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    from src.learning.public_bases import BASES,base_hash
    from src.learning.public_quality import quality_cases,preservation_verdict,correct_answer
    from src.network.artifact_store import digest_file
    for name,identifier in BASES[args.base_name].items():
        if digest_file(args.base_directory/name)[0]!=identifier:raise ValueError('Pinned public base mismatch')
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from src.learning.language_adapter import install_adapter,load_adapter,adapters_enabled
    torch.set_num_threads(4);torch.manual_seed(20261014);torch.cuda.set_per_process_memory_fraction(.25)
    tokenizer=AutoTokenizer.from_pretrained(args.base_directory,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(args.base_directory,local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,device_map={'':'cuda'}).eval()
    install_adapter(model);load_adapter(model,args.adapter,base_hash(args.base_name));results={}
    for enabled in [False,True]:
        rows=[]
        with adapters_enabled(model,enabled):
            for case in quality_cases():
                text=tokenizer.apply_chat_template([{'role':'user','content':case['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                inputs=tokenizer(text,return_tensors='pt').to('cuda')
                with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=64,do_sample=False,use_cache=True)
                answer=tokenizer.decode(out[0,inputs.input_ids.shape[1]:],skip_special_tokens=True).strip()
                rows.append({'case':case['case'],'prompt':case['prompt'],'answer':answer,'passed':correct_answer(case,answer)})
        results['candidate' if enabled else 'baseline']=rows
        print(json.dumps({'adapter':enabled,'passed':sum(row['passed'] for row in rows),'total':len(rows)}),flush=True)
    verdict=preservation_verdict(results['baseline'],results['candidate'])
    report={'base_name':args.base_name,'base_hash':base_hash(args.base_name),'adapter_id':digest_file(args.adapter)[0],
        'private_inputs':False,'results':results,'verdict':verdict}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(verdict),flush=True)


if __name__=='__main__':main()
