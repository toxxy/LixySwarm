"""Independently reconstruct the two pinned TRAIN repair artifacts on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from src.learning.language_repair_calibration import load_outcomes,verify_calibration,protocol_id
from src.learning.public_math import canonical,load_snapshot

ARTIFACTS=(('bc4dc6687d2b76fd9ec9c32cb8607c3eff5984ebcf66b02c475edbdcaf11e6c2',30717),
    ('a226cf96b7aa4b8ab47c13524c4cf7ea95a5fdf3e62dbd167db5cce5cea50ab5',31409))


def audit(training_file,outcome_files,calibration_files):
    if len(calibration_files)!=2:raise ValueError('Both physical TRAIN repair artifacts are required')
    snapshot=load_snapshot(training_file,'train')
    outcomes=load_outcomes([str(v) for v in outcome_files],snapshot)
    shards=[]
    for shard,(path,(identifier,size)) in enumerate(zip(calibration_files,ARTIFACTS)):
        path=Path(path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size!=size:
            raise ValueError('Repair artifact has another size or file type')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=identifier:raise ValueError('Repair artifact differs from its physical pin')
        value=json.loads(raw)
        if value.get('shard')!=shard or canonical(value)!=raw:
            raise ValueError('Repair artifact has another shard or encoding')
        verified=verify_calibration(value,snapshot,outcomes)
        shards.append({'shard':shard,'artifact_id':identifier,'artifact_bytes':size,'verification':verified})
    numeric=['cases','new_generation_calls','parent_correct','repair_fallback_correct','specialist_fallback_correct',
        'specialist_only_correct_after_invalid_parent','repair_only_correct_after_invalid_parent',
        'new_repair_output_tokens','new_repair_generation_seconds','new_repair_input_tokens']
    totals={key:sum(row['verification'][key] for row in shards) for key in numeric}
    return {'success':True,'protocol_id':protocol_id(),'shards':shards,'totals':totals,
        'generation_performed':False,'optimizer_updates':False,'model_promoted':False,
        'neural_routing_evaluated':False,'evaluation_data_used':False,
        'cost_scope':shards[0]['verification']['cost_scope'],'tensor_engine_loaded':'torch' in sys.modules}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-file',type=Path,required=True)
    parser.add_argument('--outcome-file',type=Path,action='append',required=True)
    parser.add_argument('--calibration-file',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=audit(args.training_file,args.outcome_file,args.calibration_file)
    if result['tensor_engine_loaded']:raise RuntimeError('The standalone independent reader must not load a tensor engine')
    if args.output:
        with args.output.open('x') as target:json.dump(result,target,ensure_ascii=False,sort_keys=True)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True),flush=True)


if __name__=='__main__':main()
