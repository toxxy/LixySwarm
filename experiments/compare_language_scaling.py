"""Compare one-PC and multi-PC reports for identical public training work."""
import argparse
import json
from pathlib import Path

from src.learning.language_scaling import compare_runs
from src.network.telemetry import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--distributed',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reports=[]
    for path in [args.baseline,args.distributed]:
        if not path.is_file() or path.stat().st_size>1024**2:parser.error('A bounded completed trial report is required')
        reports.append(json.loads(path.read_text()))
    result=compare_runs(*reports);atomic_json(args.output,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
