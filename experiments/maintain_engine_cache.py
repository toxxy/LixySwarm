"""Inspect pinned redundant code ZIPs, or reclaim them on an inactive node."""
import argparse
import json
from pathlib import Path

from src.network.engine_cache import inspect_cache,reclaim_inactive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();directory=args.directory.expanduser().absolute()
    if args.apply:result={**reclaim_inactive(directory),'dry_run':False}
    else:
        entries=inspect_cache(directory)
        result={'dry_run':True,'verified_archives':[{k:v for k,v in e.items() if k!='snapshot'} for e in entries],
            'candidate_cache_bytes':sum(e['bytes'] for e in entries),'files_removed':0}
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()
