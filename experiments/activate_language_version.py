"""Select an accepted expert locally while preserving command and resources."""
import argparse
import json
from pathlib import Path

from src.learning.public_bases import BASES
from src.network.language_activation import activate_version


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--base-name',choices=BASES,required=True)
    parser.add_argument('--release-id',required=True)
    parser.add_argument('--expected-adapter-id',required=True)
    args=parser.parse_args()
    if args.directory.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error('Node state must stay outside the checkout')
    print(json.dumps(activate_version(args.directory,args.base_name,args.release_id,expected_adapter_id=args.expected_adapter_id)))


if __name__=='__main__':main()
