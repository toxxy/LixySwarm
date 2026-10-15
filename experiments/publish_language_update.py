"""Publish a bounded, evaluated language update from known physical workers."""
import argparse
import json
from pathlib import Path

from src.learning.public_bases import BASES
from src.network.lsp import LSPIdentity
from src.release.language_publish import publish_update


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--publisher-key',type=Path,required=True)
    parser.add_argument('--base-name',choices=BASES,required=True)
    parser.add_argument('--adapter-file',type=Path,required=True)
    parser.add_argument('--report-file',type=Path,required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    if any(path.resolve().is_relative_to(repo) for path in (args.directory,args.adapter_file,args.report_file,args.publisher_key)):
        parser.error('Knowledge, reports and identity must stay outside the checkout')
    if not args.report_file.is_file() or args.report_file.is_symlink() or args.report_file.stat().st_size>256*1024:
        parser.error('Evaluation must be a bounded local file')
    identity=LSPIdentity.load(str(args.publisher_key))
    if identity is None:raise ValueError('An existing locally authorized publisher is required')
    print(json.dumps(publish_update(args.directory,args.base_name,args.adapter_file,args.report_file,identity)))


if __name__=='__main__':main()
