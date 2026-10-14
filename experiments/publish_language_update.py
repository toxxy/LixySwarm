"""Publish a bounded, evaluated language update from known physical workers."""
import argparse
import json
import hashlib
import tempfile
import time
from pathlib import Path

from src.learning.public_bases import BASES
from src.network.artifact_store import ArtifactStore,ArtifactManifest,digest_file
from src.network.lsp import LSPIdentity
from src.release.adapter_format import validate_adapter
from src.release.language_family import canonical
from src.release.language_versions import LanguageVersionRegistry,build_update


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
    store=ArtifactStore(args.directory/'artifacts',max_artifact_bytes=4*1024**3,max_total_bytes=8*1024**3)
    registry=LanguageVersionRegistry(args.directory,args.base_name)
    head=registry.head();chain=registry.history(head['release']['release_id'])
    config_id=BASES[args.base_name]['config.json']
    if not store.has(config_id):raise ValueError('Pinned public model configuration is unavailable')
    with args.report_file.open('rb') as source:encoded_report=source.read(256*1024+1)
    if len(encoded_report)>256*1024:raise ValueError('Evaluation exceeds its bound')
    report=json.loads(encoded_report)
    if not args.adapter_file.is_file() or args.adapter_file.is_symlink() or args.adapter_file.stat().st_size>32*1024**2:
        raise ValueError('Only a bounded local adapter may be staged')
    with tempfile.TemporaryDirectory(prefix='language-candidate-',dir=store.incoming) as temporary:
        snapshot=Path(temporary)/'adapter.safetensors'
        with args.adapter_file.open('rb') as source:content=source.read(32*1024**2+1)
        if len(content)>32*1024**2:raise ValueError('Adapter exceeds its bound')
        snapshot.write_bytes(content)
        validate_adapter(snapshot,json.loads(store.object_path(config_id).read_text()),head['descriptor']['base_hash'])
        identifier,size=digest_file(snapshot)
        encoded=canonical(report)
        model=ArtifactManifest(identifier,size,'model','application/octet-stream',time.time())
        evaluation=ArtifactManifest(hashlib.sha256(encoded).hexdigest(),len(encoded),'evaluation','application/json',time.time())
        # Gate before committing any candidate/report to the public namespace.
        bundle=build_update(chain,model,evaluation,report,identity)
        store.commit_generated(snapshot,kind='model',media_type='application/octet-stream')
        evaluation_path=Path(temporary)/'evaluation.json';evaluation_path.write_bytes(encoded)
        store.commit_generated(evaluation_path,kind='evaluation',media_type='application/json')
        descriptor=Path(temporary)/'descriptor.json';descriptor.write_bytes(canonical(bundle['descriptor']))
        store.commit_generated(descriptor,kind='other',media_type='application/json')
        release=registry.install(chain+[bundle],store)
    print(json.dumps({'release_id':release.release_id,'sequence':release.sequence,'model_id':release.model_artifact_id,
        'scope':'Known-operator public language pilot; active expert permissions unchanged'}))


if __name__=='__main__':main()
