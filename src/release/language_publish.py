"""Publish a bounded, evaluated language update from known physical workers."""
import json
import hashlib
import tempfile
import time
from pathlib import Path

from src.learning.public_bases import BASES
from src.network.artifact_store import ArtifactStore,ArtifactManifest,digest_file
from src.release.adapter_format import validate_adapter
from src.release.language_family import canonical
from src.release.language_versions import LanguageVersionRegistry,build_update


def publish_update(directory,base_name,adapter_file,report_file,identity,*,before_install=lambda _bundle:None):
    """Gate, snapshot and install an update; journal its signature before commit."""
    directory,adapter_file,report_file=map(Path,(directory,adapter_file,report_file))
    if not report_file.is_file() or report_file.is_symlink() or report_file.stat().st_size>256*1024:
        raise ValueError('Evaluation must be a bounded local file')
    store=ArtifactStore(directory/'artifacts',max_artifact_bytes=4*1024**3,max_total_bytes=8*1024**3)
    registry=LanguageVersionRegistry(directory,base_name)
    head=registry.head();chain=registry.history(head['release']['release_id'])
    config_id=BASES[base_name]['config.json']
    if not store.has(config_id):raise ValueError('Pinned public model configuration is unavailable')
    with report_file.open('rb') as source:encoded_report=source.read(256*1024+1)
    if len(encoded_report)>256*1024:raise ValueError('Evaluation exceeds its bound')
    report=json.loads(encoded_report)
    if not adapter_file.is_file() or adapter_file.is_symlink() or adapter_file.stat().st_size>32*1024**2:
        raise ValueError('Only a bounded local adapter may be staged')
    with tempfile.TemporaryDirectory(prefix='language-candidate-',dir=store.incoming) as temporary:
        snapshot=Path(temporary)/'adapter.safetensors'
        with adapter_file.open('rb') as source:content=source.read(32*1024**2+1)
        if len(content)>32*1024**2:raise ValueError('Adapter exceeds its bound')
        snapshot.write_bytes(content)
        validate_adapter(snapshot,json.loads(store.object_path(config_id).read_text()),head['descriptor']['base_hash'])
        identifier,size=digest_file(snapshot)
        encoded=canonical(report)
        model=ArtifactManifest(identifier,size,'model','application/octet-stream',time.time())
        evaluation=ArtifactManifest(hashlib.sha256(encoded).hexdigest(),len(encoded),'evaluation','application/json',time.time())
        # Gate before committing any candidate/report to the public namespace.
        bundle=build_update(chain,model,evaluation,report,identity)
        store.import_file(snapshot,kind='model',media_type='application/octet-stream')
        evaluation_path=Path(temporary)/'evaluation.json';evaluation_path.write_bytes(encoded)
        store.import_file(evaluation_path,kind='evaluation',media_type='application/json')
        descriptor=Path(temporary)/'descriptor.json';descriptor.write_bytes(canonical(bundle['descriptor']))
        store.import_file(descriptor,kind='other',media_type='application/json')
        before_install(bundle)
        release=registry.install(chain+[bundle],store)
    return {'release_id':release.release_id,'sequence':release.sequence,'model_id':release.model_artifact_id,
        'scope':'Known-operator public language pilot; active expert permissions unchanged'}


