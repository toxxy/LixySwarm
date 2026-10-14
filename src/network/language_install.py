"""Materialize only verified public dependencies; never execute model code here."""
import os
import shutil
import tempfile
from pathlib import Path

from src.release.language_family import load_bundle,dependencies,verify_bundle
from .artifact_store import ArtifactStore,digest_file


def materialize_family(directory,name,*,allowed=lambda:True):
    directory=Path(directory);root=directory/'public/language'
    store=ArtifactStore(root/'artifacts',max_artifact_bytes=4*1024**3,max_total_bytes=8*1024**3)
    bundle=load_bundle(root,name);release,descriptor=verify_bundle(bundle)
    return _materialize(directory,store,bundle,release,descriptor,dependencies(bundle),allowed)


def materialize_version(directory,name,release_id,*,allowed=lambda:True):
    from src.release.language_versions import LanguageVersionRegistry,validate_bundle,dependencies as version_dependencies
    directory=Path(directory);root=directory/'public/language'
    registry=LanguageVersionRegistry(root,name);bundle=registry.head()
    if bundle['release']['release_id']!=release_id:raise ValueError('Only the current verified language head may train')
    release,descriptor=validate_bundle(bundle)
    store=ArtifactStore(root/'artifacts',max_artifact_bytes=4*1024**3,max_total_bytes=8*1024**3)
    return _materialize(directory,store,bundle,release,descriptor,version_dependencies(bundle),allowed)


def _materialize(directory,store,bundle,release,descriptor,required,allowed):
    for identifier,size in required:
        if not allowed():raise RuntimeError('Preparación de lenguaje detenida')
        if not store.has(identifier) or store.manifest(identifier).size!=size:
            raise ValueError('La familia pública aún no está completa o verificada')
    models=directory/'language-models';models.mkdir(parents=True,exist_ok=True)
    destination=models/release.release_id
    declared={row['name']:(row['artifact_id'],row['bytes']) for row in descriptor['files']}
    declared['adapter.safetensors']=(release.model_artifact_id,descriptor['adapter_bytes'])
    def verify(target):
        if {p.name for p in target.iterdir()}!=set(declared):raise ValueError('Unexpected materialized language files')
        for filename,(identifier,size) in declared.items():
            if not allowed():raise RuntimeError('Preparación de lenguaje detenida')
            path=target/filename
            if path.is_symlink() or not path.is_file() or digest_file(path)!=(identifier,size):
                raise ValueError('Materialized language dependency failed verification')
    if destination.exists():verify(destination)
    else:
        with tempfile.TemporaryDirectory(prefix='.language-',dir=models) as temporary:
            staged=Path(temporary)/'base';staged.mkdir()
            for filename,(identifier,size) in declared.items():
                if not allowed():raise RuntimeError('Preparación de lenguaje detenida')
                source=store.object_path(identifier);target=staged/filename
                try:os.link(source,target)
                except OSError:shutil.copyfile(source,target)
            verify(staged)
            staged.rename(destination)
    # Preparing files neither selects a new active expert nor grants inference.
    return destination
