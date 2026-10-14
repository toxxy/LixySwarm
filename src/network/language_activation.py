"""Locally select a verified version without enabling or enlarging compute."""
from dataclasses import asdict,replace
from pathlib import Path

from src.release.language_versions import LanguageVersionRegistry
from .artifact_store import ArtifactStore
from .client_state import ClientSettings
from .language_compute import LanguageSettings
from .language_install import materialize_version


def activate_version(directory,name,release_id,*,expected_adapter_id):
    directory=Path(directory)
    original=LanguageSettings.load(directory);controls=ClientSettings.load(directory)
    if not original.enabled or original.base_name!=name or original.adapter_id!=expected_adapter_id:
        raise ValueError('Activation requires the unchanged, locally enabled expert of this family')
    registry=LanguageVersionRegistry(directory/'public/language',name)
    head=registry.head();chain=registry.history(head['release']['release_id'])
    if head['release']['release_id']!=release_id:
        raise ValueError('Activation requires the current accepted language head')
    if name=='qwen3-0.6b' and head['release']['sequence']==0:
        raise ValueError('The original small adaptation did not pass factual preservation')
    if expected_adapter_id not in {row['release']['model_artifact_id'] for row in chain}:
        raise ValueError('Activation cannot replace an unrelated locally selected expert')
    engine_fingerprint=asdict(original);policy_fingerprint=asdict(controls)
    def unchanged():
        return (asdict(LanguageSettings.load(directory))==engine_fingerprint
            and asdict(ClientSettings.load(directory))==policy_fingerprint
            and registry.head()['release']['release_id']==release_id)
    store=ArtifactStore(directory/'public/language/artifacts',max_artifact_bytes=4*1024**3,max_total_bytes=8*1024**3)
    # Repeat full artifact, tensor-format and evaluation verification before use.
    registry.install(chain,store)
    base=materialize_version(directory,name,release_id,allowed=unchanged)
    if not unchanged():raise ValueError('Local expert or controls changed during activation')
    selected=replace(original,adapter_id=head['release']['model_artifact_id'],
        base_directory=str(base),adapter_file=str(base/'adapter.safetensors'))
    selected.save(directory)
    return {'release_id':release_id,'previous_adapter_id':original.adapter_id,'adapter_id':selected.adapter_id,
        'selected':True,'compute_enabled_before':original.enabled,'resource_permissions_changed':False}
