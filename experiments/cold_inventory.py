"""Inventory and evict only independently replicated, signed public weights.

This administrative helper never stops a node or transfers a payload. The
caller holds its NodeStateLock. Remote inventories travel as compact evidence;
all replacement model bytes must still arrive through public LSP.
"""
import json
from pathlib import Path

from src.network.artifact_store import ArtifactStore,digest_file
from src.network.telemetry import verify_status
from src.release.language_versions import LanguageVersionRegistry,validate_bundle
from .cold_recovery import accounted_bytes

FAMILIES=('qwen3-0.6b','qwen3-1.7b')


def inventory(directory):
    directory=Path(directory).resolve()
    store=ArtifactStore(directory/'public/language/artifacts',max_artifact_bytes=4*1024**3)
    heads={};weights={};views=[]
    for name in FAMILIES:
        registry=LanguageVersionRegistry(directory/'public/language',name)
        head=registry.head();heads[name]=head['release']['release_id']
        for bundle in registry.history(head['release']['release_id']):
            release,descriptor=validate_bundle(bundle)
            for row in descriptor['files']:
                if row['name'].endswith('.safetensors'):
                    weights[row['artifact_id']]=row['bytes']
                    views.append((directory/'language-models'/release.release_id/row['name'],row['artifact_id'],row['bytes']))
            weights[release.model_artifact_id]=descriptor['adapter_bytes']
            views.append((directory/'language-models'/release.release_id/'adapter.safetensors',release.model_artifact_id,descriptor['adapter_bytes']))
    paths=[(store.object_path(i),i,size) for i,size in weights.items()]+views
    groups={};checked={};present=set();transferable=set()
    for path,identifier,size in paths:
        if not path.exists() and not path.is_symlink():continue
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(directory):
            raise ValueError('A public weight path escaped the local cache')
        st=path.stat();key=st.st_dev,st.st_ino
        if key not in checked:checked[key]=digest_file(path)
        if checked[key]!=(identifier,size):raise ValueError('Public weight differs from its signed descriptor')
        group=groups.setdefault(key,{'artifact_id':identifier,'bytes':size,'link_count':st.st_nlink,'paths':[]})
        relative=str(path.relative_to(directory))
        if relative not in group['paths']:group['paths'].append(relative)
        present.add(identifier)
        if path==store.object_path(identifier):
            manifest=store.manifest(identifier)
            if (manifest.artifact_id,manifest.size)!=(identifier,size):raise ValueError('Public provider manifest differs from its object')
            transferable.add(identifier)
    status=verify_status(json.loads((directory/'status.json').read_text()))
    return {'schema':'lixyswarm-cold-inventory-v1','node_id':status['node_id'],'heads':heads,'weights':weights,
        'verified_present':sorted(present),'verified_transferable':sorted(transferable),
        'inode_groups':list(groups.values()),'managed_bytes':accounted_bytes([directory]),
        'reclaimable_weight_bytes':sum(g['bytes'] for g in groups.values() if len(g['paths'])==g['link_count'])}


def validate_replica(local,provider):
    if (local.get('schema')!='lixyswarm-cold-inventory-v1' or provider.get('schema')!=local['schema']
            or provider.get('node_id')==local.get('node_id') or not provider.get('node_id')
            or provider.get('heads')!=local.get('heads') or provider.get('weights')!=local.get('weights')
            or set(provider.get('verified_transferable',[]))!=set(local['weights'])):
        raise ValueError('Eviction requires a distinct, fully verified public replica of these exact versions')


def evict_weights(directory,planned,provider):
    """Recheck every remaining path before unlinking; interrupted calls can retry.

    Only cache links listed by the original inventory are owned. A new link or
    changed file aborts the whole attempt before its first unlink. The signed
    metadata, manifests, operator files and all private files remain untouched.
    """
    directory=Path(directory).resolve();validate_replica(planned,provider)
    current=inventory(directory)
    if any(current[key]!=planned[key] for key in ['node_id','heads','weights']):
        raise ValueError('The physical identity or signed model history changed')
    declared={path:(g['artifact_id'],g['bytes']) for g in planned['inode_groups'] for path in g['paths']}
    remaining=[]
    for group in current['inode_groups']:
        if len(group['paths'])!=group['link_count']:
            raise ValueError('A public weight has an unaccounted external hardlink')
        for relative in group['paths']:
            if declared.get(relative)!=(group['artifact_id'],group['bytes']):
                raise ValueError('The cache gained a weight link outside the checkpoint')
            remaining.append(directory/relative)
    for path in remaining:path.unlink()
    return {'removed_public_links':len(remaining),'remaining_weight_links':0,
        'private_files_changed':False,'operator_files_changed':False}
