"""Local recovery primitives for a quiesced physical-node cold-install trial.

This module performs no networking, node start/stop, model promotion or cache
eviction. Callers must stop both nodes and hold their NodeStateLocks first.
Payloads stay on the same filesystem; operator files and private banks are not
copied from a trial node to the original node.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from src.artifact_io import atomic_json
from src.network.artifact_store import digest_file
from src.network.engine_package import safe_relative,verify_package
from src.network.telemetry import verify_status

HEX=re.compile(r'[0-9a-f]{64}')
MANAGED=('public','engines','language-models','engine-downloads','language-candidates')
COUNTERS=('compute-status.json','language-status.json','language-training-status.json')


def accounted_bytes(nodes,extra_roots=()):
    """Count each actual file once across old/trial/research state roots."""
    seen=set();total=0
    roots=[Path(node)/name for node in nodes for name in MANAGED]+list(map(Path,extra_roots))
    for root in roots:
        for path in root.rglob('*'):
            if path.is_symlink() or not path.is_file():continue
            st=path.stat();key=st.st_dev,st.st_ino
            if key not in seen:seen.add(key);total+=st.st_size
    return total


def checked_file(path,identifier,size):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or digest_file(path)!=(identifier,size):
        raise ValueError('Recovery payload differs from its expected public content')


def move_owned(source,destination):
    """No copy fallback: a move must never temporarily double payload storage."""
    source=Path(source);destination=Path(destination)
    if source.is_symlink():raise ValueError('Recovery source must not be a symlink')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if source.stat().st_dev!=destination.parent.stat().st_dev:
        raise ValueError('Recovery requires one filesystem; payload copies are forbidden')
    os.replace(source,destination)


def merge_partial(source,destination,*,maximum):
    """Keep the longer matching prefix; a conflicting partial is never discarded."""
    source=Path(source);destination=Path(destination)
    if source.is_symlink() or not source.is_file() or source.stat().st_size>maximum:
        raise ValueError('Invalid recovery partial')
    if destination.exists():
        if destination.is_symlink() or not destination.is_file() or destination.stat().st_size>maximum:
            raise ValueError('Invalid existing recovery partial')
        common=min(source.stat().st_size,destination.stat().st_size)
        with source.open('rb') as a,destination.open('rb') as b:
            left=common
            while left:
                count=min(left,1024**2)
                if a.read(count)!=b.read(count):raise ValueError('Recovery partial prefixes conflict')
                left-=count
        if destination.stat().st_size>=source.stat().st_size:
            source.unlink();return
    move_owned(source,destination)


def merge_artifacts(source,target,*,allowed_partials):
    """Move verified completed objects and signed-dependency partials locally.

    Object manifests are written before the atomic move. A retry after either
    side of that move is idempotent, including an interrupted recovery itself.
    Unknown incoming files remain in the trial store for inspection.
    """
    if source.root.resolve()==target.root.resolve() or source.root.resolve().is_relative_to(target.root.resolve()) or target.root.resolve().is_relative_to(source.root.resolve()):
        raise ValueError('Recovery stores must be separate')
    planned=[]
    for path in sorted(source.objects.glob('*/*')):
        identifier=path.parent.name+path.name
        if not HEX.fullmatch(identifier):raise ValueError('Unexpected recovery object path')
        manifest=source.manifest(identifier)
        checked_file(path,identifier,manifest.size)
        destination=target.object_path(identifier)
        if destination.exists():checked_file(destination,identifier,manifest.size)
        planned.append((path,destination,manifest))
    moved=[]
    for path,destination,manifest in planned:
        # Avoid calling import_file: it copies the whole object into incoming/.
        if destination.exists():
            checked_file(destination,manifest.artifact_id,manifest.size)
            target._write_manifest(manifest)
            path.unlink()
        else:
            target._write_manifest(manifest)
            move_owned(path,destination)
        moved.append(manifest.artifact_id)
    partials=[];retained=[]
    for path in sorted(source.incoming.iterdir()):
        identifier=path.stem
        if path.suffix!='.part' or not HEX.fullmatch(identifier) or identifier not in allowed_partials:
            retained.append(path.name);continue
        maximum=allowed_partials[identifier]
        if type(maximum) is not int or not 0<=maximum<=target.max_artifact_bytes:
            raise ValueError('Partial is not a bounded signed dependency')
        if target.has(identifier):
            # Completed destination is checked by has(), so its partial is disposable.
            if target.manifest(identifier).size!=maximum:raise ValueError('Recovered artifact has another size')
            if path.is_symlink() or not path.is_file() or path.stat().st_size>maximum:
                raise ValueError('Invalid redundant recovery partial')
            path.unlink()
        else:merge_partial(path,target.incoming/path.name,maximum=maximum)
        partials.append(identifier)
    return {'completed':moved,'partials':partials,'retained_incoming':retained}


def merge_engine_downloads(source_node,target_node,entries):
    """Preserve resumable package bytes and move only app-pinned installations."""
    source_node=Path(source_node);target_node=Path(target_node)
    complete={};partial={};profiles={}
    for entry in entries:
        complete[entry.sha256+'.zip']=(entry.sha256,entry.bytes)
        partial[entry.sha256+'.part']=entry.bytes
        partial[entry.sha256+'.assembling']=entry.bytes
        profiles[entry.manifest_sha256]=entry
        for part in entry.parts:
            complete[part.sha256+'.piece']=(part.sha256,part.bytes)
            partial[part.sha256+'.part']=part.bytes
    retained=[];moved=[]
    for path in sorted((source_node/'engine-downloads').glob('*')):
        destination=target_node/'engine-downloads'/path.name
        if path.name in complete:
            identifier,size=complete[path.name];checked_file(path,identifier,size)
            if destination.exists():
                checked_file(destination,identifier,size);path.unlink()
            else:move_owned(path,destination)
            moved.append(path.name)
        elif path.name in partial:
            merge_partial(path,destination,maximum=partial[path.name]);moved.append(path.name)
        else:
            # An interrupted extraction is not assumed to be a verified package.
            retained.append(path.name)
    for folder in sorted((source_node/'engines').glob('*')):
        if folder.is_symlink() or folder.name not in profiles:
            raise ValueError('Trial contains an engine outside the locally pinned catalog')
        entry=profiles[folder.name]
        verify_package(folder,entry.manifest_sha256,profile=entry.profile)
        destination=target_node/'engines'/folder.name
        if destination.exists():
            verify_package(destination,entry.manifest_sha256,profile=entry.profile)
            # Keep this verified duplicate until the caller removes the owned trial.
            retained.append('engines/'+folder.name)
        else:move_owned(folder,destination)
    return {'download_files':moved,'retained_trial_paths':retained}


def newest_status(original,trial,identity):
    """Carry forward the same identity's highest signed telemetry sequence."""
    candidates=[]
    for directory in [Path(original),Path(trial)]:
        path=directory/'status.json'
        if not path.exists():continue
        value=json.loads(path.read_text());status=verify_status(value,sender_id=identity)
        candidates.append((status['sequence'],status['timestamp_ms'],value))
    if not candidates:raise ValueError('Recovery has no signed sequence checkpoint')
    candidates.sort(key=lambda row:row[:2])
    # Equal sequences cannot describe two different signed states.
    if len(candidates)==2 and candidates[0][0]==candidates[1][0] and candidates[0][2]!=candidates[1][2]:
        raise ValueError('Signed telemetry sequence fork during recovery')
    atomic_json(Path(original)/'status.json',candidates[-1][2])
    return candidates[-1][0]


def read_counter(directory,name):
    if name not in COUNTERS:raise ValueError('Unknown recovery counter')
    path=Path(directory)/name
    if not path.exists():return 0
    if path.stat().st_size>1024:raise ValueError('Recovery counter exceeds its bound')
    value=json.loads(path.read_text())
    if set(value)!={'completed'} or type(value['completed']) is not int or not 0<=value['completed']<=10**12:
        raise ValueError('Invalid recovery counter')
    return value['completed']


def restore_counters(original,trial,before):
    """Base plus trial, never current plus trial: recovery retries do not add credit."""
    result={}
    for name in COUNTERS:
        base=before.get(name,0);addition=read_counter(trial,name)
        if type(base) is not int or not 0<=base<=10**12 or base+addition>10**12:
            raise ValueError('Invalid recovery counter checkpoint')
        current=read_counter(original,name)
        if current not in {base,base+addition}:
            raise ValueError('Original counter changed outside this recovery')
        result[name]=base+addition
    for name,total in result.items():
        if read_counter(trial,name) or (Path(original)/name).exists():
            atomic_json(Path(original)/name,{'completed':total})
    return result


def repair_view(directory,store,release_id,declared):
    """Fill a known incomplete public view without replacing unknown local files."""
    if not HEX.fullmatch(release_id):raise ValueError('Invalid recovery release')
    destination=Path(directory)/'language-models'/release_id
    if destination.is_symlink():raise ValueError('Recovery view must be a local directory')
    expected={}
    for name,(identifier,size) in declared.items():
        relative=safe_relative(name)
        if len(relative.parts)!=1:raise ValueError('Public model view filenames must be flat')
        checked_file(store.object_path(identifier),identifier,size)
        expected[name]=(identifier,size)
    if destination.exists() and {p.name for p in destination.iterdir()}-expected.keys():
        raise ValueError('Original model view contains an unknown file')
    for name,(identifier,size) in expected.items():
        target=destination/name
        if target.exists() or target.is_symlink():checked_file(target,identifier,size)
    destination.mkdir(parents=True,exist_ok=True)
    for name,(identifier,size) in expected.items():
        target=destination/name
        if not target.exists():os.link(store.object_path(identifier),target)
    return destination
