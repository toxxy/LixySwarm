"""Locally registered retained public evidence shares the own-work disk limit."""
import json
from pathlib import Path
from .public_knowledge import managed_public_bytes


def usage(directory):
    directory=Path(directory).resolve();total=managed_public_bytes(directory)
    registry=directory/'own-storage-reservations.json'
    if not registry.exists():return total
    if registry.is_symlink() or registry.stat().st_size>8192:raise ValueError('Own storage registry exceeds its local bound')
    value=json.loads(registry.read_text())
    if (not isinstance(value,dict) or set(value)!={'version','directories'} or value['version']!=1
            or not isinstance(value['directories'],list) or len(value['directories'])>32):
        raise ValueError('Invalid own public storage registry')
    root=Path.home()/'.local/state';paths=[];seen=set()
    # Immutable engine files can be shared with retained test profiles. Count
    # their physical inode once across the native and registered public roots.
    for name in ['public','engines','language-models','engine-downloads','language-candidates']:
        for item in (directory/name).rglob('*'):
            if item.is_file() and not item.is_symlink():
                stat=item.stat();seen.add((stat.st_dev,stat.st_ino))
    for raw in value['directories']:
        if not isinstance(raw,str):raise ValueError('Own storage path must be locally registered')
        path=Path(raw)
        if (not path.is_absolute() or path.is_symlink() or path.resolve()!=path
                or path.parent!=root or path==directory or directory.is_relative_to(path)):
            raise ValueError('Own storage registry may name only separate local public state roots')
        if any(path==prior for prior in paths):raise ValueError('Duplicate own storage root')
        paths.append(path)
        for item in path.rglob('*'):
            if item.is_file() and not item.is_symlink():
                stat=item.stat();key=stat.st_dev,stat.st_ino
                if key not in seen:total+=stat.st_size;seen.add(key)
    return total
