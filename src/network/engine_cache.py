"""Release only redundant, catalog-pinned code ZIPs; retain installed engines.

The preparation worker calls the internal operation while holding its existing
preparation lock. Administrative application requires both the inactive-node
lock and that same preparation lock, so it cannot alter a live cold trial.
"""
from pathlib import Path
import json

from .client_state import NodeStateLock
from .engine_catalog import cpu_engine,language_engine
from .engine_package import verify_package,_checked_digest,HEX
from .engine_profiles import engine_profile


def local_entries():
    return tuple(entry for entry in (cpu_engine(),language_engine('cpu'),language_engine('cuda')) if entry is not None)


def _owned_root(directory):
    directory=Path(directory)
    if (directory.is_symlink() or directory!=directory.resolve()
            or any((directory/name).is_symlink() for name in ['engines','engine-downloads'])):
        raise ValueError('Code-cache maintenance requires canonical owned state directories')
    return directory


def _paths(directory,entry):
    directory=_owned_root(directory)
    profile=engine_profile(entry.profile)
    if (not isinstance(entry.sha256,str) or not HEX.fullmatch(entry.sha256)
            or not isinstance(entry.manifest_sha256,str) or not HEX.fullmatch(entry.manifest_sha256)
            or type(entry.bytes) is not int or not 0<entry.bytes<=profile['max_download_bytes']):
        raise ValueError('Code-cache maintenance requires a bounded local catalog pin')
    return directory/'engine-downloads'/(entry.sha256+'.zip'),directory/'engines'/entry.manifest_sha256


def _referenced(directory,archive):
    def mentions(value):
        if isinstance(value,str):return Path(value).is_absolute() and Path(value).resolve()==archive
        if isinstance(value,list):return any(mentions(v) for v in value)
        if isinstance(value,dict):return any(mentions(v) for v in value.values())
        return False
    for name in ['compute-engine.json','language-engine.json','language-training.json']:
        path=Path(directory)/name
        if not path.exists():continue
        if path.is_symlink() or not path.is_file() or path.stat().st_size>64*1024:
            raise ValueError('Cannot inspect current engine references safely')
        if mentions(json.loads(path.read_text())):return True
    return False


def candidate_cache_bytes(directory,entries=None):
    """An upper estimate only: no download may rely on it without full checks."""
    total=0;seen=set()
    for entry in local_entries() if entries is None else entries:
        archive,installed=_paths(directory,entry)
        if (entry.sha256 not in seen and archive.is_file() and not archive.is_symlink()
                and archive.stat().st_size==entry.bytes and installed.is_dir() and not installed.is_symlink()
                and not _referenced(directory,archive)):
            seen.add(entry.sha256);total+=entry.bytes
    return total


def inspect_cache(directory,entries=None,*,check=lambda:None):
    """Read-only full verification, also safe while another owner holds locks."""
    results=[];seen=set()
    for entry in local_entries() if entries is None else entries:
        check();archive,installed=_paths(directory,entry)
        if entry.sha256 in seen or not archive.exists():continue
        seen.add(entry.sha256)
        if (archive.is_symlink() or not archive.is_file() or archive.stat().st_size!=entry.bytes
                or installed.is_symlink() or not installed.is_dir() or _referenced(directory,archive)):continue
        try:
            verify_package(installed,entry.manifest_sha256,profile=entry.profile,check=check)
        except (OSError,ValueError):continue
        if _checked_digest(archive,check)!=entry.sha256:continue
        check();stat=archive.stat()
        results.append({'archive_sha256':entry.sha256,'manifest_sha256':entry.manifest_sha256,
            'profile':entry.profile,'bytes':entry.bytes,'snapshot':[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]})
    return results


def _reclaim_locked(directory,entries=None,*,check=lambda:None):
    """Caller owns preparation lock; no code execution or settings mutation."""
    entries=local_entries() if entries is None else tuple(entries)
    verified=inspect_cache(directory,entries,check=check);removed=[]
    by_sha={entry.sha256:entry for entry in entries}
    for item in verified:
        check();archive,installed=_paths(directory,by_sha[item['archive_sha256']])
        stat=archive.stat()
        if (archive.is_symlink() or [stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]!=item['snapshot']
                or installed.is_symlink() or not installed.is_dir() or _referenced(directory,archive)):
            raise ValueError('Verified code cache changed before its retirement')
        entry=by_sha[item['archive_sha256']]
        verify_package(installed,entry.manifest_sha256,profile=entry.profile,check=check)
        check()
        if _checked_digest(archive,check)!=entry.sha256:
            raise ValueError('Code cache content changed during final verification')
        stat=archive.stat()
        if (archive.is_symlink() or [stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]!=item['snapshot']
                or _referenced(directory,archive)):
            raise ValueError('Verified code cache changed during final package verification')
        archive.unlink();removed.append({k:v for k,v in item.items() if k!='snapshot'})
    return {'removed':removed,'removed_cache_bytes':sum(item['bytes'] for item in removed),
        'installed_engines_removed':False,'models_removed':False,'operator_settings_changed':False}


def reclaim_inactive(directory,entries=None,*,check=lambda:None):
    """Offline administrator operation; a live node/trial owns its state lock."""
    directory=_owned_root(directory)
    entries=local_entries() if entries is None else tuple(entries)
    for entry in entries:_paths(directory,entry)
    with NodeStateLock(directory),NodeStateLock(directory/'preparation-lock'):
        return _reclaim_locked(directory,entries,check=check)
