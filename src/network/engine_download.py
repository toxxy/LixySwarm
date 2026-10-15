"""Bounded, resumable HTTPS preparation of an app-pinned compute package."""
from __future__ import annotations
import hashlib
import math
import re
import shutil
import ssl
import stat
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import certifi

from .engine_package import (HEX, MAX_PACKAGE_BYTES, MAX_PACKAGE_FILES,
                             digest_file, install_package, safe_relative)
from .engine_profiles import engine_profile
from .engine_catalog import engine_regular_bytes

MAX_DOWNLOAD_BYTES = 512 * 1024**2
BLOCK_BYTES = 128 * 1024
RANGE_BYTES = 32 * 1024**2


class PreparationCancelled(RuntimeError):
    pass


class DownloadInterrupted(RuntimeError):
    """An incomplete HTTP response; verified installation has not begun."""


def download_pinned(url,size,partial,*,check,opener,rate_mbps,event,progress):
    """Bound each resumable connection without trusting bytes until final SHA."""
    offset=partial.stat().st_size if partial.exists() else 0
    if offset>size:raise ValueError('Partial engine download exceeds its bound')
    while offset<size:
        check()
        end=min(size-1,offset+RANGE_BYTES-1)
        headers={'User-Agent':'LixySwarm','Accept-Encoding':'identity','Range':f'bytes={offset}-{end}'}
        with opener(urllib.request.Request(url,headers=headers),timeout=5) as response:
            status=getattr(response,'status',200)
            if status==206:
                match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                if not match or int(match[1])!=offset or not offset<=int(match[2])<=end or int(match[3])!=size:
                    raise ValueError('Engine download returned an unexpected range')
                response_end=int(match[2])
            elif status==200:
                offset=0;response_end=size-1  # Range declined: a deliberate full restart.
            else:raise ValueError('Unexpected engine download response')
            length=response.headers.get('Content-Length')
            if length is not None and (not length.isdecimal() or int(length)!=response_end-offset+1):
                raise ValueError('Engine download length differs from its response range')
            with partial.open('ab' if offset else 'wb') as stream:
                while True:
                    check();began=time.monotonic();block=response.read(BLOCK_BYTES)
                    if not block:break
                    if offset+len(block)>response_end+1:raise ValueError('Engine download exceeded its requested range')
                    stream.write(block);stream.flush();offset+=len(block);progress(offset)
                    rate=float(rate_mbps())
                    if not math.isfinite(rate) or rate<=0:raise PreparationCancelled('Presupuesto de conexión desactivado')
                    delay=len(block)*8/(rate*1_000_000)-(time.monotonic()-began)
                    while delay>0:check();waited=min(delay,.05);event.wait(waited);delay-=waited
            if offset!=response_end+1:
                raise DownloadInterrupted(f'Descarga interrumpida en {offset} de {size} bytes; se conserva para reanudarla')
    check()


def discard_verified_cache(path,identifier,size):
    """Release only this download's hash-verified, disposable cache object."""
    if not path.exists():return
    if path.is_symlink() or not path.is_file() or path.stat().st_size!=size or digest_file(path)!=identifier:
        raise ValueError('Refusing to discard an unverified engine cache object')
    path.unlink()


class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        if urlparse(newurl).scheme != "https":
            raise ValueError("Compute downloads require HTTPS throughout redirects")
        return super().redirect_request(request, response, code, message, headers, newurl)


def verified_part(part,root,*,check,opener,rate_mbps,event,progress):
    if urlparse(part.url).scheme!='https' or not HEX.fullmatch(part.sha256) or type(part.bytes) is not int or not 0<part.bytes<=1024**3:
        raise ValueError('Invalid locally pinned compute part')
    final=root/(part.sha256+'.piece');partial=root/(part.sha256+'.part')
    if final.exists():
        if final.stat().st_size!=part.bytes or digest_file(final)!=part.sha256:raise ValueError('Cached compute part failed verification')
        progress(part.bytes);return final
    download_pinned(part.url,part.bytes,partial,check=check,opener=opener,rate_mbps=rate_mbps,event=event,progress=progress)
    check()
    if partial.stat().st_size!=part.bytes or digest_file(partial)!=part.sha256:raise ValueError('Compute part failed verification')
    partial.replace(final);return final


def prepare_engine(entry, directory, *, allowed, rate_mbps, progress=lambda *_: None,
                   cancel_event=None, opener=None):
    """Download only a locally selected catalog entry, then verify and install.

    Cancellation preserves a partial archive. No package code runs here, and
    selecting the engine leaves resource and training-authority permissions as
    they were. The app explicitly requests preparation separately from compute.
    """
    profile=getattr(entry,'profile','reference-cpu');selected=engine_profile(profile)
    regular_bytes=engine_regular_bytes(entry)
    if not HEX.fullmatch(entry.sha256) or not HEX.fullmatch(entry.manifest_sha256) or type(entry.bytes) is not int or not 0 < entry.bytes <= selected['max_download_bytes'] or type(entry.installed_bytes) is not int or not 0 < entry.installed_bytes <= selected['max_package_bytes']+2*1024**2:
        raise ValueError("Invalid locally pinned engine download bounds")
    if urlparse(entry.url).scheme != "https":
        raise ValueError("Compute downloads require HTTPS")
    root = Path(directory) / "engine-downloads"
    root.mkdir(parents=True, exist_ok=True)
    archive = root / (entry.sha256 + ".zip")
    partial = root / (entry.sha256 + ".part")
    event = cancel_event or threading.Event()
    def check():
        if event.is_set() or not allowed():
            raise PreparationCancelled("Preparación detenida; se conserva la descarga para reanudarla")
    check()
    if opener is None:
        opener = urllib.request.build_opener(HTTPSRedirect(), urllib.request.HTTPSHandler(
            context=ssl.create_default_context(cafile=certifi.where()))).open
    if archive.exists() and (archive.stat().st_size != entry.bytes or digest_file(archive) != entry.sha256):
        raise ValueError("Cached engine archive failed verification")
    parts=getattr(entry,'parts',())
    if parts:
        if (selected['name']!='LixySwarmLanguageCUDA' or not 1<=len(parts)<=8
                or any(not HEX.fullmatch(p.sha256) or type(p.bytes) is not int or not 0<p.bytes<=1024**3
                       or urlparse(p.url).scheme!='https' for p in parts)
                or sum(p.bytes for p in parts)!=entry.bytes):
            raise ValueError('Invalid locally pinned multipart compute archive')
    if parts and not archive.exists():
        completed=0;verified=[]
        for part in parts:
            check();base=completed
            path=verified_part(part,root,check=check,opener=opener,rate_mbps=rate_mbps,event=event,
                progress=lambda count:progress('downloading',base+count,entry.bytes))
            verified.append(path);completed+=part.bytes
        assembling=root/(entry.sha256+'.assembling');digest=hashlib.sha256();size=0
        with assembling.open('wb') as destination:
            for path in verified:
                with path.open('rb') as source:
                    while block:=source.read(BLOCK_BYTES):
                        check();destination.write(block);digest.update(block);size+=len(block)
        check()
        if size!=entry.bytes or digest.hexdigest()!=entry.sha256:raise ValueError('Assembled compute archive failed verification')
        assembling.replace(archive)
    if not archive.exists():
        download_pinned(entry.url,entry.bytes,partial,check=check,opener=opener,rate_mbps=rate_mbps,event=event,
            progress=lambda count:progress('downloading',count,entry.bytes))
        check(); progress("verifying", 0, entry.bytes)
        if not partial.exists() or partial.stat().st_size != entry.bytes or digest_file(partial) != entry.sha256:
            if partial.exists(): partial.replace(root/(entry.sha256+".invalid"))
            raise ValueError("Engine archive hash or length mismatch")
        partial.replace(archive)
    check()
    # The complete archive is verified before its pieces can be released.
    # Cancellation during assembly still retains the individual pieces.
    for part in parts:discard_verified_cache(root/(part.sha256+'.piece'),part.sha256,part.bytes)
    # Hash the whole archive before extraction; then bound every entry as well.
    extracted = root / (entry.sha256 + ".unpacked")
    if extracted.exists(): shutil.rmtree(extracted)
    extracted.mkdir()
    try:
        # Compact CUDA packages include aliases to large libraries. Extraction
        # writes regular files; installed logical bytes also count those aliases.
        progress("extracting", 0, regular_bytes)
        with zipfile.ZipFile(archive) as zipped:
            members = zipped.infolist()
            if not 1 <= len(members) <= MAX_PACKAGE_FILES + 1:
                raise ValueError("Engine archive file count exceeds its bound")
            seen = set(); size = 0; unpacked = 0;links=[]
            for member in members:
                check()
                relative = safe_relative(member.filename)
                mode = member.external_attr >> 16
                if member.is_dir() or member.filename in seen or member.flag_bits & 1 or stat.S_ISLNK(mode) and not selected['archive_links']:
                    raise ValueError("Engine archive contains an unsupported entry")
                seen.add(member.filename)
                if stat.S_ISLNK(mode):
                    if not 0<member.file_size<=1024:raise ValueError('Engine archive link exceeds its bound')
                    link=zipped.read(member).decode('utf-8')
                    if not link or Path(link).is_absolute() or '\\' in link or ':' in link:
                        raise ValueError('Engine archive link escaped its root')
                    links.append((relative,link));continue
                size += member.file_size
                if size > regular_bytes or member.file_size > selected['max_package_bytes']:
                    raise ValueError("Engine archive exceeded its unpacked size")
                target = extracted.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, target.open("wb") as destination:
                    while block := source.read(BLOCK_BYTES):
                        check(); destination.write(block); unpacked += len(block)
                progress("extracting", unpacked, regular_bytes)
            for relative,link in links:
                check();target=extracted.joinpath(*relative.parts)
                if not (target.parent/link).resolve().is_relative_to(extracted.resolve()):
                    raise ValueError('Engine archive link escaped its root')
                target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(link)
            for relative,link in links:
                target=extracted.joinpath(*relative.parts)
                if not target.resolve().is_relative_to(extracted.resolve()) or not target.is_file():
                    raise ValueError('Engine archive link is not an internal regular file')
                size+=target.stat().st_size
                if size>entry.installed_bytes:raise ValueError('Engine archive link exceeded its unpacked size')
            if size != entry.installed_bytes:
                raise ValueError("Engine archive unpacked size does not match")
            if getattr(entry,'regular_file_bytes',None) is not None and unpacked!=regular_bytes:
                raise ValueError('Engine regular-file size differs from its app-pinned size')
        check(); progress("installing", 0, entry.installed_bytes)
        entrypoint = install_package(extracted, entry.manifest_sha256, directory,profile=profile,consume_download=True)
        discard_verified_cache(archive,entry.sha256,entry.bytes)
        progress("complete", entry.installed_bytes, entry.installed_bytes)
        return entrypoint
    finally:
        if extracted.exists():shutil.rmtree(extracted)
