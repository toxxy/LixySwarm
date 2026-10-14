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

MAX_DOWNLOAD_BYTES = 512 * 1024**2
BLOCK_BYTES = 128 * 1024


class PreparationCancelled(RuntimeError):
    pass


class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        if urlparse(newurl).scheme != "https":
            raise ValueError("Compute downloads require HTTPS throughout redirects")
        return super().redirect_request(request, response, code, message, headers, newurl)


def prepare_engine(entry, directory, *, allowed, rate_mbps, progress=lambda *_: None,
                   cancel_event=None, opener=None):
    """Download only a locally selected catalog entry, then verify and install.

    Cancellation preserves a partial archive. No package code runs here, and
    selecting the engine leaves resource and training-authority permissions as
    they were. The app explicitly requests preparation separately from compute.
    """
    if not HEX.fullmatch(entry.sha256) or not HEX.fullmatch(entry.manifest_sha256) or type(entry.bytes) is not int or not 0 < entry.bytes <= MAX_DOWNLOAD_BYTES or type(entry.installed_bytes) is not int or not 0 < entry.installed_bytes <= MAX_PACKAGE_BYTES:
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
    if not archive.exists():
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > entry.bytes:
            raise ValueError("Partial engine archive exceeds its bound")
        if offset < entry.bytes:
            headers = {"User-Agent": "LixySwarm/0.4.0a2", "Accept-Encoding": "identity"}
            if offset: headers["Range"] = f"bytes={offset}-"
            with opener(urllib.request.Request(entry.url, headers=headers), timeout=5) as response:
                status = getattr(response, "status", 200)
                if offset and status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not re.fullmatch(rf"bytes {offset}-[0-9]+/{entry.bytes}", content_range):
                        raise ValueError("Engine download resumed from an unexpected range")
                elif status == 200:
                    offset = 0  # A server may decline Range; restart deliberately.
                else:
                    raise ValueError("Unexpected engine download response")
                with partial.open("ab" if offset else "wb") as stream:
                    while True:
                        check()
                        began = time.monotonic()
                        block = response.read(BLOCK_BYTES)
                        if not block: break
                        if offset + len(block) > entry.bytes:
                            raise ValueError("Engine download exceeded its pinned size")
                        stream.write(block); stream.flush(); offset += len(block)
                        progress("downloading", offset, entry.bytes)
                        rate = float(rate_mbps())
                        if not math.isfinite(rate) or rate <= 0:
                            raise PreparationCancelled("El presupuesto de conexión está desactivado")
                        delay = len(block) * 8 / (rate * 1_000_000) - (time.monotonic() - began)
                        while delay > 0:
                            check(); waited = min(delay, 0.05); event.wait(waited); delay -= waited
        check(); progress("verifying", 0, entry.bytes)
        if not partial.exists() or partial.stat().st_size != entry.bytes or digest_file(partial) != entry.sha256:
            if partial.exists(): partial.replace(root/(entry.sha256+".invalid"))
            raise ValueError("Engine archive hash or length mismatch")
        partial.replace(archive)
    check()
    # Hash the whole archive before extraction; then bound every entry as well.
    extracted = root / (entry.sha256 + ".unpacked")
    if extracted.exists(): shutil.rmtree(extracted)
    extracted.mkdir()
    try:
        progress("extracting", 0, entry.installed_bytes)
        with zipfile.ZipFile(archive) as zipped:
            members = zipped.infolist()
            if not 1 <= len(members) <= MAX_PACKAGE_FILES + 1:
                raise ValueError("Engine archive file count exceeds its bound")
            seen = set(); size = 0; unpacked = 0
            for member in members:
                check()
                relative = safe_relative(member.filename)
                mode = member.external_attr >> 16
                if member.is_dir() or stat.S_ISLNK(mode) or member.filename in seen or member.flag_bits & 1:
                    raise ValueError("Engine archive contains an unsupported entry")
                seen.add(member.filename); size += member.file_size
                if size > entry.installed_bytes or member.file_size > MAX_PACKAGE_BYTES:
                    raise ValueError("Engine archive exceeded its unpacked size")
                target = extracted.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, target.open("wb") as destination:
                    while block := source.read(BLOCK_BYTES):
                        check(); destination.write(block); unpacked += len(block)
                progress("extracting", unpacked, entry.installed_bytes)
            if size != entry.installed_bytes:
                raise ValueError("Engine archive unpacked size does not match")
        check(); progress("installing", 0, entry.installed_bytes)
        entrypoint = install_package(extracted, entry.manifest_sha256, directory)
        progress("complete", entry.installed_bytes, entry.installed_bytes)
        return entrypoint
    finally:
        shutil.rmtree(extracted)
