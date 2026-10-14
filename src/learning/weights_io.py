"""Streaming identities for local tensor files; no network/runtime identity."""
import hashlib
from pathlib import Path


def digest_file(path):
    digest=hashlib.sha256();size=0
    with Path(path).open('rb') as source:
        while block:=source.read(1024**2):digest.update(block);size+=len(block)
    return digest.hexdigest(),size
