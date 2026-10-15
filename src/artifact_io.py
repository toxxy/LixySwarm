"""Atomic local artifact reports, independent of networking and tensor runtimes."""
import json
import os
from pathlib import Path
import tempfile
import time


def atomic_json(path: Path,value: dict) -> None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix=f'.{path.name}.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:json.dump(value,stream,sort_keys=True,allow_nan=False)
        deadline=time.monotonic()+1.
        while True:
            try:
                os.replace(temporary,path);break
            except PermissionError:
                if time.monotonic()>=deadline:raise
                time.sleep(.02)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
