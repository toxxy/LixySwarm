"""Package the native tested bundle without flattening framework symlinks."""
from __future__ import annotations
import hashlib
import json
import platform
import sys
import tarfile
import zipfile
from pathlib import Path


def main():
    dist=Path(sys.argv[1]);output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=True)
    name="LixySwarm.app" if sys.platform=="darwin" else "LixySwarm"
    bundle=dist/name
    for path in bundle.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(bundle.resolve()):
            raise ValueError("A package symlink escaped its bundle")
    machine={"AMD64":"x86_64","aarch64":"arm64"}.get(platform.machine(),platform.machine())
    extension="zip" if sys.platform=="win32" else "tar.gz"
    archive=output/f"LixySwarm-0.4.0a2-{sys.platform}-{machine}.{extension}"
    if sys.platform=="win32":
        with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zipped:
            for path in sorted(bundle.rglob("*")):
                if path.is_file():zipped.write(path,path.relative_to(dist).as_posix())
    else:
        def normalize(info):
            info.uid=info.gid=0;info.uname=info.gname="";info.mtime=0
            return info
        with tarfile.open(archive,"w:gz",dereference=False) as zipped:
            zipped.add(bundle,arcname=name,filter=normalize)
    digest=hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda:stream.read(1024**2),b""):digest.update(block)
    metadata={"archive":archive.name,"bytes":archive.stat().st_size,"sha256":digest.hexdigest(),
              "platform":sys.platform,"machine":machine}
    archive.with_name(archive.name+".json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(metadata),flush=True)


if __name__=="__main__":main()
