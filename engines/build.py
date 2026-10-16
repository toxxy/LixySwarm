"""Freeze a source-independent CPU worker from an isolated CPU environment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--own',action='store_true',help='Include the own-swarm typed operations in the local profile')
    parser.add_argument('--network',action='store_true',help='Include versioned dynamic-membership own work')
    parser.add_argument('--bootstrap',action='store_true',help='Include validation of own-lineage snapshots for new nodes')
    parser.add_argument('--programs',action='store_true',help='Include pinned public-food programs and historical retention')
    options = parser.parse_args()
    import torch
    if torch.version.cuda is not None:
        raise ValueError("Build the CPU package using the isolated CPU-only environment")
    repo = Path(__file__).resolve().parents[1]
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    licenses = output / "licenses"
    licenses.mkdir(exist_ok=True)
    paths = list({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})
    for package in importlib.metadata.distributions(path=paths):
        for file in package.files or []:
            if any(word in str(file).lower() for word in ("license", "copying", "copyright")):
                source = Path(package.locate_file(file))
                if source.is_file():
                    destination = licenses / package.metadata["Name"] / str(file).replace("..", "_")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
    for source in (Path(sys.base_prefix) / "LICENSE.txt", Path("/usr/share/doc/python3.12/copyright")):
        if source.is_file():
            shutil.copy2(source, licenses / "Python-LICENSE.txt")
            break
    command = [sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm", "--onedir",
               "--name", "LixySwarmEngineCPU", "--distpath", str(output / "dist"),
               "--workpath", str(output / "work"), "--specpath", str(output),
               "--paths", str(repo), "--add-data", f"{licenses}{os.pathsep}licenses"]
    # The lightweight contribution package exposes these through __getattr__;
    # static import analysis cannot see that runtime module name.
    command += ["--hidden-import", "src.contribution.resource_governor"]
    for module in ("PySide6", "transformers", "tiktoken", "matplotlib", "scipy", "pandas", "tensorboard"):
        command += ["--exclude-module", module]
    command.append(str(repo / "engines" / "main.py"))
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    subprocess.run(command, cwd=repo, env=environment, check=True)
    package = output / "dist" / "LixySwarmEngineCPU"
    files = []
    for path in sorted(package.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            files.append({"path": path.relative_to(package).as_posix(), "bytes": path.stat().st_size,
                          "sha256": digest.hexdigest()})
    manifest = {"version": 1, "name": "LixySwarmEngineCPU", "platform": sys.platform,
                "machine": platform.machine(), "torch": torch.__version__,
                "operations": (["reference.inference.v1", "training.gradient.v1", "reference.candidate.v1"]
                    +(["swarm.gradient.v1","swarm.candidate.v1","swarm.audit.v1"] if options.own or options.network or options.bootstrap or options.programs else [])
                    +(["swarm.gradient.v2","swarm.candidate.v2"] if options.network or options.bootstrap or options.programs else [])
                    +(["swarm.validate.v1"] if options.bootstrap or options.programs else [])
                    +(["swarm.gradient.v3","swarm.candidate.v3"] if options.programs else [])),
                "entrypoint": "LixySwarmEngineCPU.exe" if sys.platform == "win32" else "LixySwarmEngineCPU",
                "files": files}
    (package / "engine-package.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"files": len(files), "bytes": sum(f["bytes"] for f in files)}))


if __name__ == "__main__":
    main()
