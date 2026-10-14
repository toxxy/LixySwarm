"""Build an app with its own interpreter and without calculation engines."""
import argparse
import importlib.metadata
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    licenses = output / "licenses"; licenses.mkdir(exist_ok=True)
    paths = list({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})
    for package in importlib.metadata.distributions(path=paths):
        name = package.metadata["Name"]
        for file in package.files or []:
            if any(word in str(file).lower() for word in ("license", "copying", "copyright")):
                source = Path(package.locate_file(file))
                if source.is_file():
                    destination = licenses / name / str(file).replace("..", "_")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
    for candidate in (Path(sys.base_prefix) / "LICENSE.txt", Path("/usr/share/doc/python3.12/copyright")):
        if candidate.exists():
            shutil.copy2(candidate, licenses / "Python-LICENSE.txt")
            break
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--name", "LixySwarm",
               "--distpath", str(output / "dist"), "--workpath", str(output / "work"), "--specpath", str(output),
               "--paths", str(repo), "--add-data", f"{licenses}{os.pathsep}licenses", "--copy-metadata", "keyring",
               "--collect-submodules", "keyring.backends"]
    command += ["--add-data", f"{repo / 'desktop' / 'THIRD_PARTY.md'}{os.pathsep}licenses"]
    if sys.platform == "linux":
        # QtGui links the GLVND dispatch libraries even with raster/offscreen
        # rendering. PyInstaller normally excludes them as system graphics.
        listing = subprocess.check_output(["ldconfig", "-p"], text=True)
        for name in ("libEGL.so.1", "libOpenGL.so.0", "libGLX.so.0", "libGLdispatch.so.0"):
            matches = [line.split("=>", 1)[1].strip() for line in listing.splitlines() if line.strip().startswith(name + " ") and "=>" in line]
            if not matches:
                raise RuntimeError(f"Build machine is missing {name}")
            command += ["--add-binary", f"{matches[0]}{os.pathsep}."]
        for name in ("libegl1", "libopengl0", "libglx0", "libglvnd0"):
            source = Path("/usr/share/doc") / name / "copyright"
            if source.exists():
                shutil.copy2(source, licenses / f"{name}-copyright.txt")
    # Build from the isolated environment, not inherited ROS/developer paths.
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    if sys.platform in {"win32", "darwin"}:
        command.append("--windowed")
    for name in ("torch", "numpy", "tiktoken", "src.swarm", "src.network.training_worker", "src.network.gradient_aggregation"):
        command += ["--exclude-module", name]
    command.append(str(repo / "desktop" / "main.py"))
    subprocess.run(command, cwd=repo, env=environment, check=True)


if __name__ == "__main__":
    main()
