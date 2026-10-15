"""Install the bundled app for the current user; no system Python or root."""
from __future__ import annotations
import hashlib
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

VERSION = "0.4.0a11"


def bundle_digest(bundle):
    digest = hashlib.sha256()
    for path in sorted(bundle.rglob("*")):
        relative = path.relative_to(bundle).as_posix()
        if path.is_symlink():
            digest.update(f"link:{relative}:{os.readlink(path)}\n".encode())
        elif path.is_file():
            digest.update(relative.encode() + b"\0")
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()[:12]


def bundle_directory():
    if not getattr(sys, "frozen", False):
        raise RuntimeError("La instalación está disponible desde el paquete de la app")
    executable = Path(sys.executable).resolve()
    if sys.platform == "darwin":
        return next(p for p in executable.parents if p.suffix == ".app")
    return executable.parent


def install_user_app(bundle: Path, *, data_root=None, launcher_root=None):
    executable_name = "LixySwarm.exe" if sys.platform == "win32" else "LixySwarm"
    executable = bundle / "Contents" / "MacOS" / executable_name if sys.platform == "darwin" else bundle / executable_name
    digest = bundle_digest(bundle)
    if data_root is None:
        data_root = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) if sys.platform == "win32" else Path.home() / "Applications" if sys.platform == "darwin" else Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    root = Path(data_root) / "LixySwarm" / "App"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / (f"LixySwarm-{VERSION}-{digest}.app" if sys.platform == "darwin" else f"{VERSION}-{digest}")
    if not destination.exists():
        staging = root / f".install-{uuid.uuid4().hex}"
        shutil.copytree(bundle, staging, symlinks=True)
        staging.rename(destination)
    installed = destination / "Contents" / "MacOS" / executable_name if sys.platform == "darwin" else destination / executable_name
    if sys.platform == "win32":
        menu = Path(launcher_root) if launcher_root else Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        menu.mkdir(parents=True, exist_ok=True)
        launcher = menu / "LixySwarm.lnk"
        quote = lambda value: "'" + str(value).replace("'", "''") + "'"
        script = f"$s=New-Object -ComObject WScript.Shell; $l=$s.CreateShortcut({quote(launcher)}); $l.TargetPath={quote(installed)}; $l.WorkingDirectory={quote(destination)}; $l.Save()"
        import base64
        subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", base64.b64encode(script.encode("utf-16le")).decode()], check=True, capture_output=True)
    elif sys.platform == "darwin":
        launcher = Path(launcher_root) / "LixySwarm.app" if launcher_root else Path.home() / "Applications" / "LixySwarm.app"
        launcher.parent.mkdir(parents=True, exist_ok=True)
        temporary = launcher.with_name(".LixySwarm-new.app")
        if temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(destination, target_is_directory=True)
        temporary.replace(launcher)
    else:
        menu = Path(launcher_root) if launcher_root else Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / "applications"
        menu.mkdir(parents=True, exist_ok=True)
        launcher = menu / "lixyswarm.desktop"
        quote = str(installed).replace('\\', '\\\\').replace('"', '\\"').replace('`', '\\`').replace('$', '\\$').replace('%', '%%')
        launcher.write_text(f'[Desktop Entry]\nType=Application\nName=LixySwarm\nComment=Participa en la red de aprendizaje\nExec="{quote}"\nTerminal=false\nCategories=Education;Science;\n')
        launcher.chmod(0o755)
    return installed, launcher
