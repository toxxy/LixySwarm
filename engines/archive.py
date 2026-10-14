"""Archive a verified platform build for a public, app-pinned download."""
from __future__ import annotations
import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.network.engine_package import digest_file, verify_package


def flatten_macos_ci_aliases(root, metadata):
    """Actions can expand framework directory symlinks into duplicate files.

    Add only the three known aliases, each identical to its already declared
    canonical framework member. No executable or runtime file is modified.
    """
    if metadata["platform"] != "darwin": return
    declared = {item["path"]: item for item in metadata["files"]}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    extra = actual - set(declared) - {"engine-package.json"}
    aliases = {
        "_internal/Python.framework/Resources/Info.plist": "_internal/Python.framework/Versions/3.12/Resources/Info.plist",
        "_internal/Python.framework/Versions/Current/Resources/Info.plist": "_internal/Python.framework/Versions/3.12/Resources/Info.plist",
        "_internal/Python.framework/Versions/Current/Python": "_internal/Python.framework/Versions/3.12/Python",
    }
    if not extra: return
    if extra != set(aliases): raise ValueError("Unexpected files in the macOS CI package")
    for name, canonical in aliases.items():
        original = declared[canonical]
        if (root / name).stat().st_size != original["bytes"] or digest_file(root / name) != original["sha256"]:
            raise ValueError("macOS framework alias does not match its verified member")
        metadata["files"].append({**original, "path": name})
    (root / "engine-package.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    root = options.package
    path = root / "engine-package.json"
    manifest_digest = digest_file(path)
    metadata = json.loads(path.read_text())
    flatten_macos_ci_aliases(root, metadata)
    manifest_digest = digest_file(path)
    metadata = verify_package(root, manifest_digest, target_platform=metadata["platform"], target_machine=metadata["machine"])
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(options.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for name in sorted([f["path"] for f in metadata["files"]] + ["engine-package.json"]):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if name == metadata["entrypoint"] else 0o100644) << 16
            with (root / name).open("rb") as source, zipped.open(info, "w", force_zip64=True) as destination:
                for block in iter(lambda: source.read(1024**2), b""):
                    destination.write(block)
    result = {"platform": metadata["platform"], "machine": metadata["machine"],
              "archive": options.output.name, "sha256": digest_file(options.output),
              "bytes": options.output.stat().st_size, "manifest_sha256": manifest_digest,
              "installed_bytes": sum(f["bytes"] for f in metadata["files"]) + path.stat().st_size}
    options.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
