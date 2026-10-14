"""Record bytes of every regular file in a platform package."""
import hashlib
import sys
from pathlib import Path

root, output = map(Path, sys.argv[1:])
rows = []
for path in sorted(root.rglob("*")):
    if path.is_file():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(root).as_posix()}")
output.write_text("\n".join(rows) + "\n")
