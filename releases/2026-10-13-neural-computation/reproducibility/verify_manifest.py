"""Check the public release's inherited data and code without simulations."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "SHA256_MANIFEST.json").read_text(encoding="utf-8"))
for entry in manifest["files"]:
    path = (ROOT / entry["path"]).resolve()
    assert path.is_relative_to(ROOT), entry["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], entry["path"]
print(f"Verified {len(manifest['files'])} data/source files.")
