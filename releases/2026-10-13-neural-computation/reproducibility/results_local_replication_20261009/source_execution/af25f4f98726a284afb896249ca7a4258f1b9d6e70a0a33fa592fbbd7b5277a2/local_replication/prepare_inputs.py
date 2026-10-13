"""Preserve source and recover the archived current-experiment inputs."""

import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_local_replication_20261009"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Existing input differs; refusing overwrite: {path}")
    else:
        path.write_bytes(data)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "_mediation/mc_injector.py"
    historical = OUT / "source_before_fix/mc_injector.py"
    if not historical.exists():
        write_once(historical, source.read_bytes())
    archive_path = ROOT / "artifact_mc/results_mc.tar.gz"
    inputs = []
    counts = {"topologies": 0, "variants": 0}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            name = member.name
            if not member.isfile():
                continue
            if not (name.startswith("results_mc/topo/") or name.startswith("results_mc/variants/")
                    or name.startswith("results_mc/frozen/")):
                continue
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(name)
            data = archive.extractfile(member).read()
            if name.endswith(".npy"):
                adjacency = np.load(io.BytesIO(data), allow_pickle=False)
                assert adjacency.shape == (128, 128), name
                assert np.isin(adjacency, [0, 1]).all(), name
                assert not np.diag(adjacency).any(), name
                assert int(adjacency.sum()) == 1625, name
                counts["variants" if "/variants/" in name else "topologies"] += 1
            write_once(OUT / "inputs" / relative, data)
            inputs.append({"path": (Path("inputs") / relative).as_posix(), "source_member": name,
                           "sha256": sha(data), "bytes": len(data)})
    assert counts == {"topologies": 24, "variants": 144}, counts
    for task in ["CartPole", "Acrobot"]:
        for suffix in [".json", ".meta.json"]:
            source = ROOT / "results_v2_remote/results_v2/pilot/observation_stats" / f"observation_stats_{task}{suffix}"
            destination = OUT / "inputs/observation_stats" / source.name
            data = source.read_bytes()
            write_once(destination, data)
            inputs.append({"path": destination.relative_to(OUT).as_posix(), "source": source.relative_to(ROOT).as_posix(),
                           "sha256": sha(data), "bytes": len(data)})
    record = {"status": "Local rerun inputs; not recovered final historical execution code.",
              "archive_sha256": sha(archive_path.read_bytes()), "counts": counts, "inputs": inputs,
              "source_before_fix_sha256": sha(historical.read_bytes())}
    (OUT / "input_manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "counts": counts, "files": len(inputs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
