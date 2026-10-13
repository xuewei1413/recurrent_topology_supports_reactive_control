"""Portable current-exchange replication, with validated complete-cell resume."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import tarfile
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import gymnasium as gym

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))
from experiments_v2.core.controller import UnifiedSNNController
from experiments_v2.core.encoder import ObservationEncoder
from experiments_v2.core.protocol import ControllerProtocol, TASK_SPECS
from experiments_v2.core.scaffold import NeuronScaffold
from _mediation.mc_injector import derive_seed, run_cell

OUT = ROOT / "results_local_replication_20261009"
INPUTS = OUT / "inputs"
CELLS = [("none", 1.0)] + [
    (mode, alpha) for mode in ("self_exact", "energy_random")
    for alpha in (1.0, 0.25, 0.5, 0.75)
] + [("action_only", 1.0), ("action_centered", 1.0), ("time_shuffle", 1.0)]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(data):
    return json.dumps(data, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(data))
    temporary.replace(path)


def provenance():
    files = list((ROOT / "code/experiments_v2/core").glob("*.py"))
    files += list((ROOT / "code/experiments_v2/tasks").glob("*.py"))
    files += [ROOT / "_mediation/mc_injector.py"]
    files += [ROOT / "local_replication" / name for name in
              ("run_current.py", "prepare_inputs.py", "test_injector.py")]
    files += [ROOT / "local_replication/requirements.txt"]
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p.read_bytes())
              for p in sorted(files)}
    source_hash = digest(canonical(hashes))
    for path in files:
        target = OUT / "source_execution" / source_hash / path.relative_to(ROOT)
        data = path.read_bytes()
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f"Source snapshot conflict: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
    runtime = {"python": sys.version, "executable": sys.executable,
               "platform": platform.platform(), "processor": platform.processor(),
               "packages": {name: version(name) for name in
                            ("numpy", "gymnasium", "cloudpickle", "farama-notifications", "typing-extensions")},
               "math_threads": 1, "cpu_count_logical": os.cpu_count()}
    manifest = json.loads((OUT / "input_manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["inputs"]:
        path = OUT / entry["path"]
        if digest(path.read_bytes()) != entry["sha256"]:
            raise ValueError(f"Input changed: {path}")
    return {"source_sha256": source_hash, "source_files": hashes, "runtime": runtime,
            "input_manifest_sha256": digest((OUT / "input_manifest.json").read_bytes())}


def build_controller(task_name, seed, d=None, rep=None):
    task, protocol = TASK_SPECS[task_name], ControllerProtocol()
    if d is None:
        path = INPUTS / "results_mc/topo" / task_name / f"F0_topology_{task_name}_seed{seed}.npy"
    else:
        path = INPUTS / "results_mc/variants" / f"var_{task_name}_seed{seed}_d{d}_r{rep}.npy"
    encoder = ObservationEncoder.load(INPUTS / "observation_stats" / f"observation_stats_{task_name}.json")
    scaffold = NeuronScaffold.from_seed(seed, task, protocol)
    return UnifiedSNNController(scaffold, np.load(path, allow_pickle=False), encoder, protocol)


def run_lineage(job):
    task_name, seed, variants, episodes, config_hash, profile = job
    controls = json.loads((INPUTS / "results_mc/frozen/controls_mc.json").read_text(encoding="utf-8"))
    protocol = json.loads((INPUTS / "results_mc/frozen/mc_protocol.json").read_text(encoding="utf-8"))
    directory = OUT / "current" / profile / task_name
    env = gym.make(TASK_SPECS[task_name].env_id)
    summaries = []
    try:
        for d, rep in variants:
            recipient = build_controller(task_name, seed, d, rep)
            donor = build_controller(task_name, seed)
            for mode, alpha in CELLS:
                name = f"rescue_{task_name}_s{seed}_d{d}_r{rep}_{mode}_a{str(alpha).replace('.', 'p')}.json"
                path = directory / name
                if path.exists():
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if (payload["config_sha256"] != config_hash or payload["episodes"] != episodes
                            or [row["ep"] for row in payload["rows"]] != episodes
                            or not payload.get("complete")):
                        raise ValueError(f"Incomplete or incompatible existing cell: {path}")
                else:
                    started = time.perf_counter()
                    rows = run_cell(recipient, donor, env, episodes, alpha, mode,
                                    derive_seed("run", task_name, seed, d, rep, mode, alpha, "rescue"),
                                    block_len=protocol["time_shuffle_block_len"],
                                    channel_perm=protocol["channel_permute"][task_name], keep_trace=True,
                                    shuffle_perm=controls["time_shuffle"][task_name]["perm_blocks"]
                                    if mode == "time_shuffle" else None)
                    payload = {"session": "local_replication_20261009", "profile": profile,
                               "config_sha256": config_hash, "task": task_name, "seed": seed,
                               "d": d, "rep": rep, "mode": mode, "alpha": alpha,
                               "stage": "rescue", "episodes": episodes, "rows": rows,
                               "elapsed_seconds": time.perf_counter() - started, "complete": True}
                    write_json(path, payload)
                elapsed = payload["elapsed_seconds"]
                steps = sum(row["n_steps"] for row in payload["rows"])
                summaries.append({"file": str(path.relative_to(OUT)).replace("\\", "/"),
                                  "task": task_name, "seed": seed, "d": d, "rep": rep,
                                  "mode": mode, "alpha": alpha, "elapsed_seconds": elapsed,
                                  "environment_steps": steps,
                                  "mean_return": float(np.mean([row["return"] for row in payload["rows"]])),
                                  "step_assertions": sum(row["energy_assertion"]["checks"] for row in payload["rows"])})
                print(f"{task_name} s{seed} d{d} r{rep} {mode} a{alpha}: "
                      f"{summaries[-1]['mean_return']:.2f}, {elapsed:.1f}s", flush=True)
    finally:
        env.close()
    return summaries


def historical_comparison(summaries, episodes):
    comparisons = []
    with tarfile.open(ROOT / "artifact_mc/results_mc.tar.gz", "r:gz") as archive:
        lookup = {member.name: member for member in archive.getmembers()}
        for item in summaries:
            filename = Path(item["file"]).name
            member = lookup.get(f"results_mc/runs/{item['task']}/{filename}")
            if member is None:
                continue
            old = json.load(archive.extractfile(member))
            new = json.loads((OUT / item["file"]).read_text(encoding="utf-8"))
            old_rows = {row["ep"]: row for row in old["rows"]}
            if not all(ep in old_rows for ep in episodes):
                continue
            differences = [row["return"] - old_rows[row["ep"]]["return"] for row in new["rows"]]
            comparisons.append({"file": item["file"], "mode": item["mode"], "task": item["task"],
                                "mean_new_minus_archive": float(np.mean(differences)),
                                "max_absolute_return_difference": float(np.max(np.abs(differences))),
                                "identical_episode_returns": int(sum(value == 0 for value in differences)),
                                "n_episodes": len(differences),
                                "interpretation": "Changed tie handling or direction cadence" if item["mode"] in
                                ("action_only", "action_centered", "energy_random") else "Environment/source replication check"})
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("benchmark", "single", "full"), default="benchmark")
    parser.add_argument("--task", choices=("CartPole", "Acrobot", "both"), default="both")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    tasks = ["CartPole", "Acrobot"] if args.task == "both" else [args.task]
    variants = [(0, 0)] if args.profile == "benchmark" else [(d, r) for d in range(3) for r in range(2)]
    episodes = list(range(86400, 86402 if args.profile == "benchmark" else 86420))
    prov = provenance()
    config = {"provenance": prov, "protocol": asdict(ControllerProtocol()), "cells": CELLS,
              "episodes": episodes, "variants": variants, "profile": args.profile,
              "random_direction_cadence": "one per environment step", "zero_count_tie": "lowest-index argmax",
              "status": "rerun of existing seeds, not independent confirmation"}
    config_hash = digest(canonical(config))
    config_path = OUT / "current" / args.profile / "config.json"
    if config_path.exists() and config_path.read_bytes() != canonical(config):
        raise ValueError(f"Existing configuration differs; use a separate result root: {config_path}")
    write_json(config_path, config)
    jobs = []
    for task_name in tasks:
        seeds = list(range(4000, 4012)) if task_name == "CartPole" else list(range(4012, 4024))
        if args.profile != "full":
            seeds = seeds[:1]
        jobs.extend((task_name, seed, variants, episodes, config_hash, args.profile) for seed in seeds)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        summaries = [item for batch in pool.map(run_lineage, jobs) for item in batch]
    report = {"profile": args.profile, "config_sha256": config_hash,
              "wall_seconds": time.perf_counter() - started, "cells": summaries,
              "historical_comparisons": historical_comparison(summaries, episodes)}
    write_json(OUT / "current" / args.profile / f"report_{args.task}.json", report)
    print(json.dumps({"cells_completed": len(summaries), "wall_seconds": report["wall_seconds"],
                      "environment_steps": sum(item["environment_steps"] for item in summaries)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
