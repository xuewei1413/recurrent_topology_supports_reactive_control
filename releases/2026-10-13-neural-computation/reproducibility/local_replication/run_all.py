"""Explicit full-run entry; nothing starts merely by importing this module."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stage", choices=("current", "association", "all"), default="all")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    workers = ["--workers", str(args.workers)]
    commands = []
    if args.stage in ("current", "all"):
        commands += [["local_replication.run_current", "--profile", "full", *workers],
                     ["local_replication.analyze_current", "--profile", "full"]]
    if args.stage in ("association", "all"):
        commands += [["local_replication.evolve_association", "--profile", "full", *workers]]
        for cohort in ("v3", "c2"):
            commands.append(["local_replication.measure_association", "--profile", "full",
                             "--cohort", cohort, *workers])
        commands.append(["local_replication.measure_association", "--profile", "full",
                         "--cohort", "c2", "--stage", "final", *workers])
        commands.append(["local_replication.analyze_association", "--profile", "full"])
    state_path = ROOT / "results_local_replication_20261009/full_run/status.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    snapshot = state_path.parent / f"driver_{source_hash}.py"
    if not snapshot.exists():
        snapshot.write_bytes(Path(__file__).read_bytes())
    state = {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
             "stage_selection": args.stage, "workers": args.workers, "driver_sha256": source_hash,
             "total_stages": len(commands), "completed_stages": 0, "state": "running"}

    def save_state():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(state_path)

    try:
        for index, command in enumerate(commands):
            state["current_stage"] = command
            save_state()
            print("Starting: " + " ".join(command), flush=True)
            subprocess.run([sys.executable, "-u", "-m", *command], cwd=ROOT, check=True)
            state["completed_stages"] = index + 1
            save_state()
    except Exception as error:
        state["state"] = "failed"
        state["error"] = str(error)
        save_state()
        raise
    state["state"] = "complete"
    save_state()
    print("Selected stages completed. Manuscripts and historical archives were not modified.", flush=True)


if __name__ == "__main__":
    main()
