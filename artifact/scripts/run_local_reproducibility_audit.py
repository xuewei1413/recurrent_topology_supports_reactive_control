"""Run local checks for the Neural Computation submission artifact.

The audit intentionally checks the narrow claims that should be reproducible
from this repository: TO/TG/TI statistical boundaries, endpoint declarations,
manifest integrity, reproduction documentation, and figure-script inventory.
"""

from __future__ import annotations

import csv
import glob
import importlib
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifact"
PROCESSED = ARTIFACT / "processed_results"
REPORT_PATH = ARTIFACT / "REPRODUCIBILITY_REPORT.md"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


RESULTS: list[CheckResult] = []
NUMERIC_ANCHORS: list[str] = []


def record(name: str, status: str, detail: str) -> None:
    RESULTS.append(CheckResult(name=name, status=status, detail=detail))


def run_command(cmd: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_stat(
    rows: Iterable[dict[str, str]], task: str, analysis: str, contrast: str
) -> dict[str, str]:
    for row in rows:
        if (
            row.get("task") == task
            and row.get("analysis") == analysis
            and row.get("contrast") == contrast
        ):
            return row
    raise KeyError(f"Missing row: {task} / {analysis} / {contrast}")


def assert_close(label: str, value: float, expected: float, tol: float) -> None:
    delta = abs(value - expected)
    if delta > tol:
        raise AssertionError(
            f"{label}: observed {value:.6g}, expected {expected:.6g}, delta {delta:.3g}"
        )


def check_environment() -> None:
    lines = [
        f"Python {platform.python_version()}",
        f"Executable: {Path(sys.executable).name}",
        f"Platform: {platform.platform()}",
    ]
    for module_name in ["numpy", "scipy"]:
        try:
            module = importlib.import_module(module_name)
            lines.append(f"{module_name} {getattr(module, '__version__', 'unknown')}")
        except Exception as exc:  # pragma: no cover - diagnostic path
            record("Python dependencies", "FAIL", f"Cannot import {module_name}: {exc}")
            return
    record("Python dependencies", "PASS", "; ".join(lines))


def check_git_release_context() -> None:
    proc = run_command(["git", "rev-parse", "--show-toplevel"], timeout=30)
    if proc.returncode == 0:
        record("Repository context", "PASS", "Git repository detected.")
    else:
        record(
            "Repository context",
            "WARN",
            "This working directory is not a git repository; create the final tag or release in the actual submission repository.",
        )


def check_endpoint_config_and_sources() -> None:
    import numpy as np

    config_path = ARTIFACT / "configs" / "analysis_endpoints.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    controls = config["to_tg_ti_controls"]

    cart_cfg = controls["cartpole"]
    cart_path = ROOT / cart_cfg["source"]
    cart = np.load(cart_path, allow_pickle=True)
    cart_counts = Counter(item.get("mode") for item in cart if isinstance(item, dict))
    expected_cart = {mode: cart_cfg["n_per_condition"] for mode in cart_cfg["conditions"]}
    if dict(cart_counts) != expected_cart:
        raise AssertionError(f"CartPole counts {dict(cart_counts)} != {expected_cart}")

    mc_cfg = controls["mountaincar"]
    mc_paths = sorted(glob.glob(str(ROOT / mc_cfg["source_glob"])))
    mc_rows = [json.loads(Path(path).read_text(encoding="utf-8")) for path in mc_paths]
    mc_counts = Counter(row.get("mode") for row in mc_rows)
    expected_mc = {mode: mc_cfg["n_per_condition"] for mode in mc_cfg["conditions"]}
    if dict(mc_counts) != expected_mc:
        raise AssertionError(f"MountainCar counts {dict(mc_counts)} != {expected_mc}")

    record(
        "Endpoint config and source counts",
        "PASS",
        f"CartPole {dict(cart_counts)} from {cart_cfg['source']}; MountainCar {dict(mc_counts)} from {mc_cfg['source_glob']}",
    )


def regenerate_statistics() -> None:
    proc = run_command([sys.executable, str(ARTIFACT / "scripts" / "compute_statistical_boundaries.py")])
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-20:])
        record("Regenerate statistical artifact", "FAIL", tail)
        return
    expected = [
        PROCESSED / "cartpole_coevolution_seed_level.csv",
        PROCESSED / "mountaincar_coevolution_seed_level.csv",
        PROCESSED / "statistical_boundaries_to_tg_ti.csv",
        PROCESSED / "robust_statistics_to_tg_ti.csv",
        PROCESSED / "supplement_table_to_tg_ti.tex",
    ]
    missing = [str(path.relative_to(ROOT)) for path in expected if not path.exists()]
    if missing:
        record("Regenerate statistical artifact", "FAIL", f"Missing outputs: {missing}")
        return
    record(
        "Regenerate statistical artifact",
        "PASS",
        "Rebuilt seed-level CSVs, statistical summaries, robust summaries, and supplement table.",
    )


def validate_numeric_anchors() -> None:
    rows = read_csv(PROCESSED / "statistical_boundaries_to_tg_ti.csv")
    robust_rows = read_csv(PROCESSED / "robust_statistics_to_tg_ti.csv")

    checks = [
        ("CartPole", "group_mean_ci", "TO", "mean_or_diff", 498.075, 0.001),
        ("CartPole", "group_mean_ci", "TG", "mean_or_diff", 496.750, 0.001),
        ("CartPole", "group_mean_ci", "TI", "mean_or_diff", 497.825, 0.001),
        ("CartPole", "welch_pairwise_difference_ci", "TO-TI", "mean_or_diff", 0.25, 0.001),
        ("CartPole", "welch_pairwise_difference_ci", "TO-TI", "ci_low", -2.7946, 0.005),
        ("CartPole", "welch_pairwise_difference_ci", "TO-TI", "ci_high", 3.2946, 0.005),
        ("MountainCar", "group_mean_ci", "TO", "mean_or_diff", 895.4755, 0.001),
        ("MountainCar", "group_mean_ci", "TG", "mean_or_diff", 1049.3975, 0.001),
        ("MountainCar", "group_mean_ci", "TI", "mean_or_diff", 984.0194, 0.001),
        ("MountainCar", "welch_pairwise_difference_ci", "TO-TI", "mean_or_diff", -88.5439, 0.005),
        ("MountainCar", "welch_pairwise_difference_ci", "TO-TG", "mean_or_diff", -153.9220, 0.005),
        ("MountainCar", "welch_pairwise_difference_ci", "TG-TI", "mean_or_diff", 65.3781, 0.005),
    ]
    for task, analysis, contrast, field, expected, tol in checks:
        row = find_stat(rows, task, analysis, contrast)
        assert_close(f"{task} {contrast} {field}", float(row[field]), expected, tol)

    mc_success = {
        (row["task"], row["mode"]): f"{row['success_count']}/{row['n']}"
        for row in robust_rows
    }
    if mc_success[("MountainCar", "TO")] != "16/20":
        raise AssertionError("MountainCar TO success count changed")
    if mc_success[("MountainCar", "TG")] != "17/20":
        raise AssertionError("MountainCar TG success count changed")
    if mc_success[("MountainCar", "TI")] != "16/20":
        raise AssertionError("MountainCar TI success count changed")

    NUMERIC_ANCHORS.extend(
        [
            "CartPole means: TO 498.075, TG 496.750, TI 497.825 (primary metric: best_fitness; n=10 each).",
            "CartPole Welch CIs: TO-TI 0.25 [-2.79, 3.29]; TO-TG 1.32 [-1.76, 4.41]; TG-TI -1.07 [-4.39, 2.24].",
            "MountainCar means: TO 895.476, TG 1049.398, TI 984.019 (primary metric: strict_20ep_mean; n=20 each).",
            "MountainCar Welch CIs: TO-TI -88.54 [-449.89, 272.81]; TO-TG -153.92 [-511.65, 203.80]; TG-TI 65.38 [-314.69, 445.45].",
            "MountainCar success counts: TO 16/20, TG 17/20, TI 16/20.",
        ]
    )
    record("Numeric anchors", "PASS", "All manuscript-level statistical anchors match regenerated CSV files.")


def validate_manifest() -> None:
    rows = read_csv(ARTIFACT / "MANIFEST.csv")
    missing: list[str] = []
    for row in rows:
        path_text = row.get("path", "")
        if "*" in path_text:
            if not glob.glob(str(ROOT / path_text)):
                missing.append(path_text)
        elif path_text and not (ROOT / path_text).exists():
            missing.append(path_text)
    if missing:
        record("Manifest paths", "FAIL", f"Missing manifest paths: {missing}")
    else:
        record("Manifest paths", "PASS", f"All {len(rows)} manifest entries resolve locally.")


def validate_reproduction_docs() -> None:
    required_docs = {
        "README.md": [
            "public code and processed-data artifact",
            "Quick Start",
            "Regenerating Figures From Processed Results",
        ],
        "REPRODUCING.md": [
            "Main Manuscript Result Map",
            "TO/TG/TI statistics",
            "Continuous control",
        ],
        "code/experiments/README.md": [
            "Experiment Code Index",
            "Minimal Audit Path",
            "Experiment Groups",
        ],
    }
    missing: list[str] = []
    for rel_path, tokens in required_docs.items():
        path = ROOT / rel_path
        if not path.exists():
            missing.append(rel_path)
            continue
        raw = path.read_bytes()
        if b"\x00" in raw:
            missing.append(f"{rel_path} contains NUL bytes")
            continue
        text = raw.decode("utf-8", errors="replace")
        for token in tokens:
            if token not in text:
                missing.append(f"{rel_path}: {token}")
    if missing:
        record("Reproduction documentation", "FAIL", f"Missing or invalid documentation anchors: {missing}")
    else:
        record("Reproduction documentation", "PASS", "README, REPRODUCING, and experiment index are present and readable.")


def validate_figure_script_inventory() -> None:
    expected = [
        "code/plot_fig01.py",
        "code/gen_fig04_mountaincar_decoding.py",
        "code/gen_fig06_spike_timing_corrected.py",
        "code/gen_fig07_connection_ablation.py",
        "code/gen_fig08_coevolution.py",
        "code/gen_fig09_adaptation.py",
        "code/gen_fig10_cross_task.py",
        "code/gen_fig11.py",
        "code/gen_fig13_fingerprint_corrected.py",
        "code/gen_fig14_hopper.py",
        "code/gen_fig15_lunarlander_headroom.py",
    ]
    missing = [rel for rel in expected if not (ROOT / rel).exists()]
    if missing:
        record("Figure script inventory", "FAIL", f"Missing figure scripts: {missing}")
    else:
        record("Figure script inventory", "PASS", f"All {len(expected)} expected figure scripts are present.")


def write_report() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    counts = Counter(result.status for result in RESULTS)
    lines = [
        "# Reproducibility Report",
        "",
        f"Generated: {now}",
        "",
        "## Overall Status",
        "",
        f"- PASS: {counts.get('PASS', 0)}",
        f"- WARN: {counts.get('WARN', 0)}",
        f"- FAIL: {counts.get('FAIL', 0)}",
        "",
        "## Check Results",
        "",
    ]
    for result in RESULTS:
        lines.append(f"- {result.status} - {result.name}: {result.detail}")

    lines.extend(
        [
            "",
            "## Regenerated Numeric Anchors",
            "",
        ]
    )
    if NUMERIC_ANCHORS:
        lines.extend(f"- {item}" for item in NUMERIC_ANCHORS)
    else:
        lines.append("- Numeric anchors were not regenerated because an earlier check failed.")

    lines.extend(
        [
            "",
            "## Reproduction Commands",
            "",
            "```powershell",
            "python -m pip install -r artifact\\requirements.txt",
            "python artifact\\scripts\\compute_statistical_boundaries.py",
            "python artifact\\scripts\\run_local_reproducibility_audit.py",
            "python artifact\\scripts\\scan_public_artifact.py",
            "```",
            "",
            "## Boundaries And Remaining Actions",
            "",
            "- The CartPole TO/TG/TI endpoint is final best_fitness with n=10 per condition; strict 20-episode fields are released for transparency.",
            "- The MountainCar TO/TG/TI endpoint is strict_20ep_mean with n=20 per condition, plus success and goal-episode rates.",
            "- Pairwise Welch intervals are descriptive confidence intervals, not equivalence or non-inferiority tests.",
            "- This local folder may not be the final git repository; before submission, create a fixed release tag or commit identifier in the real repository.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    check_environment()
    check_git_release_context()

    for name, func in [
        ("Endpoint config and source counts", check_endpoint_config_and_sources),
        ("Regenerate statistical artifact", regenerate_statistics),
        ("Numeric anchors", validate_numeric_anchors),
        ("Manifest paths", validate_manifest),
        ("Reproduction documentation", validate_reproduction_docs),
        ("Figure script inventory", validate_figure_script_inventory),
    ]:
        try:
            func()
        except Exception as exc:  # pragma: no cover - diagnostic path
            record(name, "FAIL", str(exc))

    write_report()
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    for result in RESULTS:
        print(f"{result.status:4} {result.name}: {result.detail}")
    return 1 if any(result.status == "FAIL" for result in RESULTS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
