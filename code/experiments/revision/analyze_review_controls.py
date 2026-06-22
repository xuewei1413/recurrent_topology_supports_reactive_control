"""Pre-submission revision analyses requested by the internal review.

The script uses existing seed-level artifacts and fixed topologies to add
three lightweight controls:

1. TOST/non-inferiority sensitivity for the TO/TG/TI comparisons.
2. An architecture-matched non-spiking rate-substitution control.
3. A metadata-confounding audit for the structural fingerprint analysis.

Outputs are written to ``code/experiments/results/revision_controls``.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = ROOT / "code"
OUT_DIR = CODE_DIR / "experiments" / "results" / "revision_controls"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CODE_DIR))

from src.snn import SNNController  # noqa: E402


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# 1. TOST/non-inferiority sensitivity


def read_seed_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def welch_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = float(a.mean() - b.mean())
    va = float(a.var(ddof=1))
    vb = float(b.var(ddof=1))
    na = len(a)
    nb = len(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return {
            "diff": diff,
            "se": 0.0,
            "df": float("inf"),
            "ci95_low": diff,
            "ci95_high": diff,
        }
    df_num = (va / na + vb / nb) ** 2
    df_den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = df_num / df_den
    half = float(stats.t.ppf(0.975, df) * se)
    return {
        "diff": diff,
        "se": se,
        "df": float(df),
        "ci95_low": diff - half,
        "ci95_high": diff + half,
    }


def tost_pvalue(diff: float, se: float, df: float, delta: float) -> dict[str, float]:
    """Two one-sided tests for -delta < diff < delta."""
    if se == 0:
        p_lower = 0.0 if diff > -delta else 1.0
        p_upper = 0.0 if diff < delta else 1.0
    else:
        t_lower = (diff + delta) / se
        t_upper = (diff - delta) / se
        p_lower = float(1.0 - stats.t.cdf(t_lower, df))
        p_upper = float(stats.t.cdf(t_upper, df))
    return {
        "p_lower_diff_gt_minus_delta": p_lower,
        "p_upper_diff_lt_delta": p_upper,
        "p_tost": max(p_lower, p_upper),
    }


def run_tost_sensitivity() -> list[dict[str, Any]]:
    specs = [
        {
            "task": "CartPole",
            "path": ROOT / "artifact" / "processed_results" / "cartpole_coevolution_seed_level.csv",
            "metric": "best_fitness",
            "deltas": [2.0, 5.0, 10.0],
        },
        {
            "task": "MountainCar",
            "path": ROOT / "artifact" / "processed_results" / "mountaincar_coevolution_seed_level.csv",
            "metric": "strict_20ep_mean",
            "deltas": [100.0, 300.0, 500.0],
        },
    ]

    rows: list[dict[str, Any]] = []
    for spec in specs:
        data = read_seed_csv(spec["path"])
        groups = {
            mode: np.array(
                [float(r[spec["metric"]]) for r in data if r["mode"] == mode],
                dtype=np.float64,
            )
            for mode in ("TO", "TG", "TI")
        }
        for a, b in (("TO", "TI"), ("TO", "TG"), ("TG", "TI")):
            ws = welch_stats(groups[a], groups[b])
            tcrit_90 = float(stats.t.ppf(0.95, ws["df"]))
            min_delta_for_alpha05 = abs(ws["diff"]) + tcrit_90 * ws["se"]
            for delta in spec["deltas"]:
                p = tost_pvalue(ws["diff"], ws["se"], ws["df"], delta)
                rows.append(
                    {
                        "task": spec["task"],
                        "contrast": f"{a}-{b}",
                        "metric": spec["metric"],
                        "delta": delta,
                        "diff": ws["diff"],
                        "se": ws["se"],
                        "df": ws["df"],
                        "ci95_low": ws["ci95_low"],
                        "ci95_high": ws["ci95_high"],
                        "p_tost": p["p_tost"],
                        "equivalent_alpha05": p["p_tost"] < 0.05,
                        "min_delta_for_alpha05": min_delta_for_alpha05,
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# 2. Non-spiking rate-substitution control


class RateSubstitutionController:
    """Same topology and E/I weights, but continuous rates replace spikes/reset."""

    def __init__(
        self,
        config: dict[str, Any],
        adjacency: np.ndarray,
        neuron_types: np.ndarray,
        gain: float = 4.0,
        mode: str = "sigmoid",
    ) -> None:
        self.n = int(config["n_total"])
        self.n_in = int(config["n_input"])
        self.n_out = int(config["n_output"])
        self.tau = float(config["tau"])
        self.thresh = float(config["threshold"])
        self.t_steps = int(config["T_steps"])
        self.dt = 1.0
        self.weight_scale = float(config["weight_scale"])
        self.gain = float(gain)
        self.mode = mode
        self.w = adjacency * neuron_types[:, np.newaxis] * self.weight_scale
        self.v = np.zeros(self.n, dtype=np.float64)
        self.rates = np.zeros(self.n, dtype=np.float64)

    def reset(self) -> None:
        self.v[:] = 0.0
        self.rates[:] = 0.0

    def encode_state(self, state: np.ndarray) -> np.ndarray:
        i_ext = np.zeros(self.n_in, dtype=np.float64)
        n_dims = self.n_in // 2
        for i in range(n_dims):
            val = np.tanh(state[i]) * 4.0
            i_ext[2 * i] = max(0.0, val)
            i_ext[2 * i + 1] = max(0.0, -val)
        return i_ext

    def nonlinearity(self, v: np.ndarray) -> np.ndarray:
        x = self.gain * (v - self.thresh)
        if self.mode == "relu":
            return np.clip(x, 0.0, 1.0)
        if self.mode == "tanh":
            return np.clip(0.5 * (np.tanh(x) + 1.0), 0.0, 1.0)
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    def forward(self, state: np.ndarray) -> int:
        i_ext = self.encode_state(np.array(state, dtype=np.float64))
        leak = 1.0 - self.dt / self.tau
        accum = np.zeros(self.n, dtype=np.float64)
        for _ in range(self.t_steps):
            self.v *= leak
            if np.any(self.rates):
                self.v += self.rates @ self.w
            self.v[: self.n_in] += i_ext * self.dt / self.tau
            self.rates = self.nonlinearity(self.v)
            accum += self.rates
        return int(np.argmax(accum[-self.n_out :]))


def evaluate_cartpole_controller(
    controller: Any,
    gravity: float = 9.8,
    n_episodes: int = 20,
    seed_base: int = 70000,
) -> np.ndarray:
    env = gym.make("CartPole-v1")
    env.unwrapped.gravity = gravity
    scores = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        controller.reset()
        total = 0.0
        for _ in range(500):
            action = controller.forward(np.array(obs, dtype=np.float64))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        scores.append(total)
    env.close()
    return np.asarray(scores, dtype=np.float64)


def run_rate_substitution_control() -> dict[str, Any]:
    config = {
        "n_total": 128,
        "n_input": 8,
        "n_output": 2,
        "tau": 5.0,
        "threshold": 0.8,
        "T_steps": 20,
        "weight_scale": 0.6,
    }
    adj = np.load(ROOT / "data" / "key_npy" / "best_topology.npy")
    types = np.load(ROOT / "data" / "key_npy" / "neuron_types.npy")

    snn = SNNController(config, adj, types)
    snn_scores = evaluate_cartpole_controller(snn, n_episodes=20)

    rows = []
    best_rate: dict[str, Any] | None = None
    for mode in ("sigmoid", "tanh", "relu"):
        for gain in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
            rate_ctrl = RateSubstitutionController(config, adj, types, gain=gain, mode=mode)
            scores = evaluate_cartpole_controller(rate_ctrl, n_episodes=20)
            row = {
                "controller": "rate_substitution",
                "mode": mode,
                "gain": gain,
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "min": float(scores.min()),
                "max": float(scores.max()),
                "n_episodes": int(len(scores)),
            }
            rows.append(row)
            if best_rate is None or row["mean"] > best_rate["mean"]:
                best_rate = dict(row)

    return {
        "topology": "data/key_npy/best_topology.npy",
        "config": config,
        "snn_reference": {
            "controller": "spiking_lif",
            "mean": float(snn_scores.mean()),
            "std": float(snn_scores.std()),
            "min": float(snn_scores.min()),
            "max": float(snn_scores.max()),
            "n_episodes": int(len(snn_scores)),
        },
        "best_rate_substitution": best_rate,
        "all_rate_substitutions": rows,
        "interpretation": (
            "This is a substitution control on an SNN-evolved topology, not a "
            "freshly evolved rate-network baseline."
        ),
    }


# ---------------------------------------------------------------------------
# 3. Metadata-confounding audit for structural fingerprints


def loo_knn_predictions(x: np.ndarray, y: np.ndarray, k: int = 3) -> np.ndarray:
    preds = []
    for i in range(len(x)):
        train_x = np.delete(x, i, axis=0)
        train_y = np.delete(y, i, axis=0)
        dists = np.linalg.norm(train_x - x[i], axis=1)
        nearest = np.argsort(dists)[: min(k, len(train_x))]
        votes = Counter(train_y[nearest])
        preds.append(votes.most_common(1)[0][0])
    return np.asarray(preds)


def standardize(x: np.ndarray) -> np.ndarray:
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return (x - x.mean(axis=0)) / std


def residualize(x: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    z = np.column_stack([np.ones(len(covariates)), standardize(covariates)])
    beta, *_ = np.linalg.lstsq(z, x, rcond=None)
    return x - z @ beta


def run_fingerprint_metadata_audit() -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "code"))
    import gen_fig13_fingerprint_corrected as fp  # noqa: E402

    networks = fp.load_networks()
    x_raw, y, feature_names = fp.build_feature_matrix(networks)
    x_std = standardize(x_raw)

    task_meta = []
    for net in networks:
        meta = fp.TASK_META[net["task"]]
        task_meta.append(
            [
                meta["network_size"],
                meta["n_input"],
                meta["n_output"],
                meta["state_dim"],
                meta["action_dim"],
            ]
        )
    task_meta_arr = np.asarray(task_meta, dtype=np.float64)
    arch_meta_arr = task_meta_arr[:, :3]

    pred_struct = loo_knn_predictions(x_std, y, k=3)
    pred_arch = loo_knn_predictions(standardize(arch_meta_arr), y, k=3)
    pred_taskmeta = loo_knn_predictions(standardize(task_meta_arr), y, k=3)

    x_resid_arch = standardize(residualize(x_std, arch_meta_arr))
    x_resid_taskmeta = standardize(residualize(x_std, task_meta_arr))
    pred_resid_arch = loo_knn_predictions(x_resid_arch, y, k=3)
    pred_resid_taskmeta = loo_knn_predictions(x_resid_taskmeta, y, k=3)

    # Fixed-N subset: CartPole and Acrobot both use N=128 but differ in I/O.
    fixed_n_mask = np.isin(y, ["CartPole", "Acrobot"])
    pred_fixed_n = loo_knn_predictions(x_std[fixed_n_mask], y[fixed_n_mask], k=3)

    def acc(pred: np.ndarray, truth: np.ndarray) -> float:
        return float(np.mean(pred == truth))

    return {
        "n_networks": int(len(networks)),
        "feature_count": int(len(feature_names)),
        "task_counts": dict(sorted(Counter(y).items())),
        "accuracies": {
            "structural_features": acc(pred_struct, y),
            "architecture_metadata_only_N_nin_nout": acc(pred_arch, y),
            "task_metadata_only_N_nin_nout_state_action": acc(pred_taskmeta, y),
            "structural_residual_after_architecture_metadata": acc(pred_resid_arch, y),
            "structural_residual_after_full_size_io_metadata": acc(pred_resid_taskmeta, y),
            "fixed_N_subset_CartPole_vs_Acrobot": acc(pred_fixed_n, y[fixed_n_mask]),
        },
        "feature_names": list(feature_names),
        "interpretation": (
            "Architecture metadata is partly diagnostic because task identity is "
            "coupled to N and I/O dimensions; residual accuracies are therefore "
            "reported as a confounding audit rather than as a task-invariant "
            "fingerprint claim."
        ),
    }


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    tost_rows = run_tost_sensitivity()
    rate = run_rate_substitution_control()
    fingerprint = run_fingerprint_metadata_audit()

    write_csv_rows(OUT_DIR / "tost_sensitivity.csv", tost_rows)
    write_csv_rows(OUT_DIR / "rate_substitution_grid.csv", rate["all_rate_substitutions"])

    report = {
        "tost_sensitivity": tost_rows,
        "rate_substitution_control": rate,
        "fingerprint_metadata_audit": fingerprint,
    }
    with (OUT_DIR / "revision_controls_report.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(report), f, indent=2, ensure_ascii=False)

    with (OUT_DIR / "revision_controls_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Revision control analyses\n")
        f.write("=========================\n\n")
        f.write("TOST sensitivity (selected rows):\n")
        for row in tost_rows:
            if row["contrast"] in ("TO-TI", "TO-TG") and row["delta"] in (5.0, 500.0):
                f.write(
                    f"  {row['task']} {row['contrast']} delta={row['delta']:.0f}: "
                    f"diff={row['diff']:.2f}, pTOST={row['p_tost']:.4g}, "
                    f"min_delta_alpha05={row['min_delta_for_alpha05']:.2f}\n"
                )
        f.write("\nRate-substitution control:\n")
        f.write(
            "  SNN reference: "
            f"{rate['snn_reference']['mean']:.1f} +/- {rate['snn_reference']['std']:.1f}\n"
        )
        best = rate["best_rate_substitution"]
        f.write(
            "  Best non-spiking rate substitution: "
            f"{best['mean']:.1f} +/- {best['std']:.1f} "
            f"({best['mode']}, gain={best['gain']})\n"
        )
        f.write("\nFingerprint metadata audit:\n")
        for key, val in fingerprint["accuracies"].items():
            f.write(f"  {key}: {val:.3f}\n")

    print(f"Wrote {OUT_DIR / 'revision_controls_report.json'}")
    print(f"Wrote {OUT_DIR / 'revision_controls_summary.txt'}")


if __name__ == "__main__":
    main()
