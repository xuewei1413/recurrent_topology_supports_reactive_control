"""Generate Fig. 14: continuous-control extension on Hopper-v4 and Pendulum-v1.

The figure is built from saved result files rather than hard-coded values:
- Hopper/Pendulum reports: code/experiments/results/embodied_reflex/*.json
- Optional Hopper evolution curve fallback: code/results/embodied_reflex/hopper_evolution.log
- PPO Hopper references: code/experiments/results/ppo_hopper/ppo_hopper_results.pkl
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures_arxiv"
EMBODIED_DIR = ROOT / "code" / "experiments" / "results" / "embodied_reflex"
LEGACY_HOPPER_DIR = ROOT / "code" / "results" / "embodied_reflex"
PPO_PATH = ROOT / "code" / "experiments" / "results" / "ppo_hopper" / "ppo_hopper_results.pkl"
PENDULUM_PATH = EMBODIED_DIR / "pendulum_report.json"
REPORT_PATH = EMBODIED_DIR / "fig14_data_report.json"

FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_hopper_eval() -> dict:
    fig14_report = EMBODIED_DIR / "fig14_data_report.json"
    if fig14_report.exists():
        report = load_json(fig14_report)
        return {
            "strict_20ep_mean": report["hopper_snn_strict_20ep_mean"],
            "strict_20ep_std": report["hopper_snn_strict_20ep_sd"],
            "evo_5ep_mean": report["hopper_snn_evo_5ep_mean"],
            "adaptation": {
                gravity: {"best_score": score, "best_ws": report["gravity_best_weight_scales"][gravity]}
                for gravity, score in report["gravity_scores"].items()
            },
        }
    return load_json(LEGACY_HOPPER_DIR / "hopper_evaluation_results.json")


def load_hopper_evolution() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_path = LEGACY_HOPPER_DIR / "hopper_evolution.log"
    if not log_path.exists():
        report = load_json(EMBODIED_DIR / "fig14_data_report.json")
        points = report["evolution_key_points"]
        gens = np.array([6, 7, 8, 53], dtype=int)
        bests = np.array([points["gen6"], points["gen7"], points["gen8"], points["gen53"]], dtype=float)
        means = bests.copy()
        return gens, bests, means
    text = log_path.read_text(encoding="utf-8", errors="replace").replace("\x00", "\n")
    gens, bests, means = [], [], []
    for line in text.splitlines():
        match = re.match(r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+\d+", line)
        if not match:
            continue
        gens.append(int(match.group(1)))
        bests.append(float(match.group(2)))
        means.append(float(match.group(3)))
    if not gens:
        report = load_json(EMBODIED_DIR / "fig14_data_report.json")
        points = report["evolution_key_points"]
        gens = np.array([6, 7, 8, 53], dtype=int)
        bests = np.array([points["gen6"], points["gen7"], points["gen8"], points["gen53"]], dtype=float)
        means = bests.copy()
        return gens, bests, means
    return np.array(gens), np.array(bests), np.array(means)


def load_ppo_refs() -> dict:
    with PPO_PATH.open("rb") as f:
        ppo = pickle.load(f)
    out = {}
    for key in ("100k", "1M"):
        means = np.array(ppo[key]["means"], dtype=float)
        out[key] = {
            "mean": float(means.mean()),
            "sd_sample": float(means.std(ddof=1)),
            "sd_population": float(means.std(ddof=0)),
            "seeds": means.tolist(),
            "params": int(ppo[key]["params"]),
        }
    return out


def load_pendulum() -> dict:
    return load_json(PENDULUM_PATH)


def main() -> None:
    hopper_eval = load_hopper_eval()
    gens, bests, means = load_hopper_evolution()
    ppo = load_ppo_refs()
    pendulum = load_pendulum()

    snn_strict = float(hopper_eval["strict_20ep_mean"])
    snn_strict_sd = float(hopper_eval["strict_20ep_std"])
    snn_best = float(hopper_eval["evo_5ep_mean"])
    ppo_100k = ppo["100k"]["mean"]
    ppo_1m = ppo["1M"]["mean"]
    ppo_params = int(ppo["1M"]["params"])
    ratio_pct = snn_strict / ppo_1m * 100

    adaptation = hopper_eval["adaptation"]
    gravity_keys = ["1.0", "1.3", "1.5"]
    gravity_scores = [float(adaptation[k]["best_score"]) for k in gravity_keys]
    gravity_ws = [float(adaptation[k]["best_ws"]) for k in gravity_keys]
    gravity_pct = [score / gravity_scores[0] * 100 for score in gravity_scores]

    pend_random = float(pendulum["random_baseline"])
    pend_strict = float(pendulum["snn_final_20ep"])
    pend_best = float(pendulum["snn_best_fitness"])
    pend_strict_improvement = (pend_strict - pend_random) / abs(pend_random) * 100
    pend_best_improvement = (pend_best - pend_random) / abs(pend_random) * 100

    fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.6))

    # (a) Hopper performance comparison.
    ax = axes[0]
    labels = ["SNN\n(1 global scale)", "PPO\n100k steps", "PPO\n1M steps"]
    vals = [snn_strict, ppo_100k, ppo_1m]
    colors = ["#E86F61", "#F3B64F", "#5DA5DA"]
    bars = ax.bar(labels, vals, color=colors, alpha=0.82, edgecolor="black", linewidth=0.8)
    for idx, (bar, val) in enumerate(zip(bars, vals)):
        label = f"{val:.0f}"
        if idx == 0:
            label = f"{val:.0f}\n$\\pm${snn_strict_sd:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2, val + 55, label,
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(0, snn_strict * 0.50, "1 scale", ha="center", fontsize=8,
            color="white", fontweight="bold")
    ax.text(1, ppo_100k * 0.50, f"{ppo_params:,}\nparams", ha="center", fontsize=8,
            color="white", fontweight="bold")
    ax.text(2, ppo_1m * 0.50, f"{ppo_params:,}\nparams", ha="center", fontsize=8,
            color="white", fontweight="bold")
    ax.annotate(f"{ratio_pct:.1f}% of 1M PPO", xy=(0, snn_strict), xytext=(1.30, 1780),
                fontsize=9, ha="center", arrowprops=dict(arrowstyle="->", color="0.45"))
    ax.text(0.02, 0.94, "PPO bars: 3-seed means", transform=ax.transAxes,
            fontsize=8, ha="left", va="top", color="0.35")
    ax.set_ylabel("Mean reward")
    ax.set_title("(a) Hopper-v4 Performance", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 2500)
    ax.grid(axis="y", alpha=0.18)

    # (b) Hopper evolution curve.
    ax = axes[1]
    ax.plot(gens, bests, "o-", color="#E74C3C", markersize=3, linewidth=1.2, label="Best fitness")
    ax.plot(gens, means, "s-", color="#F39C12", markersize=2, linewidth=0.8, alpha=0.62, label="Mean fitness")
    for target in (6, 7, 8):
        idx = int(np.where(gens == target)[0][0])
        ax.annotate(f"{bests[idx]:.0f}", xy=(gens[idx], bests[idx]),
                    xytext=(gens[idx] + 3, bests[idx] - 35), fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="0.45"))
    idx53 = int(np.where(gens == 53)[0][0])
    ax.annotate(f"Gen 53: {bests[idx53]:.0f}", xy=(gens[idx53], bests[idx53]),
                xytext=(gens[idx53] + 5, bests[idx53] - 60), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="0.45"))
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("(b) Hopper-v4 Evolution", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 82)
    ax.set_ylim(0, 1100)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.16)

    # (c) Gravity retuning.
    ax = axes[2]
    xlabels = [f"{float(g):.1f}x\nws={ws:.1f}" for g, ws in zip(gravity_keys, gravity_ws)]
    bars = ax.bar(xlabels, gravity_scores, color=["#58C785", "#F3B64F", "#E86F61"],
                  alpha=0.82, edgecolor="black", linewidth=0.8)
    for bar, score, pct in zip(bars, gravity_scores, gravity_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 12,
                f"{score:.1f}\n({pct:.1f}%)", ha="center", fontsize=9,
                fontweight="bold")
    ax.set_ylabel("Mean reward")
    ax.set_title("(c) Gravity Retuning", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1200)
    ax.text(0.02, 0.94, "same topology; global scale retuned", transform=ax.transAxes,
            fontsize=8, ha="left", va="top", color="0.35")
    ax.grid(axis="y", alpha=0.18)

    # (d) Pendulum proof of concept.
    ax = axes[3]
    pend_vals = [pend_random, pend_strict, pend_best]
    pend_labels = ["Random\nbaseline", "SNN\nstrict 20-ep", "SNN\nbest"]
    bars = ax.bar(pend_labels, pend_vals, color=["#AAB7B8", "#E86F61", "#58C785"],
                  alpha=0.82, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, pend_vals):
        y = val - 40 if val < -120 else val - 28
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{val:.1f}",
                ha="center", va="top", fontsize=9, fontweight="bold")
    ax.text(0.50, 0.97,
            f"strict +{pend_strict_improvement:.1f}% vs random\nbest +{pend_best_improvement:.1f}%",
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.88))
    ax.set_ylabel("Reward (higher is better)")
    ax.set_title("(d) Pendulum-v1", fontsize=11, fontweight="bold")
    ax.set_ylim(-1450, 50)
    ax.grid(axis="y", alpha=0.18)

    fig.tight_layout(w_pad=2.0)
    for ext in ("png", "pdf"):
        out = FIG_DIR / f"fig14_embodied_control.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)

    report = {
        "hopper_snn_strict_20ep_mean": snn_strict,
        "hopper_snn_strict_20ep_sd": snn_strict_sd,
        "hopper_snn_evo_5ep_mean": snn_best,
        "ppo_100k_mean": ppo_100k,
        "ppo_100k_sd_sample": ppo["100k"]["sd_sample"],
        "ppo_100k_sd_population": ppo["100k"]["sd_population"],
        "ppo_100k_seed_means": ppo["100k"]["seeds"],
        "ppo_1m_mean": ppo_1m,
        "ppo_1m_sd_sample": ppo["1M"]["sd_sample"],
        "ppo_1m_sd_population": ppo["1M"]["sd_population"],
        "ppo_1m_seed_means": ppo["1M"]["seeds"],
        "ppo_params": ppo_params,
        "snn_ratio_to_ppo_1m_pct": ratio_pct,
        "gravity_scores": dict(zip(gravity_keys, gravity_scores)),
        "gravity_best_weight_scales": dict(zip(gravity_keys, gravity_ws)),
        "gravity_retention_pct": dict(zip(gravity_keys, gravity_pct)),
        "pendulum_random_baseline": pend_random,
        "pendulum_snn_strict_20ep": pend_strict,
        "pendulum_snn_best": pend_best,
        "pendulum_strict_improvement_pct": pend_strict_improvement,
        "pendulum_best_improvement_pct": pend_best_improvement,
        "evolution_key_points": {
            "gen6": float(bests[np.where(gens == 6)[0][0]]),
            "gen7": float(bests[np.where(gens == 7)[0][0]]),
            "gen8": float(bests[np.where(gens == 8)[0][0]]),
            "gen53": float(bests[idx53]),
            "best_fitness": float(bests.max()),
        },
    }
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
