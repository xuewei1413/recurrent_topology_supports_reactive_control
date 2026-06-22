"""Generate Fig. 08: topology/weight co-evolution controls."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "code" / "results" / "coevolution" / "coevolution_results.json"
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_data() -> dict:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH.relative_to(ROOT)}. Re-run the co-evolution "
            "experiment or use the tracked processed checkpoint for the "
            "statistical audit."
        )
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def main() -> None:
    data = load_data()
    per_seed = data["per_seed"]

    groups = {mode: [row for row in per_seed if row["mode"] == mode] for mode in ["TO", "TG", "TI"]}
    best = {mode: [row["best_fitness"] for row in rows] for mode, rows in groups.items()}
    gens = {mode: [row["generations"] for row in rows] for mode, rows in groups.items()}
    n_params = {"TO": 1, "TG": 2, "TI": 16384}

    f_value, p_value = stats.f_oneway(best["TO"], best["TG"], best["TI"])
    print(f"ANOVA: F(2,27)={f_value:.2f}, p={p_value:.2f}")
    for mode in ["TO", "TG", "TI"]:
        print(f"{mode}: {statistics.mean(best[mode]):.1f}+/-{statistics.stdev(best[mode]):.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = ["#E74C3C", "#3498DB", "#2ECC71"]
    modes = ["TO", "TG", "TI"]

    ax = axes[0]
    bp = ax.boxplot(
        [best[mode] for mode in modes],
        labels=modes,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    rng = np.random.default_rng(42)
    for i, mode in enumerate(modes):
        ax.scatter(
            rng.normal(i + 1, 0.04, len(best[mode])),
            best[mode],
            alpha=0.6,
            s=30,
            color=colors[i],
            edgecolors="black",
            linewidth=0.5,
            zorder=5,
        )
        ax.text(
            i + 1,
            statistics.mean(best[mode]) + 8,
            f"{statistics.mean(best[mode]):.1f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.axhline(y=22.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.set_ylabel("Best fitness")
    ax.set_title(f"(a) Final performance\n$F(2,27)={f_value:.2f}$, $p={p_value:.2f}$", fontsize=11)
    ax.set_ylim(0, 530)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    gen_means = [statistics.mean(gens[mode]) for mode in modes]
    gen_stds = [statistics.stdev(gens[mode]) for mode in modes]
    ax.bar(modes, gen_means, yerr=gen_stds, capsize=5, color=colors, alpha=0.6, edgecolor="black", linewidth=0.8)
    for i, (mean, sd) in enumerate(zip(gen_means, gen_stds)):
        ax.text(i, mean + sd + 5, f"{mean:.0f}+/-{sd:.0f}", ha="center", fontsize=9)
    ax.set_ylabel("Generations to solve")
    ax.set_title("(b) Convergence speed", fontsize=11)
    ax.set_ylim(0, max(mean + sd for mean, sd in zip(gen_means, gen_stds)) * 1.25)

    ax = axes[2]
    eff_vals = [statistics.mean(best[mode]) / n_params[mode] for mode in modes]
    ax.bar(modes, eff_vals, color=colors, alpha=0.6, edgecolor="black", linewidth=0.8)
    ax.set_yscale("log")
    for i, value in enumerate(eff_vals):
        label = f"{value:.0f}" if value >= 1 else f"{value:.2f}"
        ax.text(i, value * 1.3, label, ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Performance / parameter (log)")
    ax.set_title("(c) Parameter efficiency", fontsize=11)
    ax.text(0.5, -0.15, "1 param", ha="center", fontsize=8, transform=ax.get_xaxis_transform())
    ax.text(1.5, -0.15, "2 params", ha="center", fontsize=8, transform=ax.get_xaxis_transform())
    ax.text(2.5, -0.15, "~16K params", ha="center", fontsize=8, transform=ax.get_xaxis_transform())

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig08_coevolution.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig08_coevolution.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'fig08_coevolution.pdf'}")


if __name__ == "__main__":
    main()
