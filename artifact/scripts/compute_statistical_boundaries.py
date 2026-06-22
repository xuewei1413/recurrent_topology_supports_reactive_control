import csv
import glob
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifact" / "processed_results"
OUT.mkdir(parents=True, exist_ok=True)


def t_mean_ci(vals):
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    mean = float(vals.mean())
    sd = float(vals.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
    else:
        half = float("nan")
    return mean, sd, n, mean - half, mean + half


def welch_diff_ci(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = float(a.mean() - b.mean())
    va = float(a.var(ddof=1))
    vb = float(b.var(ddof=1))
    na = len(a)
    nb = len(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return diff, 0.0, float("inf"), diff, diff
    df_num = (va / na + vb / nb) ** 2
    df_den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = df_num / df_den
    half = stats.t.ppf(0.975, df) * se
    return diff, se, df, diff - half, diff + half


def wilson_ci(k, n, z=1.959963984540054):
    if n == 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return centre - half, centre + half


def load_cartpole():
    path = ROOT / "code" / "results" / "coevolution" / "coevolution_checkpoint.npy"
    data = np.load(path, allow_pickle=True)
    rows = []
    for item in data:
        strict = item.get("strict_20ep", {})
        rows.append({
            "task": "CartPole",
            "mode": item.get("mode"),
            "seed": item.get("seed"),
            "primary_metric": "best_fitness",
            "best_fitness": item.get("best_fitness"),
            "strict_20ep_mean": strict.get("mean"),
            "strict_20ep_sd": strict.get("std"),
            "strict_20ep_min": strict.get("min"),
            "strict_20ep_max": strict.get("max"),
            "success": item.get("solved"),
            "goal_episode_rate": "",
            "generations": item.get("generations"),
            "n_connections": item.get("n_connections"),
            "weight_scale": item.get("extra", {}).get("weight_scale"),
            "source_file": str(path.relative_to(ROOT)),
        })
    return rows


def load_mountaincar():
    base = ROOT / "struct_ai" / "code" / "experiments" / "results" / "mountaincar_coevolution"
    rows = []
    for path in sorted(glob.glob(str(base / "result_*.json"))):
        p = Path(path)
        item = json.loads(p.read_text(encoding="utf-8"))
        strict = item.get("strict_20ep", {})
        rows.append({
            "task": "MountainCar",
            "mode": item.get("mode"),
            "seed": item.get("seed"),
            "primary_metric": "strict_20ep_fitness_mean",
            "best_fitness": item.get("best_training_fitness"),
            "strict_20ep_mean": strict.get("fitness_mean"),
            "strict_20ep_sd": strict.get("fitness_std"),
            "strict_20ep_min": strict.get("fitness_min"),
            "strict_20ep_max": strict.get("fitness_max"),
            "success": item.get("solved_by_strict_threshold"),
            "goal_episode_rate": strict.get("goal_episode_rate"),
            "generations": item.get("generations"),
            "n_connections": item.get("n_connections"),
            "weight_scale": item.get("extra", {}).get("weight_scale"),
            "source_file": str(p.relative_to(ROOT)),
        })
    return rows


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(task, rows, value_key):
    groups = {m: [float(r[value_key]) for r in rows if r["mode"] == m] for m in ["TO", "TG", "TI"]}
    summary_rows = []
    for mode, vals in groups.items():
        mean, sd, n, lo, hi = t_mean_ci(vals)
        summary_rows.append({
            "task": task,
            "analysis": "group_mean_ci",
            "contrast": mode,
            "metric": value_key,
            "mean_or_diff": mean,
            "sd_or_se": sd,
            "n_or_df": n,
            "ci_low": lo,
            "ci_high": hi,
        })
    for a, b in [("TO", "TI"), ("TO", "TG"), ("TG", "TI")]:
        diff, se, df, lo, hi = welch_diff_ci(groups[a], groups[b])
        summary_rows.append({
            "task": task,
            "analysis": "welch_pairwise_difference_ci",
            "contrast": f"{a}-{b}",
            "metric": value_key,
            "mean_or_diff": diff,
            "sd_or_se": se,
            "n_or_df": df,
            "ci_low": lo,
            "ci_high": hi,
        })
    f_stat, p_val = stats.f_oneway(groups["TO"], groups["TG"], groups["TI"])
    summary_rows.append({
        "task": task,
        "analysis": "one_way_anova",
        "contrast": "TO/TG/TI",
        "metric": value_key,
        "mean_or_diff": f_stat,
        "sd_or_se": p_val,
        "n_or_df": "df=2,27" if task == "CartPole" else "df=2,57",
        "ci_low": "",
        "ci_high": "",
    })
    return summary_rows


def robust_rows(task, rows, value_key):
    out = []
    for mode in ["TO", "TG", "TI"]:
        subset = [r for r in rows if r["mode"] == mode]
        vals = np.asarray([float(r[value_key]) for r in subset], dtype=float)
        q1, med, q3 = np.percentile(vals, [25, 50, 75], method="linear")
        success = sum(str(r["success"]).lower() == "true" for r in subset)
        lo, hi = wilson_ci(success, len(subset))
        goal_vals = [float(r["goal_episode_rate"]) for r in subset if r.get("goal_episode_rate") not in ("", None)]
        out.append({
            "task": task,
            "mode": mode,
            "metric": value_key,
            "median": med,
            "q1": q1,
            "q3": q3,
            "success_count": success,
            "n": len(subset),
            "success_rate": success / len(subset) if subset else float("nan"),
            "success_wilson_low": lo,
            "success_wilson_high": hi,
            "goal_episode_rate_mean": float(np.mean(goal_vals)) if goal_vals else "",
        })
    return out


def fmt(x, digits=1):
    if x == "" or x is None:
        return ""
    return f"{float(x):.{digits}f}"


def write_latex_table(stats_rows, robust):
    by = {(r["task"], r["contrast"], r["analysis"]): r for r in stats_rows}
    robust_by = {(r["task"], r["mode"]): r for r in robust}
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\caption{\textbf{Statistical boundary for TO/TG/TI controls.} Mean confidence intervals and pairwise Welch mean-difference intervals are computed from seed-level final scores. They define the range of differences compatible with the data and are not formal equivalence or non-inferiority tests.}")
    lines.append(r"\label{tab:supp-stat-boundary}")
    lines.append(r"\begin{tabularx}{\textwidth}{@{}llYYY@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Task} & \textbf{Condition / contrast} & \textbf{Mean or diff. (95\% CI)} & \textbf{Median [IQR]} & \textbf{Success / goal rate} \\")
    lines.append(r"\midrule")
    for task in ["CartPole", "MountainCar"]:
        for mode in ["TO", "TG", "TI"]:
            s = by[(task, mode, "group_mean_ci")]
            r = robust_by[(task, mode)]
            mean_ci = f"${fmt(s['mean_or_diff'])}$ [${fmt(s['ci_low'])}$, ${fmt(s['ci_high'])}$]"
            med_iqr = f"${fmt(r['median'])}$ [${fmt(r['q1'])}$, ${fmt(r['q3'])}$]"
            if task == "MountainCar":
                succ = f"{int(r['success_count'])}/{int(r['n'])}; goal {100*float(r['goal_episode_rate_mean']):.1f}\%"
            else:
                succ = f"{int(r['success_count'])}/{int(r['n'])}"
            lines.append(f"{task} & {mode} & {mean_ci} & {med_iqr} & {succ} " + r"\\")
        for contrast in ["TO-TI", "TO-TG", "TG-TI"]:
            s = by[(task, contrast, "welch_pairwise_difference_ci")]
            diff_ci = f"${fmt(s['mean_or_diff'])}$ [${fmt(s['ci_low'])}$, ${fmt(s['ci_high'])}$]"
            lines.append(f"{task} & {contrast} & {diff_ci} & -- & -- " + r"\\")
        if task == "CartPole":
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\end{table}")
    (OUT / "supplement_table_to_tg_ti.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    cart = load_cartpole()
    mc = load_mountaincar()
    seed_fields = [
        "task", "mode", "seed", "primary_metric", "best_fitness",
        "strict_20ep_mean", "strict_20ep_sd", "strict_20ep_min",
        "strict_20ep_max", "success", "goal_episode_rate", "generations",
        "n_connections", "weight_scale", "source_file",
    ]
    write_csv(OUT / "cartpole_coevolution_seed_level.csv", cart, seed_fields)
    write_csv(OUT / "mountaincar_coevolution_seed_level.csv", mc, seed_fields)
    stats_rows = summarize("CartPole", cart, "best_fitness") + summarize("MountainCar", mc, "strict_20ep_mean")
    stats_fields = ["task", "analysis", "contrast", "metric", "mean_or_diff", "sd_or_se", "n_or_df", "ci_low", "ci_high"]
    write_csv(OUT / "statistical_boundaries_to_tg_ti.csv", stats_rows, stats_fields)
    robust = robust_rows("CartPole", cart, "best_fitness") + robust_rows("MountainCar", mc, "strict_20ep_mean")
    robust_fields = ["task", "mode", "metric", "median", "q1", "q3", "success_count", "n", "success_rate", "success_wilson_low", "success_wilson_high", "goal_episode_rate_mean"]
    write_csv(OUT / "robust_statistics_to_tg_ti.csv", robust, robust_fields)
    write_latex_table(stats_rows, robust)
    print("Wrote:")
    for name in [
        "cartpole_coevolution_seed_level.csv",
        "mountaincar_coevolution_seed_level.csv",
        "statistical_boundaries_to_tg_ti.csv",
        "robust_statistics_to_tg_ti.csv",
        "supplement_table_to_tg_ti.tex",
    ]:
        print(" -", OUT / name)


if __name__ == "__main__":
    main()
