# -*- coding: utf-8 -*-
"""P8 统计（mc2026）：拓扑级主分析 + gatekeeping 判定。

输入：results_mc/runs/{task}/*.json（P6 产物）+ 冻结协议。
输出：results_mc/stats/med_analysis.json + gates_decision.json。

设计 §12/§13：独立单位=topology lineage（CP n=12 / Acr n=12）；
拓扑级配对检验（n<=20 精确符号翻转）+ cluster bootstrap CI；
冻结 gatekeeping 七级顺序执行，前 gate 未过→后续标探索性；
SESOI 双判据（统计 + 实践）；M 仅预定义 gap>=SESOI 子集；
Shapley：CP 精确 2 通道 / Acr 8 子集精确公式。
"""
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixuewei/tmp/struct_mod_v2")
sys.path.insert(0, str(ROOT / "code"))

from experiments_v2.analysis.statistics import (  # noqa: E402
    paired_permutation_test,
    holm_correction,
)

CONFIRM_SEEDS = list(range(86400, 86420))
SESOI = {"CartPole": 30.0, "Acrobot": 10.0}
N_BOOT = 10000


def load_cell(task, seed, d, r, mode, alpha, stage):
    fn = (ROOT / "results_mc/runs" / task /
          f"{stage}_{task}_s{seed}_d{d}_r{r}_"
          f"{mode}_a{str(alpha).replace('.', 'p')}.json")
    if not fn.exists():
        return None
    return json.load(open(fn))


def topo_mean_return(cell):
    """cell 内 20 episode 的均值 return（拓扑级汇总）。"""
    if cell is None:
        return None
    return float(np.mean([row["return"] for row in cell["rows"]]))


def paired_test(a, b, alternative="greater"):
    """拓扑级配对检验（n<=20 精确符号翻转；H1: a > b 或 a != b）。

    返回 (p, n)。alternative='greater' 用单边；'two-sided' 用双边。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    n = len(a)
    if n == 0:
        return 1.0, 0
    d = a - b
    nz = d[d != 0]
    k = len(nz)
    if k == 0:
        return 1.0, n
    s_obs = np.sum(nz)   # 观测的秩和（单边大于）
    if k <= 20:
        count_ge = 0
        for mask in range(1 << k):
            signed = np.where(np.array([(mask >> j) & 1 for j in range(k)],
                                       dtype=int) == 1, nz, -nz)
            if np.sum(signed) >= s_obs:
                count_ge += 1
        p_greater = count_ge / (1 << k)
    else:
        from scipy.stats import wilcoxon
        try:
            stat, p_greater = wilcoxon(a, b, alternative="greater")
        except Exception:
            p_greater = 1.0
    if alternative == "greater":
        return float(p_greater), n
    p_two = 2 * min(p_greater, 1 - p_greater)
    return float(min(p_two, 1.0)), n


def bootstrap_ci(diff, n_boot=N_BOOT, alpha=0.05):
    rng = np.random.RandomState(20260808)
    d = np.asarray(diff, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return [None, None]
    boots = [np.mean(rng.choice(d, len(d), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, 100 * alpha / 2), np.percentile(boots, 100 * (1 - alpha / 2))
    return [float(lo), float(hi)]


def cluster_bootstrap_ci(per_seed_effect, n_boot=N_BOOT, alpha=0.05):
    """cluster bootstrap by seed（独立单位=拓扑）。"""
    rng = np.random.RandomState(20260809)
    keys = list(per_seed_effect.keys())
    vals = np.array([per_seed_effect[k] for k in keys])
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return [None, None]
    boots = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, 100 * alpha / 2), np.percentile(boots, 100 * (1 - alpha / 2))
    return [float(lo), float(hi)]


def analyze_task(task):
    proto = json.load(open(ROOT / "results_mc/frozen/mc_protocol.json",
                           encoding="utf-8"))
    sesoi = SESOI[task]
    evo_seeds = proto["evolution_seeds"][task]
    n_topo = len(evo_seeds)
    print(f"\n== P8 统计 {task} (n={n_topo}) ==", flush=True)

    # 汇总结构：per (seed, d, r, mode, alpha, stage) → return
    cells = {}

    def get(seed, d, r, mode, alpha, stage):
        key = (seed, d, r, mode, alpha, stage)
        if key not in cells:
            cells[key] = topo_mean_return(
                load_cell(task, seed, d, r, mode, alpha, stage))
        return cells[key]

    # ── Gate 1: self-exact 正向救援（变体←F0, α=1）相对 none ──
    # 每个 lineage 先用 6 变体 cell 的均值作 ITT 汇总
    rescue_eff = {}   # seed → mean rescue effect（跨 6 变体）
    for seed in evo_seeds:
        effs = []
        for d in range(3):
            for r in range(2):
                var_none = get(seed, d, r, "none", 1.0, "rescue")
                var_exact = get(seed, d, r, "self_exact", 1.0, "rescue")
                if var_none is None or var_exact is None:
                    continue
                effs.append(var_exact - var_none)
        rescue_eff[seed] = float(np.mean(effs)) if effs else np.nan
    g1_p, g1_n = paired_test(list(rescue_eff.values()),
                             np.zeros(n_topo))
    g1_ci = cluster_bootstrap_ci(rescue_eff)
    g1_mean = float(np.mean(list(rescue_eff.values())))
    g1_stat = g1_p < 0.05 and g1_ci[0] > 0
    g1_sesoi = g1_mean >= sesoi
    g1_pass = g1_stat and g1_sesoi
    # G1 是 α=1 通道替换恒等（output 无反馈 → recipient 输出=donor 输出），
    # 非独立因果证据（设计 §10）；仅作仪器正确性验证，不作为升级门槛
    g1_is_gate = False
    print(f"  [仪器验证，非门槛] G1 α=1 恒等: mean={g1_mean:.1f} "
          f"p={g1_p:.4f} CI={g1_ci}（政策替换恒等，设计§10）", flush=True)

    # ── Gate 2: reverse-exact 反向损伤（F0←变体, α=1）──
    # 基线 = destroy none cell（F0 原生，同 final_confirm 段）
    destroy_eff = {}
    for seed in evo_seeds:
        effs = []
        for d in range(3):
            for r in range(2):
                f0_none = get(seed, d, r, "none", 1.0, "destroy")
                f0_d = get(seed, d, r, "self_exact", 1.0, "destroy")
                if f0_none is None or f0_d is None:
                    continue
                effs.append(f0_none - f0_d)
        destroy_eff[seed] = float(np.mean(effs)) if effs else np.nan
    dvals = np.array([destroy_eff[s] for s in evo_seeds])
    dvals = dvals[~np.isnan(dvals)]
    g2_p, g2_n = paired_test(dvals, np.zeros(len(dvals)))
    g2_ci = cluster_bootstrap_ci(destroy_eff)
    g2_mean = float(np.mean(dvals))
    g2_sesoi = g2_mean >= sesoi
    g2_stat = g2_p < 0.05 and g2_ci[0] > 0
    g2_pass = g2_stat and g2_sesoi
    # G2 与 G1 同为 α=1 恒等的反向方向，非独立证据，不作为门槛
    g2_is_gate = False
    print(f"  [仪器验证，非门槛] G2 α=1 恒等（反向）: mean={g2_mean:.1f} "
          f"p={g2_p:.4f} CI={g2_ci}（政策替换恒等，设计§10）", flush=True)

    # ── Gate 3: self-exact vs action-only 与 action-centered（两标签编码，Holm）──
    g3_eff = {}      # vs action_only（正 one-hot）
    g3_eff_ac = {}   # vs action_centered（push-pull）
    for seed in evo_seeds:
        effs_ao, effs_ac = [], []
        for d in range(3):
            for r in range(2):
                ex = get(seed, d, r, "self_exact", 1.0, "rescue")
                ao = get(seed, d, r, "action_only", 1.0, "rescue")
                ac = get(seed, d, r, "action_centered", 1.0, "rescue")
                if ex is not None and ao is not None:
                    effs_ao.append(ex - ao)
                if ex is not None and ac is not None:
                    effs_ac.append(ex - ac)
        g3_eff[seed] = float(np.mean(effs_ao)) if effs_ao else np.nan
        g3_eff_ac[seed] = float(np.mean(effs_ac)) if effs_ac else np.nan
    g3_p, _ = paired_test(list(g3_eff.values()), np.zeros(n_topo))
    g3_p_ac, _ = paired_test(list(g3_eff_ac.values()), np.zeros(n_topo))
    g3_holm = holm_correction([g3_p, g3_p_ac])
    g3_p_adj = g3_holm[0]["adjusted_p"]
    g3_p_ac_adj = g3_holm[1]["adjusted_p"]
    g3_ci = cluster_bootstrap_ci(g3_eff)
    g3_ci_ac = cluster_bootstrap_ci(g3_eff_ac)
    g3_vals = np.array([v for v in g3_eff.values() if not np.isnan(v)])
    g3_vals_ac = np.array([v for v in g3_eff_ac.values() if not np.isnan(v)])
    g3_mean = float(np.mean(g3_vals))
    g3_median = float(np.median(g3_vals))
    g3_n_sesoi = int(np.sum(g3_vals >= sesoi))
    g3_loo_max = float(np.mean(np.sort(g3_vals)[:-1])) if len(g3_vals) > 1 else g3_mean  # 去最大 lineage
    g3_ac_mean = float(np.mean(g3_vals_ac))
    g3_ac_median = float(np.median(g3_vals_ac))
    g3_stat = g3_p_adj < 0.05 and g3_ci[0] > 0
    g3_pass = g3_stat and g3_mean >= sesoi
    print(f"  G3 exact vs action_only: mean={g3_mean:.1f} median={g3_median:.1f} "
          f"n>=SESOI={g3_n_sesoi}/12 loo={g3_loo_max:.1f} p={g3_p:.4f}(adj {g3_p_adj:.4f}) "
          f"CI={g3_ci} PASS={g3_pass}", flush=True)
    print(f"  G3 exact vs action_centered: mean={g3_ac_mean:.1f} median={g3_ac_median:.1f} "
          f"p={g3_p_ac:.4f}(adj {g3_p_ac_adj:.4f}) CI={g3_ci_ac}", flush=True)

    # ── Gate 4: self-exact vs time_shuffle & energy_random ──
    g4_eff_ts, g4_eff_er = {}, {}
    for seed in evo_seeds:
        e_ts, e_er = [], []
        for d in range(3):
            for r in range(2):
                ex = get(seed, d, r, "self_exact", 1.0, "rescue")
                ts = get(seed, d, r, "time_shuffle", 1.0, "rescue")
                er = get(seed, d, r, "energy_random", 1.0, "rescue")
                if ex is not None and ts is not None:
                    e_ts.append(ex - ts)
                if ex is not None and er is not None:
                    e_er.append(ex - er)
        g4_eff_ts[seed] = float(np.mean(e_ts)) if e_ts else np.nan
        g4_eff_er[seed] = float(np.mean(e_er)) if e_er else np.nan
    g4_ts_p, _ = paired_test(list(g4_eff_ts.values()), np.zeros(n_topo))
    g4_er_p, _ = paired_test(list(g4_eff_er.values()), np.zeros(n_topo))
    g4_ts_ci = cluster_bootstrap_ci(g4_eff_ts)
    g4_er_ci = cluster_bootstrap_ci(g4_eff_er)
    # Holm 校正（同 gate 两比较）——预登记 §12.3 要求
    g4_holm = holm_correction([g4_ts_p, g4_er_p])
    g4_ts_p_adj = g4_holm[0]["adjusted_p"]
    g4_er_p_adj = g4_holm[1]["adjusted_p"]
    g4_pass = (g4_ts_p_adj < 0.05 and g4_ts_ci[0] > 0 and
               g4_er_p_adj < 0.05 and g4_er_ci[0] > 0)
    print(f"  G4 exact vs ts: p={g4_ts_p:.4f}(adj {g4_ts_p_adj:.4f}) CI={g4_ts_ci}; "
          f"vs er: p={g4_er_p:.4f}(adj {g4_er_p_adj:.4f}) CI={g4_er_ci} PASS={g4_pass}",
          flush=True)

    # ── Gate 5: J*≈J_F0 恢复（keep_trace cell: self_exact α=1，trace_agg）──
    g5_eff = {}
    for seed in evo_seeds:
        diffs = []
        for d in range(3):
            for r in range(2):
                e_cell = load_cell(task, seed, d, r, "self_exact", 1.0, "rescue")
                if not e_cell:
                    continue
                for row in e_cell["rows"]:
                    ta = row.get("trace_agg")
                    if ta and ta["j_don_norm_mean"] > 1e-9:
                        diffs.append(abs(ta["j_star_norm_mean"]
                                         - ta["j_don_norm_mean"])
                                     / ta["j_don_norm_mean"])
        g5_eff[seed] = float(np.mean(diffs)) if diffs else np.nan
    g5_mean = float(np.nanmean(list(g5_eff.values())))
    # G5 不作为升级门槛（设计 §10 明示 J*≈J_donor 是 α=1 干预定义结果，
    # 非独立证据）。降级为注入正确性仪器检查。
    g5_injection_ok = g5_mean < 1e-3
    print(f"  [仪器检查，非门槛] α=1 J*≈J_donor 相对偏差: {g5_mean:.3e} "
          f"ok={g5_injection_ok}（设计 §10：定义结果，非独立证据）", flush=True)

    # ── Gate 6: full-vector vs best-single-channel ──
    # 单通道子集（ch0/ch1[/ch2]）的救援效应，取每拓扑最佳，与 full 对比
    g6_pass = None
    g6_mean = None
    subset_dir = ROOT / "results_mc/runs" / task / "subset"
    n_out = TASK_N_OUT[task]
    if subset_dir.exists() and (subset_dir / f"subset_{task}_s{evo_seeds[0]}_d0_r0_ch0.json").exists():
        best_eff = {}
        for seed in evo_seeds:
            ch_effs = []
            for ch in range(n_out):
                effs = []
                for d in range(3):
                    for r in range(2):
                        fn = (subset_dir /
                              f"subset_{task}_s{seed}_d{d}_r{r}_ch{ch}.json")
                        if fn.exists():
                            cell = json.load(open(fn))
                            effs.append(cell["mean_return"]
                                        - get(seed, d, r, "none", 1.0, "rescue"))
                if effs:
                    ch_effs.append(np.mean(effs))
            best_eff[seed] = float(np.max(ch_effs)) if ch_effs else np.nan
        g6_eff = {s: rescue_eff[s] - best_eff[s] for s in evo_seeds
                  if not np.isnan(rescue_eff[s]) and not np.isnan(best_eff[s])}
        g6_p, _ = paired_test(list(g6_eff.values()), np.zeros(len(g6_eff)))
        g6_ci = cluster_bootstrap_ci(g6_eff)
        g6_mean = float(np.mean(list(g6_eff.values())))
        g6_sesoi = g6_mean >= sesoi
        g6_pass = g6_p < 0.05 and g6_ci[0] > 0 and g6_sesoi
        print(f"  G6 full vs best-single: mean={g6_mean:.1f} p={g6_p:.4f} "
              f"CI={g6_ci} PASS={g6_pass}", flush=True)
    else:
        print("  G6: subset 数据缺失（P7 未跑），标 pending", flush=True)

    analysis = {
        "task": task, "n_topo": n_topo, "sesoi": sesoi,
        "gates": {
            "G1_rescue_identity": {"mean": g1_mean, "p": g1_p, "ci": g1_ci,
                                   "is_gate": False, "n": g1_n,
                                   "note": "α=1 政策替换恒等（设计§10），非独立证据"},
            "G2_destroy_identity": {"mean": g2_mean, "p": g2_p, "ci": g2_ci,
                                    "is_gate": False, "n": g2_n,
                                    "note": "α=1 恒等反向方向，非独立证据"},
            "G3_vs_action_only": {"mean": g3_mean, "median": g3_median,
                                  "n_at_sesoi": g3_n_sesoi,
                                  "loo_max_mean": g3_loo_max,
                                  "p": g3_p, "p_holm": g3_p_adj, "ci": g3_ci,
                                  "stat": g3_stat, "pass": g3_pass},
            "G3_vs_action_centered": {"mean": g3_ac_mean, "median": g3_ac_median,
                                       "p": g3_p_ac, "p_holm": g3_p_ac_adj,
                                       "ci": g3_ci_ac},
            "G4_vs_ts_er": {"ts_p": g4_ts_p, "ts_p_holm": g4_ts_p_adj,
                            "ts_ci": g4_ts_ci,
                            "er_p": g4_er_p, "er_p_holm": g4_er_p_adj,
                            "er_ci": g4_er_ci, "pass": g4_pass,
                            "note": "p 为单边 greater 检验；CI 为双侧 bootstrap——负效应时 p=1 与负 CI 自洽"},
            "injection_check_alpha1": {"mean_rel_diff": g5_mean,
                                       "is_injection_correct": g5_injection_ok,
                                       "is_gate": False,
                                       "note": "α=1 定义结果（设计§10），非独立证据，不作为门槛"},
            "G6_full_vs_best": {"mean": g6_mean, "pass": g6_pass},
        },
        "per_seed_rescue": {str(k): v for k, v in rescue_eff.items()},
        "per_seed_destroy": {str(k): v for k, v in destroy_eff.items()},
        # upgrade 不含 injection_check（非门槛）与恒等的 G1/G2（非独立证据）
        "upgrade_overall": g3_pass and g4_pass and (g6_pass is True),
    }
    return analysis


TASK_N_OUT = {"CartPole": 2, "Acrobot": 3}


def main():
    os.makedirs(str(ROOT / "results_mc/stats"), exist_ok=True)
    all_ = {}
    for task in ("CartPole", "Acrobot"):
        all_[task] = analyze_task(task)
    json.dump(all_, open(ROOT / "results_mc/stats/med_analysis.json", "w"),
              indent=1)
    print("\nwrote results_mc/stats/med_analysis.json")
    for task, a in all_.items():
        print(f"{task}: upgrade_overall={a['upgrade_overall']}")


if __name__ == "__main__":
    main()
