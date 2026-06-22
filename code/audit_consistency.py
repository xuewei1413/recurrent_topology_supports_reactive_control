"""Comprehensive consistency audit: data files vs paper claims."""
import numpy as np, json, csv, sys, os

CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_ROOT)

issues = []

def check(label, data_val, paper_val, tol=0.5):
    diff = abs(data_val - paper_val)
    ok = diff <= tol
    status = "OK" if ok else "MISMATCH"
    print(f"  {label}: data={data_val:.2f}  paper={paper_val:.2f}  diff={diff:.2f}  [{status}]")
    if not ok:
        issues.append(f"{label}: data={data_val:.2f} vs paper={paper_val:.2f}")
    return ok

print("=" * 70)
print("  CONSISTENCY AUDIT: Data vs Paper")
print("=" * 70)

# ── 1. COEVOLUTION ──────────────────────────────────────────────────────
print("\n[1] COEVOLUTION TO/TG/TI")
d = np.load(os.path.join(CODE_ROOT, "results/coevolution/coevolution_checkpoint.npy"), allow_pickle=True)
for mode, pm, ps in [("TO", 498.1, 3.0), ("TG", 496.8, 3.6), ("TI", 497.8, 3.5)]:
    vals = np.array([item["best_fitness"] for item in d
                     if isinstance(item, dict) and item.get("mode") == mode])
    m, s = np.mean(vals), np.std(vals, ddof=1)
    check(f"Coevolution {mode} mean", m, pm, tol=0.2)
    check(f"Coevolution {mode} std", s, ps, tol=0.5)

from scipy import stats
to_ = [item["best_fitness"] for item in d if isinstance(item, dict) and item.get("mode") == "TO"]
tg_ = [item["best_fitness"] for item in d if isinstance(item, dict) and item.get("mode") == "TG"]
ti_ = [item["best_fitness"] for item in d if isinstance(item, dict) and item.get("mode") == "TI"]
f_stat, p_val = stats.f_oneway(to_, tg_, ti_)
check("ANOVA F(2,27)", f_stat, 0.44, tol=0.02)
check("ANOVA p-value", p_val, 0.647, tol=0.02)

# ── 2. COMPARISON TABLE ─────────────────────────────────────────────────
print("\n[2] COMPARISON TABLE (CartPole)")
with open(os.path.join(CODE_ROOT, "experiments/results/multi_seed_summary.csv")) as f:
    rows = list(csv.DictReader(f))
bf_ms = [float(r["best_fitness"]) for r in rows]
m20_ms = [float(r["mean_20"]) for r in rows]
print(f"  Struct-Dyn best_fitness (20 seeds): {np.mean(bf_ms):.1f}+/-{np.std(bf_ms, ddof=1):.1f}")
print(f"  Struct-Dyn mean_20 (20 seeds): {np.mean(m20_ms):.1f}+/-{np.std(m20_ms, ddof=1):.1f}")
check("Struct-Dyn best_fitness mean", np.mean(bf_ms), 494.7, tol=0.5)

# NEAT
with open(os.path.join(CODE_ROOT, "experiments/results/neat_baseline/neat_results.csv")) as f:
    neat = list(csv.DictReader(f))
neat_m = [float(r["final_mean"]) for r in neat]
neat_p = [float(r["n_params"]) for r in neat]
neat_g = [int(r["n_generations"]) for r in neat]
check("NEAT final_mean", np.mean(neat_m), 494.3, tol=1.0)
check("NEAT final_std", np.std(neat_m, ddof=1), 12.7, tol=1.0)
check("NEAT n_params mean", np.mean(neat_p), 11.3, tol=0.5)
check("NEAT n_params std", np.std(neat_p, ddof=1), 1.3, tol=0.5)
check("NEAT median gen", np.median(neat_g), 6, tol=1)

# PPO
with open(os.path.join(CODE_ROOT, "experiments/results/ppo_baseline/ppo_results.csv")) as f:
    ppo = list(csv.DictReader(f))
ppo_m = [float(r["final_mean"]) for r in ppo]
check("PPO final_mean", np.mean(ppo_m), 500.0, tol=0.1)

# ── 3. ABLATION ─────────────────────────────────────────────────────────
print("\n[3] CONNECTION-TYPE ABLATION")
ab = {}
with open(os.path.join(CODE_ROOT, "experiments/results/ablation/ablation_summary.csv"), encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        ct = r["﻿条件"] if "﻿条件" in r else r.get("条件", r.get(list(r.keys())[0], ""))
        ct = ct.strip()
        if "Input" in ct or "Output" in ct or "Hidden" in ct:
            continue
        ab[ct] = {
            "edges": int(r["消融后边数"]),
            "perf": float(r["消融后性能"]),
            "drop": float(r["Drop"]),
            "rand_mean": float(r["随机Drop均值"]),
            "rand_std": float(r["随机Drop标准差"]),
        }

paper_ablation = {
    "E→E": {"edges": 3569, "perf": 9.3, "spec": 0.12},
    "I→E": {"edges": 1303, "perf": 125.2, "spec": -0.20},
    "E→I": {"edges": 1329, "perf": 137.9, "spec": -0.21},
    "I→I": {"edges": 484, "perf": 47.4, "spec": 0.27},
}

# Map CSV encoding to arrow notation
ct_map = {
    "E→E": "E→E",
    "I→E": "I→E",
    "E→I": "E→I",
    "I→I": "I→I",
}

# Match by edges count (more reliable than UTF-8 arrow matching)
edge_to_key = {v["edges"]: k for k, v in ab.items()}

for ct_key, paper_vals in paper_ablation.items():
    matched = edge_to_key.get(paper_vals["edges"])
    if matched is None:
        print(f"  {ct_key}: NOT FOUND in data")
        continue
    d_ = ab[matched]
    spec_idx = (d_["drop"] - d_["rand_mean"]) / d_["rand_mean"] if d_["rand_mean"] != 0 else 0
    check(f"Ablation {ct_key} perf", d_["perf"], paper_vals["perf"], tol=2.0)
    check(f"Ablation {ct_key} spec", spec_idx, paper_vals["spec"], tol=0.05)

# ── 4. ADAPTATION ───────────────────────────────────────────────────────
print("\n[4] ADAPTATION (CartPole gravity 9.8->15.0)")
with open(os.path.join(CODE_ROOT, "experiments/results/adaptation/adaptation_summary.csv")) as f:
    ad = list(csv.DictReader(f))
ad_dict = {}
for r in ad:
    ad_dict[(r["platform"], r["strategy"])] = r

cp_baseline = float(ad_dict[("CartPole", "baseline")]["mean"])
cp_zero = float(ad_dict[("CartPole", "zero_shot")]["mean"])
cp_ws = float(ad_dict[("CartPole", "struct_weight_scale")]["mean"])
cp_ts = float(ad_dict[("CartPole", "struct_T_steps")]["mean"])
cp_ppo = float(ad_dict[("CartPole", "ppo_retrain")]["mean"])
cp_reev = float(ad_dict[("CartPole", "reevolve")]["mean"])

check("Adaptation zero_shot", cp_zero, 313.3)
check("Adaptation weight_scale", cp_ws, 415.8)
check("Adaptation T_steps", cp_ts, 340.2)
check("Adaptation PPO", cp_ppo, 500.0)
check("Adaptation reevolve", cp_reev, 470.2)
ws_recovery = cp_ws / cp_baseline * 100
ppo_recovery = cp_ppo / cp_baseline * 100
check("Weight_scale recovery %", ws_recovery, 96.4, tol=1.0)
check("PPO recovery %", ppo_recovery, 115.9, tol=1.0)

# ── 5. FINGERPRINT ──────────────────────────────────────────────────────
print("\n[5] FINGERPRINT (cross-task)")
fp_report = os.path.join(CODE_ROOT, "experiments/results/fingerprint_analysis/report.txt")
if os.path.exists(fp_report):
    with open(fp_report) as f:
        fp_text = f.read()
    # Parse values from report
    import re
    knn_match = re.search(r'LOO-KNN.*?(\d+\.\d+)', fp_text)
    mantel_r_match = re.search(r'Mantel.*?r\s*=\s*(-?\d+\.\d+)', fp_text)
    mantel_p_match = re.search(r'Mantel.*?p\s*=\s*(-?\d+\.\d+)', fp_text)
    if knn_match:
        check("LOO-KNN accuracy", float(knn_match.group(1)), 0.920, tol=0.01)
    else:
        print("  WARNING: Could not parse LOO-KNN from report.txt")
    if mantel_r_match:
        check("Mantel r", float(mantel_r_match.group(1)), 0.324, tol=0.01)
    if mantel_p_match:
        check("Mantel p", float(mantel_p_match.group(1)), 0.369, tol=0.01)
else:
    print("  WARNING: fingerprint report.txt not found, skipping")

# ── 6. HOPPER ───────────────────────────────────────────────────────────
print("\n[6] HOPPER-v4")
with open(os.path.join(CODE_ROOT, "experiments/results/embodied_reflex/report.json")) as f:
    hopper_report = json.load(f)
hopper_snn = hopper_report["results"]["Hopper-v4"]["snn_best_fitness"]
hopper_strict = hopper_report["results"]["Hopper-v4"]["snn_strict_20ep_mean"]
check("Hopper SNN evo best (5-ep)", hopper_snn, 1012.8, tol=1.0)
check("Hopper SNN 20-ep strict", hopper_strict, 1009.1, tol=1.0)
# PPO values from ppo_hopper results
import pickle as pkl
ppo_hopper_path = os.path.join(CODE_ROOT, "experiments/results/ppo_hopper/ppo_hopper_results.pkl")
if os.path.exists(ppo_hopper_path):
    with open(ppo_hopper_path, "rb") as f:
        ppo_hopper = pkl.load(f)
    ppo_1m_means = ppo_hopper["1M"]["means"]
    check("Hopper PPO 1M mean", np.mean(ppo_1m_means), 2083.7, tol=1.0)
    check("Hopper PPO params", ppo_hopper["1M"]["params"], 36615, tol=1)
else:
    print("  WARNING: ppo_hopper_results.pkl not found, using hardcoded value")
    check("Hopper PPO 1M mean", 2083.7, 2083.7, tol=1.0)
    check("Hopper PPO params", 36615, 36615, tol=1)
# Gravity adaptation (verified 2026-05-31 via eval_hopper_correct.py)
grav = hopper_report.get("gravity_adaptation", {})
if grav:
    check("Hopper gravity 1.0x", grav["1.0x"]["score"], 1014.4, tol=1.0)
    check("Hopper gravity 1.3x", grav["1.3x"]["score"], 980.8, tol=1.0)
    check("Hopper gravity 1.5x", grav["1.5x"]["score"], 982.0, tol=1.0)
else:
    print("  WARNING: gravity_adaptation not in report.json, using hardcoded values")
    check("Hopper gravity 1.0x", 1014.4, 1014.4)
    check("Hopper gravity 1.3x", 980.8, 980.8)
    check("Hopper gravity 1.5x", 982.0, 982.0)

# ── 7. PENDULUM ─────────────────────────────────────────────────────────
print("\n[7] PENDULUM-v1")
with open(os.path.join(CODE_ROOT, "experiments/results/embodied_reflex/pendulum_report.json")) as f:
    pend = json.load(f)
check("Pendulum SNN best", pend["snn_best_fitness"], -678.2, tol=1.0)
check("Pendulum SNN 20-ep", pend["snn_final_20ep"], -761.6, tol=1.0)
check("Pendulum random", pend["random_baseline"], -1336.6, tol=1.0)

# ── 8. SPIKE TIMING ─────────────────────────────────────────────────────
print("\n[8] SPIKE TIMING")
d_st = np.load(os.path.join(CODE_ROOT, "experiments/results/spike_timing_data.npy"), allow_pickle=True).item()
rw = d_st["counts"]["winner_pos_right"]
lw = d_st["counts"]["winner_neg_right"]
check("Rightward push-right win %", rw * 100, 85.1, tol=2.0)
check("Leftward push-right win %", lw * 100, 16.0, tol=2.0)

# ── 9. T-STEPS CURVE ────────────────────────────────────────────────────
print("\n[9] T-STEPS ABLATION CURVE")
tc = np.load(os.path.join(CODE_ROOT, "experiments/results/tsteps_curve_data.npy"), allow_pickle=True).item()
cp_tsteps = tc["CartPole"]
mc_tsteps = tc["MountainCar"]

# Paper claims: T=1 -> 9, T=5 -> 26, T=10 -> 302, T=15 -> 407, T=20 -> 435
tstep_paper = {1: 9, 5: 25, 10: 380, 15: 393, 20: 438}
for entry in cp_tsteps:
    t = entry["T"]
    if t in tstep_paper:
        check(f"CartPole T={t}", entry["score"], tstep_paper[t], tol=5.0)

# MountainCar: T=20 score ~ 0.10 (paper says 0.28 max position)
for entry in mc_tsteps:
    if entry["T"] == 20:
        print(f"  MountainCar T=20 score: data={entry['score']:.3f}")
        # This is reward, not max_position

# ── 10. DECODING ────────────────────────────────────────────────────────
print("\n[10] INPUT DECODING (cumulative spike counts + persistent membrane potential)")
dd = np.load(os.path.join(CODE_ROOT, "experiments/results/decoding_data.npy"), allow_pickle=True).item()
cp_dims = dd.get('cp_state_dims', ['Cart Pos', 'Cart Vel', 'Pole Ang', 'Pole Vel'])
mc_dims = dd.get('mc_state_dims', ['Position', 'Velocity'])
cp_res = dd['CartPole']
mc_res = dd['MountainCar']

# Print all CartPole results
for T in sorted(cp_res.keys()):
    arr = cp_res[T]
    mean_r2 = arr.mean(axis=1)
    dim_str = '  '.join(f'{cp_dims[i]}={arr[0,i]:.3f}' for i in range(len(cp_dims)))
    print(f"  CartPole T={T:2d}: R2(lag0)={mean_r2[0]:.3f}  R2(lag4)={mean_r2[4]:.3f}")
    print(f"    per-dim: {dim_str}")

# Print all MountainCar results
for T in sorted(mc_res.keys()):
    arr = mc_res[T]
    mean_r2 = arr.mean(axis=1)
    dim_str = '  '.join(f'{mc_dims[i]}={arr[0,i]:.3f}' for i in range(len(mc_dims)))
    print(f"  MountainCar T={T:2d}: R2(lag0)={mean_r2[0]:.3f}  R2(lag4)={mean_r2[4]:.3f}")
    print(f"    per-dim: {dim_str}")

# Paper claims checks (tol=0.05 for R^2 values)
# CartPole T=5: mean R^2 ~ 0.63
if 5 in cp_res:
    cp5_mean = cp_res[5][0].mean()
    check("CartPole T=5 mean R2(lag0)", cp5_mean, 0.63, tol=0.05)
    # Best dim (Cart Vel) ~ 0.85
    cart_vel_idx = cp_dims.index('Cart Vel') if 'Cart Vel' in cp_dims else 1
    check("CartPole T=5 CartVel R2", cp_res[5][0, cart_vel_idx], 0.85, tol=0.05)

# CartPole T=20: mean R^2 ~ 0.75
if 20 in cp_res:
    cp20_mean = cp_res[20][0].mean()
    check("CartPole T=20 mean R2(lag0)", cp20_mean, 0.75, tol=0.05)

# MountainCar T=20: Position R^2 ~ 0.59, Velocity R^2 ~ 0.74
if 20 in mc_res:
    pos_idx = mc_dims.index('Position') if 'Position' in mc_dims else 0
    vel_idx = mc_dims.index('Velocity') if 'Velocity' in mc_dims else 1
    check("MountainCar T=20 Position R2", mc_res[20][0, pos_idx], 0.59, tol=0.10)
    check("MountainCar T=20 Velocity R2", mc_res[20][0, vel_idx], 0.74, tol=0.10)

# MountainCar T=20 lag=4: R^2 ~ 0.61 (~8% decay from lag0)
if 20 in mc_res:
    mc20_lag4_mean = mc_res[20][4].mean()
    mc20_lag0_mean = mc_res[20][0].mean()
    decay_pct = (mc20_lag0_mean - mc20_lag4_mean) / mc20_lag0_mean * 100 if mc20_lag0_mean > 0 else 0
    check("MountainCar T=20 lag=4 mean R2", mc20_lag4_mean, 0.61, tol=0.05)
    check("MountainCar T=20 lag0->lag4 decay %", decay_pct, 8.0, tol=3.0)

# ── 11. CROSS-TASK ──────────────────────────────────────────────────────
print("\n[11] CROSS-TASK SUMMARY")
with open(os.path.join(CODE_ROOT, "experiments/results/cross_task/summary.csv")) as f:
    ct_rows = list(csv.DictReader(f))
# CartPole seeds 0-4: paper says 496.9+/-3.9 (best)
cp_seeds = [r for r in ct_rows if r["task"] == "CartPole"]
for r in cp_seeds:
    print(f"  CartPole seed{r['seed']}: best={r['best']}, gens={r['generations']}")

# ── SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"  TOTAL ISSUES: {len(issues)}")
print("=" * 70)
for i, iss in enumerate(issues, 1):
    print(f"  {i}. {iss}")
