"""F-pilot：经验路由张量 R[d,a,τ] 估计器 + test-retest 噪声 + 线性度标定。

定义：calibration 轨迹（F0 真实 rollout，episode seeds 与评估不相交）上，
对时刻 t 的标准化输入维 d 注入 +ε·σ_d 扰动（仅 t 时刻），
重放原始未来观测，测动作 a 的 output margin 在 t+τ 的变化：
    R[d,a,τ] = mean_states[ margin'_a(t+τ) − margin_a(t+τ) ]
margin_a = spike_accum[a] − max_{b≠a} spike_accum[b]（每个 forward 的读出）。

pilot 产出（prereg §6/§9）：
  1. 线性度：ε ∈ {0.05,0.1,0.2}·σ 下 R 的余弦一致性（冻结 ε）
  2. test-retest：同拓扑、4 组随机半分校准态的 S(R1,R2) 均值/SD → SESOI_S
  3. 跨拓扑 S(R_i,R_j)：S 的自然动态范围（为 preserved/destroyed 分离定标）
  4. 计时 → 主实验预算

产出：results_v6/route_pilot/route_pilot_v6_{task}.json
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixuewei/tmp/struct_mod_v2")
sys.path.insert(0, str(ROOT / "code"))

from experiments_v2.core.protocol import TASK_SPECS, ControllerProtocol  # noqa: E402
from experiments_v2.core.weight_construction import F0  # noqa: E402
from experiments_v2.core.scaffold import NeuronScaffold  # noqa: E402
from experiments_v2.core.encoder import ObservationEncoder  # noqa: E402
from experiments_v2.core.controller import UnifiedSNNController  # noqa: E402
from experiments_v2.tasks.adapter import make_env  # noqa: E402
from experiments_v2.mechanisms.run_causal_propagation_v5 import (  # noqa: E402
    load_v3_topology,
)

TAU_MAX = 5
CAL_SEEDS = list(range(95000, 95030))   # 最多 30 episode，自适应凑够目标态数
MIN_STATES = 250


def cosine(a, b):
    a, b = a.ravel(), b.ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return None
    return float(a @ b / (na * nb))


def margin_per_action(spike_accum, n_out, readout="margin"):
    acc = spike_accum[-n_out:]
    if readout == "acc":
        # 连续读出：归一化动作活动（无 argmax 阈值噪声）
        tot = acc.sum()
        if tot < 1e-12:
            return np.zeros(n_out)
        return acc / tot
    m = np.zeros(n_out)
    for a in range(n_out):
        others = np.delete(acc, a)
        m[a] = acc[a] - (others.max() if len(others) else 0.0)
    return m


def collect_calibration(ctrl, env, task, n_snap_target, rng):
    """真实 rollout，记录 obs 序列与采样时刻的膜电位快照。

    每 episode 按自身长度采样（旧版按最短 episode 截断，短 episode 任务
    状态数被压垮）；自适应加 episode 直到 ≥ max(MIN_STATES, n_snap_target)
    或 CAL_SEEDS 耗尽。
    """
    episodes, states = [], []
    per_ep = max(1, n_snap_target // 8)
    for ep_seed in CAL_SEEDS:
        obs, _ = env.reset(seed=ep_seed)
        ctrl.reset()
        seq, snaps = [], {}
        for t in range(task.max_steps):
            seq.append(obs.copy())
            a, acc, _ = ctrl.forward(obs, return_trace=True)
            snaps[t] = (ctrl.v.copy(), acc[-ctrl.n_out:].copy())
            obs, r, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        ei = len(episodes)
        episodes.append({"seq": seq, "snaps": snaps})
        cand_t = list(range(5, max(6, len(seq) - TAU_MAX - 2)))
        n_take = min(len(cand_t), per_ep)
        idx = np.linspace(0, len(cand_t) - 1, n_take).astype(int)
        states.extend((ei, cand_t[i]) for i in idx)
        if len(states) >= max(MIN_STATES, n_snap_target):
            break
    return episodes, states


def estimate_tensor(ctrl, episodes, states, sigma, eps_frac, obs_dim, n_out,
                    readout="acc", pulse_steps=3):
    """R[d,a,τ] = mean over states of 读出 差。

    扰动：t..t+pulse_steps-1 连续 pulse_steps 步 obs[d] += eps_frac·σ_d。
    """
    R = np.zeros((obs_dim, n_out, TAU_MAX))
    se_accum = np.zeros((obs_dim, n_out, TAU_MAX))
    clone = copy.deepcopy(ctrl)
    for d in range(obs_dim):
        deltas = []
        for ei, t in states:
            ep = episodes[ei]
            v_snap = ep["snaps"][t][0]
            seq = ep["seq"]
            margins_base, margins_pert = [], []
            for branch in ("b", "p"):
                clone.v[:] = v_snap
                ms = []
                for k in range(TAU_MAX + 1):
                    o = seq[t + k].copy()
                    if branch == "p" and k < pulse_steps:
                        o[d] += eps_frac * sigma[d]
                    _, acc, _ = clone.forward(o, return_trace=True)
                    ms.append(margin_per_action(acc, n_out, readout))
                if branch == "b":
                    margins_base = ms
                else:
                    margins_pert = ms
            deltas.append([margins_pert[k] - margins_base[k]
                           for k in range(TAU_MAX + 1)])
        deltas = np.array(deltas)  # (n_states, TAU_MAX+1, n_out)
        R[d] = deltas.mean(axis=0)[1:].T           # (n_out, τ=1..TAU_MAX)
        se_accum[d] = (deltas.std(axis=0)[1:].T
                       / max(1, np.sqrt(len(states))))
    return R, se_accum


def run(task_name, n_topo, n_states, seed_base):
    task = TASK_SPECS[task_name]
    proto = ControllerProtocol()
    enc = ObservationEncoder.load(
        str(ROOT / f"results_v2/pilot/observation_stats/observation_stats_{task_name}.json"))
    sigma = enc.sigma
    obs_dim, n_out = task.obs_dim, task.n_output
    rng = np.random.RandomState(2026)

    out_dir = ROOT / "results_v6/route_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors, timing = {}, {}
    for si in range(n_topo):
        evo_seed = seed_base + si
        adj = load_v3_topology(task_name, evo_seed, str(ROOT))
        if adj is None:
            continue
        t0 = time.time()
        scaffold = NeuronScaffold.from_seed(seed=evo_seed, task=task, controller=proto)
        ctrl = UnifiedSNNController(scaffold, adj, enc, proto, condition=F0)
        env = make_env(task)
        episodes, states = collect_calibration(ctrl, env, task, n_states, rng)
        # 主估计（ε=0.1σ）
        R, se = estimate_tensor(ctrl, episodes, states, sigma, 0.1, obs_dim, n_out)
        # 线性度：子集 × 3 个 ε
        sub = states[: min(24, len(states))]
        lin = {}
        Rs = {}
        for ef in (0.05, 0.1, 0.2):
            Rs[ef], _ = estimate_tensor(ctrl, episodes, sub, sigma, ef,
                                        obs_dim, n_out)
        for e1, e2 in ((0.05, 0.1), (0.1, 0.2)):
            c = cosine(Rs[e1], Rs[e2])
            lin[f"cos_e{e1}_e{e2}"] = c
        # test-retest：4 组随机半分（签名 / 无符号 |R| 双版本，pilot 冻结其一）
        tr, tr_unsigned = [], []
        mean_abs_R = float(np.abs(R).mean())
        for rep in range(4):
            perm = rng.permutation(len(states))
            h1 = [states[i] for i in perm[: len(states) // 2]]
            h2 = [states[i] for i in perm[len(states) // 2:]]
            R1, _ = estimate_tensor(ctrl, episodes, h1, sigma, 0.1, obs_dim, n_out)
            R2, _ = estimate_tensor(ctrl, episodes, h2, sigma, 0.1, obs_dim, n_out)
            c = cosine(R1, R2)
            cu = cosine(np.abs(R1), np.abs(R2))
            if c is not None:
                tr.append(c)
            if cu is not None:
                tr_unsigned.append(cu)
        tensors[evo_seed] = {
            "R": R.tolist(), "se_mean_abs": float(np.abs(se).mean()),
            "mean_abs_R": mean_abs_R,
            "tensor_degenerate": bool(mean_abs_R < 1e-6),
            "linearity": lin, "retest_S": tr,
            "retest_S_unsigned": tr_unsigned,
            "retest_S_mean": float(np.mean(tr)), "retest_S_sd": float(np.std(tr)),
            "retest_S_unsigned_mean": (float(np.mean(tr_unsigned))
                                       if tr_unsigned else None),
            "n_states": len(states),
        }
        timing[evo_seed] = time.time() - t0
        env.close()
        print(f"  [pilot {task_name} seed={evo_seed}] "
              f"retest S={np.mean(tr):.4f}±{np.std(tr):.4f} "
              f"|R| S={np.mean(tr_unsigned) if tr_unsigned else float('nan'):.4f} "
              f"|R|_mean={mean_abs_R:.2e} "
              f"lin={lin} ({timing[evo_seed]/60:.1f} min)", flush=True)

    # 跨拓扑 S
    seeds = sorted(tensors)
    cross = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            c = cosine(np.array(tensors[seeds[i]]["R"]),
                       np.array(tensors[seeds[j]]["R"]))
            if c is not None:
                cross.append({"pair": [seeds[i], seeds[j]], "S": c})
    cross_vals = [c["S"] for c in cross]
    summary = {
        "task": task_name, "n_topo": len(seeds), "n_states": n_states,
        "retest_S_mean": float(np.mean([tensors[s]["retest_S_mean"] for s in seeds])),
        "retest_S_sd_pooled": float(np.sqrt(np.mean(
            [tensors[s]["retest_S_sd"] ** 2 for s in seeds]))),
        "retest_S_unsigned_mean": float(np.mean(
            [tensors[s]["retest_S_unsigned_mean"] for s in seeds
             if tensors[s]["retest_S_unsigned_mean"] is not None])),
        "n_degenerate": int(sum(tensors[s]["tensor_degenerate"] for s in seeds)),
        "mean_abs_R_per_topo": {str(s): tensors[s]["mean_abs_R"] for s in seeds},
        "cross_topo_S_mean": float(np.mean(cross_vals)) if cross_vals else None,
        "cross_topo_S_sd": float(np.std(cross_vals)) if cross_vals else None,
        "minutes_per_topo": float(np.mean(list(timing.values()))),
        "sesoi_S_frozen": None,  # 由 n_topo>=5 时冻结
    }
    out = {"version": "v6_route_pilot", "summary": summary,
           "per_topo": {str(k): {kk: (vv if kk != "R" else "omitted")
                                 for kk, vv in v.items()}
                        for k, v in tensors.items()},
           "cross_topo": cross}
    op = out_dir / f"route_pilot_v6_{task_name}.json"
    with open(op, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved {op}", flush=True)
    print(f"[summary] retest S={summary['retest_S_mean']:.4f} "
          f"cross-topo S={summary['cross_topo_S_mean']} "
          f"({summary['minutes_per_topo']:.1f} min/topo)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="CartPole")
    ap.add_argument("--n-topo", type=int, default=3)
    ap.add_argument("--n-states", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=3000)
    args = ap.parse_args()
    run(args.task, args.n_topo, args.n_states, args.seed_base)
