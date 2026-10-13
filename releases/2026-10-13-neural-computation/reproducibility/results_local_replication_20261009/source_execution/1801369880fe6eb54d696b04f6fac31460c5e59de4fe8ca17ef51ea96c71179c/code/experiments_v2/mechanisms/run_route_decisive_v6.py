"""Route-preserved / Route-destroyed 决定性实验（prereg_route_v1.md 主实验）。

构造（全程禁止读取任务回报；只用 calibration states 上的路由相似度）：
  RW-preserved : 恰好 h 次 HH 双边 swap，最大化保留 proxy 路由相似度 S
  RW-destroyed : 恰好 h 次 HH 双边 swap，最小化 S（同 h、同候选评估次数、
                 同 Hamming 距离）
  RW-rescue    : 从 destroyed 出发，≤5% HH 边规模的引导 swaps，最大化恢复 S
  rescue-ctrl  : 从 destroyed 出发同次数随机 swaps
  dose         : K=10 个随机度保持重接（swap 数几何分布），S 连续分布

proxy 路由矩阵（搜索目标，廉价）：
  M = w1·A + w2·A² + w3·A³ 的 [out_nodes × in_nodes] 切片（scipy.sparse），
  S = cosine(flatten(M_variant), flatten(M_F0))。
经验路由张量 R[d,a,τ]（验证仪表，非搜索目标）复用 run_route_pilot_v6，
仅在 F0/preserved/destroyed/rescue/rescue-ctrl 锚点上估计（dose 变体非优化
构造，proxy S 无 Goodhart 风险，经验-代理一致性在锚点上审计）。

行为评估：evaluate_adj，EVAL_SEEDS=93000-93019（构造从未接触）。

用法：
  hscan: --mode hscan   扫描 h 工作点（proxy-only，快）
  main : --mode main    每 (task, seed) 一个进程，结果带 resume

产出：results_v6/route_decisive/route_{decisive,hscan}_v6_{task}_{seed}.json
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path("/home/lixuewei/tmp/struct_mod_v2")
sys.path.insert(0, str(ROOT / "code"))

from experiments_v2.core.protocol import TASK_SPECS, ControllerProtocol  # noqa: E402
from experiments_v2.core.weight_construction import F0  # noqa: E402
from experiments_v2.core.scaffold import NeuronScaffold  # noqa: E402
from experiments_v2.core.encoder import ObservationEncoder  # noqa: E402
from experiments_v2.core.controller import UnifiedSNNController  # noqa: E402
from experiments_v2.mechanisms.run_causal_propagation_v5 import (  # noqa: E402
    load_v3_topology, evaluate_adj,
)
from experiments_v2.mechanisms.run_route_pilot_v6 import (  # noqa: E402
    collect_calibration, estimate_tensor, cosine,
)

EVAL_SEEDS = list(range(93000, 93020))
PROXY_W = (1.0, 0.5, 0.25)      # A, A², A³ 权重
DOSE_FRACS = np.geomspace(0.02, 0.60, 10)   # dose 变体 swap 规模（占 HH 边数比例）
RESCUE_EDGE_FRAC = 0.05          # rescue swaps 触及边数上限（占 HH 边数）


# ── proxy 路由矩阵 ────────────────────────────────────────────────────

def proxy_vec(adj, n_in, n_out, w=PROXY_W):
    """M = w1·A + w2·A² + w3·A³ 的 [out × in] 切片 flatten。"""
    A = sparse.csr_matrix(adj.astype(float))
    M = w[0] * A
    A2 = A @ A
    M = M + w[1] * A2
    M = M + w[2] * (A2 @ A)
    n = adj.shape[0]
    return np.asarray(M[n - n_out:, :n_in].todense()).ravel()


def proxy_S(vec, ref):
    na, nb = np.linalg.norm(vec), np.linalg.norm(ref)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(vec @ ref / (na * nb))


# ── HH 双边 swap（adj[j,i] = i→j；只动 hidden→hidden 边）──────────────

def hh_edge_list(adj, n_in, n_out):
    n = adj.shape[0]
    hid = np.arange(n_in, n - n_out)
    H = adj[np.ix_(hid, hid)]
    jj, ii = np.nonzero(H)          # H[jj, ii]=1 → ii→jj
    return np.stack([hid[ii], hid[jj]], axis=1)   # (m, 2) [src, dst]


def propose_swap(adj, edges, rng):
    """随机双边 swap：(a→b),(c→d) → (a→d),(c→b)。返回 (i1,i2,(a,d),(c,b)) 或 None。"""
    m = len(edges)
    if m < 2:
        return None
    i1, i2 = rng.choice(m, size=2, replace=False)
    a, b = edges[i1]
    c, d = edges[i2]
    if a == d or c == b:
        return None
    if adj[d, a] > 0 or adj[b, c] > 0:     # 新边已存在
        return None
    return i1, i2, (a, b), (c, d)


def apply_swap(adj, edges, swap):
    i1, i2, (a, b), (c, d) = swap
    adj[b, a] = 0.0
    adj[d, c] = 0.0
    adj[d, a] = 1.0
    adj[b, c] = 1.0
    edges[i1] = (a, d)
    edges[i2] = (c, b)


def random_rewire(adj, n_in, n_out, n_swaps, rng):
    """纯随机度保持重接（dose 变体 / rescue-ctrl）。"""
    child = adj.copy()
    edges = hh_edge_list(child, n_in, n_out)
    done = 0
    attempts = 0
    while done < n_swaps and attempts < n_swaps * 50 + 100:
        attempts += 1
        sw = propose_swap(child, edges, rng)
        if sw is None:
            continue
        apply_swap(child, edges, sw)
        done += 1
    return child, done


def guided_rewire(adj, ref_vec, n_in, n_out, n_swaps, n_cand, rng, mode):
    """恰好 n_swaps 步；每步从 n_cand 个候选 swap 中选使 S 最大/最小者。

    mode: "preserve"（max S）/ "destroy"（min S）/ "rescue"（max S，从 destroyed）。
    候选评估次数 = n_swaps × n_cand，preserved/destroyed 完全对称。
    """
    child = adj.copy()
    edges = hh_edge_list(child, n_in, n_out)
    done = 0
    attempts = 0
    evals = 0
    while done < n_swaps and attempts < n_swaps * 80 + 200:
        attempts += 1
        best, best_S = None, None
        cur_vec = None
        for _ in range(n_cand):
            sw = propose_swap(child, edges, rng)
            if sw is None:
                continue
            apply_swap(child, edges, sw)
            v = proxy_vec(child, n_in, n_out)
            evals += 1
            s = proxy_S(v, ref_vec)
            # 撤销
            i1, i2, (a, b), (c, d) = sw
            apply_swap(child, edges, (i1, i2, (a, d), (c, b)))
            if best_S is None or (mode != "destroy") == (s > best_S):
                best, best_S = sw, s
        if best is None:
            continue
        apply_swap(child, edges, best)
        done += 1
    return child, done, evals


# ── 图统计审计 ────────────────────────────────────────────────────────

def audit(adj, f0_adj, n_in, n_out):
    n = adj.shape[0]
    hh = slice(n_in, n - n_out)
    A = adj[hh, hh]
    A2 = A @ A
    triangles = float(np.trace(A2 @ A)) / 6.0
    recip = float(np.sum(A * A.T)) / 2.0
    evals = np.linalg.eigvals(A)
    return {
        "n_edges": int(adj.sum()),
        "hamming_vs_F0": int(np.sum(adj != f0_adj)),
        "max_in_deg_dev": int(np.abs(adj.sum(axis=0) - f0_adj.sum(axis=0)).max()),
        "max_out_deg_dev": int(np.abs(adj.sum(axis=1) - f0_adj.sum(axis=1)).max()),
        "hh_spectral_radius": float(np.abs(evals).max()),
        "hh_triangles": triangles,
        "hh_reciprocal_pairs": recip,
    }


# ── 主流程 ────────────────────────────────────────────────────────────

def run_topo(task_name, evo_seed, mode, h_frac, n_cand, n_search_seeds,
             n_states, out_dir):
    task = TASK_SPECS[task_name]
    proto = ControllerProtocol()
    enc = ObservationEncoder.load(
        str(ROOT / f"results_v2/pilot/observation_stats/observation_stats_{task_name}.json"))
    n_in, n_out = task.n_input, task.n_output
    f0_adj = load_v3_topology(task_name, evo_seed, str(ROOT))
    if f0_adj is None:
        print(f"[{task_name} {evo_seed}] topology missing", flush=True)
        return
    scaffold = NeuronScaffold.from_seed(seed=evo_seed, task=task, controller=proto)
    n_hh = len(hh_edge_list(f0_adj, n_in, n_out))
    ref_vec = proxy_vec(f0_adj, n_in, n_out)
    t0 = time.time()

    if mode == "hscan":
        rows = []
        for frac in (0.05, 0.10, 0.20, 0.30):
            h = max(2, int(round(frac * n_hh / 2)))
            for sseed in range(2):
                rng = np.random.RandomState(evo_seed * 37 + sseed)
                adj_p, _, _ = guided_rewire(f0_adj, ref_vec, n_in, n_out,
                                            h, n_cand, rng, "preserve")
                rng = np.random.RandomState(evo_seed * 91 + sseed)
                adj_d, _, _ = guided_rewire(f0_adj, ref_vec, n_in, n_out,
                                            h, n_cand, rng, "destroy")
                rows.append({
                    "frac": frac, "h": h, "search_seed": sseed,
                    "S_preserved": proxy_S(proxy_vec(adj_p, n_in, n_out), ref_vec),
                    "S_destroyed": proxy_S(proxy_vec(adj_d, n_in, n_out), ref_vec),
                })
            print(f"[hscan {task_name} {evo_seed}] frac={frac} "
                  f"S_p={rows[-1]['S_preserved']:.3f} S_d={rows[-1]['S_destroyed']:.3f}",
                  flush=True)
        out = {"version": "v6_route_hscan", "task": task_name, "seed": evo_seed,
               "n_hh": n_hh, "n_cand": n_cand, "rows": rows,
               "minutes": (time.time() - t0) / 60}
        op = out_dir / f"route_hscan_v6_{task_name}_{evo_seed}.json"
        with open(op, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved {op}", flush=True)
        return

    # ── main 模式 ──
    h = max(2, int(round(h_frac * n_hh / 2)))
    n_rescue = max(1, int(round(RESCUE_EDGE_FRAC * n_hh / 2)))
    rng_cal = np.random.RandomState(2026)

    # calibration（F0 rollout，一次）
    ctrl = UnifiedSNNController(scaffold, f0_adj, enc, proto, condition=F0)
    from experiments_v2.tasks.adapter import make_env
    env = make_env(task)
    episodes, states = collect_calibration(ctrl, env, task, n_states, rng_cal)
    env.close()
    sigma = enc.sigma
    R_F0, _ = estimate_tensor(ctrl, episodes, states, sigma, 0.1,
                              task.obs_dim, n_out)

    # 仪器 QC（prereg 修订案冻结）：|R| 退化阈值 + 2-rep 半分无符号 retest
    mean_abs_R = float(np.abs(R_F0).mean())
    rng_qc = np.random.RandomState(7)
    tr_qc = []
    for rep in range(2):
        perm = rng_qc.permutation(len(states))
        h1 = [states[i] for i in perm[: len(states) // 2]]
        h2 = [states[i] for i in perm[len(states) // 2:]]
        R1, _ = estimate_tensor(ctrl, episodes, h1, sigma, 0.1,
                                task.obs_dim, n_out)
        R2, _ = estimate_tensor(ctrl, episodes, h2, sigma, 0.1,
                                task.obs_dim, n_out)
        cu = cosine(np.abs(R1), np.abs(R2))
        if cu is not None:
            tr_qc.append(cu)
    qc = {"mean_abs_R": mean_abs_R, "retest_unsigned": tr_qc,
          "retest_unsigned_mean": float(np.mean(tr_qc)) if tr_qc else None,
          "n_states": len(states),
          "pass": bool(mean_abs_R >= 1e-4 and tr_qc
                     and np.mean(tr_qc) >= 0.5)}
    if not qc["pass"]:
        rec = {"version": "v6_route_decisive", "task": task_name,
               "seed": evo_seed, "n_hh": n_hh, "qc": qc,
               "qc_fail": True, "conditions": [], "dose": []}
        op = out_dir / f"route_decisive_v6_{task_name}_{evo_seed}.json"
        with open(op, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"[{task_name} {evo_seed}] QC FAIL "
              f"(|R|={mean_abs_R:.2e} retest={qc['retest_unsigned_mean']}), skip",
              flush=True)
        return

    rec = {"version": "v6_route_decisive", "task": task_name, "seed": evo_seed,
           "n_hh": n_hh, "h": h, "h_frac": h_frac, "n_cand": n_cand,
           "n_rescue_swaps": n_rescue, "eval_seeds": EVAL_SEEDS,
           "qc": qc, "qc_fail": False,
           "S_F0_self": 1.0, "conditions": [], "dose": []}

    def emp_S(adj):
        """经验路由相似度（prereg 修订案：无符号余弦，仪器噪声更低）。"""
        c2 = UnifiedSNNController(scaffold, adj, enc, proto, condition=F0)
        R, _ = estimate_tensor(c2, episodes, states, sigma, 0.1,
                               task.obs_dim, n_out)
        return cosine(np.abs(R), np.abs(R_F0))

    def behavior(adj):
        mean, per = evaluate_adj(task, scaffold, adj, enc, proto, EVAL_SEEDS)
        return mean, per

    # F0 行为（复用口径，主表已有，这里重算保证同评估种子）
    m0, per0 = behavior(f0_adj)
    rec["F0"] = {"return_mean": m0, "per_episode": per0}

    # preserved / destroyed × n_search_seeds；rescue 从各 destroyed 出发
    for sseed in range(n_search_seeds):
        rng = np.random.RandomState(evo_seed * 37 + sseed)
        adj_p, sw_p, ev_p = guided_rewire(f0_adj, ref_vec, n_in, n_out,
                                          h, n_cand, rng, "preserve")
        rng = np.random.RandomState(evo_seed * 91 + sseed)
        adj_d, sw_d, ev_d = guided_rewire(f0_adj, ref_vec, n_in, n_out,
                                          h, n_cand, rng, "destroy")
        rng = np.random.RandomState(evo_seed * 53 + sseed)
        adj_r, sw_r, _ = guided_rewire(adj_d, ref_vec, n_in, n_out,
                                       n_rescue, n_cand, rng, "rescue")
        rng = np.random.RandomState(evo_seed * 77 + sseed)
        adj_rc, sw_rc = random_rewire(adj_d, n_in, n_out, n_rescue, rng)

        for name, adj in (("preserved", adj_p), ("destroyed", adj_d),
                          ("rescue", adj_r), ("rescue_ctrl", adj_rc)):
            S_emp = emp_S(adj)
            m, per = behavior(adj)
            rec["conditions"].append({
                "search_seed": sseed, "condition": name,
                "S_proxy": proxy_S(proxy_vec(adj, n_in, n_out), ref_vec),
                "S_empirical": S_emp,
                "return_mean": m, "per_episode": per,
                "audit": audit(adj, f0_adj, n_in, n_out),
            })
        rec["conditions"][-4]["swaps_done"] = sw_p
        rec["conditions"][-4]["candidate_evals"] = ev_p
        rec["conditions"][-3]["swaps_done"] = sw_d
        rec["conditions"][-3]["candidate_evals"] = ev_d
        rec["conditions"][-2]["swaps_done"] = sw_r
        rec["conditions"][-1]["swaps_done"] = sw_rc
        print(f"[{task_name} {evo_seed} s{sseed}] "
              f"S_emp p={rec['conditions'][-4]['S_empirical']:.3f} "
              f"d={rec['conditions'][-3]['S_empirical']:.3f} "
              f"r={rec['conditions'][-2]['S_empirical']:.3f} | "
              f"J p={rec['conditions'][-4]['return_mean']:.1f} "
              f"d={rec['conditions'][-3]['return_mean']:.1f} "
              f"r={rec['conditions'][-2]['return_mean']:.1f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
        # 每个 search seed 落盘一次（resume 粒度）
        op = out_dir / f"route_decisive_v6_{task_name}_{evo_seed}.json"
        with open(op, "w") as f:
            json.dump(rec, f, indent=2)

    # dose 变体（纯随机，proxy S 无 Goodhart；不测经验张量）
    for k, frac in enumerate(DOSE_FRACS):
        n_sw = max(1, int(round(frac * n_hh / 2)))
        rng = np.random.RandomState(evo_seed * 131 + k)
        adj_k, sw_k = random_rewire(f0_adj, n_in, n_out, n_sw, rng)
        m, per = behavior(adj_k)
        rec["dose"].append({
            "k": k, "frac": float(frac), "swaps_done": sw_k,
            "S_proxy": proxy_S(proxy_vec(adj_k, n_in, n_out), ref_vec),
            "return_mean": m, "per_episode": per,
            "audit": audit(adj_k, f0_adj, n_in, n_out),
        })
    rec["minutes"] = (time.time() - t0) / 60
    op = out_dir / f"route_decisive_v6_{task_name}_{evo_seed}.json"
    with open(op, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"Saved {op} ({rec['minutes']:.1f} min)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", choices=["hscan", "main"], default="main")
    ap.add_argument("--h-frac", type=float, default=0.20)
    ap.add_argument("--n-cand", type=int, default=40)
    ap.add_argument("--n-search-seeds", type=int, default=3)
    ap.add_argument("--n-states", type=int, default=200)
    args = ap.parse_args()
    out_dir = ROOT / "results_v6/route_decisive"
    out_dir.mkdir(parents=True, exist_ok=True)
    op = out_dir / f"route_decisive_v6_{args.task}_{args.seed}.json"
    if args.mode == "main" and op.exists():
        with open(op) as f:
            d = json.load(f)
        if d.get("qc_fail"):
            print(f"[{args.task} {args.seed}] QC fail on record, skip", flush=True)
            return
        if d.get("dose") and len(d.get("conditions", [])) >= 4 * args.n_search_seeds:
            print(f"[{args.task} {args.seed}] already complete, skip", flush=True)
            return
    run_topo(args.task, args.seed, args.mode, args.h_frac, args.n_cand,
             args.n_search_seeds, args.n_states, out_dir)


if __name__ == "__main__":
    main()
