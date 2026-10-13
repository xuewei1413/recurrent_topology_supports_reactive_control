"""E5b-strict：严格一一等边匹配重跑（第七轮修正 P0-1/P0-2）。

第六轮 E5b-subset 实现缺陷（外审本地重构发现）：
  matchable = 池内"≥1 个同 K3 键候选"的全部骨架边，整组删除；
  匹配组贪心一一占用、候选耗尽即跳，无 n_matched == len(matchable) 校验
  → 两组删除边数不等（Acr 匹配组实际仅删 ~75%，个别拓扑 37 对 20）。

本修正（判据预先冻结，分析端 TOST）：
  1. 严格等边：按 K3 键 k，m_k = min(n_skel(k), n_pool(k))；
     skel_strict = 冻结种子从每键选 m_k 条骨架边（共 M = Σ m_k 条）；
     matched rep r = 从每键池中随机选 m_k 条（严格同数 M）；
     rand    rep r = M 条随机非骨架边（参照）。
     → 三组删除边数严格相等（M），键级配对构成最大一一匹配
     （同键内特征全同，匹配可行性仅受数量约束，Σmin 即最大匹配）。
  2. 新 episode 段 EVAL2 = 96000-96019（全库既往实验未使用；
     既往段：70000-70019 tie、83000-85099 randtie GA、93000-93059 E5/route）。
  3. 输出 per-cell drop 数组与键级匹配计数；等效性检验（TOST, ±SESOI）
     在分析端完成，不以"不显著"推等价。

产出：results_v8/e5b_strict/e5b_strict_{cohort}_{task}_{tag}.json
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixuewei/tmp/struct_mod_v2")
sys.path.insert(0, str(ROOT / "code"))

from experiments_v2.core.protocol import TASK_SPECS, ControllerProtocol  # noqa: E402
from experiments_v2.core.scaffold import NeuronScaffold  # noqa: E402
from experiments_v2.core.encoder import ObservationEncoder  # noqa: E402
from experiments_v2.mechanisms.run_causal_propagation_v5 import (  # noqa: E402
    load_v3_topology, evaluate_adj)
from experiments_v2.mechanisms.run_e2r_matched_lesion import (  # noqa: E402
    stable_seed, edge_set_hash)
from experiments_v2.mechanisms.run_e5b_support_v7 import (  # noqa: E402
    cell_features, key_of)

EVAL2_SEEDS = list(range(96000, 96020))   # 新 held-out 段（全库未使用）
R_REPS = 10


def delete_edges(adj, edges):
    child = adj.copy()
    for s, d in edges:
        child[d, s] = 0.0
    return child


def strict_match_sets(feats, skel_set, pool, tag):
    """返回 (skel_strict, key_counts, pool_by_key)。

    skel_strict：冻结种子按键选 m_k 条骨架边（与 rep 无关，全 cell 固定）。
    """
    skel_by_key = defaultdict(list)
    for e in skel_set:
        skel_by_key[key_of(feats[e], "K3")].append(e)
    pool_by_key = defaultdict(list)
    for e in pool:
        pool_by_key[key_of(feats[e], "K3")].append(e)
    common = [k for k in skel_by_key if pool_by_key.get(k)]
    key_counts = {k: (len(skel_by_key[k]), len(pool_by_key[k]),
                      min(len(skel_by_key[k]), len(pool_by_key[k])))
                  for k in common}
    rng0 = np.random.RandomState(stable_seed("e5b_strict", "skel", tag))
    skel_strict = []
    for k in common:
        m_k = key_counts[k][2]
        idx = rng0.choice(len(skel_by_key[k]), size=m_k, replace=False)
        skel_strict.extend(skel_by_key[k][i] for i in idx)
    return skel_strict, key_counts, pool_by_key


def matched_set(pool_by_key, key_counts, tag, r):
    """rep r：按键从池中随机选 m_k 条（严格一一，边数 == len(skel_strict)）。"""
    rng = np.random.RandomState(stable_seed("e5b_strict", "match", tag, r))
    out = []
    for k, (_, _, m_k) in key_counts.items():
        idx = rng.choice(len(pool_by_key[k]), size=m_k, replace=False)
        out.extend(pool_by_key[k][i] for i in idx)
    return out


def run_cell(cohort, task_name, f0_adj, scaffold_seed, tag, out_dir, extra):
    task = TASK_SPECS[task_name]
    proto = ControllerProtocol()
    enc = ObservationEncoder.load(
        str(ROOT / f"results_v2/pilot/observation_stats/"
                   f"observation_stats_{task_name}.json"))
    scaffold = NeuronScaffold.from_seed(seed=scaffold_seed, task=task,
                                        controller=proto)
    n_in, n_out = task.n_input, task.n_output
    t0 = time.time()
    feats, skel_set, n_skel = cell_features(f0_adj, scaffold, n_in, n_out)
    pool = [e for e in feats if e not in skel_set]
    skel_strict, key_counts, pool_by_key = strict_match_sets(
        feats, skel_set, pool, tag)
    M = len(skel_strict)
    rec = {"version": "v8_e5b_strict", "cohort": cohort, "task": task_name,
           "tag": tag, "scaffold_seed": scaffold_seed, "n_hh": len(feats),
           "n_skel": n_skel, "n_strict": M,
           "strict_frac_of_skel": M / max(len(skel_set), 1),
           "n_keys": len(key_counts),
           "key_counts": {str(k): list(map(int, v))
                          for k, v in key_counts.items()},
           "skel_strict_hash": edge_set_hash(skel_strict),
           "eval_seeds": EVAL2_SEEDS, "R": R_REPS}
    if M < 5:
        rec["skipped"] = f"only {M} strictly matchable skeleton edges"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"e5b_strict_{cohort}_{task_name}_{tag}.json",
                  "w") as f:
            json.dump(rec, f, indent=1)
        print(f"[{cohort} {task_name} {tag}] skip: {rec['skipped']}",
              flush=True)
        return rec

    J_F0, _ = evaluate_adj(task, scaffold, f0_adj, enc, proto, EVAL2_SEEDS)
    J_sub, _ = evaluate_adj(task, scaffold, delete_edges(f0_adj, skel_strict),
                            enc, proto, EVAL2_SEEDS)
    rec["J_F0"] = J_F0
    rec["J_del_skel_strict"] = J_sub
    rec["drop_skel_strict"] = J_F0 - J_sub

    reps = []
    for r in range(R_REPS):
        m_edges = matched_set(pool_by_key, key_counts, tag, r)
        assert len(m_edges) == M, f"strict match violated: {len(m_edges)} != {M}"
        rng_r = np.random.RandomState(stable_seed("e5b_strict", "rand", tag, r))
        r_idx = rng_r.choice(len(pool), size=M, replace=False)
        rand_edges = [pool[i] for i in r_idx]
        J_m, _ = evaluate_adj(task, scaffold, delete_edges(f0_adj, m_edges),
                              enc, proto, EVAL2_SEEDS)
        J_r, _ = evaluate_adj(task, scaffold,
                              delete_edges(f0_adj, rand_edges),
                              enc, proto, EVAL2_SEEDS)
        reps.append({"rep": r, "n_matched": len(m_edges),
                     "matched_hash": edge_set_hash(m_edges),
                     "rand_hash": edge_set_hash(rand_edges),
                     "J_del_matched": J_m, "drop_matched": J_F0 - J_m,
                     "J_del_rand": J_r, "drop_rand": J_F0 - J_r})
        print(f"  [{tag}] rep {r}: M={M} skel_strict={J_sub:.1f} "
              f"matched={J_m:.1f} rand={J_r:.1f}", flush=True)
    rec["reps"] = reps
    rec["drop_matched_mean"] = float(np.mean([x["drop_matched"] for x in reps]))
    rec["drop_rand_mean"] = float(np.mean([x["drop_rand"] for x in reps]))
    rec["elapsed_sec"] = time.time() - t0
    rec.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"e5b_strict_{cohort}_{task_name}_{tag}.json", "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[{cohort} {task_name} {tag}] done: F0={J_F0:.1f} M={M} "
          f"skel_drop={rec['drop_skel_strict']:.1f} "
          f"matched_drop={rec['drop_matched_mean']:.1f} "
          f"rand_drop={rec['drop_rand_mean']:.1f} "
          f"({rec['elapsed_sec']:.0f}s)", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["e1", "main", "c2"], required=True)
    ap.add_argument("--task", default="CartPole")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--scaffold-seed", type=int)
    ap.add_argument("--ga-seed", type=int)
    ap.add_argument("--budget", type=int)
    args = ap.parse_args()
    out_dir = ROOT / "results_v8/e5b_strict"

    if args.cohort == "e1":
        tag = f"sf{args.scaffold_seed}_ga{args.ga_seed}_b{args.budget}"
        p = (ROOT / "results_v3/e1_variance" / args.task / tag /
             f"e1_F0_topo_{args.task}_{tag}.npy")
        if not p.exists():
            print(f"missing {p}", flush=True)
            return
        if (out_dir / f"e5b_strict_e1_{args.task}_{tag}.json").exists():
            print("done, skip", flush=True)
            return
        run_cell("e1", args.task, np.load(p), args.scaffold_seed, tag, out_dir,
                 {"ga_seed": args.ga_seed, "budget": args.budget})
    else:
        sub = "topology" if args.cohort == "main" else "topology_c2"
        adj = load_v3_topology(args.task, args.seed, str(ROOT), sub)
        if adj is None:
            print(f"missing {args.task} {args.seed}", flush=True)
            return
        tag = f"seed{args.seed}"
        if (out_dir / f"e5b_strict_{args.cohort}_{args.task}_{tag}.json").exists():
            print("done, skip", flush=True)
            return
        run_cell(args.cohort, args.task, adj, args.seed, tag, out_dir,
                 {"evo_seed": args.seed})


if __name__ == "__main__":
    main()
