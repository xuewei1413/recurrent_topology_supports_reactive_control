"""v3 c2 独立确认队列测量（prereg_regime_v3 §4/§6：判据 ρ>0.40 且 p<0.01）。

队列：4260-4274（15 谱系），EVAL 89000-89019。
测量同主队列：背景族区指数 + 浴盆 cover 臂（J24 主结局）+ 输入族 + fluct + d_norm。
输出：results_regime3_c2/measure/measure_c2_CartPole_{seed}.json
"""
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixuewei/tmp/struct_mod_v2")
sys.path.insert(0, str(ROOT / "code"))

from experiments_v2.core.protocol import TASK_SPECS, ControllerProtocol  # noqa: E402
from experiments_v2.core.scaffold import NeuronScaffold  # noqa: E402
from experiments_v2.core.encoder import ObservationEncoder  # noqa: E402
from experiments_v2.core.controller import UnifiedSNNController  # noqa: E402
from experiments_v2.core.weight_construction import F0  # noqa: E402
from experiments_v2.tasks.adapter import make_env  # noqa: E402
from experiments_v2.mechanisms.run_lesion_fingerprint_v6 import (  # noqa: E402
    collect_calibration)
from experiments_v2.mechanisms.run_lesion_v2 import (  # noqa: E402
    fingerprint, fixed_groups, replay_margins)
from experiments_v2.mechanisms.run_skeleton_v6 import edge_contributions  # noqa: E402
from experiments_v2.mechanisms.run_route_decisive_v6 import hh_edge_list  # noqa: E402

TASK = "CartPole"
EVAL_SEEDS = list(range(89000, 89020))       # c2 测量段
G = 32
SESOI = 30.0
KEY_STEPS = {1, 4, 8, 12, 16, 20, 24, 28, 31, 32}
N_RAND = 10
SKEL_FRAC = 0.20
N_STATES = 250
EVO = ROOT / "results_regime3_c2/evolution"
OUT = ROOT / "results_regime3_c2/measure"


def restricted_rewire(adj, n_in, n_out, keep_mask, n_swaps, rng):
    child = adj.copy()
    edges = hh_edge_list(child, n_in, n_out)
    done, attempts = 0, 0
    while done < n_swaps and attempts < n_swaps * 80 + 200:
        attempts += 1
        m = len(edges)
        i1, i2 = rng.choice(m, size=2, replace=False)
        a, b = edges[i1]
        c, d = edges[i2]
        if keep_mask[b, a] or keep_mask[d, c] or keep_mask[d, a] or keep_mask[b, c]:
            continue
        if a == d or c == b:
            continue
        if child[d, a] > 0 or child[b, c] > 0:
            continue
        child[b, a] = 0.0
        child[d, c] = 0.0
        child[d, a] = 1.0
        child[b, c] = 1.0
        edges[i1] = (a, d)
        edges[i2] = (c, b)
        done += 1
    return child, done


def keep_mask_of(shape, keep_edges):
    keep = np.zeros(shape, dtype=bool)
    for (a, b) in keep_edges:
        keep[b, a] = True
    return keep


def one_seed(seed):
    t0 = time.time()
    task = TASK_SPECS[TASK]
    proto = ControllerProtocol()
    adj = np.load(EVO / f"F0_topology_c2_{TASK}_seed{seed}.npy").astype(int)
    n_in, n_out = task.n_input, task.n_output
    scaffold = NeuronScaffold.from_seed(seed=seed, task=task, controller=proto)
    enc = ObservationEncoder.load(
        str(ROOT / f"results_v2/pilot/observation_stats/observation_stats_{TASK}.json"))
    env = make_env(task)
    ctrl = UnifiedSNNController(scaffold, adj, enc, proto, condition=F0)
    J_F0 = ctrl.evaluate(env, EVAL_SEEDS, task.max_steps)["mean"]

    margins = []
    for s in EVAL_SEEDS:
        obs, _ = env.reset(seed=s)
        ctrl.reset()
        for _ in range(task.max_steps):
            a, acc, _ = ctrl.forward(obs, return_trace=True)
            o = acc[-2:]
            if o.sum() > 0:
                margins.append((o[0] - o[1]) / o.sum())
            obs, r, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    fluct = float(np.std(margins)) if margins else np.nan

    hh = hh_edge_list(adj, n_in, n_out)
    n_swaps = len(hh) * 10
    edges_c, contrib = edge_contributions(adj, n_in, n_out)
    n_keep = max(1, round(SKEL_FRAC * len(hh)))
    keep_skel = [tuple(edges_c[i]) for i in np.argsort(-contrib)[:n_keep]]

    sm = keep_mask_of(adj.shape, keep_skel)
    J_static = []
    for rep in range(3):
        child, _ = restricted_rewire(adj, n_in, n_out, sm, n_swaps,
                                     np.random.RandomState(seed * 131 + rep * 7))
        miss = [(a, b) for (a, b) in keep_skel if child[b, a] <= 0]
        assert not miss
        c = UnifiedSNNController(scaffold, child, enc, proto, condition=F0)
        J_static.append(c.evaluate(env, EVAL_SEEDS, task.max_steps)["mean"])
    hh_list = [tuple(e) for e in hh]
    J_rand = []
    for rep in range(N_RAND):
        rrng = np.random.RandomState(seed * 733 + rep * 101)
        keep_r = [hh_list[i] for i in rrng.choice(len(hh_list), size=n_keep,
                                                  replace=False)]
        km = keep_mask_of(adj.shape, keep_r)
        child, _ = restricted_rewire(adj, n_in, n_out, km, n_swaps, rrng)
        miss = [(a, b) for (a, b) in keep_r if child[b, a] <= 0]
        assert not miss
        c = UnifiedSNNController(scaffold, child, enc, proto, condition=F0)
        J_rand.append(c.evaluate(env, EVAL_SEEDS, task.max_steps)["mean"])

    calib = collect_calibration(ctrl, env, task, N_STATES, np.random.RandomState(2026))
    episodes, states = calib
    states = [s for s in states if s is not None]
    ge = fixed_groups(adj, n_in, n_out)
    L, base_margins, n_edges = fingerprint(adj, scaffold, enc, proto,
                                           episodes, states, ge)
    D = L.sum(axis=1)
    pairs = []
    for g1, g2 in combinations([g for g in range(len(ge)) if len(ge[g]) > 0], 2):
        eg = [tuple(e) for e in np.vstack([ge[g1], ge[g2]])]
        m = replay_margins(adj, scaffold, enc, proto, episodes, states, eg)
        Dp = float(np.abs(m[:, 1:, :] - base_margins[:, 1:, :]).sum(axis=2).mean(axis=0).sum())
        pairs.append({"g1": g1, "g2": g2, "D_pair": Dp,
                      "I_pair": Dp - float(D[g1]) - float(D[g2])})
    dnorm = float(np.sum(D / np.maximum(n_edges, 1)))

    I = np.zeros((G, G))
    for p in pairs:
        I[p["g1"], p["g2"]] = I[p["g2"], p["g1"]] = p["I_pair"]
    S = D[:, None] + D[None, :]
    C = np.where(S > 1e-9, -I / np.where(S > 1e-9, S, 1.0), 0.0)
    np.fill_diagonal(C, 0.0)
    H = np.maximum(C, 0).sum(axis=1)
    order = list(np.argsort(-H))
    arm_J = {}
    removed = []
    for step, g in enumerate(order, 1):
        removed.append(g)
        eg = [tuple(e) for e in np.vstack([ge[gg] for gg in removed])]
        if step in KEY_STEPS:
            adj_l = adj.copy()
            for (a, b) in eg:
                adj_l[b, a] = 0
            c = UnifiedSNNController(scaffold, adj_l, enc, proto, condition=F0)
            arm_J[str(step)] = c.evaluate(env, EVAL_SEEDS, task.max_steps)["mean"]

    inp = {}
    nrng = np.random.RandomState(777)
    for sig in [0.1, 0.3]:
        tot = []
        for s in EVAL_SEEDS[:10]:
            obs, _ = env.reset(seed=s)
            c2 = UnifiedSNNController(scaffold, adj, enc, proto, condition=F0)
            c2.reset()
            total = 0.0
            for _ in range(task.max_steps):
                noisy = obs + nrng.normal(0, sig, size=obs.shape)
                a = c2.forward(noisy)
                obs, r, term, trunc, _ = env.step(a)
                total += r
                if term or trunc:
                    break
            tot.append(total)
        inp[f"noise_{sig}"] = float(np.mean(tot))
    for scale in [2.0, 4.0]:
        tot = []
        for s in EVAL_SEEDS[:10]:
            env.reset(seed=s)
            env.unwrapped.state = env.unwrapped.state * scale
            c2 = UnifiedSNNController(scaffold, adj, enc, proto, condition=F0)
            c2.reset()
            obs = env.unwrapped.state
            total = 0.0
            for _ in range(task.max_steps):
                a = c2.forward(obs)
                obs, r, term, trunc, _ = env.step(a)
                total += r
                if term or trunc:
                    break
            tot.append(total)
        inp[f"init_x{scale}"] = float(np.mean(tot))

    rec = {"seed": seed, "J_F0": J_F0, "fluct_std": fluct,
           "J_static": [float(x) for x in J_static],
           "J_rand": [float(x) for x in J_rand],
           "arm_J": arm_J, "input": inp, "dnorm_sum": dnorm,
           "minutes": (time.time() - t0) / 60}
    with open(OUT / f"measure_c2_{TASK}_seed{seed}.json", "w") as f:
        json.dump(rec, f)
    return rec


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for s in [int(x) for x in sys.argv[1:]]:
        r = one_seed(s)
        print(f"seed {s}: F0={r['J_F0']:.0f} J24={r['arm_J'].get('24', 'NA')} "
              f"({r['minutes']:.1f}m)", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
