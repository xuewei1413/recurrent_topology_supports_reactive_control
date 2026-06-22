"""
实验 1C: 拓扑-权重共进化对照 (TO / TG / TI)

论文 Section 3.1.5 的核心对照实验：
  TO (Topology-Only)       — 仅进化邻接矩阵，1个标量参数
  TG (Topology+Global)     — 邻接矩阵 + 全局 weight_scale，2个参数
  TI (Topology+Independent)— 邻接矩阵 + 每条连接独立权重，~10^4 参数

每种模式 10 个独立种子，最终 20 轮严格评估。
ANOVA 检验三组最终性能是否有显著差异。

论文预期结果：
  TO:  497.4 ± 3.3   (1 scalar)
  TG:  498.4 ± 1.9   (2 params)
  TI:  497.9 ± 2.6   (~10^4 params)
  ANOVA F(2,27) = 0.42, p = 0.66 → 无统计差异
"""

import os
import sys
import json
import time
import argparse
import itertools
import numpy as np

# ── 路径设置 ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, CODE_DIR)

from src.evolution import run_evolution_coevolution, evaluate_coevolution
from src.snn import SNNController, WeightedSNNController

import gymnasium as gym

# ── 配置 ──────────────────────────────────────────────────────────────────

BASE_CONFIG = {
    'n_total':         128,
    'n_input':         8,       # 4D state × 2 (half-wave)
    'n_output':        2,
    'tau':             5.0,
    'threshold':       0.8,
    'T_steps':         20,
    'weight_scale':    0.6,
    'sparsity':        0.10,
    'pop_size':        80,
    'n_generations':   400,
    'n_elite':         16,
    'mutation_rate':   0.03,
    'n_eval_episodes': 20,
    'n_workers':       1,
    'run_baseline':    False,
    'stagnation_limit': 10,
    'mut_rate_cap':    0.10,
    'tournament_size': 3,
    'crossover_rate':  0.5,
}

MODES = ['TO', 'TG', 'TI']
N_SEEDS = 10
SEEDS = list(range(N_SEEDS))


# ── 严格评估 ─────────────────────────────────────────────────────────────

def strict_evaluate(best_adj, neuron_types, config, mode, extra, n_episodes=20):
    """20轮独立评估，返回 (mean, std, min, max)"""
    if mode == 'TI' and 'weight_matrix' in extra:
        wm = extra['weight_matrix']
        ctrl = WeightedSNNController(config, best_adj, neuron_types, wm)
    elif mode == 'TG' and 'weight_scale' in extra:
        ws = extra['weight_scale']
        W = best_adj * neuron_types[:, np.newaxis] * ws
        ctrl = WeightedSNNController(config, best_adj, neuron_types, W)
    else:
        ctrl = SNNController(config, best_adj, neuron_types)

    env = gym.make('CartPole-v1')
    scores = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=10000 + ep)
        ctrl.reset()
        total = 0.0
        for step in range(500):
            action = ctrl.forward(np.array(obs, dtype=np.float64))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        scores.append(total)
    env.close()

    scores = np.array(scores)
    return {
        'mean': float(scores.mean()),
        'std':  float(scores.std()),
        'min':  float(scores.min()),
        'max':  float(scores.max()),
    }


# ── ANOVA 统计检验 ────────────────────────────────────────────────────────

def anova_test(groups):
    """单因素 ANOVA，groups = {name: [values]}"""
    all_values = list(itertools.chain(*groups.values()))
    grand_mean = np.mean(all_values)
    n_total    = len(all_values)
    k          = len(groups)

    # 组间平方和
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups.values())
    df_between = k - 1

    # 组内平方和
    ss_within = sum(sum((x - np.mean(g))**2 for x in g) for g in groups.values())
    df_within = n_total - k

    ms_between = ss_between / df_between
    ms_within  = ss_within / df_within
    f_stat     = ms_between / ms_within

    # p-value via F distribution
    from scipy import stats
    p_value = float(1.0 - stats.f.cdf(f_stat, df_between, df_within))

    return {
        'f_stat':   float(f_stat),
        'p_value':  p_value,
        'df_between': int(df_between),
        'df_within':  int(df_within),
        'ss_between': float(ss_between),
        'ss_within':  float(ss_within),
    }


# ── 单次运行 ──────────────────────────────────────────────────────────────

def run_single(mode, seed, config):
    """运行单次进化 + 严格评估"""
    print(f"\n{'='*60}")
    print(f"  {mode} | Seed {seed}")
    print(f"{'='*60}")

    t0 = time.time()
    best_adj, neuron_types, best_hist, mean_hist, extra = \
        run_evolution_coevolution(config, mode=mode, seed=seed)
    elapsed = time.time() - t0

    gens = len(best_hist)
    solved = bool(best_hist[-1] >= 490)
    n_conns = int(best_adj.sum())

    print(f"\n[{mode} | Seed {seed}] Done: Gen={gens}, Best={best_hist[-1]:.1f}, "
          f"Solved={solved}, Time={elapsed:.1f}s, Conns={n_conns}")

    # 严格 20 轮评估
    print(f"[{mode} | Seed {seed}] Running strict 20-episode evaluation...")
    strict = strict_evaluate(best_adj, neuron_types, config, mode, extra, n_episodes=20)
    print(f"[{mode} | Seed {seed}] Strict 20-ep: {strict['mean']:.1f} "
          f"(min={strict['min']:.0f}, max={strict['max']:.0f}, std={strict['std']:.1f})")

    return {
        'mode':        mode,
        'seed':        seed,
        'generations': gens,
        'best_fitness': float(best_hist[-1]),
        'solved':      solved,
        'n_connections': n_conns,
        'elapsed_sec': elapsed,
        'strict_20ep': strict,
        'extra':       {k: (float(v) if isinstance(v, (int, float, np.floating)) else v.tolist())
                        for k, v in extra.items()} if extra else {},
    }


# ── Checkpoint 读写 ────────────────────────────────────────────────────────

def save_checkpoint(results, path):
    """保存 checkpoint"""
    data = []
    for r in results:
        entry = dict(r)
        entry['extra'] = {k: (float(v) if isinstance(v, np.floating) else v)
                          for k, v in entry['extra'].items()}
        data.append(entry)
    np.save(path, data, allow_pickle=True)
    print(f"Checkpoint saved ({len(data)}/{len(MODES)*N_SEEDS})")


def load_checkpoint(path):
    """加载 checkpoint"""
    if not os.path.exists(path):
        return []
    data = np.load(path, allow_pickle=True).tolist()
    print(f"Checkpoint loaded: {len(data)} runs done")
    return data


def is_done(done_list, mode, seed):
    return any(r['mode'] == mode and r['seed'] == seed for r in done_list)


# ── 主函数 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='TO/TG/TI Co-evolution Experiment')
    parser.add_argument('--seeds', type=int, default=N_SEEDS,
                        help='Number of seeds per mode')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--modes', nargs='+', default=MODES,
                        choices=MODES, help='Which modes to run')
    args = parser.parse_args()

    # 结果目录 — checkpoint 按模式分开，支持并行
    results_dir = os.path.join(SCRIPT_DIR, '..', '..', 'results', 'coevolution')
    os.makedirs(results_dir, exist_ok=True)
    modes_tag = '_'.join(args.modes)
    ckpt_path = os.path.join(results_dir, f'coevolution_checkpoint_{modes_tag}.npy')

    print("=" * 70)
    print("  实验 1C: TO / TG / TI 拓扑-权重共进化对照")
    print("=" * 70)
    print(f"  模式: {args.modes}")
    print(f"  每模式种子数: {args.seeds}")
    print(f"  总运行数: {len(args.modes) * args.seeds}")
    print("=" * 70)

    # 加载 checkpoint
    done = load_checkpoint(ckpt_path) if args.resume else []
    results = list(done)

    config = dict(BASE_CONFIG)

    # 运行所有 (mode, seed) 组合
    for mode in args.modes:
        for seed in range(args.seeds):
            if is_done(done, mode, seed):
                print(f"\n[{mode} | Seed {seed}] Already done, skipping.")
                continue

            result = run_single(mode, seed, config)
            results.append(result)
            save_checkpoint(results, ckpt_path)

    # ── 汇总分析 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  结果汇总")
    print("=" * 70)

    mode_results = {m: [] for m in args.modes}
    for r in results:
        mode_results[r['mode']].append(r['strict_20ep']['mean'])

    summary = {}
    for mode in args.modes:
        vals = np.array(mode_results[mode])
        print(f"\n  {mode}: mean={vals.mean():.1f} ± {vals.std():.1f}  "
              f"(n={len(vals)}, range [{vals.min():.1f}, {vals.max():.1f}])")
        summary[mode] = {
            'mean': float(vals.mean()),
            'std':  float(vals.std()),
            'n':    len(vals),
            'min':  float(vals.min()),
            'max':  float(vals.max()),
        }

    # ANOVA
    if len(args.modes) == 3 and all(len(v) >= 2 for v in mode_results.values()):
        anova = anova_test(mode_results)
        print(f"\n  ANOVA: F({anova['df_between']},{anova['df_within']}) = "
              f"{anova['f_stat']:.2f}, p = {anova['p_value']:.4f}")
        if anova['p_value'] < 0.05:
            print("  → 三组之间存在显著差异 (p < 0.05)")
        else:
            print("  → 三组之间无显著差异 (p >= 0.05)")

        # 参数效率
        param_counts = {'TO': 1, 'TG': 2, 'TI': 128*128}
        print(f"\n  参数效率 (performance / #params):")
        for mode in args.modes:
            eff = summary[mode]['mean'] / param_counts[mode]
            print(f"    {mode}: {eff:.2f} perf/param  "
                  f"({param_counts[mode]} params → {summary[mode]['mean']:.1f})")
    else:
        anova = None

    # ── 保存最终结果 ──────────────────────────────────────────────────────
    final = {
        'summary': summary,
        'anova': anova,
        'config': {k: (v if not isinstance(v, np.integer) else int(v))
                   for k, v in config.items()},
        'per_seed': results,
    }

    # 清理 per_seed 中 numpy 类型
    for r in final['per_seed']:
        for k, v in r.items():
            if isinstance(v, (np.bool_,)):
                r[k] = bool(v)
            elif isinstance(v, (np.integer, np.floating)):
                r[k] = float(v)
            elif isinstance(v, np.ndarray):
                r[k] = v.tolist()

    out_path = os.path.join(results_dir, f'coevolution_results_{modes_tag}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"\n  结果保存至: {out_path}")

    # ── 每种子详细表格 ────────────────────────────────────────────────────
    print(f"\n{'Mode':<4} {'Seed':>4} {'Gens':>5} {'Best':>7} "
          f"{'Strict':>7} {'Min':>5} {'Max':>5} {'Conns':>6}")
    print("-" * 50)
    for r in sorted(results, key=lambda x: (x['mode'], x['seed'])):
        s = r['strict_20ep']
        print(f"{r['mode']:<4} {r['seed']:4d} {r['generations']:5d} "
              f"{r['best_fitness']:7.1f} {s['mean']:7.1f} "
              f"{s['min']:5.0f} {s['max']:5.0f} {r['n_connections']:6d}")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == '__main__':
    main()
