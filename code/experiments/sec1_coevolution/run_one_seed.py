"""
实验 1C 单 seed 运行脚本（Section 3.1.5 拓扑-权重共进化对照）

用法：python run_one_seed.py --mode TO --seed 0
独立 JSON 输出，无 checkpoint，适合并行批量启动。

三种模式：
  TO (Topology-Only)       — 仅进化邻接矩阵，1 个标量参数
  TG (Topology+Global)     — 邻接矩阵 + 全局 weight_scale，2 个参数
  TI (Topology+Independent)— 邻接矩阵 + 每条连接独立权重，~16384 个参数

进化完成后进行 20 局独立严格评估（seed=10000+ep），结果保存为 JSON。

GA 配置：pop_size=80, n_elite=16, tournament_size=3,
         crossover_rate=0.5, mutation_rate=0.03 (adaptive cap=0.10),
         n_eval_episodes=20, n_generations=400

输出：
  results/coevolution/result_{MODE}_{SEED}.json
"""
import os, sys, json, time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, CODE_DIR)

from src.evolution import run_evolution_coevolution
from src.snn import SNNController, WeightedSNNController
import gymnasium as gym
import argparse

BASE_CONFIG = {
    'n_total': 128, 'n_input': 8, 'n_output': 2,
    'tau': 5.0, 'threshold': 0.8, 'T_steps': 20,
    'weight_scale': 0.6, 'sparsity': 0.10,
    'pop_size': 80, 'n_generations': 400, 'n_elite': 16,
    'mutation_rate': 0.03, 'n_eval_episodes': 20,
    'n_workers': 1, 'run_baseline': False,
    'stagnation_limit': 10, 'mut_rate_cap': 0.10,
    'tournament_size': 3, 'crossover_rate': 0.5,
}


def strict_evaluate(best_adj, neuron_types, config, mode, extra, n_episodes=20):
    if mode == 'TI' and 'weight_matrix' in extra:
        ctrl = WeightedSNNController(config, best_adj, neuron_types, extra['weight_matrix'])
    elif mode == 'TG' and 'weight_scale' in extra:
        W = best_adj * neuron_types[:, np.newaxis] * extra['weight_scale']
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
    s = np.array(scores)
    return {'mean': float(s.mean()), 'std': float(s.std()),
            'min': float(s.min()), 'max': float(s.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['TO', 'TG', 'TI'])
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()

    config = dict(BASE_CONFIG)
    mode, seed = args.mode, args.seed

    print(f"[{mode}|S{seed}] Starting...")
    t0 = time.time()
    best_adj, neuron_types, best_hist, mean_hist, extra = \
        run_evolution_coevolution(config, mode=mode, seed=seed)
    elapsed = time.time() - t0

    gens = len(best_hist)
    solved = bool(best_hist[-1] >= 490)
    n_conns = int(best_adj.sum())

    print(f"[{mode}|S{seed}] Done: Gen={gens}, Best={best_hist[-1]:.1f}, "
          f"Solved={solved}, Time={elapsed:.1f}s, Conns={n_conns}")

    print(f"[{mode}|S{seed}] Strict 20-ep eval...")
    strict = strict_evaluate(best_adj, neuron_types, config, mode, extra, n_episodes=20)
    print(f"[{mode}|S{seed}] Strict={strict['mean']:.1f} "
          f"(min={strict['min']:.0f}, max={strict['max']:.0f}, std={strict['std']:.1f})")

    result = {
        'mode': mode, 'seed': seed,
        'generations': gens,
        'best_fitness': float(best_hist[-1]),
        'solved': solved,
        'n_connections': n_conns,
        'elapsed_sec': float(elapsed),
        'strict_20ep': strict,
        'extra': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v.tolist())
                  for k, v in extra.items()} if extra else {},
    }

    # Save individual result
    results_dir = os.path.join(SCRIPT_DIR, '..', '..', 'results', 'coevolution')
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, f'result_{mode}_{seed}.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[{mode}|S{seed}] Saved to {out}")


if __name__ == '__main__':
    main()
