"""
MountainCar 5-seed rerun — 使用与论文一致的 mc_special fitness 函数
复用 exp_multitask.py 中完全相同的配置和评估逻辑。

输出: data/rerun_results/mountaincar_5seed.json
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import gymnasium as gym
from src.snn import SNNController

INPUT_SCALE = 4.0

# ── 与 exp_multitask.py 完全一致的 MountainCar 配置 ────────────────────────
MC_CONFIG = {
    'env_id': 'MountainCar-v0',
    'n_total': 64,
    'n_input': 4,
    'n_output': 3,
    'weight_scale': 0.6,
    'T_steps': 20,
    'sparsity': 0.12,
    'pop_size': 40,
    'n_generations': 300,
    'n_elite': 12,
    'mutation_rate': 0.025,
    'n_eval_episodes': 5,
    'target_sparsity': 0.15,
    'density_penalty': 200,
    'solve_threshold': 650.0,
    'fitness_fn': 'mc_special',
}

SHARED = {'tau': 5.0, 'threshold': 0.8, 'dt': 1.0}

N_SEEDS = 5
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'data', 'rerun_results', 'mountaincar_5seed.json'
)


# ── 拓扑操作（与 exp_multitask.py 一致）──────────────────────────────────

def random_topology(n, sparsity):
    adj = (np.random.rand(n, n) < sparsity).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    return adj


def mutate_density_aware(adj, rate, target):
    child = adj.copy()
    n = child.shape[0]
    max_conn = n * (n - 1)
    density = child.sum() / max_conn
    ratio = density / target if target > 0 else 1.0
    if ratio > 1.3:
        p_remove = min(rate * (1 + (ratio - 1.3) * 3), 0.3)
        p_add = rate * 0.2
    elif ratio < 0.7:
        p_remove = rate * 0.2
        p_add = min(rate * (1 + (0.7 - ratio) * 3), 0.3)
    else:
        p_remove = rate * 0.5
        p_add = rate * 0.5
    remove_mask = (child == 1.0) & (np.random.rand(n, n) < p_remove)
    add_mask = (child == 0.0) & (np.random.rand(n, n) < p_add)
    child[remove_mask] = 0.0
    child[add_mask] = 1.0
    np.fill_diagonal(child, 0.0)
    return child


# ── 评估函数（与 exp_multitask.py evaluate_mc 完全一致）──────────────────

def evaluate_mc(adj, neuron_types, config):
    cfg = {**config, **SHARED}
    ctrl = SNNController(cfg, adj, neuron_types)
    env = gym.make('MountainCar-v0')
    max_positions = []
    for _ in range(config['n_eval_episodes']):
        obs, _ = env.reset()
        ctrl.reset()
        max_pos = obs[0]
        for _ in range(config.get('max_steps', 200)):
            action = ctrl.forward(np.array(obs, dtype=np.float64))
            obs, _, terminated, truncated, _ = env.step(action)
            if obs[0] > max_pos:
                max_pos = obs[0]
            if terminated or truncated:
                break
        max_positions.append(max_pos)
    env.close()
    avg_max = float(np.mean(max_positions))
    fitness = (avg_max + 1.2) / 1.8 * 1000
    if avg_max >= 0.3:  fitness += 300
    if avg_max >= 0.45: fitness += 700
    if avg_max >= 0.5:  fitness += 1000
    n = adj.shape[0]
    density = adj.sum() / (n * (n - 1))
    over = max(0.0, density - config.get('target_sparsity', 0.15))
    fitness -= over * config.get('density_penalty', 200)
    return fitness, avg_max


# ── 进化主循环 ────────────────────────────────────────────────────────────

def run_evolution(seed):
    np.random.seed(seed)
    cfg = MC_CONFIG
    n = cfg['n_total']

    neuron_types = np.ones(n, dtype=np.float64)
    hidden_range = np.arange(cfg['n_input'], n - cfg['n_output'])
    n_inh = int(len(hidden_range) * 0.3)
    inh_idx = np.random.choice(hidden_range, n_inh, replace=False)
    neuron_types[inh_idx] = -1.0

    print(f"\n{'='*60}")
    print(f"MountainCar  Seed={seed}")
    print(f"Neurons: E={len(hidden_range)-n_inh}  I={n_inh}")
    print(f"{'='*60}")

    pop = [random_topology(n, cfg['sparsity']) for _ in range(cfg['pop_size'])]
    best_history = []
    mean_history = []
    best_adj = None
    best_fitness = -np.inf

    print(f"{'Gen':>4}  {'Best':>8}  {'Mean':>8}  {'Conns':>6}  Status")
    print("-" * 55)

    t0 = time.time()
    for gen in range(cfg['n_generations']):
        fitnesses = np.array([evaluate_mc(adj, neuron_types, cfg)[0] for adj in pop])
        order = np.argsort(fitnesses)[::-1]
        gen_best = fitnesses[order[0]]
        gen_mean = fitnesses.mean()
        best_conn = int(pop[order[0]].sum())

        best_history.append(float(gen_best))
        mean_history.append(float(gen_mean))

        if gen_best > best_fitness:
            best_fitness = gen_best
            best_adj = pop[order[0]].copy()

        solved = gen_best >= cfg['solve_threshold']
        status = "*** SOLVED ***" if solved else ""
        if (gen + 1) % 10 == 0 or solved:
            print(f"{gen+1:4d}  {gen_best:8.2f}  {gen_mean:8.2f}  {best_conn:6d}  {status}")

        if solved:
            break

        elite = [pop[i] for i in order[:cfg['n_elite']]]
        new_pop = list(elite)
        while len(new_pop) < cfg['pop_size']:
            parent = elite[np.random.randint(len(elite))]
            new_pop.append(mutate_density_aware(parent, cfg['mutation_rate'], cfg.get('target_sparsity', 0.15)))
        pop = new_pop

    elapsed = time.time() - t0
    print("-" * 55)
    print(f"Done. Best={best_fitness:.2f}  Gens={len(best_history)}  Time={elapsed:.1f}s")

    return {
        'seed': seed,
        'best_fitness': best_fitness,
        'solved_gen': len(best_history) if best_fitness >= cfg['solve_threshold'] else None,
        'history': best_history,
        'mean_history': mean_history,
        'elapsed_sec': elapsed,
    }


# ── 评估最优拓扑 ──────────────────────────────────────────────────────────

def eval_best_topology(adj, neuron_types, config, n_episodes=20):
    cfg = {**config, **SHARED}
    ctrl = SNNController(cfg, adj, neuron_types)
    env = gym.make('MountainCar-v0')
    max_positions = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        ctrl.reset()
        max_pos = obs[0]
        for _ in range(200):
            action = ctrl.forward(np.array(obs, dtype=np.float64))
            obs, _, terminated, truncated, _ = env.step(action)
            if obs[0] > max_pos:
                max_pos = obs[0]
            if terminated or truncated:
                break
        max_positions.append(max_pos)
    env.close()
    return max_positions


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    all_results = {'seeds': {}, 'config': MC_CONFIG}

    for seed in range(N_SEEDS):
        result = run_evolution(seed)
        result.pop('mean_history', None)  # 不保存 mean history 减小体积
        all_results['seeds'][str(seed)] = result

    print(f"\n{'='*60}")
    print("MountainCar 5-seed 汇总")
    print(f"{'='*60}")
    seeds_data = all_results['seeds']
    for k in sorted(seeds_data.keys(), key=lambda x: int(x)):
        s = seeds_data[k]
        solved = 'SOLVED' if s['solved_gen'] is not None else 'not solved'
        print(f"  Seed {k}: Best={s['best_fitness']:.2f}  Gens={len(s['history'])}  {solved}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
