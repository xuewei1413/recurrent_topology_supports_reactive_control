"""
实验 1A：跨任务存在性证明（5 任务 × 10 seed）

论文 Section 3.1 核心：验证纯拓扑进化在多种任务上均能收敛到高性能。
5 个任务：CartPole-v1, MountainCar-v0, Acrobot-v1, FlappyBird, Snake
每个任务 10 个独立 seed，进化最优二值邻接矩阵。

使用方式：
  批量运行：  python exp_multitask.py
  单次运行：  python exp_multitask.py --task CartPole --seed 3
  指定任务：  python exp_multitask.py --tasks CartPole FlappyBird
  并发启动（bash）：
    for task in CartPole MountainCar Acrobot FlappyBird Snake; do
      for seed in $(seq 0 9); do
        python exp_multitask.py --task $task --seed $seed &
      done
    done

输出：
  results/cross_task/{Task}_seed{N}.npz  — 拓扑 + 进化曲线
  results/cross_task/result_{Task}_{N}.json — 指标（含 20 局严格评估）
  results/cross_task/summary.csv — 汇总表
"""

import os, sys, json, time, csv, argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'snake'))

from src.snn import SNNController

INPUT_SCALE = 4.0

# ── 任务配置（论文 Table 参数） ──────────────────────────────────────

TASK_CONFIGS = {
    'CartPole': {
        'env_id': 'CartPole-v1',
        'n_total': 128, 'n_input': 8, 'n_output': 2,
        'weight_scale': 0.6, 'tau': 5.0, 'threshold': 0.8,
        'T_steps': 20, 'sparsity': 0.10,
        'pop_size': 80, 'n_generations': 400, 'n_elite': 16,
        'mutation_rate': 0.03, 'n_eval_episodes': 20,
        'stagnation_limit': 10, 'mut_rate_cap': 0.10,
        'tournament_size': 3, 'crossover_rate': 0.5,
        'solve_threshold': 490.0, 'max_steps': 500,
        'fitness_fn': 'steps',
    },
    'MountainCar': {
        'env_id': 'MountainCar-v0',
        'n_total': 64, 'n_input': 4, 'n_output': 3,
        'weight_scale': 0.6, 'tau': 5.0, 'threshold': 0.8,
        'T_steps': 20, 'sparsity': 0.12,
        'pop_size': 80, 'n_generations': 400, 'n_elite': 16,
        'mutation_rate': 0.03, 'n_eval_episodes': 10,
        'stagnation_limit': 10, 'mut_rate_cap': 0.10,
        'tournament_size': 3, 'crossover_rate': 0.5,
        'solve_threshold': 650.0, 'max_steps': 200,
        'fitness_fn': 'mc_special',
        'target_sparsity': 0.15, 'density_penalty': 200,
    },
    'Acrobot': {
        'env_id': 'Acrobot-v1',
        'n_total': 128, 'n_input': 12, 'n_output': 3,
        'weight_scale': 0.6, 'tau': 5.0, 'threshold': 0.8,
        'T_steps': 20, 'sparsity': 0.10,
        'pop_size': 80, 'n_generations': 400, 'n_elite': 16,
        'mutation_rate': 0.03, 'n_eval_episodes': 20,
        'stagnation_limit': 10, 'mut_rate_cap': 0.10,
        'tournament_size': 3, 'crossover_rate': 0.5,
        'solve_threshold': 400.0, 'max_steps': 500,
        'fitness_fn': 'acrobot_reward',
    },
    'FlappyBird': {
        'env_id': 'FlappyBird',
        'n_total': 64, 'n_input': 8, 'n_output': 2,
        'weight_scale': 0.6, 'tau': 5.0, 'threshold': 0.8,
        'T_steps': 20, 'sparsity': 0.10,
        'pop_size': 80, 'n_generations': 400, 'n_elite': 16,
        'mutation_rate': 0.03, 'n_eval_episodes': 10,
        'stagnation_limit': 10, 'mut_rate_cap': 0.10,
        'tournament_size': 3, 'crossover_rate': 0.5,
        'solve_threshold': 500.0, 'max_steps': 2000,
        'fitness_fn': 'flappy_steps',
    },
    'Snake': {
        'env_id': 'Snake',
        'n_total': 80, 'n_input': 16, 'n_output': 4,
        'weight_scale': 0.6, 'tau': 5.0, 'threshold': 0.8,
        'T_steps': 20, 'sparsity': 0.10,
        'pop_size': 80, 'n_generations': 400, 'n_elite': 16,
        'mutation_rate': 0.03, 'n_eval_episodes': 10,
        'stagnation_limit': 10, 'mut_rate_cap': 0.10,
        'tournament_size': 3, 'crossover_rate': 0.5,
        'solve_threshold': 15.0, 'max_steps': 1000,
        'fitness_fn': 'snake_score',
    },
}

N_SEEDS = 10

RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results', 'cross_task')


# ── 环境创建 ─────────────────────────────────────────────────────────

def make_env(task_name, task_cfg):
    if task_name == 'FlappyBird':
        from src.flappy_env_parametric import FlappyEnvParametric
        return FlappyEnvParametric()
    elif task_name == 'Snake':
        from snake_env import SnakeEnv
        return SnakeEnv()
    else:
        import gymnasium as gym
        return gym.make(task_cfg['env_id'])


# ── 拓扑操作 ─────────────────────────────────────────────────────────

def random_topology(n, sparsity):
    adj = (np.random.rand(n, n) < sparsity).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    return adj


def mutate(adj, rate):
    child = adj.copy()
    mask = np.random.rand(*child.shape) < rate
    child[mask] = 1.0 - child[mask]
    np.fill_diagonal(child, 0.0)
    return child


def crossover(p1, p2):
    mask = np.random.rand(*p1.shape) < 0.5
    child = np.where(mask, p1, p2).astype(np.float64)
    np.fill_diagonal(child, 0.0)
    return child


def tournament_select(pop, fitnesses, size=3):
    cands = np.random.choice(len(pop), size=min(size, len(pop)), replace=False)
    return pop[cands[np.argmax(fitnesses[cands])]]


def mutate_density_aware(adj, rate, target):
    child = adj.copy()
    n = child.shape[0]
    density = child.sum() / (n * (n - 1))
    ratio = density / target if target > 0 else 1.0
    if ratio > 1.3:
        p_rm, p_add = min(rate * (1 + (ratio - 1.3) * 3), 0.3), rate * 0.2
    elif ratio < 0.7:
        p_rm, p_add = rate * 0.2, min(rate * (1 + (0.7 - ratio) * 3), 0.3)
    else:
        p_rm, p_add = rate * 0.5, rate * 0.5
    child[(child == 1.0) & (np.random.rand(n, n) < p_rm)] = 0.0
    child[(child == 0.0) & (np.random.rand(n, n) < p_add)] = 1.0
    np.fill_diagonal(child, 0.0)
    return child


# ── 评估函数 ─────────────────────────────────────────────────────────

def evaluate_topology(adj, neuron_types, task_name, task_cfg, eval_seeds=None):
    config = {k: task_cfg[k] for k in ('n_total', 'n_input', 'n_output',
              'tau', 'threshold', 'T_steps', 'weight_scale')}
    env = make_env(task_name, task_cfg)
    scores = []

    for ep in range(task_cfg['n_eval_episodes']):
        if task_name in ('FlappyBird', 'Snake'):
            obs = env.reset()
        else:
            seed = eval_seeds[ep] if eval_seeds else None
            obs, _ = env.reset(seed=seed)

        ctrl = SNNController(config, adj, neuron_types)
        total = 0.0
        max_pos = obs[0] if task_name == 'MountainCar' else None

        for step in range(task_cfg['max_steps']):
            action = ctrl.forward(np.array(obs, dtype=np.float64) * INPUT_SCALE)

            if task_name in ('FlappyBird', 'Snake'):
                obs, reward, done, info = env.step(int(action))
                if task_name == 'FlappyBird':
                    total += 1.0
                else:
                    total += reward
                if done:
                    break
            else:
                obs, reward, terminated, truncated, _ = env.step(int(action))
                if task_cfg['fitness_fn'] == 'steps':
                    total += 1.0
                elif task_cfg['fitness_fn'] == 'acrobot_reward':
                    total += reward
                elif task_cfg['fitness_fn'] == 'mc_special':
                    if obs[0] > max_pos:
                        max_pos = obs[0]
                if terminated or truncated:
                    break

        if task_cfg['fitness_fn'] == 'mc_special':
            avg_max = max_pos if max_pos is not None else obs[0]
            fitness = (avg_max + 1.2) / 1.8 * 1000
            if avg_max >= 0.3:  fitness += 300
            if avg_max >= 0.45: fitness += 700
            if avg_max >= 0.5:  fitness += 1000
            n = adj.shape[0]
            density = adj.sum() / (n * (n - 1))
            over = max(0.0, density - task_cfg.get('target_sparsity', 0.15))
            fitness -= over * task_cfg.get('density_penalty', 200)
            scores.append(fitness)
        elif task_name == 'Snake':
            scores.append(info.get('score', 0) * 100 + total)
        elif task_name == 'FlappyBird':
            scores.append(total)
        elif task_cfg['fitness_fn'] == 'acrobot_reward':
            scores.append(500.0 + total)
        else:
            scores.append(total)

    if task_name not in ('FlappyBird', 'Snake'):
        env.close()
    return float(np.mean(scores))


# ── 进化主循环 ───────────────────────────────────────────────────────

def run_evolution(task_name, task_cfg, seed):
    np.random.seed(seed)
    n = task_cfg['n_total']
    eval_seeds = [seed * 1000 + ep for ep in range(task_cfg['n_eval_episodes'])]

    neuron_types = np.ones(n, dtype=np.float64)
    hidden_range = np.arange(task_cfg['n_input'], n - task_cfg['n_output'])
    n_inh = int(len(hidden_range) * 0.3)
    neuron_types[np.random.choice(hidden_range, n_inh, replace=False)] = -1.0

    pop = [random_topology(n, task_cfg['sparsity']) for _ in range(task_cfg['pop_size'])]
    best_history, mean_history = [], []
    best_adj, best_fitness = None, -np.inf

    base_mut = task_cfg['mutation_rate']
    cur_mut, stag = base_mut, 0
    stag_limit = task_cfg.get('stagnation_limit', 10)
    mut_cap = task_cfg.get('mut_rate_cap', 0.10)
    tourn_size = task_cfg.get('tournament_size', 3)
    cross_rate = task_cfg.get('crossover_rate', 0.5)

    tag = f"[{task_name}|S{seed}]"
    print(f"{tag} n={n}, E={len(hidden_range)-n_inh}, I={n_inh}")

    t0 = time.time()
    for gen in range(task_cfg['n_generations']):
        fitnesses = np.array([
            evaluate_topology(adj, neuron_types, task_name, task_cfg, eval_seeds)
            for adj in pop
        ])
        order = np.argsort(fitnesses)[::-1]
        gen_best = fitnesses[order[0]]

        best_history.append(float(gen_best))
        mean_history.append(float(fitnesses.mean()))

        if gen_best > best_fitness:
            best_fitness = gen_best
            best_adj = pop[order[0]].copy()
            stag = 0
            cur_mut = base_mut
        else:
            stag += 1
            if stag >= stag_limit:
                cur_mut = min(cur_mut * 1.5, mut_cap)

        solved = gen_best >= task_cfg['solve_threshold']
        if gen % 20 == 0 or solved:
            print(f"{tag} Gen {gen+1}: best={gen_best:.1f} mean={fitnesses.mean():.1f} "
                  f"mut={cur_mut:.3f}" + (" *** SOLVED ***" if solved else ""))
        if solved:
            break

        elite = [pop[i] for i in order[:task_cfg['n_elite']]]
        new_pop = list(elite)
        while len(new_pop) < task_cfg['pop_size']:
            parent = tournament_select(pop, fitnesses, tourn_size)
            if np.random.rand() < cross_rate:
                parent2 = tournament_select(pop, fitnesses, tourn_size)
                child = crossover(parent, parent2)
            else:
                child = parent.copy()
            if task_name == 'MountainCar':
                child = mutate_density_aware(child, cur_mut, task_cfg.get('target_sparsity', 0.15))
            else:
                child = mutate(child, cur_mut)
            new_pop.append(child)
        pop = new_pop

    elapsed = time.time() - t0
    gens = len(best_history)
    print(f"{tag} Done: Gen={gens}, Best={best_fitness:.1f}, Time={elapsed:.1f}s")

    return best_adj, neuron_types, best_history, mean_history, best_fitness, elapsed


# ── 严格评估 ─────────────────────────────────────────────────────────

def strict_evaluate(best_adj, neuron_types, task_name, task_cfg, n_episodes=20):
    config = {k: task_cfg[k] for k in ('n_total', 'n_input', 'n_output',
              'tau', 'threshold', 'T_steps', 'weight_scale')}
    env = make_env(task_name, task_cfg)
    scores = []

    for ep in range(n_episodes):
        if task_name in ('FlappyBird', 'Snake'):
            obs = env.reset()
        else:
            obs, _ = env.reset(seed=20000 + ep)

        ctrl = SNNController(config, best_adj, neuron_types)
        total = 0.0
        for step in range(task_cfg['max_steps']):
            action = ctrl.forward(np.array(obs, dtype=np.float64) * INPUT_SCALE)
            if task_name in ('FlappyBird', 'Snake'):
                obs, reward, done, info = env.step(int(action))
                if task_name == 'FlappyBird':
                    total += 1.0
                else:
                    total = info.get('score', 0)
                if done:
                    break
            else:
                obs, reward, terminated, truncated, _ = env.step(int(action))
                if task_cfg['fitness_fn'] == 'steps':
                    total += 1.0
                elif task_cfg['fitness_fn'] == 'acrobot_reward':
                    total += reward
                if terminated or truncated:
                    break
        scores.append(total)

    if task_name not in ('FlappyBird', 'Snake'):
        env.close()
    s = np.array(scores)
    return {'mean': float(s.mean()), 'std': float(s.std()),
            'min': float(s.min()), 'max': float(s.max())}


# ── 保存结果 ─────────────────────────────────────────────────────────

def save_result(task_name, seed, best_adj, neuron_types, best_hist, mean_hist,
                best_fitness, elapsed, strict):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    np.savez(os.path.join(RESULTS_DIR, f'{task_name}_seed{seed}.npz'),
             adj=best_adj, types=neuron_types,
             best_hist=np.array(best_hist), mean_hist=np.array(mean_hist),
             best_fitness=float(best_fitness), task=task_name, seed=seed)

    result = {
        'task': task_name, 'seed': seed,
        'generations': len(best_hist),
        'best_fitness': float(best_fitness),
        'solved': bool(best_hist[-1] >= TASK_CONFIGS[task_name]['solve_threshold']),
        'n_connections': int(best_adj.sum()),
        'elapsed_sec': float(elapsed),
        'strict_20ep': strict,
    }
    with open(os.path.join(RESULTS_DIR, f'result_{task_name}_{seed}.json'), 'w') as f:
        json.dump(result, f, indent=2)

    return result


# ── 运行单个 ─────────────────────────────────────────────────────────

def run_single(task_name, seed):
    task_cfg = TASK_CONFIGS[task_name]
    tag = f"[{task_name}|S{seed}]"

    best_adj, neuron_types, best_hist, mean_hist, best_fitness, elapsed = \
        run_evolution(task_name, task_cfg, seed)

    print(f"{tag} Strict 20-ep eval...")
    strict = strict_evaluate(best_adj, neuron_types, task_name, task_cfg, n_episodes=20)
    print(f"{tag} Strict={strict['mean']:.1f} (min={strict['min']:.0f}, max={strict['max']:.0f})")

    result = save_result(task_name, seed, best_adj, neuron_types,
                         best_hist, mean_hist, best_fitness, elapsed, strict)
    print(f"{tag} Saved.")
    return result


# ── 汇总 ─────────────────────────────────────────────────────────────

def print_summary(results):
    print('\n' + '=' * 70)
    print('  跨任务存在性实验汇总')
    print('=' * 70)
    for task_name in TASK_CONFIGS:
        rows = [r for r in results if r['task'] == task_name and r.get('best') is not None]
        if not rows:
            continue
        scores = [r['best'] for r in rows]
        solved = sum(1 for s in scores if s >= TASK_CONFIGS[task_name]['solve_threshold'])
        print(f"  {task_name:12s}: n={len(scores)}  solved={solved}/{len(scores)}  "
              f"mean={np.mean(scores):.2f}±{np.std(scores):.2f}")


def save_csv(results, path):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'task', 'seed', 'best', 'generations', 'strict_mean',
            'strict_min', 'strict_max', 'solved', 'elapsed_sec', 'error'])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'task': r.get('task', ''),
                'seed': r.get('seed', ''),
                'best': r.get('best', ''),
                'generations': r.get('generations', ''),
                'strict_mean': r.get('strict', {}).get('mean', ''),
                'strict_min': r.get('strict', {}).get('min', ''),
                'strict_max': r.get('strict', {}).get('max', ''),
                'solved': r.get('solved', ''),
                'elapsed_sec': r.get('elapsed_sec', ''),
                'error': r.get('error', ''),
            })


# ── 主函数 ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='跨任务存在性实验 (5 tasks x 10 seeds)')
    parser.add_argument('--task', type=str, help='单次运行：指定任务名')
    parser.add_argument('--seed', type=int, help='单次运行：指定 seed')
    parser.add_argument('--tasks', nargs='+', default=list(TASK_CONFIGS.keys()),
                        choices=list(TASK_CONFIGS.keys()),
                        help='批量运行：指定任务列表（默认全部）')
    parser.add_argument('--seeds', type=int, default=N_SEEDS,
                        help=f'每任务 seed 数（默认 {N_SEEDS}）')
    args = parser.parse_args()

    # 单次模式
    if args.task is not None:
        if args.seed is None:
            parser.error('--seed required when --task is specified')
        r = run_single(args.task, args.seed)
        return

    # 批量模式
    results = []
    for task_name in args.tasks:
        for seed in range(args.seeds):
            npz = os.path.join(RESULTS_DIR, f'{task_name}_seed{seed}.npz')
            if os.path.exists(npz):
                data = np.load(npz)
                r = {
                    'task': task_name, 'seed': seed,
                    'best': float(data['best_fitness']),
                    'generations': int(len(data['best_hist'])),
                    'solved': float(data['best_fitness']) >= TASK_CONFIGS[task_name]['solve_threshold'],
                    'skipped': True, 'error': '',
                }
                print(f"Skip {task_name} seed={seed} (already done, best={r['best']:.1f})")
                results.append(r)
                continue

            try:
                r = run_single(task_name, seed)
                r['best'] = r['best_fitness']
                r['skipped'] = False
                results.append(r)
            except Exception as e:
                print(f"ERROR {task_name} seed={seed}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'task': task_name, 'seed': seed, 'best': None,
                    'generations': None, 'solved': False,
                    'skipped': False, 'error': str(e),
                })

    print_summary(results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_csv(results, os.path.join(RESULTS_DIR, 'summary.csv'))
    print(f"\nSummary saved to {os.path.join(RESULTS_DIR, 'summary.csv')}")


if __name__ == '__main__':
    main()
