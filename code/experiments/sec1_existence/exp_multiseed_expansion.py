"""
1B 种子扩充：为5个任务各增加5个种子（5-9），使每任务达到10种子
支持断点续跑 — 已存在的种子自动跳过
输出格式与 cross_task/*.npz 兼容
"""
import numpy as np
import os, sys, time, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.snn import SNNController

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'cross_task')
os.makedirs(RESULTS_DIR, exist_ok=True)

INPUT_SCALE = 4.0

TASK_CONFIGS = {
    'CartPole': {
        'env_id': 'CartPole-v1', 'n_total': 128, 'n_input': 8, 'n_output': 2,
        'weight_scale': 0.6, 'tau': 5.0, 'threshold': 0.8, 'T_steps': 20,
        'sparsity': 0.10, 'pop_size': 40, 'n_generations': 400, 'n_elite': 8,
        'mutation_rate': 0.03, 'n_eval_episodes': 8, 'solve_threshold': 490.0,
        'fitness_fn': 'steps',
    },
    'MountainCar': {
        'env_id': 'MountainCar-v0', 'n_total': 64, 'n_input': 4, 'n_output': 3,
        'weight_scale': 0.6, 'tau': 5.0, 'threshold': 0.8, 'T_steps': 20,
        'sparsity': 0.12, 'pop_size': 40, 'n_generations': 300, 'n_elite': 12,
        'mutation_rate': 0.025, 'n_eval_episodes': 5,
        'target_sparsity': 0.15, 'density_penalty': 200, 'solve_threshold': 650.0,
        'fitness_fn': 'mc_special',
    },
    'Acrobot': {
        'env_id': 'Acrobot-v1', 'n_total': 128, 'n_input': 12, 'n_output': 3,
        'weight_scale': 0.6, 'tau': 5.0, 'threshold': 0.8, 'T_steps': 20,
        'sparsity': 0.10, 'pop_size': 40, 'n_generations': 250, 'n_elite': 8,
        'mutation_rate': 0.03, 'n_eval_episodes': 8, 'solve_threshold': 350.0,
        'fitness_fn': 'acrobot',
    },
    'FlappyBird': {
        'env_id': 'FlappyBird', 'n_total': 64, 'n_input': 8, 'n_output': 2,
        'weight_scale': 0.6, 'tau': 5.0, 'threshold': 0.8, 'T_steps': 20,
        'sparsity': 0.12, 'pop_size': 25, 'n_generations': 120, 'n_elite': 8,
        'mutation_rate': 0.03, 'n_eval_episodes': 1,
        'target_sparsity': 0.15, 'density_penalty': 300,
        'max_frames': 1500, 'gap_reward_w': 0.8, 'pipe_bonus': 500,
        'fitness_fn': 'flappy',
    },
    'Snake': {
        'env_id': 'Snake', 'n_total': 80, 'n_input': 16, 'n_output': 4,
        'weight_scale': 0.6, 'tau': 5.0, 'threshold': 0.8, 'T_steps': 20,
        'sparsity': 0.12, 'pop_size': 25, 'n_generations': 150, 'n_elite': 8,
        'mutation_rate': 0.03, 'n_eval_episodes': 1,
        'target_sparsity': 0.15, 'density_penalty': 300,
        'max_steps': 1000, 'food_reward': 200.0, 'step_reward': 0.1,
        'fitness_fn': 'snake',
    },
}


def random_topology(n, sparsity):
    adj = (np.random.rand(n, n) < sparsity).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    return adj


def make_neuron_types(n_total, n_input, n_output):
    types = np.ones(n_total, dtype=np.float64)
    hidden = np.arange(n_input, n_total - n_output)
    n_inh = max(int(len(hidden) * 0.3), 1)
    inh_idx = np.random.choice(hidden, n_inh, replace=False)
    types[inh_idx] = -1.0
    return types


def mutate_density_aware(adj, rate, target):
    child = adj.copy()
    n = child.shape[0]
    density = child.sum() / (n * (n - 1))
    ratio = density / target if target > 0 else 1.0
    if ratio > 1.3:
        p_remove = min(rate * (1 + (ratio - 1.3) * 3), 0.3)
        p_add = rate * 0.2
    elif ratio < 0.7:
        p_add = min(rate * (1 + (0.7 - ratio) * 3), 0.3)
        p_remove = rate * 0.2
    else:
        p_add = p_remove = rate
    rnd = np.random.rand(*child.shape)
    child[(child == 1) & (rnd < p_remove)] = 0.0
    child[(child == 0) & (rnd < p_add)] = 1.0
    np.fill_diagonal(child, 0.0)
    return child


# ── 任务特定适应度 ──

def evaluate_cartpole(adj, types, config):
    import gymnasium as gym
    env = gym.make('CartPole-v1')
    controller = SNNController(config, adj, types)
    total = 0.0
    for _ in range(config['n_eval_episodes']):
        obs, _ = env.reset()
        controller.reset()
        ep_steps = 0
        for _ in range(500):
            action = controller.forward(np.array(obs, dtype=np.float64) * INPUT_SCALE)
            obs, _, term, trunc, _ = env.step(action)
            ep_steps += 1
            if term or trunc: break
        total += ep_steps
    env.close()
    return total / config['n_eval_episodes']


def evaluate_mountaincar(adj, types, config):
    import gymnasium as gym
    env = gym.make('MountainCar-v0')
    controller = SNNController(config, adj, types)
    total_fitness = 0.0
    for _ in range(config['n_eval_episodes']):
        obs, _ = env.reset()
        controller.reset()
        max_pos = obs[0]
        for _ in range(200):
            action = controller.forward(np.array(obs, dtype=np.float64) * INPUT_SCALE)
            obs, _, term, trunc, _ = env.step(action)
            max_pos = max(max_pos, obs[0])
            if term or trunc: break
        norm_pos = (max_pos + 1.2) / 1.8
        stage_bonus = 0.0
        if max_pos >= 0.5: stage_bonus += 100.0
        if max_pos >= 0.3: stage_bonus += 50.0
        n = adj.shape[0]
        density = adj.sum() / (n * (n - 1))
        density_pen = config.get('density_penalty', 0) * abs(density - config.get('target_sparsity', 0.15))
        total_fitness += norm_pos * 500 + stage_bonus - density_pen
    env.close()
    return total_fitness / config['n_eval_episodes']


def evaluate_acrobot(adj, types, config):
    import gymnasium as gym
    env = gym.make('Acrobot-v1')
    controller = SNNController(config, adj, types)
    total = 0.0
    for _ in range(config['n_eval_episodes']):
        obs, _ = env.reset()
        controller.reset()
        ep_r = 0.0
        for _ in range(500):
            action = controller.forward(np.array(obs, dtype=np.float64) * INPUT_SCALE)
            obs, r, term, trunc, _ = env.step(action)
            ep_r += r
            if term or trunc: break
        total += (500 + ep_r)
    env.close()
    return total / config['n_eval_episodes']


def evaluate_flappy(adj, types, config):
    from src.flappy_env import FlappyBirdEnv
    controller = SNNController(config, adj, types)
    gap_w = config.get('gap_reward_w', 0.8)
    pipe_bonus = config.get('pipe_bonus', 500)
    total = 0.0
    for _ in range(config['n_eval_episodes']):
        env = FlappyBirdEnv()
        state = env.reset()
        controller.reset()
        info_final = {'score': 0, 'gap_align': 0}
        for _ in range(config['max_frames']):
            action = controller.forward(state)
            state, reward, done, info = env.step(action)
            total += 1.0
            total += info['gap_align'] * gap_w
            info_final = info
            if done:
                break
        total += info_final['score'] * pipe_bonus
    raw = total / config['n_eval_episodes']
    n = adj.shape[0]
    density = adj.sum() / (n * (n - 1))
    target = config.get('target_sparsity', 0.15)
    penalty = config.get('density_penalty', 300)
    over = max(0.0, density - target)
    raw -= over * penalty
    return raw


def evaluate_snake(adj, types, config):
    from snake.snake_env import SnakeEnv
    controller = SNNController(config, adj, types)
    total_score = 0.0
    total_steps = 0.0
    for _ in range(config['n_eval_episodes']):
        env = SnakeEnv()
        state = env.reset()
        controller.reset()
        info_final = {'score': 0, 'steps': 0}
        for _ in range(config['max_steps']):
            action = controller.forward(state)
            state, reward, done, info = env.step(action)
            info_final = info
            if done:
                break
        total_score += info_final['score']
        total_steps += info_final['steps']
    avg_score = total_score / config['n_eval_episodes']
    avg_steps = total_steps / config['n_eval_episodes']
    raw = avg_score * config['food_reward'] + avg_steps * config['step_reward']
    n = adj.shape[0]
    density = adj.sum() / (n * (n - 1))
    target = config.get('target_sparsity', 0.15)
    penalty = config.get('density_penalty', 300)
    over = max(0.0, density - target)
    raw -= over * penalty
    return raw


EVALUATORS = {
    'CartPole': evaluate_cartpole,
    'MountainCar': evaluate_mountaincar,
    'Acrobot': evaluate_acrobot,
    'FlappyBird': evaluate_flappy,
    'Snake': evaluate_snake,
}


def run_ga(task_name, config, seed):
    """Run GA for one task+seed, return (best_adj, types, best_fitness, history)."""
    np.random.seed(seed)
    n_total = config['n_total']
    n_in = config['n_input']
    n_out = config['n_output']
    sparsity = config['sparsity']

    types = make_neuron_types(n_total, n_in, n_out)
    evaluate = EVALUATORS[task_name]

    pop = [random_topology(n_total, sparsity) for _ in range(config['pop_size'])]
    best_adj = None
    best_fitness = -np.inf
    best_hist = []
    mean_hist = []

    target_sp = config.get('target_sparsity', sparsity)

    for gen in range(config['n_generations']):
        fitnesses = np.array([evaluate(adj, types, config) for adj in pop])
        order = np.argsort(fitnesses)[::-1]
        gen_best = fitnesses[order[0]]
        best_hist.append(float(gen_best))
        mean_hist.append(float(fitnesses.mean()))

        if gen_best > best_fitness:
            best_fitness = gen_best
            best_adj = pop[order[0]].copy()

        if (gen + 1) % 50 == 0:
            print(f'    [{task_name} seed{seed}] Gen {gen+1}: Best={gen_best:.1f}', flush=True)

        n_elite = config['n_elite']
        new_pop = [pop[i].copy() for i in order[:n_elite]]
        while len(new_pop) < config['pop_size']:
            parent = pop[np.random.choice(order[:n_elite + 5])]
            if task_name in ('FlappyBird', 'Snake', 'MountainCar'):
                new_pop.append(mutate_density_aware(parent, config['mutation_rate'], target_sp))
            else:
                child = parent.copy()
                mask = np.random.rand(*child.shape) < config['mutation_rate']
                child[mask] = 1.0 - child[mask]
                np.fill_diagonal(child, 0.0)
                new_pop.append(child)
        pop = new_pop

    return best_adj, types, best_fitness, best_hist, mean_hist


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=str, default='all',
                        help='Comma-separated task names, or "all"')
    parser.add_argument('--start-seed', type=int, default=5)
    parser.add_argument('--end-seed', type=int, default=9)
    args = parser.parse_args()

    if args.tasks == 'all':
        tasks = list(TASK_CONFIGS.keys())
    else:
        tasks = [t.strip() for t in args.tasks.split(',')]

    seeds = list(range(args.start_seed, args.end_seed + 1))
    print(f'Tasks: {tasks}')
    print(f'Seeds: {seeds} ({len(seeds)} per task)')
    print(f'Total runs: {len(tasks) * len(seeds)}')
    print('=' * 60)

    for task_name in tasks:
        config = TASK_CONFIGS[task_name]
        for seed in seeds:
            out_path = os.path.join(RESULTS_DIR, f'{task_name}_seed{seed}.npz')
            if os.path.exists(out_path):
                print(f'[{task_name} seed{seed}] Already exists, skipping.')
                continue

            t0 = time.time()
            print(f'\n[{task_name} seed{seed}] Starting...', flush=True)
            best_adj, types, best_fitness, best_hist, mean_hist = run_ga(task_name, config, seed)
            elapsed = time.time() - t0

            np.savez(out_path,
                     adj=best_adj, types=types, best_fitness=best_fitness,
                     best_hist=np.array(best_hist), mean_hist=np.array(mean_hist),
                     task=task_name, seed=seed, config=str(config))
            print(f'[{task_name} seed{seed}] Done in {elapsed:.0f}s. Best fitness={best_fitness:.1f}', flush=True)

    print('\nAll done.')


if __name__ == '__main__':
    main()
