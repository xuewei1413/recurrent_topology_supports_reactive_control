"""
实验 1C-Pendulum：连续控制快速验证（MuJoCo/PyBullet 不可用时）

环境：Pendulum-v1（经典倒立摆，连续动作）
- 观察：3维 [cos(theta), sin(theta), theta_dot]
- 动作：1维力矩 [-2.0, 2.0]

这虽然是简化环境，但完整验证了：
  1. ContinuousSNNController 在连续动作空间的有效性
  2. 纯拓扑进化能否习得连续控制策略
  3. 与随机策略/PPO 的对比框架

成功后可无缝迁移到 PyBullet 四足机器人。
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys, time, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.continuous_snn import ContinuousSNNController
from src.evolution import random_topology

import gymnasium as gym

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'embodied_reflex')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 配置 ──────────────────────────────────────────────────────────────────

CONFIG = {
    'env_name': 'Pendulum-v1',
    'n_total': 64,
    'n_input': 6,      # 3维观察 × 2（正负编码）
    'n_output': 1,     # 1维连续力矩
    'sparsity': 0.15,
    'weight_scale': 0.6,
    'tau': 5.0,
    'threshold': 0.8,
    'T_steps': 15,     # 连续控制用较短整合窗口
    'pop_size': 50,
    'n_generations': 120,
    'n_elite': 8,
    'mutation_rate': 0.03,
    'n_eval_episodes': 8,
    'max_steps': 200,
}


def mutate(adj, rate):
    child = adj.copy()
    mask = np.random.rand(*child.shape) < rate
    child[mask] = 1.0 - child[mask]
    np.fill_diagonal(child, 0.0)
    return child


def evaluate_pendulum(adj, neuron_types, config, render=False):
    """评估 SNN 在 Pendulum-v1 上的表现。"""
    ctrl_cfg = {
        'n_total': config['n_total'],
        'n_input': config['n_input'],
        'n_output': config['n_output'],
        'weight_scale': config['weight_scale'],
        'tau': config['tau'],
        'threshold': config['threshold'],
        'T_steps': config['T_steps'],
    }
    controller = ContinuousSNNController(ctrl_cfg, adj, neuron_types)
    env = gym.make(config['env_name'], render_mode='human' if render else None)
    total_reward = 0.0
    for _ in range(config['n_eval_episodes']):
        obs, _ = env.reset()
        controller.reset()
        ep_reward = 0.0
        for _ in range(config['max_steps']):
            action = controller.forward(np.array(obs, dtype=np.float64))
            # Pendulum 动作空间是 [-2, 2]，但 ContinuousSNNController 输出 [-1, 1]
            action_scaled = np.array([action[0] * 2.0])
            obs, reward, terminated, truncated, _ = env.step(action_scaled)
            ep_reward += reward
            if terminated or truncated:
                break
        total_reward += ep_reward
    env.close()
    return total_reward / config['n_eval_episodes']


def evaluate_random_policy(config):
    """随机策略基线。"""
    env = gym.make(config['env_name'])
    total = 0.0
    for _ in range(10):
        obs, _ = env.reset()
        ep = 0.0
        for _ in range(config['max_steps']):
            action = env.action_space.sample()
            obs, reward, done, truncated, _ = env.step(action)
            ep += reward
            if done or truncated:
                break
        total += ep
    env.close()
    return total / 10


def run_pendulum_evolution(config, seed=42):
    np.random.seed(seed)
    n = config['n_total']

    # 神经元类型
    neuron_types = np.ones(n, dtype=np.float64)
    hidden_range = np.arange(config['n_input'], n - config['n_output'])
    n_inh = int(len(hidden_range) * 0.3)
    inh_idx = np.random.choice(hidden_range, n_inh, replace=False)
    neuron_types[inh_idx] = -1.0

    pop = [random_topology(n, config['sparsity']) for _ in range(config['pop_size'])]
    best_history = []
    mean_history = []
    best_adj = None
    best_fitness = -np.inf

    print(f"\n[Pendulum-v1] Evolution | n={n} | seed={seed} | gens={config['n_generations']}")
    print(f"{'Gen':>4}  {'Best':>9}  {'Mean':>9}  {'Conns':>6}")

    start = time.time()
    for gen in range(config['n_generations']):
        fitnesses = []
        for adj in pop:
            f = evaluate_pendulum(adj, neuron_types, config)
            fitnesses.append(f)
        fitnesses = np.array(fitnesses)

        order = np.argsort(fitnesses)[::-1]
        gen_best = fitnesses[order[0]]
        gen_mean = fitnesses.mean()
        best_conn = int(pop[order[0]].sum())

        best_history.append(float(gen_best))
        mean_history.append(float(gen_mean))

        if gen_best > best_fitness:
            best_fitness = gen_best
            best_adj = pop[order[0]].copy()

        if (gen + 1) % 10 == 0 or gen == 0:
            print(f"{gen+1:4d}  {gen_best:9.1f}  {gen_mean:9.1f}  {best_conn:6d}")

        # 精英 + 变异
        new_pop = [pop[i].copy() for i in order[:config['n_elite']]]
        while len(new_pop) < config['pop_size']:
            parent = pop[np.random.choice(order[:10])]
            child = mutate(parent, config['mutation_rate'])
            new_pop.append(child)
        pop = new_pop

    elapsed = time.time() - start
    print(f"  -> Best fitness: {best_fitness:.1f} | Time: {elapsed/60:.1f} min")
    return best_adj, neuron_types, best_fitness, best_history, mean_history


def plot_results(best_hist, mean_hist, random_score, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(best_hist, label='SNN Best', color='steelblue', linewidth=2)
    ax.plot(mean_hist, label='SNN Mean', color='lightblue', alpha=0.7)
    ax.axhline(random_score, color='gray', linestyle='--', label=f'Random Baseline ({random_score:.1f})')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness (Mean Reward)')
    ax.set_title('Pendulum-v1 — SNN Topology Evolution (Continuous Control)')
    ax.legend()
    fig_path = os.path.join(out_dir, 'pendulum_evolution.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")


def main():
    print("="*70)
    print("  实验 1C-Pendulum: 连续控制快速验证")
    print("="*70)

    # 1. 随机策略基线
    print("\n[Step 1] Random policy baseline...")
    random_score = evaluate_random_policy(CONFIG)
    print(f"  Random baseline: {random_score:.1f}")

    # 2. SNN 进化
    print("\n[Step 2] SNN topology evolution...")
    best_adj, neuron_types, best_fitness, best_hist, mean_hist = \
        run_pendulum_evolution(CONFIG, seed=42)

    # 保存拓扑
    np.save(os.path.join(RESULTS_DIR, 'pendulum_best_topology.npy'), best_adj)
    np.save(os.path.join(RESULTS_DIR, 'pendulum_neuron_types.npy'), neuron_types)

    # 3. 可视化
    print("\n[Step 3] Plotting...")
    plot_results(best_hist, mean_hist, random_score, RESULTS_DIR)

    # 4. 最终评估（20 局严格测试）
    print("\n[Step 4] Final strict evaluation (20 episodes)...")
    CONFIG['n_eval_episodes'] = 20
    final_score = evaluate_pendulum(best_adj, neuron_types, CONFIG)
    print(f"  Final 20-episode mean: {final_score:.1f}")

    # 保存报告
    report = {
        'env': 'Pendulum-v1',
        'random_baseline': float(random_score),
        'snn_best_fitness': float(best_fitness),
        'snn_final_20ep': float(final_score),
        'config': {k: v for k, v in CONFIG.items() if k != 'env_name'},
    }
    with open(os.path.join(RESULTS_DIR, 'pendulum_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*70)
    print("  Done!")
    print(f"  Random: {random_score:.1f}  |  SNN Best: {best_fitness:.1f}  |  SNN Final(20ep): {final_score:.1f}")
    print("="*70)


if __name__ == '__main__':
    main()
