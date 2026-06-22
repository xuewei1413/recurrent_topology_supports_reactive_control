"""
实验 D2: NEAT 独立基线
目标：与结构动力智能方法进行系统对比。

环境：gymnasium.CartPole-v1
方法：NEAT (NeuroEvolution of Augmenting Topologies)
评估：10 个随机种子，每种子独立进化

对比维度：
1. 最终性能（20局严格评估）
2. 收敛速度（达到490步所需代数/评估次数）
3. 网络复杂度（节点数、连接数）
4. 参数效率（性能/参数数量）
5. 稳定性（多种子方差）

关键区别：
- NEAT 同时进化拓扑 + 权重（连续值）
- 我们的方法只进化拓扑（二值邻接矩阵），权重固定由神经元类型决定
"""
import os
import sys
import csv
import time
import tempfile
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import gymnasium as gym
    import neat
except ImportError as e:
    print(f"缺少依赖: {e}")
    sys.exit(1)

# ── NEAT 配置文件（代码内嵌）──────────────────────────────────────────

NEAT_CONFIG_TEXT = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 490
pop_size              = 50
reset_on_extinction   = False
no_fitness_termination = False

[DefaultGenome]
# node activation options
activation_default      = tanh
activation_mutate_rate  = 0.1
activation_options      = tanh sigmoid relu

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_init_type          = gaussian
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = 0.3
conn_delete_prob        = 0.3

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01
enabled_rate_to_true_add  = 0.0
enabled_rate_to_false_add = 0.0

feed_forward            = False
initial_connection      = full
single_structural_mutation = false
structural_mutation_surer = default

# node add/remove rates
node_add_prob           = 0.2
node_delete_prob        = 0.2

# network parameters
num_hidden              = 0
num_inputs              = 4
num_outputs             = 2

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_init_type      = gaussian
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# node time constant options (for CTRNN)
time_constant_init_mean     = 1.0
time_constant_init_stdev    = 0.0
time_constant_init_type     = gaussian
time_constant_max_value     = 10.0
time_constant_min_value     = 0.01
time_constant_mutate_power  = 0.0
time_constant_mutate_rate   = 0.0
time_constant_replace_rate  = 0.0

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_init_type        = gaussian
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""


def get_neat_config():
    """将内嵌配置写入临时文件并返回 neat.Config 对象。"""
    fd, path = tempfile.mkstemp(suffix='.cfg')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(NEAT_CONFIG_TEXT)
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            path
        )
    finally:
        os.remove(path)
    return config

# ── 配置 ──────────────────────────────────────────────────────────────

N_SEEDS = 10
MAX_GENERATIONS = 300
N_EVAL_EPISODES = 8       # 每基因组评估局数（与我们方法一致）
N_FINAL_EVAL = 20         # 最终严格评估局数
ENV_NAME = 'CartPole-v1'
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'neat_baseline')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ── 辅助函数 ──────────────────────────────────────────────────────────

def eval_genome(genome, config, n_episodes=8, seed_offset=0):
    """
    评估单个 NEAT genome 在 CartPole 上的性能。
    返回 n_episodes 局的平均步数。
    """
    net = neat.nn.RecurrentNetwork.create(genome, config)
    env = gym.make(ENV_NAME)
    total_steps = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        net.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            action = np.argmax(net.activate(obs))
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        total_steps += steps
    env.close()
    return total_steps / n_episodes


def eval_genome_detailed(genome, config, n_episodes=20, seed=0):
    """严格评估：返回每局步数列表"""
    net = neat.nn.RecurrentNetwork.create(genome, config)
    env = gym.make(ENV_NAME)
    lengths = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        net.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            action = np.argmax(net.activate(obs))
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        lengths.append(steps)
    env.close()
    return np.array(lengths)


def count_neat_params(genome):
    """
    计算 NEAT genome 的有效参数量。
    连接：每个连接1个权重
    节点：每个节点1个偏置（neat-python默认每个节点有bias）
    """
    n_connections = len(genome.connections)
    n_nodes = len(genome.nodes)
    return n_connections + n_nodes


# ── 自定义 Reporter 用于记录每代统计 ──────────────────────────────────

class GenerationLogger(neat.reporting.BaseReporter):
    def __init__(self):
        self.best_history = []
        self.mean_history = []
        self.generation = 0

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        fitnesses = [g.fitness for g in population.values()]
        best = max(fitnesses) if fitnesses else 0
        mean = np.mean(fitnesses) if fitnesses else 0
        self.best_history.append(best)
        self.mean_history.append(mean)

    def found_solution(self, config, generation, best):
        pass

    def species_stagnant(self, sid, species):
        pass

    def complete_extinction(self):
        pass


# ── 训练单个种子 ──────────────────────────────────────────────────────

def train_neat_single_seed(seed, max_generations=300):
    print(f"\n[Seed {seed + 1}/{N_SEEDS}] Training NEAT on {ENV_NAME}...")
    t0 = time.time()

    config = get_neat_config()

    p = neat.Population(config)
    logger = GenerationLogger()
    p.add_reporter(logger)
    p.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    def eval_fitness(genomes, neat_config):
        for genome_id, genome in genomes:
            genome.fitness = eval_genome(genome, neat_config, n_episodes=N_EVAL_EPISODES, seed_offset=seed * 10000)

    winner = p.run(eval_fitness, n=max_generations)

    # 最终严格评估
    final_lengths = eval_genome_detailed(winner, config, n_episodes=N_FINAL_EVAL, seed=seed + 2000)
    final_mean = float(np.mean(final_lengths))
    final_std = float(np.std(final_lengths))
    final_min = int(np.min(final_lengths))
    final_max = int(np.max(final_lengths))

    n_nodes = len(winner.nodes)
    n_conns = len(winner.connections)
    n_params = count_neat_params(winner)

    # 找到达到490的代数
    gen_solved = None
    for i, bf in enumerate(logger.best_history):
        if bf >= 490:
            gen_solved = i + 1
            break

    elapsed = time.time() - t0
    n_gens = len(logger.best_history)
    total_evals = n_gens * config.pop_size * N_EVAL_EPISODES

    print(f"[Seed {seed + 1}/{N_SEEDS}] Final eval: {final_mean:.1f} ± {final_std:.1f} (min={final_min}, max={final_max})")
    print(f"[Seed {seed + 1}/{N_SEEDS}] Network: {n_nodes} nodes, {n_conns} connections, {n_params} params")
    print(f"[Seed {seed + 1}/{N_SEEDS}] Generations: {n_gens}, Solved at gen: {gen_solved}")

    return {
        'seed': seed,
        'training_curve': list(zip(range(1, n_gens + 1), logger.best_history, logger.mean_history)),
        'final_lengths': final_lengths,
        'final_mean': final_mean,
        'final_std': final_std,
        'final_min': final_min,
        'final_max': final_max,
        'n_nodes': n_nodes,
        'n_connections': n_conns,
        'n_params': n_params,
        'n_generations': n_gens,
        'gen_solved': gen_solved,
        'total_evals': total_evals,
        'elapsed_sec': elapsed,
    }


# ── 可视化与保存 ──────────────────────────────────────────────────────

def save_results(all_results):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── CSV ───────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, 'neat_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'seed', 'final_mean', 'final_std', 'min', 'max',
            'n_nodes', 'n_connections', 'n_params',
            'n_generations', 'gen_solved', 'total_evals'
        ])
        for r in all_results:
            writer.writerow([
                r['seed'], f"{r['final_mean']:.2f}", f"{r['final_std']:.2f}",
                r['final_min'], r['final_max'],
                r['n_nodes'], r['n_connections'], r['n_params'],
                r['n_generations'], r['gen_solved'] if r['gen_solved'] else 'N/A',
                r['total_evals'],
            ])
    print(f"\nCSV saved: {csv_path}")

    # ── 训练曲线图 ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in all_results:
        if r['training_curve']:
            gens, bests, means = zip(*r['training_curve'])
            ax.plot(gens, bests, alpha=0.7, lw=1.5, label=f"Seed {r['seed']}")
    ax.axhline(490, color='red', linestyle='--', alpha=0.5, label='Solve threshold 490')
    ax.axhline(500, color='green', linestyle='--', alpha=0.5, label='Max 500')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness (mean steps)')
    ax.set_title('NEAT Training Curves (CartPole-v1, 10 seeds)')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig_path_curve = os.path.join(RESULTS_DIR, 'neat_training_curves.png')
    plt.savefig(fig_path_curve, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {fig_path_curve}")

    # ── 网络复杂度分布 ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    n_nodes_list = [r['n_nodes'] for r in all_results]
    n_conns_list = [r['n_connections'] for r in all_results]
    n_params_list = [r['n_params'] for r in all_results]

    axes[0].boxplot([n_nodes_list], labels=['Nodes'])
    axes[0].set_ylabel('Count')
    axes[0].set_title('Nodes per Network')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].boxplot([n_conns_list], labels=['Connections'])
    axes[1].set_ylabel('Count')
    axes[1].set_title('Connections per Network')
    axes[1].grid(True, alpha=0.3, axis='y')

    axes[2].boxplot([n_params_list], labels=['Parameters'])
    axes[2].set_ylabel('Count')
    axes[2].set_title('Total Parameters per Network')
    axes[2].grid(True, alpha=0.3, axis='y')

    fig.suptitle('NEAT Evolved Network Complexity (10 seeds)')
    fig_path_complex = os.path.join(RESULTS_DIR, 'neat_complexity.png')
    plt.savefig(fig_path_complex, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {fig_path_complex}")

    # ── 最终评估箱线图 ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    all_lengths = [r['final_lengths'] for r in all_results]
    bp = ax.boxplot(all_lengths, labels=[f"S{r['seed']}" for r in all_results], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('coral')
        patch.set_alpha(0.7)
    ax.axhline(490, color='red', linestyle='--', alpha=0.5, label='Solve threshold 490')
    ax.axhline(500, color='green', linestyle='--', alpha=0.5, label='Max 500')
    ax.set_xlabel('Random Seed')
    ax.set_ylabel('Episode Length (steps)')
    ax.set_title('NEAT Final Evaluation (20 episodes per seed)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig_path_box = os.path.join(RESULTS_DIR, 'neat_final_eval.png')
    plt.savefig(fig_path_box, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {fig_path_box}")

    # ── 综合对比图（NEAT vs 我们的方法 vs PPO）───────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) 最终性能对比
    methods = ['Struct-Dyn\n(Ours)', 'NEAT', 'PPO']
    means = [
        494.7,  # 我们的方法（multi_seed_summary.csv best_fitness mean）
        np.mean([r['final_mean'] for r in all_results]),
        500.0,  # PPO（来自已有结果）
    ]
    stds = [
        13.8,
        np.std([r['final_mean'] for r in all_results]),
        0.0,
    ]
    colors = ['steelblue', 'coral', 'seagreen']
    axes[0, 0].bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    axes[0, 0].set_ylabel('Final Performance (steps)')
    axes[0, 0].set_title('(a) Final Performance Comparison')
    axes[0, 0].set_ylim(0, 520)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # (b) 参数量对比（对数轴）
    params = [
        1,      # 我们的方法：1个标量（weight_scale）
        np.mean(n_params_list),
        34691,  # PPO
    ]
    axes[0, 1].bar(methods, params, color=colors, alpha=0.8, edgecolor='black')
    axes[0, 1].set_ylabel('Parameter Count (log scale)')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('(b) Parameter Count')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # (c) 收敛速度对比
    conv_gens = [r['gen_solved'] for r in all_results if r['gen_solved']]
    conv_methods = ['Struct-Dyn\n(Ours)', 'NEAT', 'PPO']
    conv_vals = [
        97,     # 我们的方法：中位数97代（来自论文20种子结果）
        np.median(conv_gens) if conv_gens else None,
        None,   # PPO用timesteps，不直接可比
    ]
    # 只画可比的
    x_pos = [0, 1]
    y_vals = [conv_vals[0], conv_vals[1]]
    axes[1, 0].bar(x_pos, y_vals, color=[colors[0], colors[1]], alpha=0.8, edgecolor='black')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([conv_methods[0], conv_methods[1]])
    axes[1, 0].set_ylabel('Generations to Solve')
    axes[1, 0].set_title('(c) Convergence Speed')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # (d) 参数效率（性能/参数）
    eff = [
        494.7 / 1,
        np.mean([r['final_mean'] / max(r['n_params'], 1) for r in all_results]),
        500.0 / 34691,
    ]
    axes[1, 1].bar(methods, eff, color=colors, alpha=0.8, edgecolor='black')
    axes[1, 1].set_ylabel('Performance / Parameter')
    axes[1, 1].set_title('(d) Parameter Efficiency')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    fig.suptitle('Comprehensive Comparison: Structural Dynamic Intelligence vs NEAT vs PPO', fontsize=14, fontweight='bold')
    fig_path_compare = os.path.join(RESULTS_DIR, 'comparison_overview.png')
    plt.savefig(fig_path_compare, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {fig_path_compare}")

    # ── 文本摘要 ────────────────────────────────────────────────
    final_means = [r['final_mean'] for r in all_results]
    final_stds_seed = [r['final_std'] for r in all_results]
    gen_solved_list = [r['gen_solved'] for r in all_results if r['gen_solved']]

    summary_lines = [
        "=" * 70,
        "                     NEAT 基线实验摘要",
        "=" * 70,
        f"环境         : {ENV_NAME}",
        f"种群大小     : 50",
        f"评估局数     : {N_EVAL_EPISODES} (训练) / {N_FINAL_EVAL} (最终)",
        f"种子数       : {N_SEEDS}",
        f"最大代数     : {MAX_GENERATIONS}",
        "",
        f"最终 20 局性能 (10 种子均值 ± std): {np.mean(final_means):.1f} ± {np.std(final_means):.1f}",
        f"最终性能范围 : {np.min(final_means):.1f} / {np.max(final_means):.1f}",
        f"训练稳定性 (10 种子最终性能方差) : {np.var(final_means):.2f}",
        "",
        f"网络复杂度   : {np.mean(n_nodes_list):.1f} ± {np.std(n_nodes_list):.1f} nodes, "
        f"{np.mean(n_conns_list):.1f} ± {np.std(n_conns_list):.1f} connections",
        f"参数量       : {np.mean(n_params_list):.1f} ± {np.std(n_params_list):.1f}",
        "",
    ]

    if gen_solved_list:
        summary_lines.append(f"解决代数中位数 : {int(np.median(gen_solved_list))}")
        summary_lines.append(f"解决代数范围   : {np.min(gen_solved_list)} / {np.max(gen_solved_list)}")
        summary_lines.append(f"解决率         : {len(gen_solved_list)}/{N_SEEDS}")
    else:
        summary_lines.append("解决代数中位数 : N/A (均未达到)")

    summary_lines.extend([
        "",
        "─" * 70,
        "每种子详细结果:",
        f"{'Seed':>5} {'FinalMean':>10} {'FinalStd':>9} {'Nodes':>6} {'Conns':>6} {'Params':>7} {'GenSolved':>10}",
        "-" * 65,
    ])
    for r in all_results:
        gs = str(r['gen_solved']) if r['gen_solved'] else 'N/A'
        summary_lines.append(
            f"{r['seed']:5d} {r['final_mean']:10.1f} {r['final_std']:9.1f} "
            f"{r['n_nodes']:6d} {r['n_connections']:6d} {r['n_params']:7d} {gs:>10}"
        )
    summary_lines.append("=" * 70)

    summary_text = "\n".join(summary_lines)
    txt_path = os.path.join(RESULTS_DIR, 'neat_results_summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"Summary saved: {txt_path}")
    print("\n" + summary_text)


# ── 主实验 ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  实验 D2: NEAT 独立基线")
    print("=" * 70)
    print(f"环境: {ENV_NAME}")
    print(f"种群: 50")
    print(f"种子: 0–{N_SEEDS - 1}")
    print(f"最大代数: {MAX_GENERATIONS}")
    print("=" * 70)

    all_results = []

    for seed in range(N_SEEDS):
        result = train_neat_single_seed(seed, max_generations=MAX_GENERATIONS)
        all_results.append(result)

    save_results(all_results)
    print("\nDone.")


if __name__ == '__main__':
    main()
