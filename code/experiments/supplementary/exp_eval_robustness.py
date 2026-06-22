"""
实验 1.2: 评估鲁棒性对比
目标：验证适应度评估的严格程度直接影响演化出的拓扑质量。

方法：
  用 4 种不同评估配置各跑 5 个随机种子，然后用统一的 20 局严格评估
  对比各配置演化出的拓扑质量。

评估配置：
  A. 3局均值  — 低噪声但容易过拟合少数初始状态
  B. 8局均值  — 当前标准（实验配置）
  C. 10局均值 — 更高标准
  D. 10局最小值 — 最严格：要求每局都表现好

输出：
  - results/evaluation_robustness.png
  - results/evaluation_robustness.csv
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os, csv, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.evolution import run_evolution
from src.snn import SNNController
import gymnasium as gym

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 基础配置 ──────────────────────────────────────────────────────────────

BASE_CONFIG = {
    'n_total'        : 128,
    'n_input'        : 8,
    'n_output'       : 2,
    'sparsity'       : 0.10,
    'weight_scale'   : 0.6,
    'tau'            : 5.0,
    'threshold'      : 0.8,
    'T_steps'        : 20,
    'pop_size'       : 40,
    'n_generations'  : 400,
    'n_elite'        : 8,
    'mutation_rate'  : 0.03,
    'run_baseline'   : False,
    'n_workers'      : 4,
}

EVAL_CONFIGS = {
    '3-mean':   {'n_eval_episodes': 3,  'eval_aggregate': 'mean'},
    '8-mean':   {'n_eval_episodes': 8,  'eval_aggregate': 'mean'},
    '10-mean':  {'n_eval_episodes': 10, 'eval_aggregate': 'mean'},
    '10-min':   {'n_eval_episodes': 10, 'eval_aggregate': 'min'},
}

N_SEEDS_PER_CONFIG = 5
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


# ── 修改 evaluate 以支持 min 聚合 ─────────────────────────────────────────

from src.evolution import evaluate as orig_evaluate
from concurrent.futures import ProcessPoolExecutor


def _eval_worker_robust(args):
    """子进程评估，供 ProcessPoolExecutor 调用"""
    adj, neuron_types, config = args
    return evaluate_with_aggregate(adj, neuron_types, config)


def evaluate_with_aggregate(adj, neuron_types, config):
    """支持 mean 和 min 两种聚合方式的评估"""
    n_eps = config['n_eval_episodes']
    aggregate = config.get('eval_aggregate', 'mean')

    controller = SNNController(config, adj, neuron_types)
    env = gym.make('CartPole-v1')
    scores = []

    for _ in range(n_eps):
        obs, _ = env.reset()
        controller.reset()
        ep_score = 0.0
        for step in range(500):
            action = controller.forward(np.array(obs, dtype=np.float64))
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_score += reward
            if terminated or truncated:
                break
        scores.append(ep_score)

    env.close()

    if aggregate == 'min':
        return min(scores)
    else:
        return np.mean(scores)


def run_evolution_custom(config):
    """
    复制 run_evolution 逻辑，但使用自定义 evaluate。
    由于 run_evolution 内部直接调用了 evaluate，我们需要重新实现。
    """
    n = config['n_total']
    
    # 神经元类型
    neuron_types = np.ones(n, dtype=np.float64)
    hidden_range = np.arange(config['n_input'], n - config['n_output'])
    n_inh = int(len(hidden_range) * 0.3)
    inh_idx = np.random.choice(hidden_range, n_inh, replace=False)
    neuron_types[inh_idx] = -1.0
    
    # 初始化种群
    from src.evolution import random_topology
    pop = [random_topology(n, config['sparsity']) for _ in range(config['pop_size'])]
    
    best_history = []
    mean_history = []
    best_adj = None
    best_fitness = -np.inf
    
    n_workers = config.get('n_workers', 1)

    for gen in range(config['n_generations']):
        if n_workers > 1:
            tasks = [(adj, neuron_types, config) for adj in pop]
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                fitnesses = np.array(list(pool.map(_eval_worker_robust, tasks)))
        else:
            fitnesses = np.array([evaluate_with_aggregate(adj, neuron_types, config) for adj in pop])
        
        order = np.argsort(fitnesses)[::-1]
        gen_best = fitnesses[order[0]]
        gen_mean = fitnesses.mean()
        best_conn = int(pop[order[0]].sum())
        
        best_history.append(gen_best)
        mean_history.append(gen_mean)
        
        if gen_best > best_fitness:
            best_fitness = gen_best
            best_adj = pop[order[0]].copy()
        
        # 通关条件：对于 min 聚合，要求 min >= 490；对于 mean，要求 mean >= 490
        solved = gen_best >= 490
        if solved:
            break
        
        # 精英保留 + 变异
        from src.evolution import mutate
        elite = [pop[i] for i in order[:config['n_elite']]]
        new_pop = list(elite)
        while len(new_pop) < config['pop_size']:
            parent = elite[np.random.randint(len(elite))]
            new_pop.append(mutate(parent, config['mutation_rate']))
        pop = new_pop
    
    return best_adj, neuron_types, best_history, mean_history


def strict_eval_20(adj, neuron_types, config):
    """统一的严格评估：20 局均值 + 最小值 + 标准差"""
    controller = SNNController(config, adj, neuron_types)
    env = gym.make('CartPole-v1')
    scores = []
    for _ in range(20):
        obs, _ = env.reset()
        controller.reset()
        ep_score = 0.0
        for step in range(500):
            action = controller.forward(np.array(obs, dtype=np.float64))
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_score += reward
            if terminated or truncated:
                break
        scores.append(ep_score)
    env.close()
    return np.mean(scores), np.min(scores), np.std(scores), scores


def run_single(config_name, eval_cfg, seed, base_config):
    """运行单个配置+种子的完整 GA"""
    np.random.seed(seed)
    
    config = {**base_config, **eval_cfg}
    
    print(f"\n[{config_name} | Seed {seed}] Starting...")
    t0 = time.time()
    
    best_adj, neuron_types, best_hist, mean_hist = run_evolution_custom(config)
    
    elapsed = time.time() - t0
    generations = len(best_hist)
    best_fitness = max(best_hist) if best_hist else 0.0
    solved = best_fitness >= 490
    
    print(f"[{config_name} | Seed {seed}] Gen={generations}, Best={best_fitness:.1f}, Solved={solved}, Time={elapsed:.1f}s")
    
    # 严格 20 局评估
    if best_adj is not None:
        mean_20, min_20, std_20, all_scores = strict_eval_20(best_adj, neuron_types, base_config)
        n_conn = int(best_adj.sum())
    else:
        mean_20, min_20, std_20, all_scores = 0.0, 0.0, 0.0, []
        n_conn = 0
    
    print(f"[{config_name} | Seed {seed}] Strict 20-eval: {mean_20:.1f} (min={min_20:.1f}, std={std_20:.1f})")
    
    return {
        'config': config_name,
        'seed': seed,
        'generations': generations,
        'best_fitness': best_fitness,
        'solved': solved,
        'mean_20': mean_20,
        'min_20': min_20,
        'std_20': std_20,
        'n_connections': n_conn,
        'elapsed_sec': elapsed,
        'all_scores_20': all_scores,
    }


def generate_report(all_results):
    """生成报告和可视化"""
    # 按配置分组
    config_names = list(EVAL_CONFIGS.keys())
    grouped = {cn: [] for cn in config_names}
    for r in all_results:
        grouped[r['config']].append(r)
    
    # 汇总统计
    print("\n" + "="*70)
    print("                     评估鲁棒性对比报告")
    print("="*70)
    print(f"{'Config':>10} {'Seeds':>6} {'Solved':>7} {'Mean20':>8} {'Min20':>8} {'Std20':>7}")
    print("-"*60)
    
    summary = {}
    for cn in config_names:
        rs = grouped[cn]
        solved_count = sum(r['solved'] for r in rs)
        mean20_vals = [r['mean_20'] for r in rs]
        min20_vals = [r['min_20'] for r in rs]
        print(f"{cn:>10} {len(rs):6d} {solved_count:6d}/{len(rs)} "
              f"{np.mean(mean20_vals):8.1f} {np.mean(min20_vals):8.1f} {np.std(mean20_vals):7.1f}")
        summary[cn] = {
            'mean_20': np.mean(mean20_vals),
            'std_20': np.std(mean20_vals),
            'mean_min20': np.mean(min20_vals),
            'solved_rate': solved_count / len(rs),
        }
    
    # 保存 CSV
    csv_path = os.path.join(RESULTS_DIR, 'evaluation_robustness.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['config', 'seed', 'generations', 'best_fitness', 'solved',
                         'mean_20', 'min_20', 'std_20', 'n_connections', 'elapsed_sec'])
        for r in all_results:
            writer.writerow([
                r['config'], r['seed'], r['generations'], r['best_fitness'],
                r['solved'], r['mean_20'], r['min_20'], r['std_20'],
                r['n_connections'], r['elapsed_sec']
            ])
    print(f"\nCSV saved: {csv_path}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('实验 1.2：评估配置对演化拓扑质量的影响', fontsize=14)
    
    colors = {'3-mean': '#e74c3c', '8-mean': '#f39c12', '10-mean': '#3498db', '10-min': '#2ecc71'}
    
    # 1. 20 局评估均值箱线图
    ax = axes[0, 0]
    data_for_box = [grouped[cn] for cn in config_names]
    bp = ax.boxplot([[r['mean_20'] for r in grouped[cn]] for cn in config_names],
                     labels=config_names, patch_artist=True)
    for patch, cn in zip(bp['boxes'], config_names):
        patch.set_facecolor(colors[cn])
    ax.axhline(490, color='red', linestyle='--', alpha=0.5, label='通关线 (490)')
    ax.set_ylabel('20局评估均值')
    ax.set_title('各配置演化拓扑的严格评估性能')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 20 局最小值箱线图（衡量最坏情况鲁棒性）
    ax = axes[0, 1]
    bp = ax.boxplot([[r['min_20'] for r in grouped[cn]] for cn in config_names],
                     labels=config_names, patch_artist=True)
    for patch, cn in zip(bp['boxes'], config_names):
        patch.set_facecolor(colors[cn])
    ax.axhline(490, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('20局评估最小值')
    ax.set_title('各配置拓扑的最坏情况表现\n(越高 = 越可靠)')
    ax.grid(True, alpha=0.3)
    
    # 3. 通关成功率对比
    ax = axes[1, 0]
    success_rates = [sum(r['solved'] for r in grouped[cn]) / len(grouped[cn]) * 100 for cn in config_names]
    bars = ax.bar(config_names, success_rates, color=[colors[cn] for cn in config_names], edgecolor='black')
    ax.set_ylabel('通关成功率 (%)')
    ax.set_title('各配置 GA 通关成功率')
    ax.set_ylim(0, 105)
    for bar, rate in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.0f}%', ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. 分布对比（violin 风格，用散点+抖动）
    ax = axes[1, 1]
    for i, cn in enumerate(config_names):
        vals = [r['mean_20'] for r in grouped[cn]]
        x_jitter = np.random.normal(i, 0.05, size=len(vals))
        ax.scatter(x_jitter, vals, color=colors[cn], alpha=0.6, s=80, edgecolors='black')
        ax.plot([i-0.2, i+0.2], [np.mean(vals)]*2, color='black', linewidth=2)
    ax.set_xticks(range(len(config_names)))
    ax.set_xticklabels(config_names)
    ax.axhline(490, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('20局评估均值')
    ax.set_title('各配置拓扑性能分布\n(黑线=均值)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'evaluation_robustness.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {fig_path}")
    
    # 保存汇总
    np.save(os.path.join(RESULTS_DIR, 'evaluation_robustness_summary.npy'), summary)
    print(f"Summary saved: {os.path.join(RESULTS_DIR, 'evaluation_robustness_summary.npy')}")


def main():
    print("="*70)
    print("  实验 1.2: 评估鲁棒性对比")
    print("="*70)
    print(f"评估配置: {list(EVAL_CONFIGS.keys())}")
    print(f"每配置种子数: {N_SEEDS_PER_CONFIG}")
    print(f"总运行次数: {len(EVAL_CONFIGS) * N_SEEDS_PER_CONFIG}")
    print("="*70)
    
    all_results = []
    checkpoint_path = os.path.join(RESULTS_DIR, 'eval_robustness_checkpoint.npy')
    
    # 尝试加载已有进度
    completed_keys = set()
    if os.path.exists(checkpoint_path):
        checkpoint = np.load(checkpoint_path, allow_pickle=True)
        all_results = list(checkpoint)
        for r in all_results:
            completed_keys.add((r['config'], r['seed']))
        print(f"Checkpoint loaded: {len(all_results)} runs done")
    
    for config_name, eval_cfg in EVAL_CONFIGS.items():
        for seed in range(1, N_SEEDS_PER_CONFIG + 1):
            if (config_name, seed) in completed_keys:
                print(f"\n[{config_name} | Seed {seed}] Already done, skipping.")
                continue
            
            result = run_single(config_name, eval_cfg, seed, BASE_CONFIG)
            all_results.append(result)
            np.save(checkpoint_path, np.array(all_results, dtype=object))
            print(f"Checkpoint saved ({len(all_results)}/{len(EVAL_CONFIGS)*N_SEEDS_PER_CONFIG})")
    
    generate_report(all_results)
    
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    print("\nDone.")


if __name__ == '__main__':
    main()
