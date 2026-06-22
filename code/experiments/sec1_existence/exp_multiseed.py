"""
实验 1.1: 多种子可复现性测试
目标：验证 GA 演化出通关拓扑是稳定现象，而非运气。

方法：
  - 用 20 个不同随机种子独立运行完整 GA 流程
  - 每个种子保存：通关代数、最终 best fitness、20局独立评估均值、拓扑连接数
  - 统计成功率、平均通关代数、fitness 分布

输出：
  - results/multi_seed_summary.npy / .csv
  - results/multi_seed_reproducibility.png
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

# ── 配置 ──────────────────────────────────────────────────────────────────

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
    'n_eval_episodes': 8,
    'run_baseline'   : False,
    'n_workers'      : 4,
}

N_SEEDS = 20
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, 'multi_seed_checkpoint.npy')


def evaluate_20_episodes(adj, neuron_types, config):
    """对给定拓扑进行 20 局独立评估，返回均值和标准差"""
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
    return np.mean(scores), np.std(scores), scores


def run_single_seed(seed, config):
    """运行单个种子的完整 GA，返回结果字典"""
    np.random.seed(seed)
    
    print(f"\n[Seed {seed}] Starting evolution...")
    t0 = time.time()
    
    best_adj, neuron_types, best_hist, mean_hist = run_evolution(config)
    
    elapsed = time.time() - t0
    generations = len(best_hist)
    best_fitness = max(best_hist) if best_hist else 0.0
    solved = best_fitness >= 490
    
    print(f"[Seed {seed}] Generations={generations}, Best={best_fitness:.1f}, Solved={solved}, Time={elapsed:.1f}s")
    
    # 对最优拓扑做 20 局独立评估
    if best_adj is not None:
        mean_20, std_20, all_scores = evaluate_20_episodes(best_adj, neuron_types, config)
        n_conn = int(best_adj.sum())
    else:
        mean_20, std_20, all_scores = 0.0, 0.0, []
        n_conn = 0
    
    print(f"[Seed {seed}] 20-episode eval: {mean_20:.1f} ± {std_20:.1f}")
    
    return {
        'seed': seed,
        'generations': generations,
        'best_fitness': best_fitness,
        'solved': solved,
        'mean_20': mean_20,
        'std_20': std_20,
        'n_connections': n_conn,
        'elapsed_sec': elapsed,
        'best_history': best_hist,
        'mean_history': mean_hist,
        'all_scores_20': all_scores,
    }


def save_checkpoint(results):
    """保存断点，支持断点续跑"""
    np.save(CHECKPOINT_PATH, np.array(results, dtype=object))
    print(f"Checkpoint saved ({len(results)}/{N_SEEDS} done)")


def load_checkpoint():
    """加载已完成的种子结果"""
    if os.path.exists(CHECKPOINT_PATH):
        data = np.load(CHECKPOINT_PATH, allow_pickle=True)
        print(f"Checkpoint loaded: {len(data)} seeds already done")
        return list(data)
    return []


def generate_report(results):
    """生成汇总表和可视化"""
    seeds = [r['seed'] for r in results]
    generations = [r['generations'] for r in results]
    best_fits = [r['best_fitness'] for r in results]
    solved = [r['solved'] for r in results]
    mean_20s = [r['mean_20'] for r in results]
    std_20s = [r['std_20'] for r in results]
    n_conns = [r['n_connections'] for r in results]
    
    success_rate = sum(solved) / len(solved) * 100
    avg_gen = np.mean([g for g, s in zip(generations, solved) if s])
    
    print("\n" + "="*70)
    print("                     多种子可复现性汇总报告")
    print("="*70)
    print(f"总种子数     : {len(results)}")
    print(f"通关成功率   : {sum(solved)}/{len(solved)} ({success_rate:.1f}%)")
    print(f"平均通关代数 : {avg_gen:.1f}" if any(solved) else "平均通关代数 : N/A (无通关)")
    print(f"20局评估均值 : {np.mean(mean_20s):.1f} ± {np.std(mean_20s):.1f}")
    print(f"20局评估范围 : [{min(mean_20s):.1f}, {max(mean_20s):.1f}]")
    print("="*70)
    
    # 详细表格
    print(f"\n{'Seed':>5} {'Gen':>5} {'BestFit':>8} {'Solved':>7} {'20Mean':>8} {'20Std':>7} {'NConn':>6}")
    print("-"*60)
    for r in results:
        print(f"{r['seed']:5d} {r['generations']:5d} {r['best_fitness']:8.1f} "
              f"{'Yes' if r['solved'] else 'No':>7} {r['mean_20']:8.1f} "
              f"{r['std_20']:7.1f} {r['n_connections']:6d}")
    
    # 保存 CSV
    csv_path = os.path.join(RESULTS_DIR, 'multi_seed_summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'generations', 'best_fitness', 'solved',
                         'mean_20', 'std_20', 'n_connections', 'elapsed_sec'])
        for r in results:
            writer.writerow([r['seed'], r['generations'], r['best_fitness'],
                             r['solved'], r['mean_20'], r['std_20'],
                             r['n_connections'], r['elapsed_sec']])
    print(f"\nCSV saved: {csv_path}")
    
    # 保存 npy
    npy_path = os.path.join(RESULTS_DIR, 'multi_seed_summary.npy')
    np.save(npy_path, np.array(results, dtype=object))
    print(f"NPY saved: {npy_path}")
    
    # ── 可视化 ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('实验 1.1：多种子可复现性测试 (n=20)', fontsize=14)
    
    # 1. 通关代数分布
    ax = axes[0, 0]
    solved_gens = [g for g, s in zip(generations, solved) if s]
    if solved_gens:
        ax.hist(solved_gens, bins=10, color='steelblue', edgecolor='white')
        ax.axvline(np.mean(solved_gens), color='red', linestyle='--', label=f'Mean={np.mean(solved_gens):.1f}')
        ax.set_xlabel('通关代数')
        ax.set_ylabel('种子数')
        ax.set_title(f'通关代数分布 (成功 {sum(solved)}/{len(solved)})')
        ax.legend()
    else:
        ax.text(0.5, 0.5, '无通关种子', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('通关代数分布')
    
    # 2. 最终 best fitness 分布
    ax = axes[0, 1]
    colors = ['#2ecc71' if s else '#e74c3c' for s in solved]
    ax.bar(range(len(best_fits)), sorted(best_fits, reverse=True), color=sorted(colors, key=lambda c: c=='#e74c3c'))
    ax.axhline(490, color='red', linestyle='--', alpha=0.7, label='通关线 (490)')
    ax.set_xlabel('种子排名')
    ax.set_ylabel('Best Fitness (8局均值)')
    ax.set_title('各种子最终 Best Fitness')
    ax.legend()
    
    # 3. 20局独立评估均值分布
    ax = axes[0, 2]
    ax.bar(range(len(mean_20s)), sorted(mean_20s, reverse=True), color='steelblue', edgecolor='white')
    ax.axhline(490, color='red', linestyle='--', alpha=0.7, label='通关线 (490)')
    ax.set_xlabel('种子排名')
    ax.set_ylabel('20局评估均值')
    ax.set_title('20局独立评估性能分布')
    ax.legend()
    
    # 4. 进化曲线叠加
    ax = axes[1, 0]
    max_gens = max(len(r['best_history']) for r in results)
    for r in results:
        hist = r['best_history']
        alpha = 0.7 if r['solved'] else 0.2
        lw = 1.5 if r['solved'] else 0.8
        ax.plot(hist, alpha=alpha, lw=lw, color='steelblue' if r['solved'] else 'gray')
    ax.axhline(490, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('进化代数')
    ax.set_ylabel('Best Fitness')
    ax.set_title('进化曲线叠加（深蓝=通关，灰=未通关）')
    ax.grid(True, alpha=0.3)
    
    # 5. 20局评估 vs 进化 best fitness 散点
    ax = axes[1, 1]
    ax.scatter(best_fits, mean_20s, c=['#2ecc71' if s else '#e74c3c' for s in solved], s=80, edgecolors='black')
    ax.plot([0, 500], [0, 500], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('进化阶段 Best Fitness (8局)')
    ax.set_ylabel('20局独立评估均值')
    ax.set_title('进化评估 vs 独立评估一致性')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. 连接数分布
    ax = axes[1, 2]
    ax.hist(n_conns, bins=10, color='coral', edgecolor='white')
    ax.axvline(np.mean(n_conns), color='red', linestyle='--', label=f'Mean={np.mean(n_conns):.0f}')
    ax.set_xlabel('连接数')
    ax.set_ylabel('种子数')
    ax.set_title('通关拓扑连接数分布')
    ax.legend()
    
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'multi_seed_reproducibility.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {fig_path}")


def main():
    print("="*70)
    print("  实验 1.1: 多种子可复现性测试")
    print("="*70)
    print(f"种子范围: 1-{N_SEEDS}")
    print(f"每种子评估: GA 8局 + 通关后20局独立评估")
    print("="*70)
    
    # 加载已有进度
    completed = load_checkpoint()
    completed_seeds = {r['seed'] for r in completed}
    
    results = list(completed)
    
    for seed in range(1, N_SEEDS + 1):
        if seed in completed_seeds:
            print(f"\n[Seed {seed}] Already done, skipping.")
            continue
        
        result = run_single_seed(seed, BASE_CONFIG)
        results.append(result)
        save_checkpoint(results)
    
    # 排序并生成报告
    results.sort(key=lambda r: r['seed'])
    generate_report(results)
    
    # 清理断点
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
    
    print("\nDone.")


if __name__ == '__main__':
    main()
