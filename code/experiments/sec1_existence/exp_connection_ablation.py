"""
实验 B1: 系统性连接类型消融
目标：精确定位拓扑中哪类连接承载核心功能。

消融条件（在 CartPole 最优拓扑上执行）：
  1. 删除全部 E→E 连接 + 随机对照
  2. 删除全部 I→E 连接 + 随机对照
  3. 删除全部 E→I 连接 + 随机对照
  4. 删除全部 I→I 连接 + 随机对照
  5. 输入层断连（删除所有指向输入层神经元的连接）
  6. 输出层断连（删除所有从输出层神经元出发的连接）
  7. 隐藏层内部断连（删除隐藏层神经元之间的所有连接）

扩展分析：
  - 连接密度变化
  - 输入→输出平均最短路径长度变化
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.snn import SNNController
import gymnasium as gym

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def evaluate_cartpole(adj, neuron_types, config, n_episodes=20):
    """CartPole 评估，默认 20 局严格评估。"""
    controller = SNNController(config, adj, neuron_types)
    env = gym.make('CartPole-v1')
    total = 0.0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        controller.reset()
        for step in range(500):
            action = controller.forward(np.array(obs, dtype=np.float64))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
    env.close()
    return total / n_episodes


def get_edge_types(adj, types, n_input, n_output):
    """
    返回各类边的索引列表。

    注意：
      - E/I 类型仅针对隐藏层神经元（types == 1 为兴奋性，types == -1 为抑制性）。
      - 输入层：0 .. n_input-1
      - 输出层：n_total-n_output .. n_total-1
      - 隐藏层：n_input .. n_total-n_output-1
    """
    n_total = adj.shape[0]
    exc_idx = set(np.where(types == 1)[0])
    inh_idx = set(np.where(types == -1)[0])

    hidden_start = n_input
    hidden_end = n_total - n_output

    e_to_e = []
    i_to_e = []
    e_to_i = []
    i_to_i = []
    input_disconnect = []
    output_disconnect = []
    hidden_internal = []

    rows, cols = np.where(adj > 0)
    for i, j in zip(rows, cols):
        # 指向输入层的连接
        if j < n_input:
            input_disconnect.append((int(i), int(j)))

        # 从输出层出发的连接
        if i >= n_total - n_output:
            output_disconnect.append((int(i), int(j)))

        # 隐藏层神经元之间的连接
        if hidden_start <= i < hidden_end and hidden_start <= j < hidden_end:
            hidden_internal.append((int(i), int(j)))

        # 基于 E/I 类型的四类连接（仅限隐藏层神经元之间）
        if i in exc_idx and j in exc_idx:
            e_to_e.append((int(i), int(j)))
        elif i in inh_idx and j in exc_idx:
            i_to_e.append((int(i), int(j)))
        elif i in exc_idx and j in inh_idx:
            e_to_i.append((int(i), int(j)))
        elif i in inh_idx and j in inh_idx:
            i_to_i.append((int(i), int(j)))

    return {
        'E→E': e_to_e,
        'I→E': i_to_e,
        'E→I': e_to_i,
        'I→I': i_to_i,
        'Input\nDisconnect': input_disconnect,
        'Output\nDisconnect': output_disconnect,
        'Hidden\nDisconnect': hidden_internal,
    }


def ablate_edges(adj, edges_to_remove):
    """删除指定边，返回新的邻接矩阵。"""
    if len(edges_to_remove) == 0:
        return adj.copy()
    adj_new = adj.copy()
    for (i, j) in edges_to_remove:
        adj_new[i, j] = 0
    return adj_new


def compute_density(adj):
    """计算有向网络连接密度（排除自连接）。"""
    n = adj.shape[0]
    possible = n * n - n
    if possible == 0:
        return 0.0
    actual = int(np.sum(adj > 0)) - int(np.sum(np.diag(adj) > 0))
    return actual / possible


def compute_avg_path_length(adj, n_input, n_output):
    """计算从输入层到输出层的平均最短有向路径长度。"""
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path
    except ImportError:
        return np.nan

    graph = csr_matrix(adj > 0)
    dist_matrix = shortest_path(csgraph=graph, directed=True, unweighted=True)

    input_idx = np.arange(n_input)
    output_idx = np.arange(adj.shape[0] - n_output, adj.shape[0])

    path_lengths = []
    for i in input_idx:
        for j in output_idx:
            d = dist_matrix[i, j]
            if np.isfinite(d) and d > 0:
                path_lengths.append(d)

    if len(path_lengths) == 0:
        return np.inf
    return float(np.mean(path_lengths))


def run_connection_ablation(adj_path, types_path, config, n_eval=20, n_random_repeats=20):
    """
    主实验函数。

    返回包含所有结果的字典，结构如下：
      {
        'baseline': float,
        'baseline_density': float,
        'baseline_path_length': float,
        'conditions': [ {name, n_edges, score_ablated, drop_specific,
                         drop_random_mean, drop_random_std, specificity,
                         random_drops, density_baseline, density_ablated,
                         path_baseline, path_ablated}, ... ]
      }
    """
    adj = np.load(adj_path)
    types = np.load(types_path)
    n_total = config['n_total']
    n_input = config['n_input']
    n_output = config['n_output']

    print("=" * 60)
    print("连接类型系统性消融实验: CartPole")
    print("=" * 60)

    # ── 基准评估 ──
    print(f"\n[1/8] 基准评估 (n_eval={n_eval})...")
    baseline = evaluate_cartpole(adj, types, config, n_eval)
    print(f"  基准性能: {baseline:.1f}")

    baseline_density = compute_density(adj)
    baseline_path = compute_avg_path_length(adj, n_input, n_output)
    print(f"  原始密度: {baseline_density:.4f}")
    if np.isfinite(baseline_path):
        print(f"  原始平均路径长度: {baseline_path:.2f}")
    else:
        print(f"  原始平均路径长度: inf")

    # 获取各类边
    edge_groups = get_edge_types(adj, types, n_input, n_output)
    condition_names = ['E→E', 'I→E', 'E→I', 'I→I', 'Input\nDisconnect', 'Output\nDisconnect', 'Hidden\nDisconnect']

    results = {
        'baseline': baseline,
        'baseline_density': baseline_density,
        'baseline_path_length': baseline_path,
        'conditions': []
    }

    all_edges = list(zip(*np.where(adj > 0)))

    # ── 逐个条件消融 ──
    for idx, name in enumerate(condition_names, start=2):
        edges = edge_groups[name]
        n_edges = len(edges)
        print(f"\n[{idx}/8] 删除全部 {name} (n_edges={n_edges})...")

        # Specific Ablation
        if n_edges > 0:
            adj_abl = ablate_edges(adj, edges)
            score_abl = evaluate_cartpole(adj_abl, types, config, n_eval)
        else:
            adj_abl = adj.copy()
            score_abl = baseline

        drop_specific = baseline - score_abl

        # 随机对照（仅前 4 个有 E/I 分类的条件）
        random_drops = []
        if name in ['E→E', 'I→E', 'E→I', 'I→I'] and n_edges > 0 and len(all_edges) >= n_edges:
            for r in range(n_random_repeats):
                rand_idx = np.random.choice(len(all_edges), size=n_edges, replace=False)
                rand_edges = [all_edges[i] for i in rand_idx]
                adj_rand = ablate_edges(adj, rand_edges)
                score_rand = evaluate_cartpole(adj_rand, types, config, n_eval)
                random_drops.append(baseline - score_rand)

        random_drops = np.array(random_drops)

        if len(random_drops) > 0:
            drop_random_mean = float(random_drops.mean())
            drop_random_std = float(random_drops.std())
            if abs(drop_random_mean) > 1e-9:
                specificity = (drop_specific - drop_random_mean) / drop_random_mean
            else:
                specificity = np.inf if drop_specific > 0 else 0.0
            print(f"  消融后性能: {score_abl:.1f} (drop={drop_specific:.1f})")
            print(f"  随机对照: mean={baseline - drop_random_mean:.1f}±{drop_random_std:.1f} (drop={drop_random_mean:.1f})")
            print(f"  特异性指数: {specificity:.2f}")
        else:
            drop_random_mean = 0.0
            drop_random_std = 0.0
            specificity = 0.0
            print(f"  消融后性能: {score_abl:.1f} (drop={drop_specific:.1f})")
            print(f"  随机对照: N/A (边数为 0 或无需对照)")
            print(f"  特异性指数: N/A")

        # 网络指标变化
        density_abl = compute_density(adj_abl)
        path_abl = compute_avg_path_length(adj_abl, n_input, n_output)
        print(f"  密度: {baseline_density:.4f} -> {density_abl:.4f}")
        if np.isfinite(path_abl):
            print(f"  平均路径长度: {baseline_path:.2f} -> {path_abl:.2f}")
        else:
            print(f"  平均路径长度: {baseline_path:.2f} -> inf (不连通)")

        results['conditions'].append({
            'name': name,
            'n_edges': n_edges,
            'score_ablated': score_abl,
            'drop_specific': drop_specific,
            'drop_random_mean': drop_random_mean,
            'drop_random_std': drop_random_std,
            'specificity': specificity,
            'random_drops': random_drops,
            'density_baseline': baseline_density,
            'density_ablated': density_abl,
            'path_baseline': baseline_path,
            'path_ablated': path_abl,
        })

    # ── 汇总表 ──
    print("\n" + "=" * 60)
    print("汇总表:")
    print(f"{'条件':<12} {'消融后':>8} {'Drop':>8} {'随机Drop':>14} {'特异性指数':>10}")
    print("-" * 60)
    for cond in results['conditions']:
        name = cond['name']
        score = cond['score_ablated']
        drop = cond['drop_specific']
        if cond['n_edges'] > 0 and name in ['E→E', 'I→E', 'E→I', 'I→I']:
            rand_str = f"{cond['drop_random_mean']:.1f}±{cond['drop_random_std']:.1f}"
            spec_str = f"{cond['specificity']:.2f}"
        else:
            rand_str = "N/A"
            spec_str = "N/A"
        print(f"{name:<12} {score:>8.1f} {drop:>8.1f} {rand_str:>14} {spec_str:>10}")
    print("=" * 60)

    return results


def plot_ablation_results(results, save_path='../results/ablation/connection_type_ablation.png'):
    """绘制消融对比图（Performance Drop）。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    conditions = [c['name'] for c in results['conditions']]
    specific_drops = [c['drop_specific'] for c in results['conditions']]
    random_means = [c['drop_random_mean'] for c in results['conditions']]
    random_stds = [c['drop_random_std'] for c in results['conditions']]

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # Specific Ablation柱子：前 4 个左偏，后 3 个居中
    red_x = x.astype(float).copy()
    red_x[:4] -= width / 2
    bars1 = ax.bar(red_x, specific_drops, width, label='Specific Ablation',
                   color='#E74C3C', edgecolor='black', zorder=3)

    # 随机消融柱子：仅前 4 个条件
    blue_x = x[:4] + width / 2
    blue_vals = random_means[:4]
    blue_errs = random_stds[:4]
    bars2 = ax.bar(blue_x, blue_vals, width, yerr=blue_errs,
                   label='Random Control (mean±std)', color='#3498DB',
                   edgecolor='black', capsize=4, zorder=3)

    ax.set_ylabel('Performance Drop', fontsize=12)
    ax.set_title('Systematic Connection-Type Ablation', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, zorder=0)

    # Value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nAblation figure saved: {save_path}")
    plt.close()


def save_csv(results, save_path='../results/ablation/ablation_summary.csv'):
    """Save CSV summary。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            '条件', '消融后边数', '消融后性能', 'Drop',
            '随机Drop均值', '随机Drop标准差', '特异性指数',
            '原始密度', '消融后密度', '原始路径长度', '消融后路径长度'
        ])
        for c in results['conditions']:
            has_rand = c['n_edges'] > 0 and c['name'] in ['E→E', 'I→E', 'E→I', 'I→I']
            writer.writerow([
                c['name'],
                c['n_edges'],
                f"{c['score_ablated']:.2f}",
                f"{c['drop_specific']:.2f}",
                f"{c['drop_random_mean']:.2f}" if has_rand else "N/A",
                f"{c['drop_random_std']:.2f}" if has_rand else "N/A",
                f"{c['specificity']:.2f}" if has_rand else "N/A",
                f"{c['density_baseline']:.4f}",
                f"{c['density_ablated']:.4f}",
                f"{c['path_baseline']:.2f}" if np.isfinite(c['path_baseline']) else "inf",
                f"{c['path_ablated']:.2f}" if np.isfinite(c['path_ablated']) else "inf",
            ])
    print(f"汇总 CSV 已保存: {save_path}")


def save_npy(results, save_path='../results/ablation/ablation_results.npy'):
    """保存完整结果（供后续分析）。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, results, allow_pickle=True)
    print(f"完整结果已保存: {save_path}")


if __name__ == '__main__':
    config = {
        'n_total': 128,
        'n_input': 8,
        'n_output': 2,
        'sparsity': 0.41,
        'weight_scale': 0.6,
        'tau': 5.0,
        'threshold': 0.8,
        'T_steps': 20,
        'dt': 1.0,
        'n_eval_episodes': 20,
    }

    adj_path = '../../../data/key_npy/best_topology.npy'
    types_path = '../../../data/key_npy/neuron_types.npy'

    if not os.path.exists(adj_path) or not os.path.exists(types_path):
        print(f"错误: 找不到 {adj_path} 或 {types_path}")
        print("请先运行主实验 evolution.py 生成最优拓扑")
        sys.exit(1)

    results = run_connection_ablation(adj_path, types_path, config)
    plot_ablation_results(results)
    save_csv(results)
    save_npy(results)

    print("\n实验 B1 完成")
