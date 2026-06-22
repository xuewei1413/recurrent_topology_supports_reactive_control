"""
实验 1.1: Motif 谱与随机基线对比
统计最优拓扑中 E→E, E→I, I→E, I→I 的 motif 密度，
与同规模随机网络对比，计算 z-score。
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体（如果可用）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_topology(adj_path, types_path):
    adj = np.load(adj_path)
    types = np.load(types_path)
    return adj, types


def count_motifs(adj, types):
    """统计四种 motif 的连接数"""
    exc_idx = np.where(types == 1)[0]
    inh_idx = np.where(types == -1)[0]
    
    ee = int(adj[np.ix_(exc_idx, exc_idx)].sum())
    ei = int(adj[np.ix_(exc_idx, inh_idx)].sum())
    ie = int(adj[np.ix_(inh_idx, exc_idx)].sum())
    ii = int(adj[np.ix_(inh_idx, inh_idx)].sum())
    
    n_exc = len(exc_idx)
    n_inh = len(inh_idx)
    n = len(types)
    
    # 密度 = 实际连接数 / 该类型子图最大可能连接数
    max_ee = n_exc * n_exc
    max_ei = n_exc * n_inh
    max_ie = n_inh * n_exc
    max_ii = n_inh * n_inh
    
    densities = {
        'E→E': ee / max_ee if max_ee > 0 else 0,
        'E→I': ei / max_ei if max_ei > 0 else 0,
        'I→E': ie / max_ie if max_ie > 0 else 0,
        'I→I': ii / max_ii if max_ii > 0 else 0,
    }
    counts = {'E→E': ee, 'E→I': ei, 'I→E': ie, 'I→I': ii}
    return densities, counts, (n_exc, n_inh)


def generate_random_network(n, n_exc, n_inh, total_conn, n_samples=1000):
    """
    生成随机网络基线：保持总连接数、类型分布不变，
    边随机分配到四种 motif 中。
    """
    exc_idx = np.arange(n_exc)
    inh_idx = np.arange(n_exc, n_exc + n_inh)
    
    max_ee = n_exc * n_exc
    max_ei = n_exc * n_inh
    max_ie = n_inh * n_exc
    max_ii = n_inh * n_inh
    max_total = max_ee + max_ei + max_ie + max_ii
    
    # 每种 motif 的最大可能连接比例
    probs = np.array([max_ee, max_ei, max_ie, max_ii]) / max_total
    
    all_densities = {k: [] for k in ['E→E', 'E→I', 'I→E', 'I→I']}
    
    for _ in range(n_samples):
        # 按概率分配总连接数到四种 motif
        counts = np.random.multinomial(total_conn, probs)
        
        # 生成每个 motif 的密度
        d_ee = counts[0] / max_ee if max_ee > 0 else 0
        d_ei = counts[1] / max_ei if max_ei > 0 else 0
        d_ie = counts[2] / max_ie if max_ie > 0 else 0
        d_ii = counts[3] / max_ii if max_ii > 0 else 0
        
        all_densities['E→E'].append(d_ee)
        all_densities['E→I'].append(d_ei)
        all_densities['I→E'].append(d_ie)
        all_densities['I→I'].append(d_ii)
    
    stats = {}
    for k in all_densities:
        arr = np.array(all_densities[k])
        stats[k] = {'mean': arr.mean(), 'std': arr.std()}
    return stats


def analyze_task(name, adj_path, types_path, n_random=1000):
    print(f"\n{'='*50}")
    print(f"任务: {name}")
    print(f"{'='*50}")
    
    adj, types = load_topology(adj_path, types_path)
    densities, counts, (n_exc, n_inh) = count_motifs(adj, types)
    n = len(types)
    total_conn = int(adj.sum())
    
    print(f"网络规模: {n} (E={n_exc}, I={n_inh})")
    print(f"总连接数: {total_conn}, 稀疏度: {total_conn/(n*n):.4f}")
    
    for k in densities:
        print(f"  {k}: count={counts[k]}, density={densities[k]:.4f}")
    
    # 随机基线
    random_stats = generate_random_network(n, n_exc, n_inh, total_conn, n_random)
    
    print(f"\n随机基线 (n={n_random}):")
    z_scores = {}
    for k in densities:
        m = random_stats[k]['mean']
        s = random_stats[k]['std']
        z = (densities[k] - m) / s if s > 0 else 0
        z_scores[k] = z
        print(f"  {k}: random_mean={m:.4f}, random_std={s:.4f}, z={z:+.3f}")
    
    return densities, z_scores, counts


def plot_motif_enrichment(results, save_path='../results/motif_enrichment.png'):
    """绘制 motif 富集图"""
    tasks = list(results.keys())
    motifs = ['E→E', 'E→I', 'I→E', 'I→I']
    
    fig, axes = plt.subplots(1, len(tasks), figsize=(4*len(tasks), 5), sharey=True)
    if len(tasks) == 1:
        axes = [axes]
    
    colors = {'E→E': '#E74C3C', 'E→I': '#3498DB', 'I→E': '#2ECC71', 'I→I': '#9B59B6'}
    
    for ax, task in zip(axes, tasks):
        densities = results[task]['densities']
        z_scores = results[task]['z_scores']
        
        x = np.arange(len(motifs))
        y = [densities[m] for m in motifs]
        z = [z_scores[m] for m in motifs]
        c = [colors[m] for m in motifs]
        
        bars = ax.bar(x, y, color=c, edgecolor='black', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(motifs, fontsize=11)
        ax.set_ylabel('Connection Density', fontsize=12)
        ax.set_title(f'{task}\nEvolved Topology', fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(y)*1.3 if max(y) > 0 else 1)
        ax.axhline(y=results[task]['random_mean']['E→E'], color='gray', 
                   linestyle='--', alpha=0.5, label='Random baseline')
        
        # 在柱子上标注 z-score
        for i, (bar, zi) in enumerate(zip(bars, z)):
            height = bar.get_height()
            ax.annotate(f'z={zi:+.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nMotif 富集图已保存: {save_path}")
    plt.close()


def plot_zscore_comparison(results, save_path='../results/motif_zscore_comparison.png'):
    """绘制跨任务 z-score 对比图"""
    tasks = list(results.keys())
    motifs = ['E→E', 'E→I', 'I→E', 'I→I']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(motifs))
    width = 0.18
    colors_task = plt.cm.tab10(np.linspace(0, 1, len(tasks)))
    
    for i, task in enumerate(tasks):
        z_scores = results[task]['z_scores']
        y = [z_scores[m] for m in motifs]
        offset = width * (i - len(tasks)/2 + 0.5)
        ax.bar(x + offset, y, width, label=task, color=colors_task[i], edgecolor='black')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(motifs, fontsize=12)
    ax.set_ylabel('Z-score (vs Random)', fontsize=12)
    ax.set_title('Motif Enrichment: Evolved vs Random Networks', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Z-score 对比图已保存: {save_path}")
    plt.close()


if __name__ == '__main__':
    # 分析所有任务
    tasks = {
        'CartPole': ('../../../data/key_npy/best_topology.npy', '../../../data/key_npy/neuron_types.npy'),
        'MountainCar': ('../../../data/key_npy/mc_best_topology.npy', '../../../data/key_npy/mc_neuron_types.npy'),
        'FlappyBird': ('../../../data/key_npy/flappy_best_topology.npy', '../../../data/key_npy/flappy_neuron_types.npy'),
        'Snake': ('../../../data/key_npy/snake_best_topology.npy', '../../../data/key_npy/snake_neuron_types.npy'),
    }
    
    results = {}
    for task_name, (adj_path, types_path) in tasks.items():
        if os.path.exists(adj_path) and os.path.exists(types_path):
            densities, z_scores, counts = analyze_task(task_name, adj_path, types_path, n_random=2000)
            # 计算随机基线的平均密度（用于画图参考线）
            n = len(np.load(types_path))
            n_exc = (np.load(types_path) == 1).sum()
            n_inh = (np.load(types_path) == -1).sum()
            total_conn = int(np.load(adj_path).sum())
            rand_stats = generate_random_network(n, n_exc, n_inh, total_conn, 100)
            rand_mean = {k: rand_stats[k]['mean'] for k in rand_stats}
            results[task_name] = {
                'densities': densities,
                'z_scores': z_scores,
                'counts': counts,
                'random_mean': rand_mean
            }
        else:
            print(f"跳过 {task_name}: 文件不存在")
    
    # 绘图
    plot_motif_enrichment(results)
    plot_zscore_comparison(results)
    
    print("\n" + "="*50)
    print("实验 1.1 完成")
    print("="*50)
