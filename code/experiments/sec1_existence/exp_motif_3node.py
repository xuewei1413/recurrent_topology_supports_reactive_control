"""
实验 1.1b: 三节点 Motif (Triadic Motif) 谱分析
枚举所有有序三元组，提取 6-bit 边模式 + 3-bit 类型模式，
与随机网络对比计算 z-score。
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os, time

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def count_triad_motifs(adj, types):
    """
    统计所有有序三元组 (i,j,k) 的 motif 频率。
    边模式: 6-bit 编码 [i->j, j->i, i->k, k->i, j->k, k->j]
    类型模式: 3-bit 编码 [type_i, type_j, type_k], I=1, E=0
    返回: counts[512], 以及 motif 描述字典
    """
    n = len(types)
    type_bits = (types == -1).astype(np.int32)  # I=1, E=0
    counts = np.zeros(512, dtype=np.int64)
    
    for i in range(n):
        ti = type_bits[i]
        row_i = adj[i, :]
        col_i = adj[:, i]
        for j in range(n):
            if i == j:
                continue
            tj = type_bits[j]
            a_ij = int(row_i[j])
            a_ji = int(col_i[j])
            row_j = adj[j, :]
            col_j = adj[:, j]
            for k in range(n):
                if k == i or k == j:
                    continue
                tk = type_bits[k]
                # 6-bit edge pattern
                pat = (a_ij |
                       (a_ji << 1) |
                       (int(row_i[k]) << 2) |
                       (int(col_i[k]) << 3) |
                       (int(row_j[k]) << 4) |
                       (int(col_j[k]) << 5))
                # 3-bit type pattern
                tpat = ti | (tj << 1) | (tk << 2)
                idx = pat * 8 + tpat
                counts[idx] += 1
    return counts


def generate_random_network(n, total_conn):
    """生成随机有向网络，保持总连接数"""
    adj = np.zeros((n, n), dtype=np.float64)
    indices = np.random.choice(n * n, size=total_conn, replace=False)
    adj.flat[indices] = 1.0
    return adj


def motif_index_to_description(idx):
    """将 motif 索引转换为人类可读描述"""
    pat = idx // 8
    tpat = idx % 8
    
    # 边模式
    edges = []
    edge_names = ['i->j', 'j->i', 'i->k', 'k->i', 'j->k', 'k->j']
    for bit in range(6):
        if pat & (1 << bit):
            edges.append(edge_names[bit])
    edge_str = ', '.join(edges) if edges else 'none'
    n_edges = len(edges)
    
    # 类型模式
    types = []
    type_names = ['i', 'j', 'k']
    for bit in range(3):
        t = 'I' if (tpat >> bit) & 1 else 'E'
        types.append(f"{type_names[bit]}={t}")
    type_str = ', '.join(types)
    
    # 结构分类
    if n_edges == 0:
        struct = "empty"
    elif n_edges == 1:
        struct = "single_edge"
    elif n_edges == 2:
        # 判断两条边的关系
        e = [i for i in range(6) if pat & (1 << i)]
        if e == [0, 1]:
            struct = "mutual_ij"
        elif e == [2, 3]:
            struct = "mutual_ik"
        elif e == [4, 5]:
            struct = "mutual_jk"
        elif (0 in e and 4 in e) or (0 in e and 5 in e) or (1 in e and 4 in e) or (1 in e and 5 in e):
            struct = "chain_ij_jk"
        elif (2 in e and 5 in e) or (3 in e and 4 in e):
            struct = "chain_ik_kj"
        else:
            struct = "two_edges_unconnected"
    elif n_edges == 3:
        # 检查是否有闭环
        # i->j->k->i: 需要 0,4,3 或 1,5,2
        if (pat & 1) and (pat & 16) and (pat & 8):
            struct = "cycle_ijk"
        elif (pat & 2) and (pat & 32) and (pat & 4):
            struct = "cycle_ikj"
        else:
            struct = "three_edges_open"
    elif n_edges == 4:
        struct = "four_edges"
    elif n_edges == 5:
        struct = "five_edges"
    else:
        struct = "fully_connected"
    
    return f"[{struct}] {type_str} | edges: {edge_str}"


def analyze_triadic_motifs(task_name, adj_path, types_path, n_random=100):
    print(f"\n{'='*60}")
    print(f"Triadic Motif Analysis: {task_name}")
    print(f"{'='*60}")
    
    adj = np.load(adj_path)
    types = np.load(types_path)
    n = len(types)
    total_conn = int(adj.sum())
    
    print(f"Network: N={n}, conn={total_conn}, sparse={total_conn/(n*n):.4f}")
    print("Counting evolved motifs...")
    t0 = time.time()
    evolved_counts = count_triad_motifs(adj, types)
    print(f"  Done in {time.time()-t0:.2f}s, total triples={evolved_counts.sum()}")
    
    # 随机基线
    print(f"Generating {n_random} random networks...")
    random_counts = np.zeros((n_random, 512), dtype=np.float64)
    for r in range(n_random):
        rand_adj = generate_random_network(n, total_conn)
        random_counts[r, :] = count_triad_motifs(rand_adj, types)
    
    random_mean = random_counts.mean(axis=0)
    random_std = random_counts.std(axis=0)
    
    # 计算 z-score
    z_scores = np.zeros(512)
    for idx in range(512):
        if random_std[idx] > 0:
            z_scores[idx] = (evolved_counts[idx] - random_mean[idx]) / random_std[idx]
        else:
            z_scores[idx] = 0
    
    # 筛选显著模式
    significant = []
    for idx in range(512):
        if abs(z_scores[idx]) >= 2.0 and evolved_counts[idx] > 0:
            significant.append({
                'idx': idx,
                'evolved': int(evolved_counts[idx]),
                'random_mean': random_mean[idx],
                'random_std': random_std[idx],
                'z': z_scores[idx],
                'desc': motif_index_to_description(idx)
            })
    
    significant.sort(key=lambda x: abs(x['z']), reverse=True)
    
    print(f"\nSignificant motifs (|z| >= 2.0): {len(significant)}")
    print("-" * 80)
    for m in significant[:20]:
        print(f"z={m['z']:+.2f} | evolved={m['evolved']} | random={m['random_mean']:.1f}±{m['random_std']:.1f}")
        print(f"  {m['desc']}")
    
    return {
        'evolved_counts': evolved_counts,
        'random_mean': random_mean,
        'random_std': random_std,
        'z_scores': z_scores,
        'significant': significant
    }


def plot_triadic_zscore(results, save_path='../results/triadic_zscore.png'):
    """绘制三节点 motif z-score 分布"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (task_name, data) in enumerate(results.items()):
        ax = axes[idx]
        z = data['z_scores']
        
        # 只画非零的
        mask = data['evolved_counts'] > 0
        z_nonzero = z[mask]
        
        ax.hist(z_nonzero, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(x=2, color='red', linestyle='--', label='|z|=2')
        ax.axvline(x=-2, color='red', linestyle='--')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Z-score', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{task_name} (n={len(z_nonzero)} motifs)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle('Triadic Motif Z-score Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nZ-score 分布图已保存: {save_path}")
    plt.close()


def plot_top_motifs(results, top_n=10, save_path='../results/triadic_top_motifs.png'):
    """绘制每个任务 top 富集/贫乏的 motif"""
    n_tasks = len(results)
    fig, axes = plt.subplots(1, n_tasks, figsize=(5*n_tasks, 6), sharey=True)
    if n_tasks == 1:
        axes = [axes]
    
    for ax, (task_name, data) in zip(axes, results.items()):
        sig = data['significant'][:top_n]
        if not sig:
            ax.text(0.5, 0.5, 'No significant motifs', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(task_name)
            continue
        
        labels = [f"M{s['idx']}" for s in sig]
        z_vals = [s['z'] for s in sig]
        colors = ['#E74C3C' if z > 0 else '#3498DB' for z in z_vals]
        
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, z_vals, color=colors, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Z-score', fontsize=11)
        ax.set_title(f'{task_name}\nTop {top_n} Significant', fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.axvline(x=2, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=-2, color='red', linestyle='--', alpha=0.5)
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Top motifs 图已保存: {save_path}")
    plt.close()


if __name__ == '__main__':
    tasks = {
        'CartPole': ('../../../data/key_npy/best_topology.npy', '../../../data/key_npy/neuron_types.npy'),
        'MountainCar': ('../../../data/key_npy/mc_best_topology.npy', '../../../data/key_npy/mc_neuron_types.npy'),
        'FlappyBird': ('../../../data/key_npy/flappy_best_topology.npy', '../../../data/key_npy/flappy_neuron_types.npy'),
        'Snake': ('../../../data/key_npy/snake_best_topology.npy', '../../../data/key_npy/snake_neuron_types.npy'),
    }
    
    results = {}
    for task_name, (adj_path, types_path) in tasks.items():
        if os.path.exists(adj_path) and os.path.exists(types_path):
            data = analyze_triadic_motifs(task_name, adj_path, types_path, n_random=50)
            results[task_name] = data
        else:
            print(f"跳过 {task_name}: 文件不存在")
    
    # 保存数据
    np.save('../results/triadic_motif_analysis.npy', results, allow_pickle=True)
    
    # 绘图
    plot_triadic_zscore(results)
    plot_top_motifs(results, top_n=15)
    
    print("\n" + "="*60)
    print("实验 1.1b (三节点 Motif) 完成")
    print("="*60)
