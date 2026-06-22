"""
实验 1B：跨任务结构指纹分析

目标：验证"不同任务诱导出的最优拓扑具有可区分的结构指纹"。

方法：
  1. 收集 5 个任务的最优拓扑（每个任务 10 个种子，共 50 网络）
  2. 提取与网络规模无关的归一化结构特征（19 维）
  3. 判别分析：KNN 留一法分类任务类别
  4. 无监督聚类：t-SNE + 层次聚类纯度分析
  5. 相似度相关：结构距离 vs 任务距离（Mantel 检验）

输出：
  - results/fingerprint_analysis/fingerprint_tsne.png
  - results/fingerprint_analysis/confusion_matrix.png
  - results/fingerprint_analysis/feature_importance.png
  - results/fingerprint_analysis/report.txt
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np

try:
    import threadpoolctl
    # Patch _OpenBLASModule to skip broken OpenBLAS queries
    class _PatchedOpenBLASModule(threadpoolctl._OpenBLASModule):
        def get_version(self):
            return None
        def get_num_threads(self):
            return 1
        def set_num_threads(self, num):
            pass
    threadpoolctl._OpenBLASModule = _PatchedOpenBLASModule
except Exception:
    pass
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import pearsonr
from collections import Counter
import os, sys, json, warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# ── 配置 ──────────────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
OUT_DIR = os.path.join(RESULTS_DIR, 'fingerprint_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

# 任务元数据（从已有实验代码归纳）
TASK_META = {
    'CartPole': {
        'n_input': 8, 'n_output': 2, 'state_dim': 4, 'action_dim': 2,
        'discrete': 1, 'time_scale': 500, 'control_type': 'balance',
        'network_size': 128,
    },
    'MountainCar': {
        'n_input': 4, 'n_output': 3, 'state_dim': 2, 'action_dim': 3,
        'discrete': 1, 'time_scale': 200, 'control_type': 'propulsion',
        'network_size': 64,
    },
    'Acrobot': {
        'n_input': 12, 'n_output': 3, 'state_dim': 6, 'action_dim': 3,
        'discrete': 1, 'time_scale': 500, 'control_type': 'swing-up',
        'network_size': 128,
    },
    'FlappyBird': {
        'n_input': 4, 'n_output': 2, 'state_dim': 2, 'action_dim': 2,
        'discrete': 1, 'time_scale': 9999, 'control_type': 'avoidance',
        'network_size': 64,
    },
    'Snake': {
        'n_input': 12, 'n_output': 4, 'state_dim': 6, 'action_dim': 4,
        'discrete': 1, 'time_scale': 9999, 'control_type': 'navigation',
        'network_size': 80,
    },
}

CONTROL_TYPE_ENCODE = {'balance': 0, 'propulsion': 1, 'swing-up': 2, 'avoidance': 3, 'navigation': 4}


# ── 数据加载 ──────────────────────────────────────────────────────────────

def load_all_networks():
    """加载 cross_task 目录下的所有拓扑（5 任务 × 10 种子 = 50 网络）。"""
    networks = []

    cross_dir = os.path.join(RESULTS_DIR, 'cross_task')
    if os.path.isdir(cross_dir):
        for fname in sorted(os.listdir(cross_dir)):
            if not fname.endswith('.npz'):
                continue
            data = np.load(os.path.join(cross_dir, fname))
            task = str(data['task'])
            if task not in TASK_META:
                continue
            meta = TASK_META[task]
            networks.append({
                'name': fname.replace('.npz', ''),
                'task': task,
                'seed': int(data['seed']),
                'adj': data['adj'],
                'types': data['types'],
                'n_input': meta['n_input'],
                'n_output': meta['n_output'],
                'fitness': float(data['best_fitness']),
            })

    print(f"Loaded {len(networks)} networks from {len(set(n['task'] for n in networks))} tasks.")
    for task in sorted(set(n['task'] for n in networks)):
        count = sum(1 for n in networks if n['task'] == task)
        print(f"  {task}: {count} networks")
    return networks


# ── 特征提取 ──────────────────────────────────────────────────────────────

def directed_clustering_coeff(adj):
    """有向网络全局聚类系数（基于三元组）。"""
    n = adj.shape[0]
    adj_bin = (adj > 0).astype(int)
    C = 0.0
    count = 0
    for i in range(n):
        neighbors = np.where(adj_bin[i] | adj_bin[:, i])[0]
        if len(neighbors) < 2:
            continue
        for j in neighbors:
            for k in neighbors:
                if j >= k:
                    continue
                # 三角形计数：i-j, i-k, j-k 任意方向
                tri = adj_bin[i, j] + adj_bin[j, i] + adj_bin[i, k] + adj_bin[k, i] + adj_bin[j, k] + adj_bin[k, j]
                if tri >= 3:
                    C += 1.0
        count += len(neighbors) * (len(neighbors) - 1) // 2
    return C / max(count, 1)


def characteristic_path_length(adj):
    """基于 BFS 的平均最短路径长度（有向图，弱连通分量）。"""
    n = adj.shape[0]
    adj_bin = (adj > 0).astype(int)
    total_len = 0
    total_pairs = 0
    for src in range(n):
        dist = np.full(n, -1)
        dist[src] = 0
        queue = [src]
        while queue:
            u = queue.pop(0)
            for v in np.where(adj_bin[u])[0]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        valid = dist[dist > 0]
        if len(valid) > 0:
            total_len += valid.sum()
            total_pairs += len(valid)
    return total_len / max(total_pairs, 1)


def modularity_ei(adj, types):
    """以 E/I 类型作为社团的模块化指数。"""
    n = adj.shape[0]
    m = adj.sum()
    if m == 0:
        return 0.0
    communities = (types == -1).astype(int)  # 0=E, 1=I
    Q = 0.0
    for i in range(n):
        for j in range(n):
            if communities[i] == communities[j]:
                k_i_out = adj[i].sum()
                k_j_in = adj[:, j].sum()
                Q += (adj[i, j] - k_i_out * k_j_in / m)
    return Q / m


def degree_entropy(adj):
    """度分布的香农熵（入度+出度）。"""
    n = adj.shape[0]
    degrees = adj.sum(axis=0) + adj.sum(axis=1)
    probs = degrees / degrees.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def assortativity(adj, types):
    """基于 E/I 类型的同配性：同类神经元是否更倾向于相互连接。"""
    n = adj.shape[0]
    same_type_edges = 0
    total_edges = 0
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                total_edges += 1
                if types[i] == types[j]:
                    same_type_edges += 1
    if total_edges == 0:
        return 0.0
    p_same = same_type_edges / total_edges
    # 随机期望：P(E)*P(E) + P(I)*P(I)
    p_e = (types == 1).sum() / n
    p_i = 1 - p_e
    p_random = p_e * p_e + p_i * p_i
    return (p_same - p_random) / max(1 - p_random, 1e-6)


def information_transfer_efficiency(adj):
    """信息传递效率 = 所有可达节点对的最短路径长度倒数的平均值。"""
    n = adj.shape[0]
    adj_bin = (adj > 0).astype(int)
    eff = 0.0
    count = 0
    for src in range(n):
        dist = np.full(n, -1)
        dist[src] = 0
        queue = [src]
        while queue:
            u = queue.pop(0)
            for v in np.where(adj_bin[u])[0]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for d in dist:
            if d > 0:
                eff += 1.0 / d
                count += 1
    return eff / max(count, 1)


def extract_features(net):
    """提取单个网络的归一化结构特征向量。"""
    adj = net['adj']
    types = net['types']
    n = adj.shape[0]
    n_in = net['n_input']
    n_out = net['n_output']
    hidden = np.arange(n_in, n - n_out)

    exc = types == 1
    inh = types == -1
    total_conn = adj.sum()

    f = {}

    # ── 全局图论 ──
    f['density'] = total_conn / (n * (n - 1)) if n > 1 else 0.0
    f['clustering'] = directed_clustering_coeff(adj)
    f['path_length'] = characteristic_path_length(adj) / n  # 归一化到 [0,1]
    f['modularity_ei'] = modularity_ei(adj, types)
    f['degree_entropy'] = degree_entropy(adj) / np.log2(n) if n > 1 else 0.0
    f['assortativity'] = assortativity(adj, types)

    # ── E/I 结构 ──
    f['ei_ratio'] = exc.sum() / n
    f['ee_ratio'] = adj[exc][:, exc].sum() / max(total_conn, 1)
    f['ei_ratio_conn'] = adj[exc][:, inh].sum() / max(total_conn, 1)
    f['ie_ratio'] = adj[inh][:, exc].sum() / max(total_conn, 1)
    f['ii_ratio'] = adj[inh][:, inh].sum() / max(total_conn, 1)
    f['exc_indeg'] = adj[:, exc].sum(axis=0).mean() / max(n, 1)
    f['inh_indeg'] = adj[:, inh].sum(axis=0).mean() / max(n, 1)

    # ── 层连接 ──
    if len(hidden) > 0:
        f['in_to_hidden'] = adj[hidden[:, None], np.arange(n_in)].sum() / max(len(hidden) * n_in, 1)
        f['hidden_to_out'] = adj[np.arange(n - n_out, n)[:, None], hidden].sum() / max(n_out * len(hidden), 1)
        f['hidden_internal'] = adj[hidden[:, None], hidden].sum() / max(len(hidden) ** 2, 1)
    else:
        f['in_to_hidden'] = f['hidden_to_out'] = f['hidden_internal'] = 0.0

    # ── 动态 ──
    f['info_efficiency'] = information_transfer_efficiency(adj)
    f['inv_path_len'] = 1.0 / max(f['path_length'] * n, 1e-6)

    # ── 规模无关的稀疏度 ──
    f['sparsity'] = 1.0 - f['density']

    return f


# ── 分析 ──────────────────────────────────────────────────────────────────

def build_feature_matrix(networks):
    """返回特征矩阵 X (n_networks × n_features) 和标签 y。"""
    feature_dicts = [extract_features(n) for n in networks]
    feature_names = sorted(feature_dicts[0].keys())
    X = np.array([[fd[k] for k in feature_names] for fd in feature_dicts])
    y = np.array([n['task'] for n in networks])
    return X, y, feature_names, networks


def knn_loo_accuracy(X, y, k=3):
    """留一法 KNN 分类准确率。"""
    from collections import Counter
    correct = 0
    for i in range(len(X)):
        train_X = np.delete(X, i, axis=0)
        train_y = np.delete(y, i, axis=0)
        dists = np.linalg.norm(train_X - X[i], axis=1)
        nearest = np.argsort(dists)[:k]
        votes = Counter(train_y[nearest])
        pred = votes.most_common(1)[0][0]
        if pred == y[i]:
            correct += 1
    return correct / len(X)


def compute_task_distance_matrix(networks):
    """基于任务元数据的手工距离矩阵。"""
    tasks = sorted(set(n['task'] for n in networks))
    task_to_idx = {t: i for i, t in enumerate(tasks)}
    n_tasks = len(tasks)

    # 数值特征
    numeric = ['state_dim', 'action_dim', 'time_scale', 'network_size']
    cat = ['control_type']

    feat = {}
    for t in tasks:
        m = TASK_META[t]
        vec = [m[k] for k in numeric]
        vec.append(CONTROL_TYPE_ENCODE[m['control_type']])
        feat[t] = np.array(vec, dtype=float)

    # 标准化
    all_vecs = np.array([feat[t] for t in tasks])
    mean = all_vecs.mean(axis=0)
    std = all_vecs.std(axis=0)
    std[std == 0] = 1.0
    for t in tasks:
        feat[t] = (feat[t] - mean) / std

    D = np.zeros((n_tasks, n_tasks))
    for i, ti in enumerate(tasks):
        for j, tj in enumerate(tasks):
            D[i, j] = np.linalg.norm(feat[ti] - feat[tj])
    return D, tasks


def mantel_test(struct_dist, task_dist, n_permutations=1000):
    """Mantel 检验：两个距离矩阵的相关性。"""
    # 展平上三角
    triu_idx = np.triu_indices_from(struct_dist, k=1)
    s = struct_dist[triu_idx]
    t = task_dist[triu_idx]
    r_obs, _ = pearsonr(s, t)

    count = 0
    for _ in range(n_permutations):
        perm = np.random.permutation(len(s))
        r_perm, _ = pearsonr(s[perm], t)
        if abs(r_perm) >= abs(r_obs):
            count += 1
    p_value = count / n_permutations
    return r_obs, p_value


# ── 可视化 ────────────────────────────────────────────────────────────────

def plot_tsne(X, y, networks, out_dir):
    """t-SNE 降维可视化。"""
    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("[Warning] scikit-learn not installed. Skipping t-SNE plot.")
        return

    X_std = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=min(10, len(X)-1), random_state=42, init='pca')
    X2 = tsne.fit_transform(X_std)

    tasks = sorted(set(y))
    colors = plt.cm.tab10(np.linspace(0, 1, len(tasks)))
    task_color = {t: colors[i] for i, t in enumerate(tasks)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for task in tasks:
        mask = y == task
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   c=[task_color[task]], label=task, s=120, alpha=0.8, edgecolors='black')
        # 标注种子号
        for i in np.where(mask)[0]:
            ax.annotate(str(networks[i]['seed']), (X2[i, 0], X2[i, 1]), fontsize=7, alpha=0.6)

    ax.set_title('Cross-Task Structural Fingerprint (t-SNE)', fontsize=14)
    ax.legend(title='Task', loc='best')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    fig_path = os.path.join(out_dir, 'fingerprint_tsne.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")


def plot_confusion_matrix(y_true, y_pred, tasks, out_dir):
    """绘制混淆矩阵。"""
    n = len(tasks)
    cm = np.zeros((n, n), dtype=int)
    task_idx = {t: i for i, t in enumerate(tasks)}
    for yt, yp in zip(y_true, y_pred):
        cm[task_idx[yt], task_idx[yp]] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tasks, rotation=45, ha='right')
    ax.set_yticklabels(tasks)
    ax.set_xlabel('Predicted Task')
    ax.set_ylabel('True Task')
    ax.set_title('Structural Fingerprint Classification (LOO-KNN)')

    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black', fontsize=12)

    fig_path = os.path.join(out_dir, 'confusion_matrix.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")


def plot_feature_importance(X, y, feature_names, out_dir):
    """基于随机森林的特征重要性。"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("[Warning] scikit-learn not installed. Skipping feature importance plot.")
        return

    X_std = StandardScaler().fit_transform(X)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_std, y)
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, importances[order], color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title('Structural Feature Importance (Random Forest)')
    fig_path = os.path.join(out_dir, 'feature_importance.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")


def plot_similarity_heatmaps(struct_dist, task_dist, tasks, out_dir):
    """结构距离与任务距离的对比热图。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, dist_mat, title in [(axes[0], struct_dist, 'Structural Fingerprint Distance'),
                                  (axes[1], task_dist, 'Task Metadata Distance')]:
        im = ax.imshow(dist_mat, cmap='viridis')
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(tasks)))
        ax.set_yticks(range(len(tasks)))
        ax.set_xticklabels(tasks, rotation=45, ha='right')
        ax.set_yticklabels(tasks)
        ax.set_title(title)

    fig_path = os.path.join(out_dir, 'similarity_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")


# ── 主函数 ────────────────────────────────────────────────────────────────

def main():
    print("="*70)
    print("  实验 1B: 跨任务结构指纹分析")
    print("="*70)

    # 1. 加载数据
    print("\n[Step 1] Loading networks...")
    networks = load_all_networks()
    if len(networks) < 5:
        print("[Error] Not enough networks. Need at least 5.")
        return

    # 2. 提取特征
    print("\n[Step 2] Extracting structural features...")
    X, y, feature_names, networks = build_feature_matrix(networks)
    print(f"  Feature matrix: {X.shape}")
    print(f"  Features: {', '.join(feature_names)}")

    # 3. 标准化
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std

    # 4. 判别分析：留一法 KNN
    print("\n[Step 3] Discriminant analysis (LOO-KNN)...")
    acc = knn_loo_accuracy(X_norm, y, k=min(3, len(networks)-1))
    n_tasks = len(set(y))
    random_baseline = 1.0 / n_tasks
    print(f"  LOO-KNN accuracy: {acc:.3f} (random baseline: {random_baseline:.3f})")
    print(f"  Lift over random: {acc/random_baseline:.2f}x")

    # 5. 无监督聚类纯度
    print("\n[Step 4] Unsupervised clustering purity...")
    Z = linkage(X_norm, method='ward')
    cluster_labels = fcluster(Z, n_tasks, criterion='maxclust')
    # 计算纯度：每个簇中最频繁任务的比例
    purity_sum = 0
    for cid in range(1, n_tasks + 1):
        members = [y[i] for i in range(len(y)) if cluster_labels[i] == cid]
        if members:
            most_common = Counter(members).most_common(1)[0][1]
            purity_sum += most_common
    purity = purity_sum / len(y)
    print(f"  Hierarchical clustering purity (K={n_tasks}): {purity:.3f}")

    # 6. 结构距离 vs 任务距离
    print("\n[Step 5] Structural distance vs Task distance...")
    tasks = sorted(set(y))
    # 每个任务的平均指纹
    task_fingerprints = {}
    for t in tasks:
        mask = y == t
        task_fingerprints[t] = X_norm[mask].mean(axis=0)
    struct_dist = squareform(pdist(np.array([task_fingerprints[t] for t in tasks]), metric='euclidean'))
    task_dist, task_list = compute_task_distance_matrix(networks)
    # 对齐任务顺序
    task_idx = {t: i for i, t in enumerate(tasks)}
    task_dist_aligned = np.zeros((len(tasks), len(tasks)))
    for i, ti in enumerate(tasks):
        for j, tj in enumerate(tasks):
            task_dist_aligned[i, j] = task_dist[task_list.index(ti), task_list.index(tj)]

    r_obs, p_val = mantel_test(struct_dist, task_dist_aligned, n_permutations=1000)
    print(f"  Mantel test: r={r_obs:.3f}, p={p_val:.4f}")
    if p_val < 0.05:
        print("  → Structure fingerprint distance SIGNIFICANTLY correlates with task similarity!")
    else:
        print("  → No significant correlation (possibly due to small sample size).")

    # 7. 可视化
    print("\n[Step 6] Generating visualizations...")
    plot_tsne(X, y, networks, OUT_DIR)

    # KNN 混淆矩阵
    y_pred = []
    for i in range(len(X_norm)):
        train_X = np.delete(X_norm, i, axis=0)
        train_y = np.delete(y, i, axis=0)
        dists = np.linalg.norm(train_X - X_norm[i], axis=1)
        nearest = np.argsort(dists)[:3]
        votes = Counter(train_y[nearest])
        y_pred.append(votes.most_common(1)[0][0])
    plot_confusion_matrix(y, np.array(y_pred), tasks, OUT_DIR)
    plot_feature_importance(X, y, feature_names, OUT_DIR)
    plot_similarity_heatmaps(struct_dist, task_dist_aligned, tasks, OUT_DIR)

    # 8. 保存报告
    report_path = os.path.join(OUT_DIR, 'report.txt')
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("  实验 1B: 跨任务结构指纹分析报告\n")
        f.write("="*70 + "\n\n")
        f.write(f"样本量: {len(networks)} 个网络，来自 {n_tasks} 个任务\n")
        f.write(f"特征维度: {len(feature_names)}\n\n")
        f.write("[判别分析]\n")
        f.write(f"  LOO-KNN 准确率: {acc:.3f} (随机基线: {random_baseline:.3f}, 提升: {acc/random_baseline:.2f}x)\n\n")
        f.write("[无监督聚类]\n")
        f.write(f"  层次聚类纯度 (K={n_tasks}): {purity:.3f}\n\n")
        f.write("[距离矩阵相关]\n")
        f.write(f"  Mantel 检验: r={r_obs:.3f}, p={p_val:.4f}\n\n")
        f.write("[特征均值表]\n")
        f.write(f"{'Feature':<20} " + " ".join(f"{t:<12}" for t in tasks) + "\n")
        for fi, fname in enumerate(feature_names):
            vals = []
            for t in tasks:
                mask = y == t
                vals.append(X[mask, fi].mean())
            f.write(f"{fname:<20} " + " ".join(f"{v:<12.4f}" for v in vals) + "\n")

    print(f"\nReport saved: {report_path}")
    print("Done.")


if __name__ == '__main__':
    main()
