"""
实验 1.3: 关键神经元与回路选择性消融
目标：证明拓扑中的特定结构承担特定功能。
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.snn import SNNController
from src.evolution import evaluate
import gymnasium as gym

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def quick_evaluate_cartpole(adj, neuron_types, config, n_episodes=5):
    """CartPole 快速评估"""
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


def quick_evaluate_mountaincar(adj, neuron_types, config, n_episodes=5):
    """MountainCar 快速评估"""
    controller = SNNController(config, adj, neuron_types)
    env = gym.make('MountainCar-v0')
    total = 0.0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        controller.reset()
        for step in range(200):
            action = controller.forward(np.array(obs, dtype=np.float64))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
    env.close()
    return total / n_episodes


def quick_evaluate(adj, neuron_types, config, n_episodes=5, task='CartPole'):
    """快速评估：根据任务选择环境"""
    if task == 'CartPole':
        return quick_evaluate_cartpole(adj, neuron_types, config, n_episodes)
    elif task == 'MountainCar':
        return quick_evaluate_mountaincar(adj, neuron_types, config, n_episodes)
    else:
        raise ValueError(f"Unknown task: {task}")


def ablate_neuron(adj, neuron_idx):
    """消融单个神经元：删除其所有入边和出边"""
    adj_new = adj.copy()
    adj_new[neuron_idx, :] = 0
    adj_new[:, neuron_idx] = 0
    return adj_new


def find_eie_motifs(adj, types):
    """
    找到所有 E->I->E 三节点回路。
    返回涉及的边列表 [(i,j), (j,k), ...] 以及唯一边集合。
    """
    exc_idx = np.where(types == 1)[0]
    inh_idx = np.where(types == -1)[0]
    
    edges = set()
    motifs = []
    
    for i in exc_idx:
        for j in inh_idx:
            if adj[i, j] > 0:  # E->I
                for k in exc_idx:
                    if adj[j, k] > 0:  # I->E
                        motifs.append((i, j, k))
                        edges.add((i, j))
                        edges.add((j, k))
    
    return list(edges), motifs


def ablate_edges(adj, edges_to_remove):
    """删除指定边"""
    adj_new = adj.copy()
    for (i, j) in edges_to_remove:
        adj_new[i, j] = 0
    return adj_new


def run_ablation_experiment(task_name, adj_path, types_path, config, n_eval=5):
    print(f"\n{'='*60}")
    print(f"选择性消融实验: {task_name}")
    print(f"{'='*60}")
    
    adj = np.load(adj_path)
    types = np.load(types_path)
    n = len(types)
    
    # 1. 基准性能
    print(f"\n[1/4] 基准评估 (n_eval={n_eval})...")
    baseline = quick_evaluate(adj, types, config, n_eval, task=task_name)
    print(f"  基准性能: {baseline:.1f}")
    
    # 2. 单个神经元消融
    print(f"\n[2/4] 单个神经元消融 (N={n})...")
    neuron_scores = []
    for i in range(n):
        adj_abl = ablate_neuron(adj, i)
        score = quick_evaluate(adj_abl, types, config, n_eval, task=task_name)
        drop = baseline - score
        neuron_scores.append({'idx': i, 'score': score, 'drop': drop, 'type': 'E' if types[i]==1 else 'I'})
        if i % 20 == 0:
            print(f"  进度: {i}/{n}, 当前drop={drop:.1f}")
    
    neuron_scores.sort(key=lambda x: x['drop'], reverse=True)
    
    print(f"\n  Top 10 关键神经元（消融后性能下降最大）:")
    for ns in neuron_scores[:10]:
        print(f"    Neuron {ns['idx']} ({ns['type']}): score={ns['score']:.1f}, drop={ns['drop']:.1f}")
    
    # 3. E->I->E 回路选择性消融
    print(f"\n[3/4] E->I->E 回路选择性消融...")
    eie_edges, eie_motifs = find_eie_motifs(adj, types)
    print(f"  找到 {len(eie_motifs)} 个 E->I->E 回路, 涉及 {len(eie_edges)} 条唯一边")
    
    adj_eie_abl = ablate_edges(adj, eie_edges)
    score_eie = quick_evaluate(adj_eie_abl, types, config, n_eval, task=task_name)
    drop_eie = baseline - score_eie
    print(f"  E->I->E消融后: {score_eie:.1f} (drop={drop_eie:.1f})")
    
    # 4. 随机消融对照（同数量边）
    print(f"\n[4/4] 随机消融对照 (n_repeats=20)...")
    n_edges_remove = len(eie_edges)
    all_edges = list(zip(*np.where(adj > 0)))
    random_drops = []
    
    for r in range(20):
        edges_to_remove = [all_edges[i] for i in np.random.choice(len(all_edges), size=n_edges_remove, replace=False)]
        adj_rand_abl = ablate_edges(adj, edges_to_remove)
        score_rand = quick_evaluate(adj_rand_abl, types, config, n_eval, task=task_name)
        random_drops.append(baseline - score_rand)
    
    random_drops = np.array(random_drops)
    print(f"  随机消融 drop: mean={random_drops.mean():.1f}, std={random_drops.std():.1f}")
    print(f"  E->I->E消融 drop: {drop_eie:.1f}")
    print(f"  差异 (EIE - Random): {drop_eie - random_drops.mean():.1f}")
    
    return {
        'task': task_name,
        'baseline': baseline,
        'neuron_scores': neuron_scores,
        'eie_edges': eie_edges,
        'eie_motifs': eie_motifs,
        'drop_eie': drop_eie,
        'random_drops': random_drops,
        'n_edges_removed': n_edges_remove,
    }


def plot_neuron_ablation(results, save_path='../results/neuron_ablation.png'):
    """绘制单个神经元消融结果"""
    n_tasks = len(results)
    fig, axes = plt.subplots(1, n_tasks, figsize=(5*n_tasks, 5), sharey=True)
    if n_tasks == 1:
        axes = [axes]
    
    for ax, (task_name, data) in zip(axes, results.items()):
        scores = data['neuron_scores']
        drops = [s['drop'] for s in scores]
        types = [1 if s['type'] == 'E' else 0 for s in scores]
        colors = ['#E74C3C' if t == 1 else '#3498DB' for t in types]
        
        # 按 drop 排序后绘制
        sorted_idx = np.argsort(drops)[::-1]
        y_pos = np.arange(len(drops))
        
        ax.barh(y_pos, [drops[i] for i in sorted_idx], 
                color=[colors[i] for i in sorted_idx], edgecolor='black', alpha=0.8)
        ax.invert_yaxis()
        ax.set_xlabel('Performance Drop', fontsize=11)
        ax.set_ylabel('Neuron (sorted by importance)', fontsize=11)
        ax.set_title(f'{task_name}\nSingle-Neuron Ablation', fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n神经元消融图已保存: {save_path}")
    plt.close()


def plot_selective_ablation_comparison(results, save_path='../results/selective_ablation_comparison.png'):
    """绘制选择性消融对比"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    tasks = list(results.keys())
    x = np.arange(len(tasks))
    width = 0.35
    
    eie_drops = [results[t]['drop_eie'] for t in tasks]
    rand_means = [results[t]['random_drops'].mean() for t in tasks]
    rand_stds = [results[t]['random_drops'].std() for t in tasks]
    
    bars1 = ax.bar(x - width/2, eie_drops, width, label='E→I→E motif ablation', 
                   color='#E74C3C', edgecolor='black')
    bars2 = ax.bar(x + width/2, rand_means, width, yerr=rand_stds, 
                   label='Random ablation (mean±std)', color='#3498DB', 
                   edgecolor='black', capsize=5)
    
    ax.set_ylabel('Performance Drop', fontsize=12)
    ax.set_title('Selective vs Random Ablation', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # 标注数值
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"选择性消融对比图已保存: {save_path}")
    plt.close()


if __name__ == '__main__':
    # CartPole 配置
    config_cartpole = {
        'n_total': 128,
        'n_input': 8,
        'n_output': 2,
        'sparsity': 0.41,
        'weight_scale': 0.6,
        'tau': 5.0,
        'threshold': 0.8,
        'T_steps': 20,
        'dt': 1.0,
        'n_eval_episodes': 5,
    }
    
    # MountainCar 配置
    config_mc = {
        'n_total': 64,
        'n_input': 4,
        'n_output': 3,
        'sparsity': 0.20,
        'weight_scale': 0.6,
        'tau': 5.0,
        'threshold': 0.8,
        'T_steps': 20,
        'dt': 1.0,
        'n_eval_episodes': 5,
    }
    
    tasks = {
        'CartPole': ('../../../data/key_npy/best_topology.npy', '../../../data/key_npy/neuron_types.npy', config_cartpole),
        'MountainCar': ('../../../data/key_npy/mc_best_topology.npy', '../../../data/key_npy/mc_neuron_types.npy', config_mc),
    }
    
    results = {}
    for task_name, (adj_path, types_path, cfg) in tasks.items():
        if os.path.exists(adj_path) and os.path.exists(types_path):
            data = run_ablation_experiment(task_name, adj_path, types_path, cfg, n_eval=5)
            results[task_name] = data
        else:
            print(f"跳过 {task_name}: 文件不存在")
    
    # 保存结果
    np.save('../results/selective_ablation_results.npy', results, allow_pickle=True)
    
    # 绘图
    plot_neuron_ablation(results)
    plot_selective_ablation_comparison(results)
    
    print("\n" + "="*60)
    print("实验 1.3 (选择性消融) 完成")
    print("="*60)
