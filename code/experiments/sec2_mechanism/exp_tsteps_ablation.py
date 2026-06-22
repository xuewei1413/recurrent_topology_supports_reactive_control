"""
实验 2A：T_steps 连续消融曲线（Section 3.2.1）

量化 SNN 时间通道长度 T_steps 对控制性能的影响，找到功能拐点。
固定最优拓扑，仅改变 T_steps 从 1 到 30，观察性能变化。

方法：
  1. 加载 CartPole 最优拓扑（results/best_topology.npy）
  2. 对每个 T_steps 值 (1,2,3,5,7,10,15,20,25,30) 跑 30 局 CartPole
  3. 绘制性能曲线，标注拐点

论文预期（Section 3.2.1, Fig 2, Table 3）：
  T=1: 9, T=5: 25, T=10: 380, T=15: 393, T=20: 438, T=25: 400

输出：
  results/tsteps_curve.png
  results/tsteps_curve_data.npy
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.snn import SNNController
import gymnasium as gym

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def evaluate_with_tsteps(adj, types, base_config, T_steps, n_episodes=20, task='CartPole'):
    """使用指定 T_steps 评估"""
    cfg = base_config.copy()
    cfg['T_steps'] = T_steps
    
    if task == 'CartPole':
        env = gym.make('CartPole-v1')
        max_steps = 500
        use_reward = True
    elif task == 'MountainCar':
        env = gym.make('MountainCar-v0')
        max_steps = 200
        use_reward = False
    elif task == 'FlappyBird':
        return None
    else:
        raise ValueError(f"Unknown task: {task}")
    
    totals = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        controller = SNNController(cfg, adj, types)
        if use_reward:
            ep_total = 0.0
        else:
            ep_total = obs[0]  # MountainCar: 追踪最大位置
        for step in range(max_steps):
            action = controller.forward(np.array(obs, dtype=np.float64))
            obs, reward, terminated, truncated, _ = env.step(action)
            if use_reward:
                ep_total += reward
            else:
                if obs[0] > ep_total:
                    ep_total = obs[0]
            if terminated or truncated:
                break
        totals.append(ep_total)
    env.close()
    return float(np.mean(totals))


def run_tsteps_curve(adj_path, types_path, config, task_name, T_list):
    print(f"\n{'='*50}")
    print(f"T_steps 消融曲线: {task_name}")
    print(f"{'='*50}")
    
    adj = np.load(adj_path)
    types = np.load(types_path)
    
    results = []
    for T in T_list:
        score = evaluate_with_tsteps(adj, types, config, T, n_episodes=20, task=task_name)
        results.append({'T': T, 'score': score})
        print(f"  T={T:2d}: {score:.1f}")
    
    return results


def plot_tsteps_curves(all_results, save_path='../results/tsteps_curve.png'):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    colors = {'CartPole': '#E74C3C', 'MountainCar': '#3498DB', 'FlappyBird': '#2ECC71'}
    markers = {'CartPole': 'o', 'MountainCar': 's', 'FlappyBird': '^'}
    
    for task_name, results in all_results.items():
        if results is None:
            continue
        T_vals = [r['T'] for r in results]
        scores = [r['score'] for r in results]
        
        if task_name == 'CartPole':
            ax = ax1
            ax.set_ylabel('CartPole: Average Steps Survived', fontsize=12, color=colors[task_name])
            ax.tick_params(axis='y', labelcolor=colors[task_name])
        elif task_name == 'MountainCar':
            ax = ax2
            ax.set_ylabel('MountainCar: Average Max Position', fontsize=12, color=colors[task_name])
            ax.tick_params(axis='y', labelcolor=colors[task_name])
        else:
            ax = ax1
        
        ax.plot(T_vals, scores, marker=markers.get(task_name, 'o'),
                color=colors.get(task_name, 'black'), linewidth=2.5,
                markersize=8, label=task_name)
    
    ax1.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('T_steps (Channel Length)', fontsize=13)
    ax1.set_title('Performance vs Temporal Integration Window', fontsize=14, fontweight='bold')
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=11)
    
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nT_steps 曲线图已保存: {save_path}")
    plt.close()


if __name__ == '__main__':
    T_list = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]
    
    configs = {
        'CartPole': {
            'n_total': 128, 'n_input': 8, 'n_output': 2,
            'sparsity': 0.4, 'weight_scale': 0.6,
            'tau': 5.0, 'threshold': 0.8, 'T_steps': 20,
        },
        'MountainCar': {
            'n_total': 64, 'n_input': 4, 'n_output': 3,
            'sparsity': 0.12, 'weight_scale': 0.6,
            'tau': 5.0, 'threshold': 0.8, 'T_steps': 20,
        },
    }
    
    paths = {
        'CartPole': ('../../../data/key_npy/best_topology.npy', '../../../data/key_npy/neuron_types.npy'),
        'MountainCar': ('../../../data/key_npy/mc_best_topology.npy', '../../../data/key_npy/mc_neuron_types.npy'),
    }
    
    all_results = {}
    for task, cfg in configs.items():
        adj_path, types_path = paths[task]
        if os.path.exists(adj_path) and os.path.exists(types_path):
            res = run_tsteps_curve(adj_path, types_path, cfg, task, T_list)
            all_results[task] = res
        else:
            print(f"跳过 {task}: 文件不存在")
    
    plot_tsteps_curves(all_results)
    np.save('../results/tsteps_curve_data.npy', all_results, allow_pickle=True)
    
    print("\n" + "="*50)
    print("实验 2.2 完成")
    print("="*50)
