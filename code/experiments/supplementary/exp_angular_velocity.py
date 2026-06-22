"""
实验 1.4: 角速度专项因果测试
屏蔽位置信息，只给角速度，测试网络是否仍能维持平衡。
如果网络依赖内部动态积分来估计位置，那么仅角速度输入应能部分工作。
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.snn import SNNController
import gymnasium as gym

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def evaluate_with_masked_input(adj, types, config, mask, n_episodes=20):
    """
    评估网络，但屏蔽某些输入维度。
    mask: 4元素布尔数组，True=保留该维度，False=屏蔽（设为0）
    注意：SNN使用半波整流编码，所以输入是8维（每维正负分离）
    """
    env = gym.make('CartPole-v1')
    total = 0.0
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        controller = SNNController(config, adj, types)
        
        for step in range(500):
            # 应用mask：屏蔽指定维度
            masked_obs = obs.copy()
            for dim in range(4):
                if not mask[dim]:
                    masked_obs[dim] = 0.0
            
            action = controller.forward(np.array(masked_obs, dtype=np.float64))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
    
    env.close()
    return total / n_episodes


def run_angular_velocity_test(adj_path, types_path, config):
    print("="*60)
    print("实验 1.4: 角速度专项因果测试")
    print("="*60)
    
    adj = np.load(adj_path)
    types = np.load(types_path)
    
    # 四种输入条件
    conditions = {
        'Full (all)': [True, True, True, True],
        'Position only': [True, True, False, False],
        'Velocity only': [False, False, True, True],
        'Cart pos only': [True, False, False, False],
        'Pole angle only': [False, True, False, False],
        'Cart vel only': [False, False, True, False],
        'Pole vel only': [False, False, False, True],
    }
    
    results = {}
    for name, mask in conditions.items():
        score = evaluate_with_masked_input(adj, types, config, np.array(mask), n_episodes=20)
        results[name] = score
        print(f"  {name}: {score:.1f}")
    
    return results


def plot_av_test(results, save_path='../results/angular_velocity_test.png'):
    """绘制角速度测试结果"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    names = list(results.keys())
    scores = list(results.values())
    
    colors = ['#2ECC71' if s > 400 else '#F39C12' if s > 100 else '#E74C3C' for s in scores]
    
    bars = ax.barh(range(len(names)), scores, color=colors, edgecolor='black', alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel('Average Score (max=500)', fontsize=12)
    ax.set_title('CartPole: Performance with Masked Inputs', fontsize=13, fontweight='bold')
    ax.axvline(x=500, color='green', linestyle='--', alpha=0.5, label='Perfect')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # 标注数值
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.annotate(f'{score:.1f}',
                   xy=(width, bar.get_y() + bar.get_height()/2),
                   xytext=(5, 0), textcoords="offset points",
                   ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n角速度测试图已保存: {save_path}")
    plt.close()


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
    }
    
    results = run_angular_velocity_test('../../../data/key_npy/best_topology.npy', '../../../data/key_npy/neuron_types.npy', config)
    plot_av_test(results)
    
    print("\n" + "="*60)
    print("实验 1.4 完成")
    print("="*60)
