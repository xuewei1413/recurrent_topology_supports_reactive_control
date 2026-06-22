"""
实验 2.4: 痕迹擦除测试
在 T_steps 前向传播的中间时刻重置隐藏层膜电位为 0，
证明中间神经状态对最终决策是必要的（动态痕迹的因果证据）。
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.snn import SNNController
import gymnasium as gym

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def forward_with_erasure(controller, state, erase_step=None):
    """
    运行 T_steps 前向，可选在 erase_step 重置隐藏层膜电位。
    erase_step=None: 正常前向
    erase_step=k: 在第 k 步后（0-indexed）将隐藏层 v 设为 0
    """
    n = controller.n
    T = controller.T
    dt = controller.dt
    tau = controller.tau
    
    v = np.zeros(n)
    spikes = np.zeros(n)
    
    I_ext = controller.encode_state(state)
    
    for t in range(T):
        v *= (1 - dt / tau)
        if spikes.any():
            v += spikes @ controller.W
        v[:controller.n_in] += I_ext * dt / tau
        
        fired = v >= controller.thresh
        spikes = fired.astype(np.float64)
        v[fired] = 0.0
        
        # 痕迹擦除
        if erase_step is not None and t == erase_step:
            v[controller.n_in:-controller.n_out] = 0.0  # 只重置隐藏层
    
    action = int(np.argmax(spikes[-controller.n_out:]))
    return action


def evaluate_with_erasure(adj, types, config, erase_step=None, n_episodes=20):
    """评估带痕迹擦除的网络"""
    env = gym.make('CartPole-v1')
    total = 0.0
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        controller = SNNController(config, adj, types)
        
        for step in range(500):
            action = forward_with_erasure(controller, np.array(obs, dtype=np.float64), erase_step)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
    
    env.close()
    return total / n_episodes


def run_trace_erasure_experiment(adj_path, types_path, config):
    print("="*60)
    print("实验 2.4: 痕迹擦除测试")
    print("="*60)
    
    adj = np.load(adj_path)
    types = np.load(types_path)
    T = config['T_steps']
    
    # 基准（无擦除）
    print(f"\n基准评估 (T={T}, 无擦除)...")
    baseline = evaluate_with_erasure(adj, types, config, erase_step=None, n_episodes=20)
    print(f"  基准性能: {baseline:.1f}")
    
    # 不同擦除时刻
    erase_steps = list(range(T))  # 0, 1, ..., T-1
    results = []
    
    for es in erase_steps:
        score = evaluate_with_erasure(adj, types, config, erase_step=es, n_episodes=20)
        drop = baseline - score
        results.append({'erase_step': es, 'score': score, 'drop': drop})
        print(f"  擦除时刻 t={es}: score={score:.1f}, drop={drop:.1f}")
    
    return baseline, results


def plot_trace_erasure(baseline, results, T, save_path='../results/trace_erasure.png'):
    """绘制痕迹擦除结果"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    erase_steps = [r['erase_step'] for r in results]
    scores = [r['score'] for r in results]
    drops = [r['drop'] for r in results]
    
    # 左图：性能随擦除时刻变化
    ax = axes[0]
    ax.plot(erase_steps, scores, 'o-', color='#E74C3C', linewidth=2, markersize=6)
    ax.axhline(y=baseline, color='black', linestyle='--', label=f'Baseline ({baseline:.1f})')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Erase Step', fontsize=12)
    ax.set_ylabel('Average Score', fontsize=12)
    ax.set_title('Performance vs Trace Erasure Timing', fontsize=13, fontweight='bold')
    ax.set_xticks(erase_steps)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 右图：性能下降量
    ax = axes[1]
    colors = ['#2ECC71' if d < 0 else '#E74C3C' for d in drops]
    ax.bar(erase_steps, drops, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xlabel('Erase Step', fontsize=12)
    ax.set_ylabel('Performance Drop', fontsize=12)
    ax.set_title('Performance Drop from Baseline', fontsize=13, fontweight='bold')
    ax.set_xticks(erase_steps)
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Trace Erasure Test (T_steps={T})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n痕迹擦除图已保存: {save_path}")
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
    
    baseline, results = run_trace_erasure_experiment(
        '../../../data/key_npy/best_topology.npy',
        '../../../data/key_npy/neuron_types.npy',
        config
    )
    
    plot_trace_erasure(baseline, results, config['T_steps'])
    
    print("\n" + "="*60)
    print("实验 2.4 完成")
    print("="*60)
