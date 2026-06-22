"""
实验 2B：膜电位轨迹可视化（Section 3.2.2, Fig 3）

展示隐藏层神经元的持续脉冲活动，而非稳定平衡态。
选择 5 个代表性隐藏神经元，记录完整一局 CartPole 的膜电位时间序列。

方法：
  1. 加载 CartPole 最优拓扑和神经元类型
  2. 运行一局 CartPole（500 步），每步记录 5 个隐藏神经元的膜电位
  3. 绘制膜电位时间序列，标注发放事件

关键观察：神经元持续脉冲（非稳态），验证 SNN 的动态计算依赖于
中间状态的时间结构，而非静态特征。

输出：
  results/membrane_traces.png
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.snn import SNNController

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# CartPole 标准配置（与论文一致）
CONFIG = {
    'n_total': 128,
    'n_input': 8,
    'n_output': 2,
    'sparsity': 0.41,
    'weight_scale': 0.6,
    'tau': 5.0,
    'threshold': 0.8,
    'T_steps': 20,
}

# 论文 fig03 展示的 5 个代表性隐藏神经元
REPRESENTATIVE_NEURONS = [8, 38, 68, 98, 125]


def forward_with_trace(controller, state):
    """运行一次完整 T_steps 前向传播，记录每步膜电位和脉冲。"""
    n = controller.n
    T = controller.T
    dt = controller.dt
    tau = controller.tau

    v = controller.v.copy()
    spikes = np.zeros(n)

    mem_trace = np.zeros((T, n))
    spike_trace = np.zeros((T, n))

    I_ext = controller.encode_state(state)

    for t in range(T):
        v *= (1 - dt / tau)
        if spikes.any():
            v += spikes @ controller.W
        v[:controller.n_in] += I_ext * dt / tau

        fired = v >= controller.thresh
        spikes = fired.astype(np.float64)
        v[fired] = 0.0

        mem_trace[t] = v.copy()
        spike_trace[t] = spikes.copy()

    controller.v = v.copy()
    action = int(np.argmax(spike_trace.sum(axis=0)[-controller.n_out:]))
    return mem_trace, spike_trace, action


def collect_cartpole_traces(adj, types, config, max_steps=2000):
    """运行 CartPole 一局，收集所有时间步的膜电位轨迹。"""
    import gymnasium as gym

    env = gym.make('CartPole-v1')
    obs, _ = env.reset()
    controller = SNNController(config, adj, types)

    # 只记录代表性神经元的膜电位
    all_mem = []       # (steps, n_neurons) — 每步最后一个 T_step 的膜电位快照
    all_spikes = []
    states = []

    for step in range(max_steps):
        states.append(obs.copy())
        mem, spikes, action = forward_with_trace(controller, np.array(obs, dtype=np.float64))
        # 取每步中各神经元的平均膜电位
        all_mem.append(mem.mean(axis=0))
        all_spikes.append(spikes.sum(axis=0))

        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    env.close()
    return np.array(states), np.array(all_mem), np.array(all_spikes)


def plot_membrane_traces(all_mem, spike_accum, neurons, save_path='../results/membrane_traces.png'):
    """绘制代表性隐藏神经元的膜电位轨迹（对应论文 fig03）。"""
    fig, axes = plt.subplots(len(neurons), 1, figsize=(14, 2.5 * len(neurons)), sharex=True)

    time_axis = np.arange(all_mem.shape[0])

    for ax, nid in zip(axes, neurons):
        trace = all_mem[:, nid]
        ax.plot(time_axis, trace, color='#2980B9', linewidth=0.8, alpha=0.9)

        # 标记发放时刻
        spike_times = np.where(spike_accum[:, nid] > 0)[0]
        for st in spike_times:
            ax.axvline(x=st, color='#E74C3C', alpha=0.15, linewidth=0.5)

        ax.set_ylabel(f'N{nid}', fontsize=11, fontweight='bold')
        ax.set_ylim(-0.1, 1.0)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel('Simulation Timestep', fontsize=12)
    axes[0].set_title('Membrane-Potential Trajectories (CartPole, Hidden Neurons)',
                       fontsize=13, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"膜电位轨迹图已保存: {save_path}")
    plt.close()


def plot_spiking_activity(all_spikes, neurons, save_path='../results/spiking_activity.png'):
    """绘制代表性神经元的脉冲栅格图。"""
    fig, ax = plt.subplots(figsize=(14, 4))

    for i, nid in enumerate(neurons):
        spike_times = np.where(all_spikes[:, nid] > 0)[0]
        ax.scatter(spike_times, [i] * len(spike_times), marker='|', s=20,
                   color='#E74C3C', linewidth=0.8)

    ax.set_yticks(range(len(neurons)))
    ax.set_yticklabels([f'N{n}' for n in neurons])
    ax.set_xlabel('Simulation Timestep', fontsize=12)
    ax.set_title('Spike Raster (CartPole, Representative Hidden Neurons)', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.2, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"脉冲栅格图已保存: {save_path}")
    plt.close()


def plot_firing_rate_summary(all_spikes, config, save_path='../results/firing_rate_summary.png'):
    """汇总所有隐藏神经元的发放率分布。"""
    n_in = config['n_input']
    n_out = config['n_output']
    hidden_idx = np.arange(n_in, config['n_total'] - n_out)

    firing_rates = all_spikes[:, hidden_idx].mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(firing_rates, bins=30, color='#3498DB', edgecolor='black', alpha=0.8)
    axes[0].set_xlabel('Mean Firing Rate (spikes/step)', fontsize=11)
    axes[0].set_ylabel('Neuron Count', fontsize=11)
    axes[0].set_title('Hidden-Layer Firing Rate Distribution', fontsize=12, fontweight='bold')
    axes[0].axvline(firing_rates.mean(), color='red', linestyle='--', label=f'Mean={firing_rates.mean():.3f}')
    axes[0].legend()

    active = (firing_rates > 0).sum()
    silent = len(firing_rates) - active
    axes[1].bar(['Active', 'Silent'], [active, silent], color=['#2ECC71', '#BDC3C7'], edgecolor='black')
    axes[1].set_ylabel('Neuron Count', fontsize=11)
    axes[1].set_title(f'Active vs Silent ({active}/{len(firing_rates)} active)', fontsize=12, fontweight='bold')

    plt.suptitle('CartPole Hidden-Layer Activity Summary', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"发放率汇总图已保存: {save_path}")
    plt.close()


if __name__ == '__main__':
    print("=" * 60)
    print("实验 2.1: CartPole 膜电位轨迹可视化")
    print("=" * 60)

    adj = np.load('../../../data/key_npy/best_topology.npy')
    types = np.load('../../../data/key_npy/neuron_types.npy')

    print("\n收集 CartPole 膜电位轨迹 (最多 2000 步)...")
    states, all_mem, all_spikes = collect_cartpole_traces(adj, types, CONFIG, max_steps=2000)
    print(f"  实际跑了 {len(states)} 步")

    n_in = CONFIG['n_input']
    n_out = CONFIG['n_output']
    hidden_idx = np.arange(n_in, CONFIG['n_total'] - n_out)
    active_hidden = (all_spikes[:, hidden_idx].sum(axis=0) > 0).sum()
    print(f"  活跃隐藏神经元: {active_hidden}/{len(hidden_idx)}")

    # 验证论文中的代表性神经元是否在范围内且活跃
    valid_neurons = [n for n in REPRESENTATIVE_NEURONS if n in hidden_idx]
    print(f"  代表性神经元: {valid_neurons}")

    # 如果论文指定的神经元不活跃，自动替换为最活跃的
    for nid in valid_neurons:
        if all_spikes[:, nid].sum() == 0:
            print(f"  警告: N{nid} 不活跃，将替换")

    active_neurons = hidden_idx[np.argsort(all_spikes[:, hidden_idx].sum(axis=0))[-5:]][::-1]
    neurons_to_plot = valid_neurons if all(all_spikes[:, n].sum() > 0 for n in valid_neurons) else list(active_neurons)
    print(f"  绘图神经元: {neurons_to_plot}")

    # 绘图
    plot_membrane_traces(all_mem, all_spikes, neurons_to_plot)
    plot_spiking_activity(all_spikes, neurons_to_plot)
    plot_firing_rate_summary(all_spikes, CONFIG)

    print("\n" + "=" * 60)
    print("实验 2.1 完成")
    print("=" * 60)
