"""
实验 2D：输出层脉冲时序分析（Section 3.2.4, Fig 6）

验证动态痕迹是否系统性影响输出神经元的脉冲时间。

核心逻辑：
  选取 CartPole "平衡点附近"的步（|pole_angle| < 0.05），
  这些步的当前输入（角度）几乎相同，但角速度不同（反映不同历史趋势）。
  如果动态痕迹影响输出编码，则不同角速度下输出脉冲时序应有系统性差异。

方法：
  1. 加载 CartPole 最优拓扑
  2. 运行 100 局 CartPole，收集所有 |angle| < 0.05 的步
  3. 按角速度正负分组，对比两组的输出脉冲时序分布
  4. 统计检验两组时序是否有显著差异

输出：
  results/output_spike_timing.png
  results/spike_timing_data.npy
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.snn import SNNController
import gymnasium as gym

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def forward_with_full_trace(controller, state):
    """运行T_steps并返回完整脉冲轨迹"""
    n = controller.n
    T = controller.T
    dt = controller.dt
    tau = controller.tau
    
    v = np.zeros(n)
    spikes = np.zeros(n)
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
        spike_trace[t] = spikes.copy()
    
    action = int(np.argmax(spike_trace.sum(axis=0)[-controller.n_out:]))
    return spike_trace, action


def collect_timing_data(adj, types, config, n_episodes=100):
    """
    收集数据：每步记录当前状态 + 输出层脉冲时间。
    返回列表，每个元素是 dict。
    """
    env = gym.make('CartPole-v1')
    max_steps = 500
    
    records = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        controller = SNNController(config, adj, types)
        
        for step in range(max_steps):
            spike_trace, action = forward_with_full_trace(
                controller, np.array(obs, dtype=np.float64)
            )
            
            # 输出层脉冲轨迹
            output_spikes = spike_trace[:, -config['n_output']:]
            
            # 计算每个输出神经元的首次发放时间（T_steps内未发放则为-1）
            first_spike_times = []
            for out_idx in range(config['n_output']):
                spike_times = np.where(output_spikes[:, out_idx] > 0)[0]
                if len(spike_times) > 0:
                    first_spike_times.append(int(spike_times[0]))
                else:
                    first_spike_times.append(-1)
            
            # 总发放次数
            spike_counts = output_spikes.sum(axis=0).astype(int).tolist()
            
            records.append({
                'cart_pos': obs[0],
                'cart_vel': obs[1],
                'pole_angle': obs[2],
                'pole_vel': obs[3],
                'action': action,
                'first_spike': first_spike_times,  # [left_time, right_time]
                'spike_counts': spike_counts,       # [left_count, right_count]
                'output_spikes': output_spikes,     # (T, n_out)
            })
            
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    
    env.close()
    return records


def analyze_timing_by_history(records, angle_thresh=0.05):
    """
    分析平衡点附近、不同角速度条件下的脉冲时序差异。
    """
    # 筛选：|pole_angle| < angle_thresh（当前输入相似）
    filtered = [r for r in records if abs(r['pole_angle']) < angle_thresh]
    print(f"平衡点附近样本数: {len(filtered)} / {len(records)}")
    
    if len(filtered) < 50:
        print("警告：样本数不足，放宽阈值到 0.10")
        angle_thresh = 0.10
        filtered = [r for r in records if abs(r['pole_angle']) < angle_thresh]
        print(f"放宽后样本数: {len(filtered)}")
    
    # 按角速度分组
    pos_vel = [r for r in filtered if r['pole_vel'] > 0]   # 向右倒的趋势
    neg_vel = [r for r in filtered if r['pole_vel'] < 0]   # 向左倒的趋势
    
    print(f"  角速度>0 (向右倒): {len(pos_vel)}")
    print(f"  角速度<0 (向左倒): {len(neg_vel)}")
    
    # 提取首次发放时间（只考虑有发放的样本）
    def extract_times(group, neuron_idx):
        times = [r['first_spike'][neuron_idx] for r in group if r['first_spike'][neuron_idx] >= 0]
        return np.array(times)
    
    left_pos = extract_times(pos_vel, 0)   # 角速度>0时，左推神经元的首次发放时间
    right_pos = extract_times(pos_vel, 1)  # 角速度>0时，右推神经元的首次发放时间
    left_neg = extract_times(neg_vel, 0)   # 角速度<0时，左推神经元的首次发放时间
    right_neg = extract_times(neg_vel, 1)  # 角速度<0时，右推神经元的首次发放时间
    
    # 统计
    results = {
        'left_pos': left_pos,
        'right_pos': right_pos,
        'left_neg': left_neg,
        'right_neg': right_neg,
        'angle_thresh': angle_thresh,
    }
    
    print(f"\n首次发放时间统计 (|angle|<{angle_thresh}):")
    print(f"  角速度>0: 左推神经元 mean={left_pos.mean():.1f} (n={len(left_pos)}), "
          f"右推神经元 mean={right_pos.mean():.1f} (n={len(right_pos)})")
    print(f"  角速度<0: 左推神经元 mean={left_neg.mean():.1f} (n={len(left_neg)}), "
          f"右推神经元 mean={right_neg.mean():.1f} (n={len(right_neg)})")
    
    # t-test：同一神经元在不同历史条件下的发放时间差异
    if len(left_pos) > 5 and len(left_neg) > 5:
        t_left, p_left = stats.ttest_ind(left_pos, left_neg)
        print(f"\n  左推神经元 t-test: t={t_left:.2f}, p={p_left:.4f}")
    if len(right_pos) > 5 and len(right_neg) > 5:
        t_right, p_right = stats.ttest_ind(right_pos, right_neg)
        print(f"  右推神经元 t-test: t={t_right:.2f}, p={p_right:.4f}")
    
    return results


def plot_timing_results(results, save_path='../results/output_spike_timing.png'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # 图1：首次发放时间分布（小提琴图/箱线图）
    ax = axes[0]
    data_to_plot = [
        results['left_neg'], results['left_pos'],
        results['right_neg'], results['right_pos']
    ]
    labels = ['Left\n(vel<0)', 'Left\n(vel>0)', 'Right\n(vel<0)', 'Right\n(vel>0)']
    colors = ['#3498DB', '#85C1E9', '#E74C3C', '#F1948A']
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('First Spike Time (step within T)', fontsize=11)
    ax.set_title(f'Output First Spike Time\n(|pole_angle|<{results["angle_thresh"]:.2f})',
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # 图2：均值对比
    ax = axes[1]
    categories = ['Left neuron', 'Right neuron']
    neg_means = [results['left_neg'].mean(), results['right_neg'].mean()]
    pos_means = [results['left_pos'].mean(), results['right_pos'].mean()]
    neg_stds = [results['left_neg'].std(), results['right_neg'].std()]
    pos_stds = [results['left_pos'].std(), results['right_pos'].std()]
    
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, neg_means, width, yerr=neg_stds, label='vel<0', color='#3498DB', alpha=0.8, capsize=4)
    ax.bar(x + width/2, pos_means, width, yerr=pos_stds, label='vel>0', color='#E74C3C', alpha=0.8, capsize=4)
    ax.set_ylabel('Mean First Spike Time', fontsize=11)
    ax.set_title('Mean First Spike Time by History', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # 图3：发放次数对比
    ax = axes[2]
    # 这里我们用原始records重新计算发放次数的均值
    # 但results中没有保存原始records，所以我们只展示首次发放时间的差异显著性
    # 改为：发放时间差异的方向性分析
    
    # 计算每个历史条件下，哪个神经元先发放的比例
    def compute_first_neuron_ratio(recs, neuron_idx):
        # 需要重新从records计算...但results中没有保留records
        # 改为绘制发放时间的散点图
        pass
    
    # 改为：发放时间差异的直方图
    if len(results['left_neg']) > 0 and len(results['left_pos']) > 0:
        ax.hist(results['left_neg'], bins=15, alpha=0.5, color='#3498DB', label='Left neuron (vel<0)')
        ax.hist(results['left_pos'], bins=15, alpha=0.5, color='#85C1E9', label='Left neuron (vel>0)')
    if len(results['right_neg']) > 0 and len(results['right_pos']) > 0:
        ax.hist(results['right_neg'], bins=15, alpha=0.5, color='#E74C3C', label='Right neuron (vel<0)')
        ax.hist(results['right_pos'], bins=15, alpha=0.5, color='#F1948A', label='Right neuron (vel>0)')
    ax.set_xlabel('First Spike Time', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of First Spike Times', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n时序分析图已保存: {save_path}")
    plt.close()


def analyze_spike_count_patterns(records, angle_thresh=0.05):
    """额外分析：不同历史条件下的发放次数模式"""
    filtered = [r for r in records if abs(r['pole_angle']) < angle_thresh]
    pos_vel = [r for r in filtered if r['pole_vel'] > 0]
    neg_vel = [r for r in filtered if r['pole_vel'] < 0]
    
    # 计算每个条件下，输出层发放次数的均值
    def mean_counts(group):
        if len(group) == 0:
            return np.array([0.0, 0.0])
        counts = np.array([r['spike_counts'] for r in group])
        return counts.mean(axis=0)
    
    counts_pos = mean_counts(pos_vel)
    counts_neg = mean_counts(neg_vel)
    
    print(f"\n输出层发放次数均值 (|angle|<{angle_thresh}):")
    print(f"  角速度>0: 左推={counts_pos[0]:.2f}, 右推={counts_pos[1]:.2f}")
    print(f"  角速度<0: 左推={counts_neg[0]:.2f}, 右推={counts_neg[1]:.2f}")
    
    # 胜者比例（哪个神经元发放更多）
    def winner_ratio(group, winner_idx):
        if len(group) == 0:
            return 0.0
        wins = sum(1 for r in group if r['spike_counts'][winner_idx] > r['spike_counts'][1-winner_idx])
        return wins / len(group)
    
    print(f"\n胜者比例（发放次数更多）:")
    print(f"  角速度>0: 左推胜={winner_ratio(pos_vel, 0):.2%}, 右推胜={winner_ratio(pos_vel, 1):.2%}")
    print(f"  角速度<0: 左推胜={winner_ratio(neg_vel, 0):.2%}, 右推胜={winner_ratio(neg_vel, 1):.2%}")
    
    return {
        'counts_pos': counts_pos,
        'counts_neg': counts_neg,
        'winner_pos_left': winner_ratio(pos_vel, 0),
        'winner_pos_right': winner_ratio(pos_vel, 1),
        'winner_neg_left': winner_ratio(neg_vel, 0),
        'winner_neg_right': winner_ratio(neg_vel, 1),
    }


if __name__ == '__main__':
    config = {
        'n_total': 128, 'n_input': 8, 'n_output': 2,
        'sparsity': 0.41, 'weight_scale': 0.6,
        'tau': 5.0, 'threshold': 0.8, 'T_steps': 20,
    }
    
    print("=" * 60)
    print("实验 2.9: 输出层脉冲时序分析")
    print("=" * 60)
    
    adj = np.load('../../../data/key_npy/best_topology.npy')
    types = np.load('../../../data/key_npy/neuron_types.npy')
    
    print("\n收集数据 (100 episodes)...")
    records = collect_timing_data(adj, types, config, n_episodes=100)
    print(f"总样本数: {len(records)}")
    
    # 主分析
    print("\n" + "-" * 50)
    print("分析1: 首次发放时间 vs 历史条件")
    print("-" * 50)
    timing_results = analyze_timing_by_history(records, angle_thresh=0.05)
    
    # 额外分析
    print("\n" + "-" * 50)
    print("分析2: 发放次数模式")
    print("-" * 50)
    count_results = analyze_spike_count_patterns(records, angle_thresh=0.05)
    
    # 绘图
    plot_timing_results(timing_results)
    
    # 保存数据
    np.save('../results/spike_timing_data.npy', {
        'timing': timing_results,
        'counts': count_results,
    }, allow_pickle=True)
    
    print("\n" + "=" * 60)
    print("实验 2.9 完成")
    print("=" * 60)
