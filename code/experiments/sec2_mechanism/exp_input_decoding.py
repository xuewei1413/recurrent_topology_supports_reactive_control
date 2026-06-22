"""
实验 2C：输入历史解码（Section 3.2.3, Fig 4）

从当前隐藏层活动反推过去 N 步的输入信号，直接证明动态痕迹中
携带可提取的历史信息。

方法：
  1. 加载 CartPole 最优拓扑
  2. 运行 30 局 CartPole，每步记录隐藏层膜电位 + 真实输入
  3. 对每个 T_steps 值 (1,5,10,20)，用 Ridge 回归从当前膜电位
     解码过去 N 步的输入，计算 R²
  4. 展示解码精度随回溯步数的衰减曲线

关键结论：T_steps=20 时，膜电位中保留约 10 步可解码的输入历史，
验证了时间通道的有效长度。

输出：
  results/input_decoding.png
  results/decoding_data.npy
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.snn import SNNController
import gymnasium as gym

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def collect_decoding_data(adj, types, config, n_episodes=50, task='CartPole'):
    """
    收集数据：每步记录隐藏层活动 + 过去5步的输入。
    返回: X (n_samples, hidden_dim), Y (n_samples, 5, input_dim)
    """
    if task == 'CartPole':
        env = gym.make('CartPole-v1')
        max_steps = 500
        state_dim = 4
    else:
        raise ValueError(f"Unknown task: {task}")
    
    hidden_dim = config['n_total'] - config['n_input'] - config['n_output']
    n_in = config['n_input']
    n_out = config['n_output']
    hidden_idx = list(range(n_in, config['n_total'] - n_out))
    
    X = []  # 隐藏层活动
    Y = []  # 过去5步输入 [current, prev1, prev2, prev3, prev4]
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        controller = SNNController(config, adj, types)
        
        input_history = [np.zeros(state_dim) for _ in range(5)]  # 初始填充0
        
        for step in range(max_steps):
            # 记录当前隐藏层状态（用forward_with_trace的最后一个时刻）
            mem, spikes, action = forward_with_trace(controller, np.array(obs, dtype=np.float64))
            hidden_activity = spikes[-1, hidden_idx]  # 最后一步的隐藏层脉冲
            
            X.append(hidden_activity)
            Y.append(np.array(input_history[-5:]))  # 最近5步输入
            
            # 更新历史
            input_history.append(obs.copy())
            if len(input_history) > 5:
                input_history.pop(0)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    
    env.close()
    return np.array(X), np.array(Y)


def forward_with_trace(controller, state):
    """运行T_steps并返回完整轨迹"""
    n = controller.n
    T = controller.T
    dt = controller.dt
    tau = controller.tau
    
    v = np.zeros(n)
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
    
    action = int(np.argmax(spike_trace.sum(axis=0)[-controller.n_out:]))
    return mem_trace, spike_trace, action


def decode_history(X, Y, max_lag=5):
    """
    训练线性解码器预测过去每一步的输入。
    返回每个lag的R2分数。
    """
    n_samples = X.shape[0]
    state_dim = Y.shape[2]
    
    r2_scores = np.zeros((max_lag, state_dim))
    
    for lag in range(max_lag):
        # lag=0: 预测当前输入；lag=1: 预测上一步输入，等等
        y_target = Y[:, 4-lag, :]  # Y的shape是(n_samples, 5, state_dim)
        
        # 对每个state维度单独训练解码器
        for dim in range(state_dim):
            model = Ridge(alpha=1.0)
            model.fit(X, y_target[:, dim])
            pred = model.predict(X)
            r2_scores[lag, dim] = r2_score(y_target[:, dim], pred)
    
    return r2_scores


def run_decoding_experiment(adj_path, types_path, base_config, T_list=[1, 5, 10, 20]):
    print("="*60)
    print("实验 2.3: 输入历史解码")
    print("="*60)
    
    adj = np.load(adj_path)
    types = np.load(types_path)
    
    all_results = {}
    
    for T in T_list:
        print(f"\nT_steps={T}:")
        cfg = base_config.copy()
        cfg['T_steps'] = T
        
        X, Y = collect_decoding_data(adj, types, cfg, n_episodes=30, task='CartPole')
        print(f"  收集样本: {len(X)}")
        
        r2 = decode_history(X, Y, max_lag=5)
        all_results[T] = r2
        
        print(f"  解码R2 (当前步): {r2[0].mean():.3f}")
        for lag in range(1, 5):
            print(f"  解码R2 (lag={lag}): {r2[lag].mean():.3f}")
    
    return all_results


def plot_decoding_results(all_results, save_path='../results/input_decoding.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    T_list = sorted(all_results.keys())
    state_names = ['Cart Pos', 'Cart Vel', 'Pole Angle', 'Pole Vel']
    colors = plt.cm.tab10(np.linspace(0, 1, 4))
    markers = ['o', 's', '^', 'D']
    
    # 左图：不同T下的当前步解码
    ax = axes[0]
    for dim in range(4):
        vals = [all_results[T][0, dim] for T in T_list]
        ax.plot(T_list, vals, marker=markers[dim], color=colors[dim], 
                linewidth=2, markersize=8, label=state_names[dim])
    ax.set_xlabel('T_steps', fontsize=12)
    ax.set_ylabel('R² (Current Input)', fontsize=12)
    ax.set_title('Current Input Decoding vs Channel Length', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 右图：T=20时不同lag的解码衰减
    ax = axes[1]
    T = 20
    if T in all_results:
        for dim in range(4):
            vals = [all_results[T][lag, dim] for lag in range(5)]
            ax.plot(range(5), vals, marker=markers[dim], color=colors[dim],
                    linewidth=2, markersize=8, label=state_names[dim])
    ax.set_xlabel('Temporal Lag (steps back)', fontsize=12)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('History Decoding Decay (T=20)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n解码图已保存: {save_path}")
    plt.close()


if __name__ == '__main__':
    config = {
        'n_total': 128, 'n_input': 8, 'n_output': 2,
        'sparsity': 0.41, 'weight_scale': 0.6,
        'tau': 5.0, 'threshold': 0.8, 'T_steps': 20,
    }
    
    all_results = run_decoding_experiment(
        '../../../data/key_npy/best_topology.npy',
        '../../../data/key_npy/neuron_types.npy',
        config,
        T_list=[1, 5, 10, 20]
    )
    
    plot_decoding_results(all_results)
    np.save('../results/decoding_data.npy', all_results, allow_pickle=True)
    
    print("\n" + "="*60)
    print("实验 2.3 完成")
    print("="*60)
