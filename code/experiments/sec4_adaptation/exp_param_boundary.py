"""
C1-精简：参数Boundary扫描
CartPole (gravity × pole_length) + FlappyBird (pipe_speed)
固定标准拓扑，纯评估，无需进化
"""
import os, sys, json, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from src.snn import SNNController
from src.flappy_env_parametric import FlappyEnvParametric

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SAVE_DIR = 'results/boundary_scan'
os.makedirs(SAVE_DIR, exist_ok=True)

# ── CartPole 配置 ─────────────────────────────────────────────────────────
CP_CONFIG = {
    'n_total': 128, 'n_input': 8, 'n_output': 2,
    'weight_scale': 0.6, 'tau': 5.0, 'threshold': 0.8, 'T_steps': 20,
}
CP_GRAVITIES = [4.9, 9.8, 15.0, 20.0]
CP_LENGTHS = [0.3, 0.5, 0.7, 1.0]

# ── FlappyBird 配置 ───────────────────────────────────────────────────────
FB_CONFIG = {
    'n_total': 64, 'n_input': 8, 'n_output': 2,
    'weight_scale': 0.6, 'tau': 5.0, 'threshold': 0.8, 'T_steps': 20,
}
FB_SPEEDS = [2, 3, 4, 5, 6, 8]


def evaluate_cartpole(adj, types, gravity, length, n_episodes=20):
    config = CP_CONFIG.copy()
    scores = []
    for _ in range(n_episodes):
        env = gym.make('CartPole-v1')
        env.unwrapped.gravity = gravity
        env.unwrapped.length = length
        ctrl = SNNController(config, adj, types)
        obs, _ = env.reset()
        ctrl.reset()
        total = 0
        for step in range(500):
            action = ctrl.forward(np.array(obs, dtype=np.float64))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        env.close()
        scores.append(total)
    return float(np.mean(scores)), float(np.std(scores))


def evaluate_flappy(adj, types, pipe_speed, n_episodes=50):
    config = FB_CONFIG.copy()
    scores = []
    for _ in range(n_episodes):
        env = FlappyEnvParametric(pipe_speed=pipe_speed, gravity=0.45, pipe_gap=140,
                                   pipe_frequency=100, seed=np.random.randint(0, 100000))
        ctrl = SNNController(config, adj, types)
        state = env.reset()
        ctrl.reset()
        total_reward = 0
        while True:
            action = ctrl.forward(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                scores.append(info['score'])
                break
    return float(np.mean(scores)), float(np.std(scores))


def run_cartpole_scan():
    print("[C1] Loading CartPole topology...")
    adj = np.load('../../../data/key_npy/best_topology.npy')
    types = np.load('../../../data/key_npy/neuron_types.npy')

    results = []
    for g in CP_GRAVITIES:
        for l in CP_LENGTHS:
            print(f"  Evaluating gravity={g}, length={l} ...")
            mean, std = evaluate_cartpole(adj, types, g, l)
            results.append({'gravity': g, 'length': l, 'mean': mean, 'std': std})
            print(f"    mean={mean:.1f} ± {std:.1f}")

    # 保存 CSV
    csv_path = os.path.join(SAVE_DIR, 'cartpole_boundary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['gravity', 'length', 'mean', 'std'])
        writer.writeheader()
        writer.writerows(results)

    # 绘制热力图
    means = np.array([r['mean'] for r in results]).reshape(len(CP_GRAVITIES), len(CP_LENGTHS))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(means, cmap='RdYlGn', aspect='auto', vmin=0, vmax=500)
    ax.set_xticks(np.arange(len(CP_LENGTHS)))
    ax.set_yticks(np.arange(len(CP_GRAVITIES)))
    ax.set_xticklabels([str(l) for l in CP_LENGTHS])
    ax.set_yticklabels([str(g) for g in CP_GRAVITIES])
    ax.set_xlabel('Pole Length')
    ax.set_ylabel('Gravity')
    ax.set_title('CartPole Parameter Boundary Scan\n(Mean Steps (20 Episodes))')

    # 标注数值和区域
    for i in range(len(CP_GRAVITIES)):
        for j in range(len(CP_LENGTHS)):
            val = means[i, j]
            if val >= 490:
                zone = '舒适'
                color = 'darkgreen'
            elif val >= 200:
                zone = 'Boundary'
                color = 'darkgoldenrod'
            else:
                zone = 'Failure'
                color = 'darkred'
            ax.text(j, i, f'{val:.0f}\n{zone}', ha='center', va='center', fontsize=9, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Mean Steps')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'cartpole_boundary.png'), dpi=150)
    print(f"Saved: {SAVE_DIR}/cartpole_boundary.png")
    return results


def run_flappy_scan():
    print("[C1] Loading FlappyBird topology...")
    adj = np.load('../../../data/key_npy/flappy_best_topology.npy')
    types = np.load('../../../data/key_npy/flappy_neuron_types.npy')

    results = []
    for speed in FB_SPEEDS:
        print(f"  Evaluating pipe_speed={speed} ...")
        mean, std = evaluate_flappy(adj, types, speed)
        results.append({'pipe_speed': speed, 'mean': mean, 'std': std})
        print(f"    mean={mean:.1f} ± {std:.1f}")

    csv_path = os.path.join(SAVE_DIR, 'flappy_boundary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['pipe_speed', 'mean', 'std'])
        writer.writeheader()
        writer.writerows(results)

    # 绘制曲线
    speeds = [r['pipe_speed'] for r in results]
    means = [r['mean'] for r in results]
    stds = [r['std'] for r in results]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(speeds, means, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax.fill_between(speeds, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.2, color='steelblue')

    # 标注区域
    baseline_speed = 4
    baseline_mean = means[FB_SPEEDS.index(baseline_speed)]
    ax.axhline(baseline_mean, color='green', linestyle='--', alpha=0.5, label=f'Standard Env Baseline ({baseline_mean:.0f})')

    for i, (s, m) in enumerate(zip(speeds, means)):
        if m >= baseline_mean * 0.8:
            zone = 'Comfort'
            color = 'green'
        elif m >= baseline_mean * 0.3:
            zone = 'Boundary'
            color = 'orange'
        else:
            zone = 'Failure'
            color = 'red'
        ax.text(s, m + stds[i] + 2, zone, ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')

    ax.set_xlabel('Pipe Speed')
    ax.set_ylabel('Mean Score (50 Episodes)')
    ax.set_title('FlappyBird Parameter Boundary Scan')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'flappy_boundary.png'), dpi=150)
    print(f"Saved: {SAVE_DIR}/flappy_boundary.png")
    return results


def main():
    print("=" * 55)
    print("C1-精简: 参数Boundary扫描")
    print("=" * 55)
    cp_results = run_cartpole_scan()
    fb_results = run_flappy_scan()
    print("\n[C1] All done!")


if __name__ == '__main__':
    main()
