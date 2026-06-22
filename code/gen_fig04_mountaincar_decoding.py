"""
MountainCar input-decoding experiment with CORRECT INPUT_SCALE.

Verifies that the optimal decoding window for MountainCar is T≈20,
matching the claim in the paper. Uses INPUT_SCALE=4.0 as in the GA.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.snn import SNNController
import gymnasium as gym

INPUT_SCALE = 4.0

CONFIG = {
    'n_total': 64,
    'n_input': 4,
    'n_output': 3,
    'sparsity': 0.41,
    'weight_scale': 0.6,
    'tau': 5.0,
    'threshold': 0.8,
    'T_steps': 20,
    'dt': 1.0,
}

TOPO_PATH = os.path.join(os.path.dirname(__file__),
                         'experiments', 'results', 'cross_task', 'MountainCar_seed0.npz')
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures_arxiv')


def forward_with_trace(controller, state):
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


def collect_data(adj, types, config, n_episodes=80):
    env = gym.make('MountainCar-v0')
    n_in = config['n_input']
    n_out = config['n_output']
    hidden_idx = list(range(n_in, config['n_total'] - n_out))
    state_dim = 2  # MountainCar: position, velocity

    X = []
    Y_current = []
    Y_history = [[] for _ in range(5)]

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=100 + ep)
        controller = SNNController(config, adj, types)

        history = [np.zeros(state_dim) for _ in range(5)]

        for step in range(200):
            scaled = np.array(obs, dtype=np.float64) * INPUT_SCALE
            spike_trace, action = forward_with_trace(controller, scaled)
            hidden_activity = spike_trace[-1, hidden_idx]

            X.append(hidden_activity)
            Y_current.append(obs.copy())
            for lag in range(5):
                Y_history[lag].append(history[-(lag + 1)].copy())

            history.append(obs.copy())
            if len(history) > 5:
                history.pop(0)

            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    env.close()
    return (np.array(X), np.array(Y_current),
            [np.array(y) for y in Y_history])


def run_decoding(adj, types, config, T_list):
    results = {}
    for T in T_list:
        cfg = config.copy()
        cfg['T_steps'] = T
        print(f"  T={T}: collecting data...", end='', flush=True)
        X, Y_cur, Y_hist = collect_data(adj, types, cfg, n_episodes=80)
        print(f" {len(X)} samples", flush=True)

        # Current input decoding
        r2_current = np.zeros(2)
        for dim in range(2):
            model = Ridge(alpha=1.0)
            model.fit(X, Y_cur[:, dim])
            r2_current[dim] = r2_score(Y_cur[:, dim], model.predict(X))

        # History decoding at lag 1-4
        r2_history = np.zeros((4, 2))
        for lag in range(4):
            for dim in range(2):
                model = Ridge(alpha=1.0)
                model.fit(X, Y_hist[lag + 1][:, dim])
                r2_history[lag, dim] = r2_score(Y_hist[lag + 1][:, dim],
                                                  model.predict(X))

        results[T] = {
            'current': r2_current,
            'history': r2_history,
        }
        print(f"    Current R2: pos={r2_current[0]:.3f}, vel={r2_current[1]:.3f}")
        print(f"    Lag-4 R2:   pos={r2_history[3, 0]:.3f}, vel={r2_history[3, 1]:.3f}")

    return results


def plot_results(results, save_dir):
    T_list = sorted(results.keys())
    dims = ['Position', 'Velocity']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) Current input R² vs T
    ax = axes[0]
    for d, (name, color, marker) in enumerate(zip(
            dims, ['#E74C3C', '#3498DB'], ['o', 's'])):
        vals = [results[T]['current'][d] for T in T_list]
        ax.plot(T_list, vals, marker=marker, color=color,
                linewidth=2, markersize=8, label=name)
    ax.set_xlabel('$T_{steps}$', fontsize=12)
    ax.set_ylabel('$R^2$ (Current Input)', fontsize=12)
    ax.set_title('(a) MountainCar: Current Input Decoding', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    # Mark optimal T
    pos_r2 = [results[T]['current'][0] for T in T_list]
    best_T = T_list[np.argmax(pos_r2)]
    ax.axvline(best_T, color='gray', linestyle='--', alpha=0.5)
    ax.annotate(f'Peak T={best_T}', xy=(best_T, max(pos_r2)),
                xytext=(best_T + 3, max(pos_r2) - 0.05),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))

    # (b) History decay at optimal T
    ax = axes[1]
    T_plot = best_T
    if T_plot in results:
        for d, (name, color, marker) in enumerate(zip(
                dims, ['#E74C3C', '#3498DB'], ['o', 's'])):
            vals = [results[T_plot]['current'][d]] + \
                   [results[T_plot]['history'][lag, d] for lag in range(4)]
            ax.plot(range(5), vals, marker=marker, color=color,
                    linewidth=2, markersize=8, label=name)
    ax.set_xlabel('Temporal Lag (steps back)', fontsize=12)
    ax.set_ylabel('$R^2$', fontsize=12)
    ax.set_title(f'(b) History Decay at T={T_plot}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # (c) Comparison table
    ax = axes[2]
    ax.axis('off')
    table_data = []
    for T in T_list:
        r = results[T]
        table_data.append([
            f'{T}',
            f'{r["current"][0]:.3f}',
            f'{r["current"][1]:.3f}',
            f'{r["history"][3, 0]:.3f}',
            f'{r["history"][3, 1]:.3f}',
        ])
    table = ax.table(
        cellText=table_data,
        colLabels=['T', 'Pos R²', 'Vel R²', 'Pos lag4', 'Vel lag4'],
        loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    # Highlight best T
    best_idx = T_list.index(best_T)
    for j in range(5):
        table[(best_idx + 1, j)].set_facecolor('#d5f5e3')
    ax.set_title('(c) Full Results', fontsize=12, fontweight='bold')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, 'fig04_mountaincar_decoding.png')
    pdf_path = os.path.join(save_dir, 'fig04_mountaincar_decoding.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {png_path} ({os.path.getsize(png_path):,} bytes)")
    print(f"Saved: {pdf_path} ({os.path.getsize(pdf_path):,} bytes)")


if __name__ == '__main__':
    print("=" * 60)
    print("MountainCar Input Decoding (with INPUT_SCALE=4.0)")
    print("=" * 60)

    data = np.load(TOPO_PATH)
    adj = data['adj'].astype(np.float64)
    types = data['types'].astype(np.float64)
    print(f"Topology: {adj.shape[0]} neurons, density={adj.sum()/adj.size:.3f}")

    T_list = [1, 5, 10, 15, 20, 25, 30]
    results = run_decoding(adj, types, CONFIG, T_list)

    # Find optimal T
    pos_r2 = {T: results[T]['current'][0] for T in T_list}
    best_T = max(pos_r2, key=pos_r2.get)
    print(f"\nOptimal T for position decoding: T={best_T} (R²={pos_r2[best_T]:.3f})")

    # Decay analysis at T=20
    if 20 in results:
        r = results[20]
        lag4_pos = r['history'][3, 0]
        lag0_pos = r['current'][0]
        decay_pct = (1 - lag4_pos / lag0_pos) * 100 if lag0_pos > 0 else 0
        print(f"\nAt T=20:")
        print(f"  Current pos R²: {lag0_pos:.3f}")
        print(f"  Lag-4 pos R²:   {lag4_pos:.3f}")
        print(f"  Decay:          {decay_pct:.1f}%")

    plot_results(results, OUT_DIR)

    # Save data
    data_path = os.path.join(os.path.dirname(__file__), 'experiments', 'results',
                             'mountaincar_decoding_data.npy')
    np.save(data_path, results, allow_pickle=True)
    print(f"Data saved: {data_path}")
    print("=" * 60)
