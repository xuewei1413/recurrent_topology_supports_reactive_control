"""
Fig 04: Input Decoding — Recurrent Connection Topology Encodes Reactive Control

Demonstrates that a linear decoder can reconstruct input history from hidden-layer
activity, with the optimal decoding window matching the task timescale:
  - CartPole: T ~ 5 (fast reactive control)
  - MountainCar: T ~ 20 (momentum-based, longer memory needed)

Runs decoding experiments for both tasks across multiple T_steps values,
showing how R^2 of current-input decoding peaks at the task-appropriate T.
Also shows history-decoding decay for the best T.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import sys, os

CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, CODE_ROOT)
from src.snn import SNNController, INPUT_SCALE
import gymnasium as gym

# ── Global plot style ──────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'lines.markersize': 7,
})


# ── Helper: run SNN for T_steps and return membrane/spike traces ──────
def forward_with_trace(controller, state):
    """Run T-step forward propagation using the controller's PERSISTENT
    membrane potential.  This matches the dynamics used during actual control
    (where v carries over between steps), giving realistic hidden-layer
    representations for the decoding analysis."""
    n = controller.n
    T = controller.T
    dt = controller.dt
    tau = controller.tau

    v = controller.v.copy()      # ← persistent state, not zeros
    spikes = np.zeros(n)
    mem_trace = np.zeros((T, n))
    spike_trace = np.zeros((T, n))

    I_ext = controller.encode_state(np.asarray(state, dtype=np.float64) * INPUT_SCALE)
    leak = 1.0 - dt / tau

    for t in range(T):
        v *= leak
        if spikes.any():
            v += spikes @ controller.W
        v[:controller.n_in] += I_ext * dt / tau
        fired = v >= controller.thresh
        spikes = fired.astype(np.float64)
        v[fired] = 0.0
        mem_trace[t] = v.copy()
        spike_trace[t] = spikes.copy()

    controller.v = v.copy()    # persist membrane potential for next step

    action = int(np.argmax(spike_trace.sum(axis=0)[-controller.n_out:]))
    return mem_trace, spike_trace, action


# ── Collect hidden activity + input history for one task ──────────────
def collect_data(adj, types, config, n_episodes=30, env_id='CartPole-v1',
                 state_dim=4, max_steps=500, history_len=10):
    """
    Feature vector = spike-count accumulation over T_steps for hidden neurons.
    This is the natural state representation — the same quantity the network
    uses to decide its output action. More T_steps gives the recurrent dynamics
    more time to encode input history into the spike pattern.
    """
    n_in = config['n_input']
    n_out = config['n_output']
    hidden_idx = list(range(n_in, config['n_total'] - n_out))

    X_list, Y_list = [], []
    env = gym.make(env_id)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        controller = SNNController(config, adj, types)
        input_history = [np.zeros(state_dim) for _ in range(history_len)]

        for step in range(max_steps):
            mem_trace, spike_trace, action = forward_with_trace(
                controller, np.asarray(obs, dtype=np.float64))
            # Spike count accumulation — the network's actual decision variable
            hidden_activity = spike_trace.sum(axis=0)[hidden_idx]

            X_list.append(hidden_activity)
            Y_list.append(np.array(input_history[-history_len:]))

            input_history.append(obs.copy())
            if len(input_history) > history_len:
                input_history.pop(0)

            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    env.close()
    return np.array(X_list), np.array(Y_list)


# ── Ridge decoding: predict input at each lag from hidden activity ────
def decode_history(X, Y, max_lag=10):
    """
    Train Ridge regression on 80%, test on 20%.
    Skips lags where the target is essentially constant (all zeros from padding).
    """
    n_samples, state_dim = X.shape[0], Y.shape[2]
    n_train = int(0.8 * n_samples)
    X_train, X_test = X[:n_train], X[n_train:]
    r2_scores = np.zeros((max_lag, state_dim))

    for lag in range(max_lag):
        y_target = Y[:, max_lag - 1 - lag, :]
        y_train, y_test = y_target[:n_train], y_target[n_train:]
        for dim in range(state_dim):
            # Skip if target has zero variance (padding artifacts)
            if np.std(y_train[:, dim]) < 1e-8:
                r2_scores[lag, dim] = 0.0
                continue
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train[:, dim])
            pred = model.predict(X_test)
            r2_scores[lag, dim] = r2_score(y_test[:, dim], pred)
    return r2_scores


# ── Main experiment runner ────────────────────────────────────────────
def run_task_decoding(task_name, adj_path, env_id, state_dim, n_total,
                      n_input, n_output, max_steps, T_list, n_episodes=30):
    print(f"\n{'='*60}")
    print(f"  Task: {task_name}")
    print(f"{'='*60}")

    data = np.load(adj_path)
    adj = data['adj']
    types = data['types']

    all_results = {}
    for T in T_list:
        cfg = {
            'n_total': n_total, 'n_input': n_input, 'n_output': n_output,
            'tau': 5.0, 'threshold': 0.8, 'weight_scale': 0.6,
            'sparsity': 0.1, 'T_steps': T,
        }
        print(f"  T_steps={T} ...", end=' ', flush=True)
        X, Y = collect_data(adj, types, cfg, n_episodes=n_episodes,
                            env_id=env_id, state_dim=state_dim,
                            max_steps=max_steps, history_len=10)
        r2 = decode_history(X, Y, max_lag=10)
        all_results[T] = r2
        mean_r2 = r2.mean(axis=1)
        print(f"samples={len(X)}, R2(lag0)={mean_r2[0]:.3f}, "
              f"R2(lag4)={mean_r2[4]:.3f}")
    return all_results


# ══════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════
def plot_fig04(cp_results, mc_results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # ── Color / marker palette ────────────────────────────────────────
    cp_colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0']
    mc_colors = ['#b2182b', '#ef8a62']
    markers = ['o', 's', '^', 'D']
    cp_state = ['Cart Pos', 'Cart Vel', 'Pole Ang', 'Pole Vel']
    mc_state = ['Position', 'Velocity']

    # ── Panel A: CartPole — current-input R^2 vs T_steps ──────────────
    ax = axes[0]
    T_list_cp = sorted(cp_results.keys())
    for dim in range(4):
        vals = [cp_results[T][0, dim] for T in T_list_cp]
        ax.plot(T_list_cp, vals, marker=markers[dim], color=cp_colors[dim],
                label=cp_state[dim])
    # Find and mark the peak T
    mean_per_T = {T: cp_results[T][0].mean() for T in T_list_cp}
    peak_T = max(mean_per_T, key=mean_per_T.get)
    ax.axvline(peak_T, color='gray', ls='--', lw=1, alpha=0.7,
               label=f'Peak $T={peak_T}$')
    ax.set_xlabel('$T_{\\mathrm{steps}}$ (channel length)')
    ax.set_ylabel('$R^2$ (current input)')
    ax.set_title('(a) CartPole', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.set_ylim(-0.05, 1.05)

    # ── Panel B: MountainCar — current-input R^2 vs T_steps ───────────
    ax = axes[1]
    T_list_mc = sorted(mc_results.keys())
    mc_markers = ['o', 's']
    for dim in range(2):
        vals = [mc_results[T][0, dim] for T in T_list_mc]
        ax.plot(T_list_mc, vals, marker=mc_markers[dim], color=mc_colors[dim],
                label=mc_state[dim])
    mean_per_T_mc = {T: mc_results[T][0].mean() for T in T_list_mc}
    peak_T_mc = max(mean_per_T_mc, key=mean_per_T_mc.get)
    ax.axvline(peak_T_mc, color='gray', ls='--', lw=1, alpha=0.7,
               label=f'Peak $T={peak_T_mc}$')
    ax.set_xlabel('$T_{\\mathrm{steps}}$ (channel length)')
    ax.set_ylabel('$R^2$ (current input)')
    ax.set_title('(b) MountainCar', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.set_ylim(-0.05, 1.05)

    # ── Panel C: History decoding decay at peak T for each task ───────
    ax = axes[2]
    best_T_cp = peak_T
    mean_r2_cp = cp_results[best_T_cp].mean(axis=1)
    lags = np.arange(len(mean_r2_cp))
    ax.plot(lags, mean_r2_cp, marker='o', color='#2166ac',
            label=f'CartPole ($T={best_T_cp}$)')

    best_T_mc = peak_T_mc
    mean_r2_mc = mc_results[best_T_mc].mean(axis=1)
    lags_mc = np.arange(len(mean_r2_mc))
    ax.plot(lags_mc, mean_r2_mc, marker='s', color='#b2182b',
            label=f'MountainCar ($T={best_T_mc}$)')

    ax.set_xlabel('Temporal lag (steps back)')
    ax.set_ylabel('$R^2$ (mean over state dims)')
    ax.set_title('(c) History decoding decay', fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    png_path = os.path.join(save_dir, 'fig04_input_decoding.png')
    pdf_path = os.path.join(save_dir, 'fig04_input_decoding.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")
    return png_path, pdf_path


# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    CROSS_TASK = os.path.join(CODE_ROOT, 'experiments', 'results', 'cross_task')
    FIG_DIR = os.path.abspath(os.path.join(CODE_ROOT, '..', 'outputs', 'figures'))

    # T_steps values to sweep
    T_list = [1, 3, 5, 10, 15, 20, 30]

    # ── CartPole ──────────────────────────────────────────────────────
    cp_results = run_task_decoding(
        task_name='CartPole',
        adj_path=os.path.join(CROSS_TASK, 'CartPole_seed0.npz'),
        env_id='CartPole-v1',
        state_dim=4,
        n_total=128, n_input=8, n_output=2,
        max_steps=500,
        T_list=T_list,
        n_episodes=30,
    )

    # ── MountainCar ───────────────────────────────────────────────────
    mc_results = run_task_decoding(
        task_name='MountainCar',
        adj_path=os.path.join(CROSS_TASK, 'MountainCar_seed9.npz'),
        env_id='MountainCar-v0',
        state_dim=2,
        n_total=64, n_input=4, n_output=3,
        max_steps=200,
        T_list=T_list,
        n_episodes=30,
    )

    # ── Plot & save ───────────────────────────────────────────────────
    png_path, pdf_path = plot_fig04(cp_results, mc_results, FIG_DIR)

    # ── Report key data values ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("KEY DATA VALUES")
    print("=" * 60)
    cp_dims = ['CartPos', 'CartVel', 'PoleAng', 'PoleVel']
    for T in sorted(cp_results.keys()):
        r2_mean = cp_results[T].mean(axis=1)
        print(f"  CartPole T={T:2d}  R2(lag0)={r2_mean[0]:.3f}  "
              f"R2(lag4)={r2_mean[4]:.3f}  R2(lag9)={r2_mean[9]:.3f}")
        # Per-dimension at lag=0
        r2 = cp_results[T][0]
        dim_str = '  '.join(f'{cp_dims[i]}={r2[i]:.3f}' for i in range(4))
        print(f"    per-dim: {dim_str}")
    print()
    mc_dims = ['Position', 'Velocity']
    for T in sorted(mc_results.keys()):
        r2_mean = mc_results[T].mean(axis=1)
        print(f"  MountainCar T={T:2d}  R2(lag0)={r2_mean[0]:.3f}  "
              f"R2(lag4)={r2_mean[4]:.3f}  R2(lag9)={r2_mean[9]:.3f}")
        # Per-dimension at lag=0
        r2 = mc_results[T][0]
        dim_str = '  '.join(f'{mc_dims[i]}={r2[i]:.3f}' for i in range(2))
        print(f"    per-dim: {dim_str}")

    # File sizes
    for p in (png_path, pdf_path):
        sz_kb = os.path.getsize(p) / 1024
        print(f"  {os.path.basename(p)}: {sz_kb:.1f} KB")
