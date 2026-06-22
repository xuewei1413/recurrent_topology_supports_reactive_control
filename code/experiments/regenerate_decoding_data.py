"""
Regenerate decoding_data.npy using the CORRECT methodology from run_fig04.py.

Key differences from the stale exp_input_decoding.py:
  1. Uses persistent membrane potential (controller.v) instead of zeros
  2. Uses cumulative spike counts (spike_trace.sum(axis=0)) instead of single-timestep binary spikes
  3. Includes both CartPole AND MountainCar
  4. Uses 80/20 train/test split (not training-set-only R^2)
  5. Runs for T_steps = [1, 5, 10, 15, 20] for both tasks

Output: experiments/results/decoding_data.npy with format:
  {
    'CartPole': { T: r2_array(shape=(10, 4)), ... },
    'MountainCar': { T: r2_array(shape=(10, 2)), ... },
    'T_list': [1, 5, 10, 15, 20],
    'cp_state_dims': ['Cart Pos', 'Cart Vel', 'Pole Ang', 'Pole Vel'],
    'mc_state_dims': ['Position', 'Velocity'],
  }
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import sys, os

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_ROOT)
from src.snn import SNNController, INPUT_SCALE
import gymnasium as gym


def forward_with_trace(controller, state):
    """Run T-step forward propagation using the controller's PERSISTENT
    membrane potential.  This matches the dynamics used during actual control
    (where v carries over between steps), giving realistic hidden-layer
    representations for the decoding analysis."""
    n = controller.n
    T = controller.T
    dt = controller.dt
    tau = controller.tau

    v = controller.v.copy()      # persistent state, not zeros
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


def collect_data(adj, types, config, n_episodes=30, env_id='CartPole-v1',
                 state_dim=4, max_steps=500, history_len=10):
    """Collect hidden activity (cumulative spike counts) + input history."""
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
            # Spike count accumulation -- the network's actual decision variable
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


def decode_history(X, Y, max_lag=10):
    """Train Ridge regression on 80%, test on 20%."""
    n_samples, state_dim = X.shape[0], Y.shape[2]
    n_train = int(0.8 * n_samples)
    X_train, X_test = X[:n_train], X[n_train:]
    r2_scores = np.zeros((max_lag, state_dim))

    for lag in range(max_lag):
        y_target = Y[:, max_lag - 1 - lag, :]
        y_train, y_test = y_target[:n_train], y_target[n_train:]
        for dim in range(state_dim):
            if np.std(y_train[:, dim]) < 1e-8:
                r2_scores[lag, dim] = 0.0
                continue
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train[:, dim])
            pred = model.predict(X_test)
            r2_scores[lag, dim] = r2_score(y_test[:, dim], pred)
    return r2_scores


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


if __name__ == '__main__':
    CROSS_TASK = os.path.join(CODE_ROOT, 'experiments', 'results', 'cross_task')
    RESULTS_DIR = os.path.join(CODE_ROOT, 'experiments', 'results')

    # T_steps values to sweep (all 5 values the audit expects)
    T_list = [1, 5, 10, 15, 20]

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

    # ── Save results ─────────────────────────────────────────────────
    save_data = {
        'CartPole': cp_results,
        'MountainCar': mc_results,
        'T_list': T_list,
        'cp_state_dims': ['Cart Pos', 'Cart Vel', 'Pole Ang', 'Pole Vel'],
        'mc_state_dims': ['Position', 'Velocity'],
    }

    save_path = os.path.join(RESULTS_DIR, 'decoding_data.npy')
    np.save(save_path, save_data, allow_pickle=True)
    print(f"\nSaved: {save_path}")

    # ── Report key values ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("KEY DATA VALUES FOR AUDIT")
    print("=" * 60)
    cp_dims = ['Cart Pos', 'Cart Vel', 'Pole Ang', 'Pole Vel']
    for T in sorted(cp_results.keys()):
        r2_mean = cp_results[T].mean(axis=1)
        r2 = cp_results[T][0]
        dim_str = '  '.join(f'{cp_dims[i]}={r2[i]:.3f}' for i in range(4))
        print(f"  CartPole T={T:2d}  R2(lag0)={r2_mean[0]:.3f}  "
              f"R2(lag4)={r2_mean[4]:.3f}")
        print(f"    per-dim: {dim_str}")

    mc_dims = ['Position', 'Velocity']
    for T in sorted(mc_results.keys()):
        r2_mean = mc_results[T].mean(axis=1)
        r2 = mc_results[T][0]
        dim_str = '  '.join(f'{mc_dims[i]}={r2[i]:.3f}' for i in range(2))
        print(f"  MountainCar T={T:2d}  R2(lag0)={r2_mean[0]:.3f}  "
              f"R2(lag4)={r2_mean[4]:.3f}")
        print(f"    per-dim: {dim_str}")

    # ── Verify paper claims ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PAPER CLAIM VERIFICATION")
    print("=" * 60)

    # CartPole T=5: R^2 ~ 0.63, best dim (Cart Vel) ~ 0.85
    if 5 in cp_results:
        cp5 = cp_results[5]
        mean_r2_0 = cp5[0].mean()
        best_dim_r2 = cp5[0].max()
        best_dim_idx = cp5[0].argmax()
        print(f"  CartPole T=5: mean R2(lag0)={mean_r2_0:.3f} (paper ~0.63)")
        print(f"  CartPole T=5: best dim={cp_dims[best_dim_idx]} R2={best_dim_r2:.3f} (paper ~0.85 for CartVel)")

    # CartPole T=20: R^2 ~ 0.75
    if 20 in cp_results:
        cp20 = cp_results[20]
        mean_r2_0 = cp20[0].mean()
        print(f"  CartPole T=20: mean R2(lag0)={mean_r2_0:.3f} (paper ~0.75)")

    # MountainCar T=20: Position R^2 ~ 0.59, Velocity R^2 ~ 0.74
    if 20 in mc_results:
        mc20 = mc_results[20]
        pos_r2 = mc20[0, 0]
        vel_r2 = mc20[0, 1]
        print(f"  MountainCar T=20: Position R2={pos_r2:.3f} (paper ~0.59)")
        print(f"  MountainCar T=20: Velocity R2={vel_r2:.3f} (paper ~0.74)")

    # MountainCar T=20 lag=4: R^2 ~ 0.61 (~8% decay)
    if 20 in mc_results:
        mc20 = mc_results[20]
        mean_lag0 = mc20[0].mean()
        mean_lag4 = mc20[4].mean()
        decay_pct = (mean_lag0 - mean_lag4) / mean_lag0 * 100
        print(f"  MountainCar T=20 lag=4: mean R2={mean_lag4:.3f} (paper ~0.61)")
        print(f"  MountainCar T=20 decay lag0->lag4: {decay_pct:.1f}% (paper ~8%)")

    print("\nDone.")
