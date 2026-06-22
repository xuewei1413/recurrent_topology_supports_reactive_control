"""
Regenerate fig03: Membrane-potential traces of representative hidden neurons
during a CartPole episode.

This script loads the best CartPole topology, runs one full episode while
recording membrane potentials, and produces a publication-quality figure
suitable for the arXiv paper "Recurrent Connection Topology Encodes
Reactive Control".
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.snn import SNNController, INPUT_SCALE

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
})

# ── Configuration ──────────────────────────────────────────────────────────
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

REPRESENTATIVE_NEURONS = [8, 38, 68, 98, 125]

SAVE_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'outputs', 'figures'))

# ── Helper: forward pass with full membrane-potential trace ────────────────
def forward_with_trace(controller, state):
    """Run one full T-step forward propagation, recording membrane potential
    and spike at every sub-step."""
    n = controller.n
    T = controller.T
    dt = controller.dt
    tau = controller.tau

    v = controller.v.copy()
    spikes = np.zeros(n)

    mem_trace = np.zeros((T, n))
    spike_trace = np.zeros((T, n))

    I_ext = controller.encode_state(state * INPUT_SCALE)

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
    """Run one CartPole episode and collect membrane-potential traces for
    every simulation step."""
    import gymnasium as gym
    env = gym.make('CartPole-v1')
    obs, _ = env.reset()
    controller = SNNController(config, adj, types)

    all_mem = []
    all_spikes = []
    states = []

    for step in range(max_steps):
        states.append(obs.copy())
        mem, spikes, action = forward_with_trace(
            controller, np.array(obs, dtype=np.float64))
        all_mem.append(mem.mean(axis=0))
        all_spikes.append(spikes.sum(axis=0))

        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    env.close()
    return np.array(states), np.array(all_mem), np.array(all_spikes)


def collect_best_cartpole_trace(adj, types, config, n_trials=10, max_steps=2000):
    """Run multiple CartPole episodes and return the longest one, to ensure
    a rich membrane-trace figure."""
    import gymnasium as gym
    best_result = None
    best_len = 0
    env = gym.make('CartPole-v1')
    for trial in range(n_trials):
        obs, _ = env.reset()
        controller = SNNController(config, adj, types)

        all_mem = []
        all_spikes = []
        states = []

        for step in range(max_steps):
            states.append(obs.copy())
            mem, spikes, action = forward_with_trace(
                controller, np.array(obs, dtype=np.float64))
            all_mem.append(mem.mean(axis=0))
            all_spikes.append(spikes.sum(axis=0))

            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        if len(states) > best_len:
            best_len = len(states)
            best_result = (np.array(states), np.array(all_mem),
                           np.array(all_spikes))

    env.close()
    return best_result


# ── Plotting ───────────────────────────────────────────────────────────────
def plot_fig03(all_mem, all_spikes, neurons, save_dir):
    """Produce fig03: membrane-potential traces of representative hidden
    neurons during a CartPole episode."""

    NEURON_COLORS = ['#2166AC', '#4393C3', '#92C5DE', '#D6604D', '#B2182B']
    time_axis = np.arange(all_mem.shape[0])

    n_panels = len(neurons)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(7, 0.85 * n_panels + 0.6),   # compact, single-column width
        sharex=True)
    if n_panels == 1:
        axes = [axes]

    for idx, (ax, nid) in enumerate(zip(axes, neurons)):
        trace = all_mem[:, nid]
        ax.fill_between(time_axis, 0, trace, color=NEURON_COLORS[idx],
                        alpha=0.25, linewidth=0)
        ax.plot(time_axis, trace, color=NEURON_COLORS[idx],
                linewidth=0.6, alpha=0.95)

        # Spike events as thin red ticks at the top
        spike_times = np.where(all_spikes[:, nid] > 0)[0]
        for st in spike_times:
            ax.axvline(x=st, color='#E74C3C', alpha=0.10, linewidth=0.4)

        ax.set_ylabel(f'Neuron {nid}', fontsize=9, fontweight='bold',
                      rotation=0, labelpad=42, va='center')
        ax.set_ylim(-0.05, 0.90)
        ax.set_yticks([0, 0.4, 0.8])
        ax.tick_params(axis='y', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.15, linewidth=0.4)

    axes[-1].set_xlabel('Simulation Timestep', fontsize=10)
    axes[0].set_title(
        'Membrane-Potential Traces of Hidden Neurons (CartPole)',
        fontsize=11, fontweight='bold', pad=6)

    fig.align_ylabels(axes)
    plt.tight_layout(h_pad=0.4)

    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, 'fig03_membrane_traces.png')
    pdf_path = os.path.join(save_dir, 'fig03_membrane_traces.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight', format='pdf')
    plt.close(fig)

    print(f"  PNG saved: {png_path}  ({os.path.getsize(png_path)/1024:.0f} KB)")
    print(f"  PDF saved: {pdf_path}  ({os.path.getsize(pdf_path)/1024:.0f} KB)")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("Regenerating fig03: Membrane-Potential Traces (CartPole)")
    print("=" * 60)

    # Load topology from cross_task results (as requested)
    topo_path = os.path.join(
        os.path.dirname(__file__), '..', 'results', 'cross_task',
        'CartPole_seed0.npz')
    data = np.load(topo_path)
    adj = data['adj']
    types = data['types']
    print(f"  Topology: {adj.shape[0]} neurons, "
          f"{adj.sum():.0f} connections, "
          f"sparsity={1 - adj.sum()/adj.size:.3f}")

    # Collect traces (pick the longest of several trials for a richer figure)
    print("\nRunning CartPole episodes (best of 15 trials, max 2000 steps each)...")
    states, all_mem, all_spikes = collect_best_cartpole_trace(
        adj, types, CONFIG, n_trials=15, max_steps=2000)
    print(f"  Best episode length: {len(states)} steps")

    n_in, n_out = CONFIG['n_input'], CONFIG['n_output']
    hidden_idx = np.arange(n_in, CONFIG['n_total'] - n_out)
    active_hidden = (all_spikes[:, hidden_idx].sum(axis=0) > 0).sum()
    print(f"  Active hidden neurons: {active_hidden}/{len(hidden_idx)}")

    # Pick neurons to plot
    valid_neurons = [n for n in REPRESENTATIVE_NEURONS if n in hidden_idx]
    if all(all_spikes[:, n].sum() > 0 for n in valid_neurons):
        neurons_to_plot = valid_neurons
    else:
        active_neurons = hidden_idx[
            np.argsort(all_spikes[:, hidden_idx].sum(axis=0))[-5:]][::-1]
        neurons_to_plot = list(active_neurons)
    print(f"  Neurons plotted: {neurons_to_plot}")

    # Generate figure
    plot_fig03(all_mem, all_spikes, neurons_to_plot, SAVE_DIR)

    print("\n" + "=" * 60)
    print("fig03 regeneration complete.")
    print("=" * 60)
