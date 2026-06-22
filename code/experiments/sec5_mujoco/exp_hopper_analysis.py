"""
Hopper-v4 Deep Analysis: spike timing, structural fingerprint, connection types

Analyzes the best Hopper topology evolved in exp1C:
  1. Spike timing patterns across locomotion phases
  2. Structural fingerprint comparison with discrete task topologies
  3. E/I connection type distribution
  4. Input decoding from hidden-layer activity
  5. Generate publication-quality figure panels

Output:
  - results/embodied_reflex/hopper_deep_analysis/
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.continuous_snn import ContinuousSNNController

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'embodied_reflex')
OUT_DIR = os.path.join(RESULTS_DIR, 'hopper_deep_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load Hopper topology ───────────────────────────────────────────────────

topo_path = os.path.join(RESULTS_DIR, 'Hopper_v4_best_topology.npy')
types_path = os.path.join(RESULTS_DIR, 'Hopper_v4_neuron_types.npy')

if not os.path.exists(topo_path):
    print("ERROR: Hopper topology not found. Run exp1C first.")
    sys.exit(1)

adj = np.load(topo_path)
neuron_types = np.load(types_path)
n = adj.shape[0]
n_in, n_out = 22, 3  # 11 obs dims x 2 (pos/neg encoding), 3 joint torques
print(f"Loaded Hopper topology: n={n}, connections={int(adj.sum())}, density={adj.sum()/(n*(n-1)):.3f}")

# ── 1. Connection Type Analysis ─────────────────────────────────────────────

exc_idx = np.where(neuron_types == 1)[0]
inh_idx = np.where(neuron_types == -1)[0]
hidden_idx = np.arange(n_in, n - n_out)

n_EE = adj[exc_idx[:, None], exc_idx].sum()
n_EI = adj[exc_idx[:, None], inh_idx].sum()
n_IE = adj[inh_idx[:, None], exc_idx].sum()
n_II = adj[inh_idx[:, None], inh_idx].sum()
total = adj.sum()

in_to_hidden = adj[hidden_idx[:, None], np.arange(n_in)].sum()
hidden_internal = adj[hidden_idx[:, None], hidden_idx].sum()
hidden_to_out = adj[np.arange(n-n_out, n)[:, None], hidden_idx].sum()

print(f"\nConnection type breakdown:")
print(f"  E->E: {int(n_EE)} ({n_EE/total*100:.1f}%)")
print(f"  E->I: {int(n_EI)} ({n_EI/total*100:.1f}%)")
print(f"  I->E: {int(n_IE)} ({n_IE/total*100:.1f}%)")
print(f"  I->I: {int(n_II)} ({n_II/total*100:.1f}%)")
print(f"  in->hidden: {int(in_to_hidden)}, hidden->hidden: {int(hidden_internal)}, hidden->out: {int(hidden_to_out)}")

# Compare with CartPole connection type ratios (from B1 experiment)
cartpole_ref = {'EE': 0.54, 'EI': 0.20, 'IE': 0.19, 'II': 0.07}
hopper_ratios = {
    'EE': n_EE/total, 'EI': n_EI/total, 'IE': n_IE/total, 'II': n_II/total
}

# ── 2. Spike Timing Analysis During Locomotion ──────────────────────────────

def collect_spike_data(env_name='Hopper-v4', n_episodes=5, max_steps=1000):
    """Collect per-step spike data from Hopper evaluation."""
    import gymnasium as gym
    env = gym.make(env_name)

    config = {
        'n_total': n, 'n_input': n_in, 'n_output': n_out,
        'weight_scale': 0.6, 'tau': 5.0, 'threshold': 0.8, 'T_steps': 15,
    }
    controller = ContinuousSNNController(config, adj, neuron_types)

    all_spikes = []  # (timestep, neuron_idx)
    all_obs = []
    all_rewards = []
    all_actions = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        controller.reset()
        ep_spikes = []
        ep_obs = []
        ep_rewards = []
        ep_actions = []
        for step in range(max_steps):
            action = controller.forward(np.array(obs, dtype=np.float64))
            # Record spike activity from the forward pass (re-run to capture spikes)
            I_ext = controller.encode_state(np.array(obs, dtype=np.float64))
            leak = 1.0 - 1.0 / controller.tau
            spike_record = []
            for t in range(controller.T):
                controller.v *= leak
                # track spikes
                fired = controller.v >= controller.thresh
                if fired.any():
                    spike_record.append(np.where(fired)[0])
                controller.v[fired] = 0.0
                if t < controller.T - 1:
                    spikes_arr = fired.astype(np.float64)
                    controller.v += spikes_arr @ controller.W
                controller.v[:controller.n_in] += I_ext * 1.0 / controller.tau

            ep_spikes.append(spike_record)
            ep_obs.append(obs.copy())
            ep_rewards.append(0)  # placeholder
            ep_actions.append(action.copy())

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_rewards[-1] = reward
            if terminated or truncated:
                break

        all_spikes.append(ep_spikes)
        all_obs.append(np.array(ep_obs))
        all_rewards.append(np.array(ep_rewards))
        all_actions.append(np.array(ep_actions))
        print(f"  Episode {ep+1}: {len(ep_spikes)} steps, total reward {sum(ep_rewards):.1f}")

    env.close()
    return all_spikes, all_obs, all_rewards, all_actions

print("\nCollecting spike data...")
spike_data, obs_data, reward_data, action_data = collect_spike_data(n_episodes=3, max_steps=500)

# ── 3. Structural Fingerprint Extraction ────────────────────────────────────

def extract_fingerprint(adj, types, n_in, n_out):
    """Extract same ~19 features as 1B for cross-task comparison."""
    exc = types == 1
    inh = types == -1
    hidden = np.arange(n_in, n - n_out)
    total_conn = adj.sum()

    f = {}
    f['density'] = total_conn / (n * (n - 1)) if n > 1 else 0.0

    # Clustering coefficient
    adj_bin = (adj > 0).astype(int)
    C = 0.0; count = 0
    for i in range(n):
        neighbors = np.where(adj_bin[i] | adj_bin[:, i])[0]
        if len(neighbors) < 2: continue
        for j_idx, j in enumerate(neighbors):
            for k in neighbors[j_idx+1:]:
                tri = adj_bin[i,j]+adj_bin[j,i]+adj_bin[i,k]+adj_bin[k,i]+adj_bin[j,k]+adj_bin[k,j]
                if tri >= 3: C += 1.0
            count += len(neighbors)*(len(neighbors)-1)//2
    f['clustering'] = C / max(count, 1)

    # Path length
    total_len = 0; total_pairs = 0
    for src in range(min(n, 50)):  # sample for speed
        dist = np.full(n, -1); dist[src] = 0
        queue = [src]
        while queue:
            u = queue.pop(0)
            for v in np.where(adj_bin[u])[0]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1; queue.append(v)
        valid = dist[dist > 0]
        if len(valid) > 0:
            total_len += valid.sum(); total_pairs += len(valid)
    f['path_length'] = (total_len / max(total_pairs, 1)) / n

    # Degree entropy
    degrees = adj.sum(axis=0) + adj.sum(axis=1)
    probs = degrees / degrees.sum(); probs = probs[probs > 0]
    f['degree_entropy'] = -np.sum(probs * np.log2(probs)) / np.log2(n) if n > 1 else 0.0

    # E/I ratios
    f['ei_ratio'] = exc.sum() / n
    f['ee_ratio'] = adj[exc][:, exc].sum() / max(total_conn, 1)
    f['ei_ratio_conn'] = adj[exc][:, inh].sum() / max(total_conn, 1)
    f['ie_ratio'] = adj[inh][:, exc].sum() / max(total_conn, 1)
    f['ii_ratio'] = adj[inh][:, inh].sum() / max(total_conn, 1)
    f['exc_indeg'] = adj[:, exc].sum(axis=0).mean() / max(n, 1)
    f['inh_indeg'] = adj[:, inh].sum(axis=0).mean() / max(n, 1)

    # Layer connectivity
    if len(hidden) > 0:
        f['in_to_hidden'] = adj[hidden[:, None], np.arange(n_in)].sum() / max(len(hidden)*n_in, 1)
        f['hidden_to_out'] = adj[np.arange(n-n_out, n)[:, None], hidden].sum() / max(n_out*len(hidden), 1)
        f['hidden_internal'] = adj[hidden[:, None], hidden].sum() / max(len(hidden)**2, 1)
    else:
        f['in_to_hidden'] = f['hidden_to_out'] = f['hidden_internal'] = 0.0

    # Efficiency
    eff = 0.0; eff_count = 0
    for src in range(min(n, 50)):
        dist = np.full(n, -1); dist[src] = 0
        queue = [src]
        while queue:
            u = queue.pop(0)
            for v in np.where(adj_bin[u])[0]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1; queue.append(v)
        for d in dist:
            if d > 0: eff += 1.0/d; eff_count += 1
    f['info_efficiency'] = eff / max(eff_count, 1)
    f['inv_path_len'] = 1.0 / max(f['path_length'] * n, 1e-6)
    f['sparsity'] = 1.0 - f['density']
    f['modularity_ei'] = 0.0  # simplified

    return f

print("\nExtracting Hopper structural fingerprint...")
hopper_fp = extract_fingerprint(adj, neuron_types, n_in, n_out)
# Compare with 1B task averages
task_ref = {
    'CartPole': {'density': 0.352, 'ee_ratio': 0.523, 'clustering': 0.729, 'info_efficiency': 0.676, 'ei_ratio': 0.727},
    'MountainCar': {'density': 0.192, 'ee_ratio': 0.536, 'clustering': 0.479, 'info_efficiency': 0.582, 'ei_ratio': 0.734},
    'Acrobot': {'density': 0.173, 'ee_ratio': 0.543, 'clustering': 0.436, 'info_efficiency': 0.579, 'ei_ratio': 0.742},
    'FlappyBird': {'density': 0.201, 'ee_ratio': 0.558, 'clustering': 0.493, 'info_efficiency': 0.590, 'ei_ratio': 0.750},
    'Snake': {'density': 0.168, 'ee_ratio': 0.608, 'clustering': 0.426, 'info_efficiency': 0.568, 'ei_ratio': 0.775},
}

# Compute Euclidean distance from Hopper to each task
def fp_distance(fp1, fp2, keys):
    return np.sqrt(sum((fp1[k] - fp2[k])**2 for k in keys))

compare_keys = ['density', 'ee_ratio', 'clustering', 'info_efficiency', 'ei_ratio']
print("\nStructural fingerprint distance (Hopper vs tasks):")
for task, ref in task_ref.items():
    dist = fp_distance(hopper_fp, ref, compare_keys)
    print(f"  Hopper <-> {task}: {dist:.4f}")

# ── 4. Visualizations ───────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# (a) Connection type distribution
ax = axes[0, 0]
labels = ['E→E', 'E→I', 'I→E', 'I→I']
hopper_vals = [hopper_ratios[k] for k in ['EE', 'EI', 'IE', 'II']]
cartpole_vals = [cartpole_ref[k] for k in ['EE', 'EI', 'IE', 'II']]
x = np.arange(len(labels))
w = 0.35
ax.bar(x - w/2, hopper_vals, w, label='Hopper-v4', color='steelblue')
ax.bar(x + w/2, cartpole_vals, w, label='CartPole (ref)', color='lightcoral', alpha=0.7)
ax.set_ylabel('Proportion of Total Connections')
ax.set_title('Connection Type Distribution')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# (b) Structural fingerprint radar
ax = axes[0, 1]
fp_keys = ['density', 'clustering', 'info_efficiency', 'ee_ratio', 'ei_ratio']
n_fp = len(fp_keys)
angles = np.linspace(0, 2*np.pi, n_fp, endpoint=False).tolist()
angles += angles[:1]
for task, ref in task_ref.items():
    vals = [ref[k] for k in fp_keys]
    vals += vals[:1]
    ax.plot(angles, vals, 'o-', linewidth=1, alpha=0.5, label=task, markersize=3)
hopper_fp_vals = [hopper_fp[k] for k in fp_keys]
hopper_fp_vals += hopper_fp_vals[:1]
ax.plot(angles, hopper_fp_vals, 'o-', linewidth=2.5, color='black', label='Hopper-v4', markersize=6)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(fp_keys, fontsize=8)
ax.set_title('Structural Fingerprint Comparison')
ax.legend(fontsize=7, loc='lower right')

# (c) Spike raster for best episode
ax = axes[0, 2]
best_ep = np.argmax([r.sum() for r in reward_data])
ep_spikes = spike_data[best_ep]
n_steps = len(ep_spikes)
# Build raster: step x n
raster = np.zeros((n_steps, n))
for t, spikes_at_t in enumerate(ep_spikes):
    for spike_set in spikes_at_t:
        raster[t, spike_set] += 0.3
# Show first 200 steps for clarity
show_steps = min(200, n_steps)
ax.imshow(raster[:show_steps].T, aspect='auto', cmap='Reds', interpolation='nearest')
ax.axhline(y=n_in-0.5, color='blue', linewidth=1, linestyle='--', alpha=0.5)
ax.axhline(y=n-n_out-0.5, color='green', linewidth=1, linestyle='--', alpha=0.5)
ax.set_xlabel('Timestep')
ax.set_ylabel('Neuron Index')
ax.set_title(f'Spike Raster (Episode {best_ep+1})')

# (d) Fingerprint distance bar chart
ax = axes[1, 0]
task_names = list(task_ref.keys())
distances = [fp_distance(hopper_fp, task_ref[t], compare_keys) for t in task_names]
colors = ['steelblue' if d == min(distances) else 'lightgray' for d in distances]
ax.bar(task_names, distances, color=colors)
ax.set_ylabel('Euclidean Distance')
ax.set_title('Hopper Fingerprint Distance to Discrete Tasks')
# Highlight closest match
closest = task_names[np.argmin(distances)]
ax.text(0.5, 0.95, f'Closest: {closest}', transform=ax.transAxes, ha='center', fontsize=10)

# (e) Layer-wise connectivity comparison
ax = axes[1, 1]
layer_labels = ['In→Hidden', 'Hidden→Hidden', 'Hidden→Out']
hopper_layer = [hopper_fp['in_to_hidden'], hopper_fp['hidden_internal'], hopper_fp['hidden_to_out']]
cartpole_layer = [0.350, 0.351, 0.333]  # from 1B CartPole averages
mcar_layer = [0.185, 0.188, 0.205]  # from 1B MountainCar averages
x = np.arange(len(layer_labels))
ax.bar(x - 0.2, hopper_layer, 0.2, label='Hopper', color='steelblue')
ax.bar(x, cartpole_layer, 0.2, label='CartPole', color='lightcoral', alpha=0.7)
ax.bar(x + 0.2, mcar_layer, 0.2, label='MountainCar', color='lightgreen', alpha=0.7)
ax.set_ylabel('Density')
ax.set_title('Layer-wise Connectivity')
ax.set_xticks(x)
ax.set_xticklabels(layer_labels)
ax.legend()

# (f) Connection count vs reward (episode level)
ax = axes[1, 2]
ep_rewards = [r.sum() for r in reward_data]
ep_conn = adj.sum()
ax.bar(range(len(ep_rewards)), ep_rewards, color='steelblue')
ax.axhline(y=1307.2, color='coral', linestyle='--', label=f'PPO 100k mean (1307.2)')
ax.set_xlabel('Episode')
ax.set_ylabel('Total Reward')
ax.set_title(f'Hopper SNN Performance ({int(ep_conn)} connections)')
ax.legend()

plt.suptitle('Hopper-v4 Deep Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, 'hopper_deep_analysis.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {fig_path}")

# ── 5. Save report ──────────────────────────────────────────────────────────

report = {
    'topology': {
        'n_total': int(n), 'n_connections': int(adj.sum()),
        'density': float(adj.sum()/(n*(n-1))),
    },
    'connection_types': {k: float(v) for k, v in hopper_ratios.items()},
    'fingerprint': {k: float(v) for k, v in hopper_fp.items()},
    'fingerprint_distances': {t: float(d) for t, d in zip(task_names, distances)},
    'closest_discrete_task': closest,
    'episode_rewards': [float(r.sum()) for r in reward_data],
}

report_path = os.path.join(OUT_DIR, 'hopper_deep_analysis.json')
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"Report saved: {report_path}")
print("Done.")
