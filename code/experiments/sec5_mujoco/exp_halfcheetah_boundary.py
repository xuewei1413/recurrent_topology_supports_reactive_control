"""
HalfCheetah-v4 Local Optimum Escape Experiment
Pre-registered: see results/preregistration_halfcheetah_escape.md

Strategies:
  S1: Quality-Diversity (simplified MAP-Elites)
  S3: Fitness Shaping (reward forward velocity, penalize stillness)
  S4: Larger Population + More Generations
  S5: Per-Output Weight Scales (6 independent ws, addresses C(6,2)=15 bottleneck)
"""
import numpy as np
import gymnasium as gym
import os, sys, time, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.continuous_snn import ContinuousSNNController

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'embodied_reflex', 'halfcheetah_escape')
os.makedirs(OUT_DIR, exist_ok=True)

ENV_NAME = 'HalfCheetah-v4'
N_TOTAL = 96
N_IN = 34   # 17 obs x 2
N_OUT = 6
SPARSITY = 0.12
N_EVAL_STRICT = 10  # strict eval episodes

BASE_CONFIG = {
    'n_total': N_TOTAL, 'n_input': N_IN, 'n_output': N_OUT,
    'weight_scale': 0.6, 'tau': 5.0, 'threshold': 0.8, 'T_steps': 15,
}


def random_topology(n, sparsity):
    adj = (np.random.rand(n, n) < sparsity).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    return adj


def mutate_topology(adj, rate):
    child = adj.copy()
    mask = np.random.rand(*child.shape) < rate
    child[mask] = 1.0 - child[mask]
    np.fill_diagonal(child, 0.0)
    return child


def make_neuron_types():
    types = np.ones(N_TOTAL, dtype=np.float64)
    hidden = np.arange(N_IN, N_TOTAL - N_OUT)
    n_inh = int(len(hidden) * 0.3)
    inh_idx = np.random.choice(hidden, n_inh, replace=False)
    types[inh_idx] = -1.0
    return types


def evaluate_strict(adj, types, config, n_eps=N_EVAL_STRICT):
    """10-episode strict evaluation, returns mean reward."""
    controller = ContinuousSNNController(config, adj, types)
    env = gym.make(ENV_NAME)
    total = 0.0
    for _ in range(n_eps):
        obs, _ = env.reset()
        controller.reset()
        ep = 0.0
        for _ in range(1000):
            action = controller.forward(np.array(obs, dtype=np.float64))
            obs, r, term, trunc, _ = env.step(action)
            ep += r
            if term or trunc:
                break
        total += ep
    env.close()
    return total / n_eps


# ══════════════════════════════════════════════════════════════════════════════
# S3: Fitness Shaping
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_shaped(adj, types, config, n_eps=5):
    """Fitness that rewards forward velocity, punishes standing still."""
    controller = ContinuousSNNController(config, adj, types)
    env = gym.make(ENV_NAME)
    total = 0.0
    for _ in range(n_eps):
        obs, _ = env.reset()
        controller.reset()
        prev_x = obs[0]  # initial x position
        ep_fitness = 0.0
        for _ in range(1000):
            action = controller.forward(np.array(obs, dtype=np.float64))
            obs, r, term, trunc, _ = env.step(action)
            forward_vel = (obs[0] - prev_x) / env.dt if hasattr(env, 'dt') else (obs[0] - prev_x) / 0.05
            prev_x = obs[0]
            # Shaped: base reward + velocity bonus - stillness penalty
            shaped_r = r + 5.0 * max(0, forward_vel) - 2.0 * abs(action).mean()
            ep_fitness += shaped_r
            if term or trunc:
                break
        total += ep_fitness
    env.close()
    return total / n_eps


def run_s3_fitness_shaping(seed=42):
    """S3: Standard GA but with shaped fitness."""
    print('\n' + '='*60)
    print('S3: Fitness Shaping (velocity bonus + action penalty)')
    print('='*60)

    np.random.seed(seed)
    types = make_neuron_types()
    config = BASE_CONFIG.copy()

    pop_size = 40
    n_gen = 80
    pop = [random_topology(N_TOTAL, SPARSITY) for _ in range(pop_size)]

    best_adj = None
    best_fitness = -np.inf
    best_hist = []
    mean_hist = []

    for gen in range(n_gen):
        fitnesses = np.array([evaluate_shaped(adj, types, config) for adj in pop])
        order = np.argsort(fitnesses)[::-1]
        gen_best = fitnesses[order[0]]
        best_hist.append(float(gen_best))
        mean_hist.append(float(fitnesses.mean()))

        if gen_best > best_fitness:
            best_fitness = gen_best
            best_adj = pop[order[0]].copy()

        if (gen + 1) % 20 == 0:
            print(f'  Gen {gen+1:3d}: Best={gen_best:.1f}, Mean={fitnesses.mean():.1f}', flush=True)

        new_pop = [pop[i].copy() for i in order[:3]]
        while len(new_pop) < pop_size:
            parent = pop[np.random.choice(order[:10])]
            new_pop.append(mutate_topology(parent, 0.03))
        pop = new_pop

    # Strict eval
    strict_score = evaluate_strict(best_adj, types, config)
    print(f'  S3 Best shaped: {best_fitness:.1f}, Strict 10-ep: {strict_score:.1f}')
    return best_adj, types, strict_score


# ══════════════════════════════════════════════════════════════════════════════
# S4: Larger Population + More Generations
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_standard(adj, types, config, n_eps=5):
    controller = ContinuousSNNController(config, adj, types)
    env = gym.make(ENV_NAME)
    total = 0.0
    for _ in range(n_eps):
        obs, _ = env.reset()
        controller.reset()
        ep = 0.0
        for _ in range(1000):
            action = controller.forward(np.array(obs, dtype=np.float64))
            obs, r, term, trunc, _ = env.step(action)
            ep += r
            if term or trunc: break
        total += ep
    env.close()
    return total / n_eps


def run_s4_larger_population(seed=43):
    """S4: Double population, double generations."""
    print('\n' + '='*60)
    print('S4: Larger Population (80 pop × 120 gen)')
    print('='*60)

    np.random.seed(seed)
    types = make_neuron_types()
    config = BASE_CONFIG.copy()

    pop_size = 80
    n_gen = 120
    pop = [random_topology(N_TOTAL, SPARSITY) for _ in range(pop_size)]

    best_adj = None
    best_fitness = -np.inf
    best_hist = []

    for gen in range(n_gen):
        fitnesses = np.array([evaluate_standard(adj, types, config) for adj in pop])
        order = np.argsort(fitnesses)[::-1]
        gen_best = fitnesses[order[0]]
        best_hist.append(float(gen_best))

        if gen_best > best_fitness:
            best_fitness = gen_best
            best_adj = pop[order[0]].copy()

        if (gen + 1) % 30 == 0:
            print(f'  Gen {gen+1:3d}: Best={gen_best:.1f}, Mean={fitnesses.mean():.1f}', flush=True)

        new_pop = [pop[i].copy() for i in order[:5]]
        while len(new_pop) < pop_size:
            parent = pop[np.random.choice(order[:20])]
            new_pop.append(mutate_topology(parent, 0.03))
        pop = new_pop

    strict_score = evaluate_strict(best_adj, types, config)
    print(f'  S4 Best: {best_fitness:.1f}, Strict 10-ep: {strict_score:.1f}')
    return best_adj, types, strict_score


# ══════════════════════════════════════════════════════════════════════════════
# S5: Per-Output Weight Scales
# ══════════════════════════════════════════════════════════════════════════════

class PerOutputScaleController(ContinuousSNNController):
    """SNN with independent weight_scale per output neuron."""

    def __init__(self, config, adjacency, neuron_types, output_scales):
        """
        output_scales: array of length n_out, one scale per output neuron.
        The weight matrix scales columns of output neurons by these values.
        """
        super().__init__(config, adjacency, neuron_types)
        # Override W with per-output scaling
        scale_vec = np.ones(config['n_total']) * config['weight_scale']
        scale_vec[-config['n_output']:] = output_scales
        self.W = adjacency * neuron_types[:, np.newaxis] * scale_vec[np.newaxis, :]


def evaluate_s5(adj, types, base_config, output_scales, n_eps=5):
    """Evaluate with per-output weight scales."""
    config = base_config.copy()
    controller = PerOutputScaleController(config, adj, types, output_scales)
    env = gym.make(ENV_NAME)
    total = 0.0
    for _ in range(n_eps):
        obs, _ = env.reset()
        controller.reset()
        ep = 0.0
        for _ in range(1000):
            action = controller.forward(np.array(obs, dtype=np.float64))
            obs, r, term, trunc, _ = env.step(action)
            ep += r
            if term or trunc: break
        total += ep
    env.close()
    return total / n_eps


def run_s5_per_output_scales(seed=44):
    """S5: Evolve topology + grid-search per-output weight scales."""
    print('\n' + '='*60)
    print('S5: Per-Output Weight Scales (6 independent parameters)')
    print('='*60)
    print('  This directly addresses the C(6,2)=15 coordination bottleneck')
    print('  by giving each joint its own signal gain.')

    np.random.seed(seed)
    types = make_neuron_types()
    config = BASE_CONFIG.copy()

    # Phase 1: Evolve topology with standard single ws (faster search)
    pop_size = 40
    n_gen = 60
    pop = [random_topology(N_TOTAL, SPARSITY) for _ in range(pop_size)]

    best_adj = None
    best_fitness = -np.inf

    print('  Phase 1: Topology evolution (60 gen, ws=0.6 fixed)...')
    for gen in range(n_gen):
        fitnesses = np.array([evaluate_standard(adj, types, config) for adj in pop])
        order = np.argsort(fitnesses)[::-1]
        gen_best = fitnesses[order[0]]

        if gen_best > best_fitness:
            best_fitness = gen_best
            best_adj = pop[order[0]].copy()

        if (gen + 1) % 20 == 0:
            print(f'    Gen {gen+1:3d}: Best={gen_best:.1f}', flush=True)

        new_pop = [pop[i].copy() for i in order[:3]]
        while len(new_pop) < pop_size:
            parent = pop[np.random.choice(order[:10])]
            new_pop.append(mutate_topology(parent, 0.03))
        pop = new_pop
    print(f'  Phase 1 done. Best: {best_fitness:.1f}')

    # Phase 2: Grid search per-output weight scales
    print('  Phase 2: Per-output weight scale grid search...')
    best_overall = -np.inf
    best_scales = None

    # Smart search: test coarse grid first, then refine
    coarse_range = [0.1, 0.3, 0.6, 1.0, 1.5]
    best_config_score = -np.inf
    # Test a subset of combinations (too many for full grid: 5^6 = 15625)
    # Strategy: test all at same scale first, then vary one at a time
    for base_ws in coarse_range:
        scales = np.full(N_OUT, base_ws)
        score = evaluate_s5(best_adj, types, config, scales, n_eps=3)
        if score > best_config_score:
            best_config_score = score
            best_scales = scales.copy()
    print(f'    Best uniform ws: {best_scales[0]:.1f} -> {best_config_score:.1f}')

    # Fine-tune: vary each output's scale by +/- 0.2 around best uniform
    for out_idx in range(N_OUT):
        for delta in [-0.2, 0.2]:
            test_scales = best_scales.copy()
            test_scales[out_idx] = np.clip(best_scales[out_idx] + delta, 0.1, 2.0)
            if np.array_equal(test_scales, best_scales):
                continue
            score = evaluate_s5(best_adj, types, config, test_scales, n_eps=3)
            if score > best_config_score:
                best_config_score = score
                best_scales = test_scales.copy()
                print(f'    Improved: output[{out_idx}]={best_scales[out_idx]:.1f} -> {best_config_score:.1f}')

    # Final strict eval
    strict_score = evaluate_s5(best_adj, types, config, best_scales, n_eps=N_EVAL_STRICT)
    print(f'  S5 Best output_scales: {best_scales.tolist()}')
    print(f'  S5 Best 3-ep: {best_config_score:.1f}, Strict 10-ep: {strict_score:.1f}')

    return best_adj, types, strict_score, best_scales


# ══════════════════════════════════════════════════════════════════════════════
# S1: Simplified QD (MAP-Elites)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_with_descriptor(adj, types, config, n_eps=3):
    """Evaluate and return (fitness, behavior_descriptor)."""
    controller = ContinuousSNNController(config, adj, types)
    env = gym.make(ENV_NAME)
    total_r = 0.0
    final_x = 0.0
    max_forward = 0.0
    prev_x = None
    for _ in range(n_eps):
        obs, _ = env.reset()
        controller.reset()
        prev_x = obs[0]
        ep_r = 0.0
        for _ in range(1000):
            action = controller.forward(np.array(obs, dtype=np.float64))
            obs, r, term, trunc, _ = env.step(action)
            ep_r += r
            if term or trunc:
                break
        total_r += ep_r
        final_x = obs[0]
        max_forward = max(max_forward, final_x - prev_x)
    env.close()
    # Behavior descriptor: (mean forward displacement, torso height variation proxy)
    return total_r / n_eps, np.array([max_forward, 0.0])


def run_s1_qd(seed=45):
    """S1: Simplified MAP-Elites with forward displacement descriptor."""
    print('\n' + '='*60)
    print('S1: Quality-Diversity (MAP-Elites simplified)')
    print('='*60)

    np.random.seed(seed)
    types = make_neuron_types()
    config = BASE_CONFIG.copy()

    # Behavior grid: forward_displacement × dummy_dim (just 5 bins)
    n_bins = 5
    archive = {}  # bin_idx -> (adj, fitness)
    pop_size = 40
    n_gen = 60

    pop = [random_topology(N_TOTAL, SPARSITY) for _ in range(pop_size)]

    for gen in range(n_gen):
        for adj in pop:
            fitness, bd = evaluate_with_descriptor(adj, types, config)
            bin_idx = min(int((bd[0] + 1) / 2 * n_bins), n_bins - 1)
            if bin_idx not in archive or fitness > archive[bin_idx][1]:
                archive[bin_idx] = (adj.copy(), fitness)

        # Select parents from archive (diverse bins)
        archive_entries = list(archive.values())
        new_pop = [e[0].copy() for e in archive_entries[:min(5, len(archive_entries))]]
        while len(new_pop) < pop_size:
            parent = archive_entries[np.random.randint(len(archive_entries))][0]
            new_pop.append(mutate_topology(parent, 0.05))
        pop = new_pop

        if (gen + 1) % 20 == 0:
            best_in_archive = max(e[1] for e in archive_entries)
            print(f'  Gen {gen+1:3d}: Archive size={len(archive)}, Best={best_in_archive:.1f}', flush=True)

    # Best from archive
    best_entry = max(archive.values(), key=lambda x: x[1])
    best_adj, best_fitness_qd = best_entry

    strict_score = evaluate_strict(best_adj, types, config)
    print(f'  S1 Best (archive): {best_fitness_qd:.1f}, Strict 10-ep: {strict_score:.1f}')
    return best_adj, types, strict_score


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default='all',
                        choices=['S1', 'S3', 'S4', 'S5', 'all'])
    args = parser.parse_args()

    # Pre-registered thresholds
    F_BOUNDARY = -200  # below this = true boundary
    F_PARTIAL = 0      # between -200 and 0 = soft boundary
    F_SUCCESS = 200    # above this = local optimum hypothesis confirmed

    RANDOM_BASELINE = -314.6

    print('='*70)
    print('  HalfCheetah-v4 Local Optimum Escape Experiment')
    print(f'  Pre-registered thresholds:')
    print(f'    F_boundary: strict < {F_BOUNDARY} -> true boundary')
    print(f'    F_partial:  {F_BOUNDARY} <= strict < {F_PARTIAL} -> soft boundary')
    print(f'    F_success:  strict >= {F_SUCCESS} -> local optimum')
    print(f'    Random baseline: {RANDOM_BASELINE:.1f}')
    print('='*70)

    results = {}

    if args.strategy in ('S1', 'all'):
        adj_s1, types_s1, score_s1 = run_s1_qd()
        results['S1_QD'] = {'score': score_s1}

    if args.strategy in ('S3', 'all'):
        adj_s3, types_s3, score_s3 = run_s3_fitness_shaping()
        results['S3_FitnessShaping'] = {'score': score_s3}

    if args.strategy in ('S4', 'all'):
        adj_s4, types_s4, score_s4 = run_s4_larger_population()
        results['S4_LargerPop'] = {'score': score_s4}

    if args.strategy in ('S5', 'all'):
        adj_s5, types_s5, score_s5, scales_s5 = run_s5_per_output_scales()
        results['S5_PerOutputScales'] = {'score': score_s5, 'scales': scales_s5.tolist()}

    # ── Final Report ──
    print('\n' + '='*70)
    print('  FINAL REPORT: HalfCheetah Escape Experiment')
    print('='*70)
    header = f'  {"Strategy":<25} {"10-ep Strict":>12} {"vs Random":>10} {"Verdict":>20}'
    sep = f'  {"-"*70}'
    base = f'  {"Original GA (baseline)":<25} {-415.5:>12.1f} {-32.1:>9.1f}% {"(baseline)":>20}'
    print(header)
    print(sep)
    print(base)

    best_score = -415.5
    best_strategy = 'Original GA'

    for strat_name, r in results.items():
        s = r['score']
        vs_random = (s - RANDOM_BASELINE) / abs(RANDOM_BASELINE) * 100
        if s < F_BOUNDARY:
            verdict = 'TRUE BOUNDARY'
        elif s < F_PARTIAL:
            verdict = 'SOFT BOUNDARY'
        elif s < F_SUCCESS:
            verdict = 'WEAK SUCCESS'
        else:
            verdict = 'SUCCESS (local opt)'
        print(f'  {strat_name:<25} {s:>12.1f} {vs_random:>9.1f}% {verdict:>20}')

        if s > best_score:
            best_score = s
            best_strategy = strat_name

    print(f'\n  Best strategy: {best_strategy} (score={best_score:.1f})')

    if best_score < F_BOUNDARY:
        conclusion = ('TRUE BOUNDARY: 1-parameter typed-graph framework cannot '
                      'solve 6D multi-joint coordination. Multiple escape '
                      'strategies confirmed the boundary.')
    elif best_score < F_PARTIAL:
        conclusion = ('SOFT BOUNDARY: Framework shows limited capability at 6D. '
                      'Better than random but far from learned methods.')
    else:
        conclusion = ('LOCAL OPTIMUM: Original GA was stuck. Improved strategies '
                      'found superior topologies. Framework can scale to 6D.')

    print(f'\n  CONCLUSION: {conclusion}')

    # Save report
    report = {
        'experiment': 'HalfCheetah Local Optimum Escape',
        'pre_registration': 'results/preregistration_halfcheetah_escape.md',
        'thresholds': {'F_boundary': F_BOUNDARY, 'F_partial': F_PARTIAL, 'F_success': F_SUCCESS},
        'random_baseline': RANDOM_BASELINE,
        'original_ga_score': -415.5,
        'results': {k: {'score': float(v['score'])} for k, v in results.items()},
        'best_strategy': best_strategy,
        'best_score': float(best_score),
        'conclusion': conclusion,
    }
    report_path = os.path.join(OUT_DIR, 'escape_experiment_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nReport saved: {report_path}')


if __name__ == '__main__':
    main()
