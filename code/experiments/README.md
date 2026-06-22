# Experiment Code Index

This directory contains the experiment scripts used by the manuscript. Scripts
are grouped by result section. Most scripts can be run directly, but full
re-evolution and RL baselines are compute intensive. Processed outputs used by
the submitted manuscript are stored in `code/experiments/results/`.

## Directory Structure

```text
code/
  src/                         Core SNN controllers and evolution utilities
  experiments/
    sec1_existence/            Discrete-task existence, baselines, ablations
    sec1_coevolution/          TO/TG/TI topology and weight-control experiments
    sec2_mechanism/            T-step, membrane, decoding, and timing analyses
    sec4_adaptation/           Low-dimensional adaptation and boundary scans
    sec5_mujoco/               Pendulum, Hopper, HalfCheetah boundary scripts
    supplementary/             Additional robustness and exploratory controls
    revision/                  Review-motivated controls and sensitivity checks
    utils/                     Batch runners and figure-reproduction helpers
    results/                   Processed public result files
```

## Minimal Audit Path

For the reviewer-facing statistical audit, use the top-level artifact scripts
instead of rerunning the expensive experiments:

```powershell
python artifact\scripts\compute_statistical_boundaries.py
python artifact\scripts\run_local_reproducibility_audit.py
```

## Experiment Groups

### `sec1_existence/`: discrete-task existence and baselines

| File | Purpose | Main outputs |
| --- | --- | --- |
| `exp_multitask.py` | Evolves fixed-weight topologies for CartPole, MountainCar, Acrobot, FlappyBird, and Snake. | `results/cross_task/` |
| `exp_multiseed.py` | CartPole multi-seed reproducibility check. | `results/multi_seed_summary.*` |
| `exp_multiseed_expansion.py` | Five-task multi-seed expansion. | `results/cross_task/` |
| `exp_connection_ablation.py` | Connection-type ablation analysis. | `results/ablation/` |
| `exp_cross_task_fingerprint.py` | Structural feature and task-fingerprint analysis. | `results/fingerprint_analysis*` |
| `exp_baseline_neat.py` | NEAT CartPole baseline. | `results/neat_baseline/` |
| `exp_baseline_ppo.py`, `exp_baseline_ppo_v2.py` | PPO CartPole baselines. | `results/ppo_baseline/` |
| `exp_baseline_comparison.py` | Combines SNN, NEAT, and PPO comparison data. | `results/comparison/` |
| `snake/snake_env.py` | Custom Snake environment. | Used by `exp_multitask.py`. |

### `sec1_coevolution/`: topology/weight co-evolution controls

| File | Purpose | Main outputs |
| --- | --- | --- |
| `exp_coevolution_to_tg_ti.py` | CartPole TO/TG/TI comparison. | `code/results/coevolution/` |
| `exp_mountaincar_coevolution_to_tg_ti.py` | MountainCar TO/TG/TI comparison. | `struct_ai/code/experiments/results/mountaincar_coevolution/` |
| `exp_lunarlander_coevolution_to_tg_ti.py` | LunarLander TO/TG/TI headroom audit. | `results/lunarlander_coevolution/` |
| `run_one_seed.py` | Single-seed runner for parallel execution. | Per-seed JSON files. |

### `sec2_mechanism/`: candidate propagation-dynamics analyses

| File | Purpose | Main outputs |
| --- | --- | --- |
| `exp_tsteps_ablation.py` | Varies internal rollout length `T_steps`. | `results/tsteps_curve_data*.npy` |
| `exp_membrane_traces.py` | Records membrane and spike traces. | `outputs/figures/fig03_membrane_traces.pdf` |
| `exp_trace_erasure.py` | Tests intermediate membrane-state reset. | Console/results files. |
| `exp_input_decoding.py` | Ridge decoding of current and lagged inputs. | `results/decoding_data.npy` |
| `exp_spike_timing.py` | Spike timing and count under selected history conditions. | `results/spike_timing_data.npy` |
| `exp_weight_scale_ablation.py` | Weight-scale robustness. | `results/weight_scale_ablation_data.npy` |

### `sec4_adaptation/`: adaptation and boundary scans

| File | Purpose | Main outputs |
| --- | --- | --- |
| `exp_adaptation_compare.py` | Compares zero-shot, global-scale, and T-step retuning after CartPole gravity change. | `outputs/figures/fig09_adaptation.pdf` |
| `exp_param_boundary.py` | Scans CartPole gravity/length and FlappyBird pipe-speed boundaries. | `results/boundary_scan/*.csv` |

### `sec5_mujoco/`: continuous-control proof-of-concept and boundary tests

| File | Purpose | Main outputs |
| --- | --- | --- |
| `exp_pendulum.py` | Pendulum-v1 fixed-weight SNN proof-of-concept. | `results/embodied_reflex/pendulum_report.json` |
| `exp_hopper_evolution.py` | Hopper-v4 topology evolution. | `results/embodied_reflex/` |
| `exp_hopper_analysis.py` | Hopper strict evaluation and gravity analysis. | `results/embodied_reflex/hopper_deep_analysis/` |
| `exp_halfcheetah_boundary.py` | HalfCheetah-v4 action-dimensionality boundary. | `results/embodied_reflex/halfcheetah_escape/` |
| `exp_ant_boundary_audit.py` | Additional high-dimensional boundary audit. | Optional exploratory output. |

### `supplementary/` and `revision/`

| File | Purpose |
| --- | --- |
| `supplementary/exp_eval_robustness.py` | Tests sensitivity to evaluation episode count. |
| `supplementary/exp_selective_ablation.py` | Exploratory selective neuron/circuit ablation. |
| `supplementary/exp_angular_velocity.py` | Input-dimension perturbation control. |
| `revision/analyze_review_controls.py` | Review-motivated rate-substitution and sensitivity summaries. |

## Common Parameters

Unless a script overrides them, the fixed-weight SNN uses LIF-like dynamics with
one global `weight_scale`, fixed neuron signs, and a binary recurrent adjacency
matrix evolved by a genetic algorithm. The manuscript and `REPRODUCING.md`
document the per-task network sizes, `T_steps`, seed counts, and endpoint
interpretation caveats.

## Re-running Everything

The utility below attempts to run the experiment groups in sequence, but it is
not the recommended reviewer path because it can take a long time and may need
optional dependencies:

```powershell
python code\experiments\utils\run_all.py
```

For review, prefer the processed-data audit in `artifact/` and targeted reruns
of individual scripts.
