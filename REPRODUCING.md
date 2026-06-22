# Reproducing the Paper Results

This document maps manuscript results to public repository files. It is written
for reviewers who want to audit numbers quickly and for readers who want to
rerun selected experiments.

## Environment Levels

### Level 1: Statistical audit only

Use this for the fastest reviewer check.

```powershell
python -m pip install -r artifact\requirements.txt
python artifact\scripts\compute_statistical_boundaries.py
python artifact\scripts\run_local_reproducibility_audit.py
```

Inputs:

- `code/results/coevolution/coevolution_checkpoint.npy`
- `struct_ai/code/experiments/results/mountaincar_coevolution/result_*.json`
- `artifact/configs/analysis_endpoints.json`

Outputs:

- `artifact/processed_results/cartpole_coevolution_seed_level.csv`
- `artifact/processed_results/mountaincar_coevolution_seed_level.csv`
- `artifact/processed_results/statistical_boundaries_to_tg_ti.csv`
- `artifact/processed_results/robust_statistics_to_tg_ti.csv`
- `artifact/processed_results/supplement_table_to_tg_ti.tex`
- `artifact/REPRODUCIBILITY_REPORT.md`

### Level 2: Figure regeneration from processed results

Install the broader plotting and experiment dependencies:

```powershell
python -m pip install -r code\requirements.txt
```

Then run the figure scripts listed below. Scripts save PDFs and/or PNGs into
`outputs/figures/` unless noted otherwise.

### Level 3: Full experiment reruns

The evolutionary and RL baseline experiments can be rerun from
`code/experiments/`, but they are compute intensive and may require optional
packages such as `neat-python`, `stable-baselines3`, and MuJoCo. The scripts use
fixed seeds where practical, but evolutionary wall-clock time and stochastic
optimization noise should be expected.

## Main Manuscript Result Map

| Manuscript item | Claim/result audited | Primary scripts | Public inputs | Public outputs |
| --- | --- | --- | --- | --- |
| Table: multitask proficiency | Fixed-weight topology can reach task-specific proficiency on five reactive tasks | `code/experiments/sec1_existence/exp_multitask.py`, `code/plot_fig01.py` | `code/experiments/results/cross_task/*.npz`, `code/experiments/results/cross_task/result_*.json`, `code/experiments/results/cross_task/summary.csv` | `outputs/figures/fig01_evolution_curve.pdf` |
| TO/TG/TI statistics | Topology-only, topology+global-scale, and topology+independent-weight controls on CartPole and MountainCar | `artifact/scripts/compute_statistical_boundaries.py`, `code/experiments/sec1_coevolution/*.py` | `code/results/coevolution/coevolution_checkpoint.npy`, `struct_ai/code/experiments/results/mountaincar_coevolution/result_*.json` | `artifact/processed_results/*.csv`, `artifact/processed_results/supplement_table_to_tg_ti.tex` |
| LunarLander headroom | Additional TO/TG/TI headroom audit | `code/experiments/sec1_coevolution/exp_lunarlander_coevolution_to_tg_ti.py`, `code/gen_fig15_lunarlander_headroom.py` | `code/experiments/results/lunarlander_coevolution/*.json`, `*.csv` | `outputs/figures/fig15_lunarlander_headroom.pdf` |
| PPO/NEAT comparison | CartPole positioning against NEAT and PPO baselines | `code/experiments/sec1_existence/exp_baseline_neat.py`, `code/experiments/sec1_existence/exp_baseline_ppo.py`, `code/experiments/sec1_existence/exp_baseline_comparison.py` | `code/experiments/results/neat_baseline/*`, `code/experiments/results/ppo_baseline/*` | `outputs/figures/fig12_comparison.pdf` |
| T-step ablation | Performance depends on internal rollout length | `code/experiments/sec2_mechanism/exp_tsteps_ablation.py` | `code/experiments/results/tsteps_curve_data*.npy` | `outputs/figures/fig02_tsteps_curve.pdf` |
| Membrane traces | Hidden dynamics remain active during internal SNN rollout | `code/experiments/sec2_mechanism/exp_membrane_traces.py` | processed trace arrays in `code/experiments/results/` | `outputs/figures/fig03_membrane_traces.pdf` |
| Input-history decoding | Descriptive history information can be decoded from internal activity | `code/experiments/sec2_mechanism/exp_input_decoding.py`, `code/gen_fig04_mountaincar_decoding.py` | `code/experiments/results/decoding_data.npy`, `code/mountaincar_T20_r2.npy` | `outputs/figures/fig04_input_decoding.pdf` |
| Spike timing | Spike-count/timing shifts under selected history conditions | `code/experiments/sec2_mechanism/exp_spike_timing.py`, `code/gen_fig06_spike_timing_corrected.py` | `code/experiments/results/spike_timing_data.npy` | `outputs/figures/fig06_spike_timing.pdf` |
| Rate substitution control | Rate-only substitution remains below the evolved SNN endpoint | `code/experiments/revision/analyze_review_controls.py` | `code/experiments/results/revision_controls/*` | `code/experiments/results/revision_controls/revision_controls_report.json` |
| Connection ablation | Connection-type ablations identify sensitivity to input/hidden link classes | `code/experiments/sec1_existence/exp_connection_ablation.py`, `code/gen_fig07_connection_ablation.py` | `code/experiments/results/ablation/*` | `outputs/figures/fig07_connection_ablation.pdf` |
| Structural fingerprints | Topologies are configuration-separable but confounded by architecture metadata | `code/experiments/sec1_existence/exp_cross_task_fingerprint.py`, `code/gen_fig13_fingerprint_corrected.py` | `code/experiments/results/cross_task/*.npz` | `outputs/figures/fig13_fingerprint.pdf`, `code/experiments/results/fingerprint_analysis_corrected/report.*` |
| Continuous control | Hopper and Pendulum proof-of-concept; HalfCheetah boundary | `code/experiments/sec5_mujoco/*.py`, `code/gen_fig14_hopper.py` | `code/experiments/results/embodied_reflex/*`, `code/experiments/results/ppo_hopper/ppo_hopper_results.pkl` | `outputs/figures/fig14_embodied_control.pdf` |
| Cross-task transfer | Zero-shot cross-task transfer is weak | `code/gen_fig10_cross_task.py` | `code/experiments/results/cross_task/*.npz` | `outputs/figures/fig10_cross_task.pdf` |
| Low-dimensional adaptation | Global-scale and T-step retuning after CartPole gravity change | `code/experiments/sec4_adaptation/exp_adaptation_compare.py`, `code/gen_fig09_adaptation.py` | `code/experiments/results/cross_task/CartPole_seed0.npz` | `outputs/figures/fig09_adaptation.pdf` |
| Boundary scans | CartPole and FlappyBird parameter boundary scans | `code/experiments/sec4_adaptation/exp_param_boundary.py`, `code/gen_fig11.py` | `code/experiments/sec4_adaptation/results/boundary_scan/*.csv` | `outputs/figures/fig11_boundary_scan.pdf` |

## Supplementary Result Map

| Supplement item | Script/data | Notes |
| --- | --- | --- |
| TO/TG/TI statistical-boundary table | `artifact/processed_results/supplement_table_to_tg_ti.tex` | Generated by `artifact/scripts/compute_statistical_boundaries.py`. |
| Weight-scale robustness | `code/experiments/sec2_mechanism/exp_weight_scale_ablation.py`, `code/experiments/results/weight_scale_ablation_data.npy` | Figure stored as `outputs/figures/fig05_weight_robustness.pdf`. |
| Selective ablation | `code/experiments/supplementary/exp_selective_ablation.py`, `code/experiments/results/selective_ablation_results.npy` | Exploratory supplementary analysis. |
| Evaluation robustness | `code/experiments/supplementary/exp_eval_robustness.py` | Evaluates sensitivity to sparse GA fitness evaluation. |
| Angular-velocity input test | `code/experiments/supplementary/exp_angular_velocity.py` | Exploratory input-dimension control. |

## Manuscript And Final Figures

The public Git repository is intentionally limited to code, processed data, and
audit scripts. Manuscript source and final figure PDFs are distributed outside
this repository. The figure scripts can regenerate reviewable PDFs/PNGs into
`outputs/figures/` from the included processed data where the corresponding
processed inputs are public.

## Expected Review-Audit Commands

For the clean reviewer artifact, these are the commands reviewers should run
first:

```powershell
python artifact\scripts\compute_statistical_boundaries.py
python artifact\scripts\run_local_reproducibility_audit.py
git status --short
```

A clean checkout should remain unchanged except for regenerated files whose
contents match the tracked versions.
