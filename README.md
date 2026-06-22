# Recurrent Connection Topology Supports Reactive Control

This repository is the public code and processed-data artifact for the manuscript:

**Recurrent Connection Topology Supports Reactive Control through Propagation
Dynamics in Fixed-Weight Spiking Neural Networks**

The project asks whether binary recurrent connection topology, under fixed
synaptic magnitudes and fixed neuron signs, can support reactive control in
spiking neural networks. The repository is organized for reviewer audit: it
contains core implementation, experiment scripts, processed result files, and
scripts that regenerate the main statistical checks.

This repository intentionally excludes the manuscript source, final PDF figure
folder, local tools, private machine logs, local cluster launch files, legacy
drafts, and duplicate raw result folders.

## What This Artifact Supports

- Audit the CartPole and MountainCar TO/TG/TI topology/weight-control results.
- Inspect the scripts used for the discrete-task, mechanism, adaptation,
  fingerprint, and continuous-control analyses.
- Regenerate result tables and selected figures from processed results.
- Re-run selected experiments locally, subject to compute and optional package
  availability.

## Repository Layout

- `code/src/`: core SNN controllers, continuous-control controller, evolution
  utilities, and custom environments.
- `code/experiments/`: experiment scripts grouped by manuscript section.
- `code/experiments/results/`: processed result files used by figure and audit
  scripts.
- `code/gen_fig*.py` and `code/plot_fig01.py`: figure regeneration scripts.
- `artifact/`: curated audit package with endpoint declarations, processed
  statistical tables, and local reproducibility checks.
- `struct_ai/`: curated MountainCar TO/TG/TI result subset used by the audit.
- `REPRODUCING.md`: result-by-result reproduction map.
- `UPLOAD_CHECKLIST.md`: public/private file triage notes for the artifact.

## Quick Start

Create a Python environment with the lightweight audit dependencies:

```powershell
python -m pip install -r artifact\requirements.txt
```

Regenerate the TO/TG/TI statistical tables and run the local audit:

```powershell
python artifact\scripts\compute_statistical_boundaries.py
python artifact\scripts\run_local_reproducibility_audit.py
```

The audit writes `artifact/REPRODUCIBILITY_REPORT.md` and verifies:

- endpoint declarations for CartPole and MountainCar TO/TG/TI controls;
- seed counts and processed source files;
- regenerated Welch confidence intervals and robust summaries;
- manifest path integrity;
- reproducibility documentation anchors.

## Regenerating Figures From Processed Results

Figure scripts write to `outputs/figures/` by default, creating the folder if it
is absent. Install the broader code dependencies in `code/requirements.txt`:

```powershell
python -m pip install -r code\requirements.txt
python code\plot_fig01.py
python code\gen_fig04_mountaincar_decoding.py
python code\gen_fig06_spike_timing_corrected.py
python code\gen_fig07_connection_ablation.py
python code\gen_fig09_adaptation.py
python code\gen_fig10_cross_task.py
python code\gen_fig13_fingerprint_corrected.py
python code\gen_fig14_hopper.py
python code\gen_fig15_lunarlander_headroom.py
```

Some scripts perform new simulation or optional baseline evaluation and may need
Gymnasium, MuJoCo, Stable-Baselines3, or NEAT dependencies. See
`REPRODUCING.md` and `code/experiments/README.md` for the result-to-script map.

## Statistical Boundary Note

The TO/TG/TI pairwise intervals are Welch 95% confidence intervals computed from
seed-level final scores. They define the range of differences compatible with
the observed data. They are not formal equivalence or non-inferiority tests.

## Reproducibility Boundaries

- Processed data are included for the reported manuscript values, but full raw
  cluster logs are excluded from Git history.
- Continuous-control Hopper and Pendulum results are proof-of-concept results,
  not multi-seed benchmark claims.
- Cross-task structural fingerprints are configuration-separable and confounded
  by task-specific architecture metadata, as stated in the manuscript.
- Custom FlappyBird and Snake environments are included in source form.

## Citation

If you use this artifact, cite the manuscript and the release tag or commit hash
used for your reproduction attempt.
