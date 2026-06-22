# Recurrent Connection Topology Supports Reactive Control

This repository contains the manuscript source, analysis code, processed results,
and reproducibility artifact for:

**Recurrent Connection Topology Supports Reactive Control through Propagation
Dynamics in Fixed-Weight Spiking Neural Networks**

The project studies whether recurrent connection topology, with fixed synaptic
weight magnitudes and neuron signs, can support reactive control through
propagation dynamics in spiking neural networks.

## Repository Layout

- `paper/`: LaTeX manuscript and supplementary material.
- `figures_arxiv/`: PDF figures used by the manuscript and supplement.
- `code/`: core SNN implementation, experiment scripts, figure scripts, and
  processed result files.
- `struct_ai/`: MountainCar TO/TG/TI control scripts and processed per-seed
  result files used by the submission artifact.
- `artifact/`: submission artifact with endpoint declarations, seed-level
  processed tables, statistical-boundary regeneration scripts, and a local
  reproducibility report.

## Minimal Reproducibility Check

From the repository root:

```powershell
python -m pip install -r artifact\requirements.txt
python artifact\scripts\compute_statistical_boundaries.py
python artifact\scripts\run_local_reproducibility_audit.py
```

The audit verifies:

- CartPole and MountainCar TO/TG/TI endpoint declarations.
- Seed counts and source files used for statistical-boundary reporting.
- Regenerated seed-level CSVs, Welch confidence intervals, robust summaries, and
  the supplementary LaTeX table.
- Manuscript text anchors for endpoint boundaries and code/data availability.
- LaTeX compilation when `tools/tectonic/tectonic.exe` is available locally.

## Statistical Boundary

The TO/TG/TI pairwise intervals are Welch 95% confidence intervals computed from
seed-level final scores. They define the range of differences compatible with
the observed data and are not formal equivalence or non-inferiority tests.

## Notes

- Large local tools, legacy drafts, logs, and duplicate result folders are
  intentionally excluded by `.gitignore`.
- A fixed release tag or commit identifier should be created for submission.
- Any raw logs not included in this repository can be provided by the
  corresponding author upon reasonable request.
