# Submission Artifact

This artifact records the processed data and scripts needed to audit the main
TO/TG/TI statistical comparisons in the manuscript.

## Contents

- `processed_results/cartpole_coevolution_seed_level.csv`: seed-level CartPole
  TO/TG/TI data. The primary manuscript metric for this audit is
  `best_fitness`; strict 20-episode re-evaluation columns are also included to
  make the metric boundary explicit.
- `processed_results/mountaincar_coevolution_seed_level.csv`: seed-level
  MountainCar TO/TG/TI data. The primary manuscript metric is
  `strict_20ep_mean`.
- `processed_results/statistical_boundaries_to_tg_ti.csv`: group mean 95% CIs,
  Welch pairwise mean-difference 95% CIs, and one-way ANOVA summaries for the
  TO/TG/TI controls.
- `processed_results/robust_statistics_to_tg_ti.csv`: median, IQR, success
  rate, Wilson success-rate intervals, and MountainCar goal-episode rates.
- `processed_results/supplement_table_to_tg_ti.tex`: LaTeX table included by
  the supplementary material.
- `configs/analysis_endpoints.json`: machine-readable endpoint declaration for
  the CartPole and MountainCar TO/TG/TI audits.
- `scripts/compute_statistical_boundaries.py`: regenerates the three CSV files
  above from the processed result files in this repository.
- `scripts/run_local_reproducibility_audit.py`: runs the local artifact audit,
  verifies numeric anchors, checks the manifest, and checks reproduction docs and
  figure-script inventory.
- `requirements.txt` and `environment.yml`: minimal Python environments for the
  artifact scripts.

## Statistical interpretation

The pairwise intervals are Welch 95% confidence intervals computed from
seed-level final scores. They define the range of differences compatible with
the observed data. They are not formal equivalence or non-inferiority tests.

## Local reproduction

From the repository root:

```powershell
python -m pip install -r artifact\requirements.txt
python artifact\scripts\compute_statistical_boundaries.py
python artifact\scripts\run_local_reproducibility_audit.py
```

The audit writes `artifact/REPRODUCIBILITY_REPORT.md`. It is a local validation
record rather than a substitute for a fixed public release tag.

## Submission checklist

Before submission, archive the repository with a fixed tag or commit identifier
(e.g., `v1.0-submission-artifact`) and cite that release in the manuscript's
code/data availability statement.
