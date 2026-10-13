# Neural Computation Study Artifact

## Study

**Control Robustness in Evolved Spiking Neural Networks: Linking Performance after Rewiring and Deletion**

Authors: Xuewei Li, Bowen Zhao, Guangxu Mei, Huanci Li, and Xin Ding.

This artifact supports a study of connectivity changes and control performance in evolved spiking networks. It provides the completed local rerun, paired connectivity graphs and measurements, executed code snapshots, and separate historical data supporting the other reported interventions. The private submission manuscript and cover letter are not included.

## Contents

- `reproducibility/results_local_replication_20261009/`: local network matrices, evolution records, rewiring and deletion measurements, current-injection episodes, calibration checks, source snapshots, and analysis outputs.
- `reproducibility/local_replication/`: local simulation and analysis entry points, dependencies, and instrument tests.
- `reproducibility/code/experiments_v2/`: controller and topology-search modules used by the local execution.
- `reproducibility/_mediation/`: output-current injection and statistical analysis modules.
- `reproducibility/paper/analysis/`: reported-value analyses, historical audits, and figure source data.
- `reproducibility/paper/figures/v5/`: scientific figures and underlying tables.
- `reproducibility/REPRODUCE.md`: execution and analysis instructions.
- `reproducibility/SHA256_MANIFEST.json`: path, size, and SHA-256 for every inherited data/source file.

## Interpretation

The main association concerns specified rewiring and response-guided deletion conditions. New evaluation episodes reuse the same networks. The local rerun evolves networks again over the original seed ranges; it does not recover deleted historical adjacency matrices or add an independent confirmatory cohort. Output-current analyses use revised controls and are supporting evidence, not identification of a common integration mediator.

The historical artifact elsewhere in this repository remains available but describes an earlier manuscript and different analyses. Use this dated folder for the current study.

## AI Assistance

Generative AI tools assisted with experimental code development, language editing, and translation. The authors reviewed the resulting code and manuscript materials and retain responsibility for the work.

## Versioning

Cite this dated folder together with the repository commit used. No publication DOI or acceptance status is claimed. This release does not assign a new license; any existing per-file licenses continue to apply.
