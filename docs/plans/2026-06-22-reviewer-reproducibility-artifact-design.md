# Reviewer Reproducibility Artifact Design

Date: 2026-06-22

## Goal

Prepare a clean public repository for the paper "Recurrent Connection Topology Supports Reactive Control through Propagation Dynamics in Fixed-Weight Spiking Neural Networks". The repository should help reviewers reproduce or audit the manuscript's main and supplementary results without exposing local machine files, legacy drafts, private paths, or large duplicate logs.

## Scope

Use an artifact-first layout:

- `README.md`: high-level overview, setup, minimal audit commands, and full reproduction map.
- `REPRODUCING.md`: result-by-result mapping from paper claims/figures/tables to scripts, processed inputs, and expected outputs.
- `artifact/`: curated processed tables, endpoint declarations, statistical regeneration scripts, and audit report.
- `code/`: core SNN implementation, experiment scripts, figure scripts, and processed result files needed by the manuscript.
- `paper/`: LaTeX source for the manuscript and supplement.
- `figures_arxiv/`: final manuscript figures.
- `struct_ai/`: only the MountainCar co-evolution subset needed for TO/TG/TI auditing.

## Exclusions

Do not upload local tools, third-party writing/review folders, SSH files, root-level legacy drafts, external PDFs, raw duplicate logs, local cluster launch scripts, archives, or folders with private paths unless they are rewritten and explicitly needed for reproduction.

## Validation

Before pushing:

1. Regenerate statistical boundary tables.
2. Run the local reproducibility audit.
3. Scan tracked files for credentials, private keys, local paths, and private network markers.
4. Confirm `README.md` and experiment docs are readable text.
5. Push a clean `main` and fixed `v1.0-submission-artifact` tag.
