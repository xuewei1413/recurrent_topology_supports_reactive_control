# Reproducing the Reported Analyses

Run the commands below from this `reproducibility/` directory. The published graph files and measurements are matched to one another and must not be mixed with historical CSV outcomes.

## Environment

The local execution used Python 3.12.14, NumPy 1.26.4, Gymnasium 1.2.2, and SciPy 1.13.1. Create a separate environment and install the pinned dependencies:

```powershell
python -m pip install -r local_replication/requirements.txt
python -m pip install -r local_replication/requirements-association.txt
```

For LunarLander simulations outside the local CartPole/Acrobot rerun, Box2D dependencies are additionally required. The full original server environment is not available.

## Recompute Statistics Without New Simulations

```powershell
python -m local_replication.analyze_current --profile full
python -m local_replication.analyze_association --profile full
python -m paper.analysis.compare_local_rerun_20261012
```

The principal outputs are:

- `results_local_replication_20261009/current/full/analysis_CartPole.json`
- `results_local_replication_20261009/current/full/analysis_Acrobot.json`
- `results_local_replication_20261009/association/full/analysis_both.json`
- `paper/analysis/2026-10-12_local_comparison.json`

These commands write new analysis outputs and source snapshots; use a working copy when preserving the published artifact byte-for-byte. Completed historical package checks reproduced all four summaries exactly. The current release also includes a fresh verification report after extraction.

## Instrument Tests

```powershell
python -m unittest local_replication.test_association_summaries local_replication.test_calibration_guard local_replication.test_injector
```

The tests check analysis aggregation, calibration-window validation, and current-injection behavior, including zero-count ties and strength matching.

## Local Simulation Rerun

```powershell
python -m local_replication.run_all --workers 4
```

This is a compute-intensive command, not a quick audit. Completed compatible cells are reused; incomplete network evolution restarts under the recorded rules. Read `local_replication/README.md` before running. The entry point covers the key output-current conditions and the 55 association networks, not every historical experiment.

## Historical and Local Sources

Current source snapshots and inputs are preserved under `results_local_replication_20261009/`. Historical structural controls, matched lesions, retention conditions, and tie-rule re-evolution remain distinct, with processed records and supporting archives included where available. The deleted historical association graphs cannot be recovered from the new matrices. Equal-score matching is a limited comparison of matchable tied-score connections, not a controlled high-versus-low path-count experiment.

## Verification and Access

`SHA256_MANIFEST.json` records all inherited file hashes. `verify_manifest.py` checks them without running simulations. `PORTABLE_VERIFICATION.json` records the extraction-based analysis checks. Repository access is at https://github.com/xuewei1413/recurrent_topology_supports_reactive_control/tree/main/releases/2026-10-13-neural-computation.
