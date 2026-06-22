# Reproducibility Report

Generated: 2026-06-22 05:44:06 UTC

## Overall Status

- PASS: 9
- WARN: 0
- FAIL: 0

## Check Results

- PASS - Python dependencies: Python 3.11.7; Executable: python.exe; Platform: Windows-10-10.0.26200-SP0; numpy 1.26.4; scipy 1.13.1
- PASS - Repository context: Git repository detected.
- PASS - Endpoint config and source counts: CartPole {'TO': 10, 'TG': 10, 'TI': 10} from code/results/coevolution/coevolution_checkpoint.npy; MountainCar {'TG': 20, 'TI': 20, 'TO': 20} from struct_ai/code/experiments/results/mountaincar_coevolution/result_*.json
- PASS - Regenerate statistical artifact: Rebuilt seed-level CSVs, statistical summaries, robust summaries, and supplement table.
- PASS - Numeric anchors: All manuscript-level statistical anchors match regenerated CSV files.
- PASS - Manifest paths: All 12 manifest entries resolve locally.
- PASS - Manuscript text anchors: Endpoint, statistical-boundary, supplement, and availability anchors are present.
- PASS - LaTeX compilation: draft_v12_full.tex: Compiled to paper\_compile_check\draft_v12_full.pdf
- PASS - LaTeX compilation: draft_v12_compact_supplement.tex: Compiled to paper\_compile_check\draft_v12_compact_supplement.pdf

## Regenerated Numeric Anchors

- CartPole means: TO 498.075, TG 496.750, TI 497.825 (primary metric: best_fitness; n=10 each).
- CartPole Welch CIs: TO-TI 0.25 [-2.79, 3.29]; TO-TG 1.32 [-1.76, 4.41]; TG-TI -1.07 [-4.39, 2.24].
- MountainCar means: TO 895.476, TG 1049.398, TI 984.019 (primary metric: strict_20ep_mean; n=20 each).
- MountainCar Welch CIs: TO-TI -88.54 [-449.89, 272.81]; TO-TG -153.92 [-511.65, 203.80]; TG-TI 65.38 [-314.69, 445.45].
- MountainCar success counts: TO 16/20, TG 17/20, TI 16/20.

## Reproduction Commands

```powershell
python -m pip install -r artifact\requirements.txt
python artifact\scripts\compute_statistical_boundaries.py
python artifact\scripts\run_local_reproducibility_audit.py
.\tools\tectonic\tectonic.exe --outdir paper\_compile_check paper\draft_v12_full.tex
.\tools\tectonic\tectonic.exe --outdir paper\_compile_check paper\draft_v12_compact_supplement.tex
```

## Boundaries And Remaining Actions

- The CartPole TO/TG/TI endpoint is final best_fitness with n=10 per condition; strict 20-episode fields are released for transparency.
- The MountainCar TO/TG/TI endpoint is strict_20ep_mean with n=20 per condition, plus success and goal-episode rates.
- Pairwise Welch intervals are descriptive confidence intervals, not equivalence or non-inferiority tests.
- This local folder may not be the final git repository; before submission, create a fixed release tag or commit identifier in the real repository.
