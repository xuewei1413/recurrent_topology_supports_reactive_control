# GitHub Upload Checklist

Repository: `https://github.com/xuewei1413/recurrent_topology_supports_reactive_control`

This checklist separates a clean, reviewable public artifact from local working
files. The goal is to remove the "empty repository" review risk without pushing
private machine paths, duplicate logs, third-party tools, or legacy drafts.

## P0: Upload First

These files should be uploaded before submission or review. Together they provide
manuscript source, figure source, processed data, statistical regeneration, and a
local audit path.

```powershell
git add README.md .gitignore UPLOAD_CHECKLIST.md
git add artifact

git add paper\draft_v12_full.tex paper\draft_v12_compact_supplement.tex
git add figures_arxiv\*.pdf

git add code\requirements.txt code\audit_consistency.py
git add code\src
git add code\experiments\README.md code\experiments\regenerate_decoding_data.py
git add code\experiments\sec1_existence code\experiments\sec1_coevolution code\experiments\sec2_mechanism code\experiments\sec4_adaptation code\experiments\sec5_mujoco code\experiments\supplementary code\experiments\utils code\experiments\revision

git add code\plot_fig01.py code\gen_fig04_mountaincar_decoding.py code\gen_fig06_spike_timing_corrected.py code\gen_fig07_connection_ablation.py code\gen_fig08_coevolution.py code\gen_fig09_adaptation.py code\gen_fig10_cross_task.py code\gen_fig13_fingerprint_corrected.py code\gen_fig14_hopper.py code\gen_fig15_lunarlander_headroom.py code\run_ppo_hopper.py

git add code\experiments\results\ablation\ablation_results.npy code\experiments\results\ablation\ablation_summary.csv
git add code\experiments\results\adaptation\adaptation_summary.csv
git add code\experiments\results\comparison\comparison_analysis_summary.txt
git add code\experiments\results\cross_task
git add code\experiments\results\decoding_data.npy code\experiments\results\spike_timing_data.npy code\experiments\results\tsteps_curve_data.npy code\experiments\results\tsteps_curve_data_full.npy code\experiments\results\weight_scale_ablation_data.npy code\experiments\results\triadic_motif_analysis.npy code\experiments\results\selective_ablation_results.npy
git add code\experiments\results\embodied_reflex\*.json code\experiments\results\embodied_reflex\*.npy
git add code\experiments\results\fingerprint_analysis_corrected\report.json code\experiments\results\fingerprint_analysis_corrected\report.txt
git add code\experiments\results\lunarlander_coevolution\lunarlander_coevolution_results.json code\experiments\results\lunarlander_coevolution\lunarlander_coevolution_summary.csv code\experiments\results\lunarlander_coevolution\lunarlander_tost_summary.csv
git add code\experiments\results\multi_seed_summary.csv code\experiments\results\multi_seed_summary.npy
git add code\experiments\results\neat_baseline\*.csv code\experiments\results\neat_baseline\*.txt
git add code\experiments\results\ppo_baseline\*.csv code\experiments\results\ppo_baseline\*.txt code\experiments\results\ppo_baseline\ppo_training_curves.json
git add code\experiments\results\revision_controls

git add code\mountaincar_decoding_results.txt code\mountaincar_T20_r2.npy
git add code\results\coevolution\coevolution_checkpoint.npy

git add struct_ai\code\src
git add struct_ai\code\experiments\sec1_coevolution\exp_mountaincar_coevolution_to_tg_ti.py
git add struct_ai\code\experiments\results\mountaincar_coevolution\result_*.json
git add struct_ai\code\experiments\results\mountaincar_coevolution\mountaincar_coevolution_results.json
git add struct_ai\code\experiments\results\mountaincar_coevolution\mountaincar_coevolution_summary.csv
```

## P0 Upload Notes

- `paper/draft_v12_full.tex` now says the repository is "being prepared for public release". After the GitHub release exists, replace that wording with the exact tag or commit URL.
- Prefer `figures_arxiv/*.pdf` as tracked manuscript figures. PNG previews can be attached to a release if needed.
- `artifact/scripts/run_local_reproducibility_audit.py` should pass after the P0 set is staged because it checks endpoint tables, manuscript anchors, manifest paths, and LaTeX compilation when Tectonic exists locally.
- `code/plot_fig01.py`, `code/gen_fig11.py`, and `code/gen_fig13_fingerprint.py` had local absolute paths; use the patched portable versions before upload. Prefer `code/gen_fig13_fingerprint_corrected.py` for the submitted Fig. 13.

## Upload After Quick Cleanup

These are useful, but should be checked before becoming public.

- `code/gen_fig11.py`: legacy boundary-scan figure generator, now path-portable but not central to the current full draft if Fig. 11 numbering changed.
- `code/gen_fig13_fingerprint.py`: legacy fingerprint generator; upload only if you want to preserve provenance alongside `code/gen_fig13_fingerprint_corrected.py`.
- `struct_ai/run_mountaincar_parallel*.sh`: contains local cluster paths; either rewrite with configurable `ROOT` or keep out of the public artifact.
- `paper/_make_compact.py`, `paper/_make_method_overview.py`, `paper/_regen_fig07_connection_ablation.py`, `paper/_regen_fig12_comparison.py`: safe as helper scripts if you want full manuscript-generation provenance.

## Optional Release Assets

Attach these as GitHub release assets rather than normal tracked files if size or clutter matters.

- `paper/draft_v12_full.pdf`
- `paper/draft_v12_compact_supplement.pdf`
- `figures_arxiv/*.png`
- `struct_ai/code/experiments/results/mountaincar_coevolution/topology_*.npz`
- Raw experiment logs after path/privacy cleanup.

## Do Not Upload

Keep these out of Git history.

- `Nature-Skill/`: third-party writing/review skill, unrelated to this paper artifact.
- `tools/`: local Tectonic binary and other local tools.
- `.claude/`, SSH temp files, `known_hosts_tmp`, `empty_ssh_config`, `ssh_*.txt`.
- `upload_and_launch.py`: contains SSH/password-style remote execution logic and should never be public.
- `remote_run_*.sh`, `run_mountaincar_parallel_*.sh`, `run_lunarlander_parallel.sh`: local cluster launch paths; rewrite before upload if needed.
- Root legacy drafts and compiled files: `draft_cn_v11*`, `draft_en_v11*`, `main_manuscript.pdf`, `manuscript_with_supplement.pdf`, `supplementary.pdf`.
- External/reference PDFs unrelated to this paper, including `Yongheng Mu-IPM【联合通讯】.pdf` and `（ICLR2025）StructRAG.pdf`.
- `paper/_compile_check/`, `paper/_full_build/`, `paper/_compact_build/`, `paper/format_review_shots/`, and LaTeX `.aux/.log/.out` files.
- Root `data/`, root `results/`, and `server_results_mc/`: duplicate or legacy result locations with local paths/logs. Use the curated processed results listed in P0 instead.
- `__pycache__/`, `*.pyc`, `*.pyc.*`, `.pytest_cache/`, and experiment `.log` files.
- Archives or snapshots such as `files.zip`, `arxiv_submission.tar.gz`, and `struct_ai_patch.tar.gz`.

## Pre-Push Checks

Run these before pushing:

```powershell
git status --short
git diff --cached --stat
python artifact\scripts\compute_statistical_boundaries.py
python artifact\scripts\run_local_reproducibility_audit.py
rg -n "(password|passwd|api[_-]?key|secret|token|PRIVATE KEY|Authorization|Bearer|192\.168|/home/|C:\\|E:\\|/Users/)" . --glob '!Nature-Skill/**' --glob '!tools/**' --glob '!**/__pycache__/**' --glob '!*.pdf' --glob '!*.png' --glob '!*.npy' --glob '!*.npz' --glob '!*.pyc*'
```

A few `PASS` strings in audit reports are expected; any real credential, private IP, or absolute local path in staged files should be removed or justified before push.

## Commit And Release

After checking the staged files:

```powershell
git commit -m "Prepare fixed-weight SNN reproducibility artifact"
git tag v1.0-submission-artifact
git push origin main
git push origin v1.0-submission-artifact
```

After pushing, update the manuscript code-availability sentence with the exact release URL or commit hash, then recompile the PDF.
