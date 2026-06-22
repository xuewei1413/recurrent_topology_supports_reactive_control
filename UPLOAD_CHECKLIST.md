# GitHub Upload Checklist

Repository: `https://github.com/xuewei1413/recurrent_topology_supports_reactive_control`

This checklist records the current public artifact policy. The repository is a
code and processed-data artifact for reviewer audit, not a manuscript source
archive.

## Upload Scope

Tracked public material should include:

- `README.md`, `REPRODUCING.md`, and `code/experiments/README.md`.
- `artifact/` configs, processed TO/TG/TI tables, regeneration scripts, audit
  report, and sensitive-scan script.
- `code/src/` core implementation.
- `code/experiments/` experiment scripts and curated processed result files.
- `code/gen_fig*.py` and `code/plot_fig01.py` figure regeneration scripts.
- `struct_ai/` MountainCar TO/TG/TI subset used by the audit.

## Explicit Exclusions

Do not upload:

- `paper/`: manuscript source, drafts, compiled PDFs, and helper drafts.
- `figures_arxiv/`: final manuscript figure PDFs/PNGs.
- `docs/plans/`: local planning notes.
- `tools/`, `Nature-Skill/`, `.claude/`, SSH temp files, and local launch files.
- Root legacy drafts, archives, external PDFs, duplicate result folders, and
  private machine logs.

## Pre-Push Checks

```powershell
python artifact\scripts\compute_statistical_boundaries.py
python artifact\scripts\run_local_reproducibility_audit.py
python artifact\scripts\scan_public_artifact.py
git diff --cached --check
```

## Release

After checks pass, commit and tag the artifact:

```powershell
git commit -m "Update reviewer reproducibility artifact"
git tag -f v1.0-submission-artifact
git push origin main
git push --force origin v1.0-submission-artifact
```
