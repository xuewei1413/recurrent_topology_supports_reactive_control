"""Scan tracked/public artifact files for common sensitive strings.

The scan is conservative and intended for pre-push review. It ignores binary
arrays, PDFs, generated images, and local tool folders. Known manuscript terms
such as "spiking token" are not expected in this repository; any hit should be
reviewed before pushing.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKIP_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".npy", ".npz", ".pyc", ".zip", ".gz"}
SKIP_PARTS = {".git", "tools", "Nature-Skill", ".claude", "__pycache__"}
PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)(github_pat|ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]+"),
    re.compile(r"(?i)\b(?:api[_-]?key|secret|token|authorization|bearer|password|passwd)\b\s*[:=]\s*[^,;\s]{6,}"),
    re.compile(r"\b(?:10|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b"),
    re.compile(r"(?:[A-Za-z]:\\Users\\|/Users/|/home/|E:\\|C:\\)", re.IGNORECASE),
]
ALLOW_SUBSTRINGS = {
    "weight_scale",  # model parameter, not a credential
    "endpoint boundary",
    "rg -n",
}


def tracked_files() -> list[Path]:
    proc = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git ls-files failed")
    return [ROOT / line.strip() for line in proc.stdout.splitlines() if line.strip()]


def should_skip(path: Path) -> bool:
    rel_parts = set(path.relative_to(ROOT).parts)
    return bool(rel_parts & SKIP_PARTS) or path.suffix.lower() in SKIP_SUFFIXES


def main() -> int:
    findings: list[str] = []
    for path in tracked_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel == "artifact/scripts/scan_public_artifact.py":
            continue
        if should_skip(path) or not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            lowered = line.lower()
            if any(allowed in lowered for allowed in ALLOW_SUBSTRINGS):
                continue
            for pattern in PATTERNS:
                if pattern.search(line):
                    findings.append(f"{rel}:{lineno}: {line.strip()[:180]}")
                    break
    if findings:
        print("Potential sensitive/public-artifact findings:")
        for item in findings:
            print(item)
        return 1
    print("No sensitive-string findings in tracked public artifact files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
