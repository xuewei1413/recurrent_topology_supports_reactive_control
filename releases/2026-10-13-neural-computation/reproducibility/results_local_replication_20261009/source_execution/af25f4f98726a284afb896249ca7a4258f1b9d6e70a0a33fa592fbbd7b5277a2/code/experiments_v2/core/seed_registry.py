"""
Seed Registry 与冻结工具（Phase 6）

设计文档 §11.1 / §11.2：
  Pilot：
    - 每任务使用 3--5 个专用 pilot seeds
    - Pilot seeds 永不进入确认性统计
    - Pilot 后生成冻结配置和 seed registry，并记录哈希

  Confirmatory：
    - 每任务至少 20 个全新 evolution seeds
    - 固定配置、代码版本、训练/验证/测试 seed 集合
    - 每个最终条件使用相同的 100 个 held-out test episodes
    - 测试集只运行一次

冻结产物：
    protocol_v2_frozen.yaml
    seed_registry_v2.json
    observation_stats_v2.json
    analysis_spec_v2.md
    pilot_decision_log.md
    上述文件的哈希清单
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SeedRegistry:
    """完整的 seed 注册表。

    train/val/test 严格不重叠。
    pilot_seeds 独立于 confirmatory_evolution_seeds。
    """

    pilot_evolution_seeds: tuple[int, ...]      # 3-5 个 pilot 用
    confirmatory_evolution_seeds: tuple[int, ...]  # 20+ 个确认性用

    # 每个 evolution seed 内部的 train/val/test split
    train_episode_seeds: tuple[int, ...]        # GA 训练用
    validation_episode_seeds: tuple[int, ...]   # gain 优化模型选择用
    test_episode_seeds: tuple[int, ...]         # 100 个 held-out，只跑一次

    def __post_init__(self):
        pilot = set(self.pilot_evolution_seeds)
        confirm = set(self.confirmatory_evolution_seeds)
        if pilot & confirm:
            raise ValueError(f"pilot/confirmatory overlap: {pilot & confirm}")

        train = set(self.train_episode_seeds)
        val = set(self.validation_episode_seeds)
        test = set(self.test_episode_seeds)
        if train & val:
            raise ValueError(f"train/val overlap: {train & val}")
        if train & test:
            raise ValueError(f"train/test overlap: {train & test}")
        if val & test:
            raise ValueError(f"val/test overlap: {val & test}")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: str | Path) -> SeedRegistry:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SeedRegistry(**data)

    def hash(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_default_seed_registry(
    n_pilot_seeds: int = 5,
    n_confirmatory_seeds: int = 20,
    n_train: int = 100,
    n_validation: int = 20,
    n_test: int = 100,
) -> SeedRegistry:
    """生成默认的 seed registry。

    seed 区间：
      pilot evolution: 1000..1000+N-1
      confirmatory evolution: 2000..2000+N-1
      train episodes: 10000..10000+N-1
      validation episodes: 20000..20000+N-1
      test episodes: 30000..30000+N-1
    """
    return SeedRegistry(
        pilot_evolution_seeds=tuple(range(1000, 1000 + n_pilot_seeds)),
        confirmatory_evolution_seeds=tuple(range(2000, 2000 + n_confirmatory_seeds)),
        train_episode_seeds=tuple(range(10000, 10000 + n_train)),
        validation_episode_seeds=tuple(range(20000, 20000 + n_validation)),
        test_episode_seeds=tuple(range(30000, 30000 + n_test)),
    )


# ── 哈希清单 ──────────────────────────────────────────────────────────

def compute_file_hash(path: str | Path) -> str:
    """计算文件的 SHA256。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def create_hash_manifest(
    files: dict[str, str | Path],
    output_path: str | Path,
) -> dict:
    """为冻结产物创建哈希清单。

    Args:
        files: {name: filepath}
        output_path: manifest 保存路径

    Returns:
        {name: {path, sha256}}
    """
    manifest = {}
    for name, path in files.items():
        path = Path(path)
        if path.exists():
            manifest[name] = {
                "path": str(path),
                "sha256": compute_file_hash(path),
            }
        else:
            manifest[name] = {"path": str(path), "sha256": None, "missing": True}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def verify_hash_manifest(manifest_path: str | Path) -> tuple[bool, list[str]]:
    """验证哈希清单中的所有文件。

    Returns:
        (all_ok, error_messages)
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    errors = []
    for name, info in manifest.items():
        path = Path(info["path"])
        if not path.exists():
            errors.append(f"{name}: file missing at {path}")
            continue
        current_hash = compute_file_hash(path)
        if current_hash != info["sha256"]:
            errors.append(
                f"{name}: hash mismatch "
                f"(expected {info['sha256'][:16]}..., got {current_hash[:16]}...)"
            )

    return len(errors) == 0, errors
