"""
结果 Schema 定义与严格验证（Phase 0 补强）

实施计划 Phase 0 验收：
  - 结果文件包含所有自由度、拓扑哈希、模块标签哈希、E/I 哈希和 seed split
  - 缺少字段时分析脚本必须失败，不能填默认值继续运行

本模块定义结果 JSON 的必填字段，并提供严格验证函数。
任何缺失或类型不符都会抛出异常，不静默填充默认值。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ── 单 seed topology 结果 schema ──────────────────────────────────────

TOPOLOGY_RESULT_REQUIRED_FIELDS = {
    # 协议元信息
    "task": str,
    "evolution_seed": int,
    "protocol_version": str,

    # 哈希审计
    "topology_hash": str,
    "module_label_hash": str,
    "ei_type_hash": str,
    "edge_count": int,
    "expected_edge_count": int,

    # seed split
    "validation_seeds_used": list,
}

TOPOLOGY_RESULT_CONDITIONAL_FIELDS = {
    # F0 结果
    "f0_best_fitness": (float, int),
    "f0_validation_mean": (float, int),
    "f0_validation_std": (float, int),

    # RA/RW 结果
    "ra_validation_means": list,
    "ra_validation_mean_of_means": (float, int),
    "rw_validation_means": list,
    "rw_validation_mean_of_means": (float, int),
}


# ── gain 结果 schema ──────────────────────────────────────────────────

GAIN_RESULT_REQUIRED_FIELDS = {
    "task": str,
    "evolution_seed": str,  # JSON 中可能被序列化为 str
}

GAIN_RESULT_CONDITION_FIELDS = {
    # 每个条件（F0/G1/M4/N/E）的 test 结果
    "test": dict,
}


# ── 验证函数 ──────────────────────────────────────────────────────────

class SchemaValidationError(Exception):
    """Schema 验证失败。"""

    def __init__(self, errors: list[str]):
        self.errors = errors
        message = f"Schema validation failed ({len(errors)} errors):\n  " + "\n  ".join(errors)
        super().__init__(message)


def _check_type(value: Any, expected_type) -> bool:
    """检查值是否符合期望类型（支持 union 类型元组）。"""
    if isinstance(expected_type, tuple):
        return any(isinstance(value, t) for t in expected_type)
    return isinstance(value, expected_type)


def validate_topology_result(data: dict, strict: bool = True) -> list[str]:
    """验证 topology 结果 JSON 的完整性。

    Args:
        data:   加载的 JSON 字典
        strict: 如果 True，条件字段也必须存在

    Returns:
        错误消息列表（空列表 = 通过）

    设计文档 Phase 0 验收：
      缺少字段时分析脚本必须失败，不能填默认值继续运行。
    """
    errors = []

    # 必填字段
    for field, expected_type in TOPOLOGY_RESULT_REQUIRED_FIELDS.items():
        if field not in data:
            errors.append(f"missing required field: '{field}'")
        elif data[field] is None:
            errors.append(f"required field is None: '{field}'")
        elif not _check_type(data[field], expected_type):
            actual = type(data[field]).__name__
            expected = expected_type.__name__ if isinstance(expected_type, type) else str(expected_type)
            errors.append(f"field '{field}' type mismatch: expected {expected}, got {actual}")

    # 条件字段（strict 模式）
    fields_to_check = (
        list(TOPOLOGY_RESULT_CONDITIONAL_FIELDS.items()) if strict else []
    )
    for field, expected_type in fields_to_check:
        if field in data and data[field] is not None:
            if not _check_type(data[field], expected_type):
                actual = type(data[field]).__name__
                expected = expected_type.__name__ if isinstance(expected_type, type) else str(expected_type)
                errors.append(f"field '{field}' type mismatch: expected {expected}, got {actual}")
        elif strict:
            errors.append(f"missing conditional field (strict mode): '{field}'")

    # 边数一致性
    if "edge_count" in data and "expected_edge_count" in data:
        if data["edge_count"] != data["expected_edge_count"]:
            errors.append(
                f"edge_count {data['edge_count']} != expected_edge_count "
                f"{data['expected_edge_count']}"
            )

    # 哈希长度（SHA256 = 64 hex chars）
    for hash_field in ["topology_hash", "module_label_hash", "ei_type_hash"]:
        if hash_field in data and isinstance(data[hash_field], str):
            if len(data[hash_field]) != 64:
                errors.append(
                    f"{hash_field} length {len(data[hash_field])} != 64 (SHA256 hex)"
                )

    # validation seeds 非空
    if "validation_seeds_used" in data:
        if isinstance(data["validation_seeds_used"], list) and len(data["validation_seeds_used"]) == 0:
            errors.append("validation_seeds_used is empty")

    return errors


def validate_gain_result(data: dict) -> list[str]:
    """验证 gain 结果 JSON 的完整性。"""
    errors = []

    for field, expected_type in GAIN_RESULT_REQUIRED_FIELDS.items():
        if field not in data:
            errors.append(f"missing required field: '{field}'")

    # conditions 必须存在且非空
    conditions = data.get("conditions", {})
    if not conditions:
        errors.append("conditions is empty or missing")
    else:
        for cond_name, cond_data in conditions.items():
            if not isinstance(cond_data, dict):
                errors.append(f"condition '{cond_name}' is not a dict")
                continue

            if "error" in cond_data:
                continue  # 错误条件跳过进一步验证

            # 有 test 结果的条件必须有 raw_return_mean
            if "test" in cond_data:
                test = cond_data["test"]
                if not isinstance(test, dict):
                    errors.append(f"condition '{cond_name}' test is not a dict")
                elif "raw_return_mean" not in test:
                    errors.append(
                        f"condition '{cond_name}' test missing 'raw_return_mean'"
                    )
                elif "test_seeds" not in test:
                    errors.append(
                        f"condition '{cond_name}' test missing 'test_seeds'"
                    )

    return errors


def validate_result_file(
    path: str | Path,
    result_type: str = "topology",
    strict: bool = True,
) -> tuple[bool, list[str]]:
    """验证单个结果文件。

    Args:
        path:        结果文件路径
        result_type: "topology" 或 "gain"
        strict:      严格模式

    Returns:
        (ok, errors)
    """
    path = Path(path)
    if not path.exists():
        return False, [f"file not found: {path}"]

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"JSON decode error: {e}"]

    if result_type == "topology":
        errors = validate_topology_result(data, strict)
    elif result_type == "gain":
        errors = validate_gain_result(data)
    else:
        return False, [f"unknown result_type: {result_type}"]

    return len(errors) == 0, errors


def validate_results_dir(
    results_dir: str | Path,
    result_type: str = "topology",
    strict: bool = True,
    raise_on_error: bool = False,
) -> dict:
    """验证目录中所有结果文件。

    Returns:
        {
            'n_files': int,
            'n_valid': int,
            'n_invalid': int,
            'errors': {filename: [error_messages]},
        }
    """
    results_dir = Path(results_dir)
    pattern = "*.json"
    files = sorted(results_dir.glob(pattern))

    summary = {
        "n_files": len(files),
        "n_valid": 0,
        "n_invalid": 0,
        "errors": {},
    }

    for fpath in files:
        if "summary" in fpath.name:
            continue  # 跳过汇总文件

        ok, errors = validate_result_file(fpath, result_type, strict)
        if ok:
            summary["n_valid"] += 1
        else:
            summary["n_invalid"] += 1
            summary["errors"][fpath.name] = errors

    if raise_on_error and summary["n_invalid"] > 0:
        raise SchemaValidationError(
            [f"{fn}: {e}" for fn, errs in summary["errors"].items() for e in errs]
        )

    return summary


# ── Protocol 元信息嵌入工具 ────────────────────────────────────────────

def get_protocol_metadata(
    protocol_version: str = "v2.0",
) -> dict:
    """获取当前代码版本的 protocol 元信息，用于嵌入结果 JSON。

    设计文档 Phase 0：protocol version 和代码 commit/hash 字段。
    """
    import hashlib
    import subprocess
    from datetime import datetime

    # git commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        code_commit = result.stdout.strip()[:12] if result.returncode == 0 else "unknown"
    except Exception:
        code_commit = "unknown"

    return {
        "protocol_version": protocol_version,
        "code_commit": code_commit,
        "timestamp": datetime.now().isoformat(),
    }


def embed_protocol_metadata(
    result: dict,
    protocol_version: str = "v2.0",
    config_hash: str | None = None,
) -> dict:
    """将 protocol 元信息嵌入结果字典。

    确保每个结果文件都包含完整的审计追溯信息。
    """
    meta = get_protocol_metadata(protocol_version)
    result["protocol_version"] = meta["protocol_version"]
    result["code_commit"] = meta["code_commit"]
    result["timestamp"] = meta["timestamp"]
    if config_hash is not None:
        result["config_hash"] = config_hash
    return result
