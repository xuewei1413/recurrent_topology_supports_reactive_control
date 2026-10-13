"""
统计分析（Phase 11 / v3 / v4 mechanism）

设计文档 §12 统计分析：
  - 独立 evolved topology 是主要统计单位，不把 episode 当作独立生物/算法重复
  - 每个 topology 内先计算相同 test episodes 上的条件均值和配对差值
  - 每任务报告均值、中位数、标准差、四分位距、bootstrap 95% CI 和配对效应量
  - 主要对比使用配对置换检验；四任务多重比较使用 Holm 校正
  - 同时使用跨任务分层模型估计总体自由度效应，但不替代任务级结果
  - 等价性或非劣效性必须在 pilot 后、确认性运行前定义 SESOI
  - raw return 为主终点

设计文档 §13 结果决策规则：
  - F0 > RA 且 F0 > RW：支持具体 evolved topology 的独立贡献
  - F0 > RA 但 F0 不优于 RW：支持度分布或粗结构贡献
  - M4 获得 E 大部分增益且处于预设等价边界内：支持低维模块调制
  - E > M4 > F0：支持分层增益

v4 修复要点（protocol_v4_mechanism.json §statistics）：
  - paired_connection: by_evolution_seed （不是 glob 顺序）
  - sign_flip_exact: n<=20 时枚举全部 2^n 符号组合
  - holm_monotone: 应用 cumulative max 保证单调
  - tost_paired: 用配对差值的标准误和 df=n-1
  - bootstrap_ci: 保留 bootstrap CI
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats

_HERE = Path(__file__).resolve().parent
_CODE_DIR = _HERE.parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))


# ── 描述性统计 ────────────────────────────────────────────────────────

@dataclass
class DescriptiveStats:
    mean: float
    median: float
    std: float
    iqr: float
    n: int
    bootstrap_ci_low: float
    bootstrap_ci_high: float

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "iqr": self.iqr,
            "n": self.n,
            "bootstrap_ci_95": [self.bootstrap_ci_low, self.bootstrap_ci_high],
        }


def descriptive_statistics(
    values: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> DescriptiveStats:
    """计算描述性统计量。

    设计文档 §12：均值、中位数、标准差、四分位距、bootstrap 95% CI。
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    mean = float(values.mean())
    median = float(np.median(values))
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    q75, q25 = np.percentile(values, [75, 25])
    iqr = float(q75 - q25)

    # Bootstrap CI
    rng = np.random.RandomState(seed)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)
    alpha = 1 - ci_level
    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return DescriptiveStats(mean, median, std, iqr, n, ci_low, ci_high)


# ── 配对数据规整工具 ─────────────────────────────────────────────────

def _align_by_seed(
    paired_data: dict[str, dict[int, float]],
    cond_a: str,
    cond_b: str,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """将两个条件按 evolution_seed 对齐。

    Args:
        paired_data: {condition: {evolution_seed: value}}
        cond_a, cond_b: 两个对比的条件

    Returns:
        (a_values, b_values, common_seeds, missing_seeds)
        a_values / b_values 顺序与 common_seeds 严格一致
        missing_seeds = a 中有但 b 没有 + b 中有但 a 没有的 seed 并集
    """
    a_map = paired_data.get(cond_a, {})
    b_map = paired_data.get(cond_b, {})

    common = sorted(set(a_map.keys()) & set(b_map.keys()))
    missing = sorted((set(a_map.keys()) | set(b_map.keys())) - set(common))

    a_values = np.array([a_map[s] for s in common], dtype=np.float64)
    b_values = np.array([b_map[s] for s in common], dtype=np.float64)
    return a_values, b_values, common, missing


def _coerce_paired_input(
    a, b
) -> tuple[np.ndarray, np.ndarray]:
    """接受两种输入格式：
    - 两个等长 array（向后兼容旧 API）
    - dict {seed: value}（v3/v4 paired loader 输出），按 seed 对齐

    返回对齐后的 (a_values, b_values)。
    """
    if isinstance(a, dict) and isinstance(b, dict):
        common = sorted(set(a.keys()) & set(b.keys()))
        a_values = np.array([a[s] for s in common], dtype=np.float64)
        b_values = np.array([b[s] for s in common], dtype=np.float64)
        return a_values, b_values

    a_values = np.asarray(a, dtype=np.float64)
    b_values = np.asarray(b, dtype=np.float64)
    if len(a_values) != len(b_values):
        raise ValueError(
            f"Length mismatch (a={len(a_values)}, b={len(b_values)}); "
            "pass dicts keyed by evolution_seed for proper pairing"
        )
    return a_values, b_values


# ── 配对置换检验（v4: 精确符号翻转）──────────────────────────────────

def paired_permutation_test(
    a,
    b,
    n_permutations: int = 10000,
    seed: int = 42,
    exact_threshold: int = 20,
) -> dict:
    """配对符号翻转检验（v4 修正版）。

    设计文档 §12：主要对比使用配对置换检验。
    protocol_v4_mechanism.json §statistics.sign_flip_exact: true。

    对每对配对差值 d_i = a_i - b_i，零假设下 d_i 的符号可互换，
    检验统计量为 T = mean(d_i)（等价于 mean(a) - mean(b)）。

    Args:
        a, b:           两个条件的数据
                        - array-like（向后兼容，需等长）
                        - dict {seed: value}（推荐，按 seed 自动对齐）
        n_permutations: 仅当 n > exact_threshold 时使用（Monte Carlo 次数）
        seed:           Monte Carlo 随机种子
        exact_threshold: n <= 此值时枚举全部 2^n 符号组合（默认 20）

    Returns:
        {
            'observed_diff': float,
            'raw_p':         float,           # 未校正 p 值
            'p_value':       float,           # 同 raw_p（向后兼容）
            'cohens_d':      float,
            'n_pairs':       int,
            'n_missing':     int,             # 仅当输入为 dict 时计算
            'method':        'exact_sign_flip' | 'monte_carlo_sign_flip',
            'n_permutations': int,
        }
    """
    is_dict_input = isinstance(a, dict) and isinstance(b, dict)
    if is_dict_input:
        common = sorted(set(a.keys()) & set(b.keys()))
        missing = sorted((set(a.keys()) | set(b.keys())) - set(common))
        a_values = np.array([a[s] for s in common], dtype=np.float64)
        b_values = np.array([b[s] for s in common], dtype=np.float64)
        n_missing = len(missing)
    else:
        a_values, b_values = _coerce_paired_input(a, b)
        n_missing = 0

    n = len(a_values)
    differences = a_values - b_values
    observed_diff = float(differences.mean()) if n > 0 else 0.0

    if n == 0:
        return {
            "observed_diff": 0.0,
            "raw_p": 1.0,
            "p_value": 1.0,
            "cohens_d": 0.0,
            "n_pairs": 0,
            "n_missing": n_missing,
            "method": "no_data",
            "n_permutations": 0,
        }

    if n <= exact_threshold:
        # ── Exact sign-flip test: 枚举全部 2^n 符号组合 ──
        # 每种组合的统计量 T = mean(signs * d)
        # 双侧 p = #{|T_perm| >= |T_obs|} / 2^n
        # 用 vectorized 方式：对 n<=20, 2^20 = 1,048,576，可承受
        n_combos = 1 << n  # 2^n
        # 构造所有 ±1 符号矩阵：(n_combos, n)
        # 用 bit 表示：bit k 表示第 k 个差值的符号
        signs_all = np.where(
            (np.arange(n_combos)[:, None] >> np.arange(n)[None, :]) & 1,
            1.0,
            -1.0,
        )
        perm_stats = signs_all @ differences / n  # (n_combos,)
        abs_obs = abs(observed_diff)
        count_extreme = int(np.sum(np.abs(perm_stats) >= abs_obs))
        # Exact p = (count + 1) / (N + 1) 太保守；exact 通常用 count / N
        # 这里 N = 2^n 包含了 observed 本身，所以 count/N 是标准 exact p
        raw_p = count_extreme / n_combos
        method = "exact_sign_flip"
        n_perms_report = n_combos
    else:
        # ── Monte Carlo sign-flip: random ±1 sign vectors ──
        # p = (extreme_count + 1) / (n_permutations + 1)
        # 参考 Davison & Hinkley (1997)，避免 p=0
        rng = np.random.RandomState(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
        perm_stats = (signs * differences[None, :]).mean(axis=1)
        abs_obs = abs(observed_diff)
        count_extreme = int(np.sum(np.abs(perm_stats) >= abs_obs))
        raw_p = (count_extreme + 1) / (n_permutations + 1)
        method = "monte_carlo_sign_flip"
        n_perms_report = n_permutations

    # Cohen's d for paired = mean(d) / std(d, ddof=1)
    if n > 1 and differences.std(ddof=1) > 0:
        cohens_d = float(differences.mean() / differences.std(ddof=1))
    else:
        cohens_d = 0.0

    return {
        "observed_diff": observed_diff,
        "raw_p": float(raw_p),
        "p_value": float(raw_p),  # 向后兼容
        "cohens_d": cohens_d,
        "n_pairs": int(n),
        "n_missing": int(n_missing),
        "method": method,
        "n_permutations": int(n_perms_report),
    }


# ── Holm 校正（v4: 保证单调）────────────────────────────────────────

def holm_correction(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Holm 多重比较校正（v4 monotonicity-fixed 版）。

    设计文档 §12：四任务多重比较使用 Holm 校正。
    protocol_v4_mechanism.json §statistics.holm_monotone: true。

    步骤：
      1. 按 raw p 升序排序：p_(1) <= p_(2) <= ... <= p_(m)
      2. 原始 Holm 调整：raw_adj_(i) = (m - i + 1) * p_(i)，截断到 [0,1]
      3. 应用 cumulative max 保证单调不减：
            adj_(1) = raw_adj_(1)
            adj_(i) = max(raw_adj_(i), adj_(i-1))
      4. reject_(i) = (adj_(i) <= alpha) 且前面的都被 reject

    Returns:
        list of dict（顺序与输入 p_values 一致），每个元素：
            {
                'original_index': int,
                'raw_p':         float,
                'adjusted_p':    float,    # 单调不减
                'reject_h0':     bool,
                'holm_alpha':    float,    # 第 i 步的 step-down alpha
                'rank':          int,      # 1-indexed sort rank
            }
    """
    n = len(p_values)
    if n == 0:
        return []

    # 1. 升序排序
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    # 2. 原始 Holm 调整 + 3. cumulative max
    adjusted: list[tuple[int, float, float, int]] = []  # (orig_idx, raw_p, adj_p, rank)
    prev_adj = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):  # rank = 0..n-1
        raw_adj = p * (n - rank)
        raw_adj = min(raw_adj, 1.0)
        adj_p = max(raw_adj, prev_adj) if rank > 0 else raw_adj
        adj_p = min(adj_p, 1.0)
        adjusted.append((orig_idx, p, adj_p, rank + 1))
        prev_adj = adj_p

    # 4. reject: step-down —— 一旦遇到不拒绝就停止
    results = [None] * n
    stopped = False
    for orig_idx, p, adj_p, rank in adjusted:
        # step-down alpha: alpha / (n - rank + 1)
        step_alpha = alpha / (n - rank + 1)
        if stopped:
            reject = False
        else:
            reject = (p <= step_alpha) and (adj_p <= alpha)
            if not reject:
                stopped = True
        results[orig_idx] = {
            "original_index": orig_idx,
            "raw_p": float(p),
            "adjusted_p": float(adj_p),
            "reject_h0": bool(reject),
            "holm_alpha": float(step_alpha),
            "rank": int(rank),
        }

    return results


# ── 等价性/非劣效性检验（v4: 配对 TOST）─────────────────────────────

def tost_equivalence(
    a,
    b,
    sesoi: float,
    alpha: float = 0.05,
) -> dict:
    """配对 TOST 等价性检验（v4 修正版）。

    设计文档 §12：等价性或非劣效性必须在 pilot 后定义 SESOI。
    protocol_v4_mechanism.json §statistics.tost_paired: true。

    H0: |mean(a) - mean(b)| >= sesoi
    H1: |mean(a) - mean(b)| < sesoi  (等价)

    两个单侧检验：
      H0_lower: diff <= -sesoi   →  t_lower = (diff + sesoi) / SE,  p_lower = 1 - T.cdf(t_lower)
      H0_upper: diff >= +sesoi   →  t_upper = (diff - sesoi) / SE,  p_upper = T.cdf(t_upper)
    p_tost = max(p_lower, p_upper)，等价 iff p_tost < alpha。

    v4 改动：使用配对差值 d_i = a_i - b_i 的标准误：
      SE = std(d, ddof=1) / sqrt(n),  df = n - 1
    （旧版用 Welch SE，会高估 SE 并损失自由度。）

    Args:
        a, b:   配对数据。可以是 array-like（需等长）或 dict {seed: value}
        sesoi:  Smallest Effect Size of Interest（raw return 单位）
        alpha:  显著性水平

    Returns:
        {
            'diff_mean':   float,
            'paired_se':   float,
            'df':          float,
            'sesoi':       float,
            'p_lower':     float,
            'p_upper':     float,
            'p_tost':      float,
            'equivalent':  bool,
            'n_pairs':     int,
            # 向后兼容字段
            'diff':        float,   # 同 diff_mean
            'se':          float,   # 同 paired_se
        }
    """
    is_dict_input = isinstance(a, dict) and isinstance(b, dict)
    if is_dict_input:
        common = sorted(set(a.keys()) & set(b.keys()))
        missing = sorted((set(a.keys()) | set(b.keys())) - set(common))
        a_values = np.array([a[s] for s in common], dtype=np.float64)
        b_values = np.array([b[s] for s in common], dtype=np.float64)
        n_missing = len(missing)
    else:
        a_values, b_values = _coerce_paired_input(a, b)
        n_missing = 0

    n = len(a_values)
    differences = a_values - b_values
    diff_mean = float(differences.mean()) if n > 0 else 0.0

    if n < 2:
        return {
            "diff_mean": diff_mean,
            "paired_se": 0.0,
            "df": 0.0,
            "sesoi": float(sesoi),
            "p_lower": 1.0,
            "p_upper": 1.0,
            "p_tost": 1.0,
            "equivalent": False,
            "n_pairs": int(n),
            "n_missing": int(n_missing),
            # bc
            "diff": diff_mean,
            "se": 0.0,
        }

    # 配对 SE: std(d, ddof=1) / sqrt(n),  df = n - 1
    d_std = float(differences.std(ddof=1))
    paired_se = d_std / np.sqrt(n) if d_std > 0 else 0.0
    df = float(n - 1)

    if paired_se == 0:
        # 差值为常数：等价 iff |diff_mean| < sesoi
        equivalent = bool(abs(diff_mean) < sesoi)
        p_lower = 0.0 if (diff_mean + sesoi) > 0 else 1.0
        p_upper = 0.0 if (diff_mean - sesoi) < 0 else 1.0
        p_tost = max(p_lower, p_upper)
        return {
            "diff_mean": diff_mean,
            "paired_se": 0.0,
            "df": df,
            "sesoi": float(sesoi),
            "p_lower": float(p_lower),
            "p_upper": float(p_upper),
            "p_tost": float(p_tost),
            "equivalent": equivalent,
            "n_pairs": int(n),
            "n_missing": int(n_missing),
            "diff": diff_mean,
            "se": 0.0,
        }

    t_lower = (diff_mean + sesoi) / paired_se
    t_upper = (diff_mean - sesoi) / paired_se
    p_lower = float(1.0 - stats.t.cdf(t_lower, df))
    p_upper = float(stats.t.cdf(t_upper, df))
    p_tost = float(max(p_lower, p_upper))

    return {
        "diff_mean": diff_mean,
        "paired_se": float(paired_se),
        "df": df,
        "sesoi": float(sesoi),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": p_tost,
        "equivalent": bool(p_tost < alpha),
        "n_pairs": int(n),
        "n_missing": int(n_missing),
        # 向后兼容字段
        "diff": diff_mean,
        "se": float(paired_se),
    }


# ── 主对比分析 ────────────────────────────────────────────────────────

PRIMARY_CONTRASTS = [
    ("F0", "RA"),   # 结构充分性
    ("F0", "RW"),   # 结构特异性
    ("G1", "F0"),   # 全局 gain 增益
    ("M4", "G1"),   # 模块 gain 增益
    ("N", "M4"),    # 逐神经元 gain 增益
    ("E", "N"),     # 逐边 gain 增益
]

# v4 等价性对比（M4/M2/M8/N vs E）
EQUIVALENCE_CONTRASTS = [
    ("M4", "E"),
    ("M2", "E"),
    ("M8", "E"),
    ("N", "E"),
]


def analyze_task(
    task_name: str,
    topology_level_means: dict[str, list[float]],
    sesoi: float | None = None,
    n_permutations: int = 10000,
) -> dict:
    """分析单个任务的全部配对对比（向后兼容入口）。

    旧 API：topology_level_means = {condition: [per-topology test means]}
    列表顺序隐含配对（不安全，v4 起推荐用 analyze_task_paired）。

    本函数继续工作，但内部仍按列表顺序对齐，请优先使用
    analyze_task_paired，它通过 evolution_seed 严格对齐。
    """
    conditions = list(topology_level_means.keys())

    # 描述性统计
    descriptive = {}
    for cond in conditions:
        vals = np.array(topology_level_means[cond])
        descriptive[cond] = descriptive_statistics(vals).to_dict()

    # 配对对比
    contrast_results = []
    p_values_for_holm = []
    contrast_names_for_holm = []

    for cond_a, cond_b in PRIMARY_CONTRASTS:
        if cond_a not in topology_level_means or cond_b not in topology_level_means:
            continue

        a = np.array(topology_level_means[cond_a])
        b = np.array(topology_level_means[cond_b])

        n_min = min(len(a), len(b))
        if n_min < 2:
            continue

        result = paired_permutation_test(a[:n_min], b[:n_min], n_permutations)
        contrast_name = f"{cond_a}-{cond_b}"
        result["contrast"] = contrast_name
        contrast_results.append(result)

        p_values_for_holm.append(result["raw_p"])
        contrast_names_for_holm.append(contrast_name)

    # Holm 校正
    holm_results = holm_correction(p_values_for_holm) if p_values_for_holm else []
    holm_map = {contrast_names_for_holm[i]: holm_results[i]
                for i in range(len(holm_results))}

    for cr in contrast_results:
        cr["holm"] = holm_map.get(cr["contrast"], {})

    # 等价性检验（旧 API：用列表，TOST 内部按列表顺序配对）
    equivalence = {}
    if sesoi is not None:
        for cond_a, cond_b in [("M4", "E"), ("N", "E")]:
            if cond_a in topology_level_means and cond_b in topology_level_means:
                a = np.array(topology_level_means[cond_a])
                b = np.array(topology_level_means[cond_b])
                n_min = min(len(a), len(b))
                if n_min >= 2:
                    eq = tost_equivalence(a[:n_min], b[:n_min], sesoi)
                    equivalence[f"{cond_a}-{cond_b}"] = eq

    # 结果决策规则
    decisions = _apply_decision_rules(descriptive, contrast_results)

    return {
        "task": task_name,
        "descriptive": descriptive,
        "contrasts": contrast_results,
        "equivalence": equivalence,
        "decision_rules": decisions,
    }


def analyze_task_paired(
    task_name: str,
    paired_data: dict[str, dict[int, float]],
    sesoi: float | None = None,
    n_permutations: int = 10000,
    primary_contrasts: list[tuple[str, str]] | None = None,
    equivalence_contrasts: list[tuple[str, str]] | None = None,
) -> dict:
    """v4 推荐的任务级分析入口：严格按 evolution_seed 配对。

    设计文档 §12 + protocol_v4_mechanism.json §statistics.paired_connection:
    'by_evolution_seed'。

    Args:
        task_name:               任务名
        paired_data:             {condition: {evolution_seed: value}}
                                 由 load_paired_condition_data 返回
        sesoi:                   等价性检验的 SESOI（如定义）
        n_permutations:          Monte Carlo 置换次数（n>20 时）
        primary_contrasts:       主对比列表（默认 PRIMARY_CONTRASTS）
        equivalence_contrasts:   等价性对比列表（默认 EQUIVALENCE_CONTRASTS）

    Returns:
        {
            'task':           str,
            'descriptive':    {condition: DescriptiveStats.to_dict()},
            'n_seeds':        {condition: int},     # 每条件 seed 数
            'contrasts':      [contrast_result, ...],
            'equivalence':    {contrast_name: tost_result}  (如 sesoi 定义),
            'decision_rules': {...},
        }
        其中 contrast_result 包含：
            contrast, observed_diff, raw_p, cohens_d,
            n_pairs, n_missing, common_seeds, missing_seeds,
            ci_95 (bootstrap on paired diffs),
            holm: {raw_p, adjusted_p, reject_h0, holm_alpha, rank},
            method: 'exact_sign_flip' | 'monte_carlo_sign_flip',
    """
    contrasts_to_run = primary_contrasts if primary_contrasts is not None else PRIMARY_CONTRASTS
    equiv_to_run = equivalence_contrasts if equivalence_contrasts is not None else EQUIVALENCE_CONTRASTS

    conditions = list(paired_data.keys())

    # 描述性统计（每条件全部 seed）
    descriptive = {}
    n_seeds = {}
    for cond, seed_map in paired_data.items():
        vals = np.array(list(seed_map.values()), dtype=np.float64)
        n_seeds[cond] = int(len(vals))
        if len(vals) > 0:
            descriptive[cond] = descriptive_statistics(vals).to_dict()
        else:
            descriptive[cond] = DescriptiveStats(0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0).to_dict()

    # 主对比：按 seed 配对
    contrast_results = []
    p_values_for_holm = []
    contrast_names_for_holm = []

    for cond_a, cond_b in contrasts_to_run:
        if cond_a not in paired_data or cond_b not in paired_data:
            continue

        a_values, b_values, common_seeds, missing_seeds = _align_by_seed(
            paired_data, cond_a, cond_b
        )
        if len(a_values) < 2:
            continue

        result = paired_permutation_test(a_values, b_values, n_permutations)
        contrast_name = f"{cond_a}-{cond_b}"
        result["contrast"] = contrast_name
        result["common_seeds"] = list(common_seeds)
        result["missing_seeds"] = list(missing_seeds)
        # _align_by_seed 知道有多少 seed 缺失，覆盖 paired_permutation_test
        # 内部对数组输入报告的 n_missing=0
        result["n_missing"] = len(missing_seeds)

        # 配对差值的 bootstrap CI
        diffs = a_values - b_values
        ds = descriptive_statistics(diffs)
        result["diff_mean"] = ds.mean
        result["diff_std"] = ds.std
        result["ci_95"] = [ds.bootstrap_ci_low, ds.bootstrap_ci_high]

        contrast_results.append(result)
        p_values_for_holm.append(result["raw_p"])
        contrast_names_for_holm.append(contrast_name)

    # Holm 校正（单调保证）
    holm_results = holm_correction(p_values_for_holm) if p_values_for_holm else []
    holm_map = {contrast_names_for_holm[i]: holm_results[i]
                for i in range(len(holm_results))}

    for cr in contrast_results:
        cr["holm"] = holm_map.get(cr["contrast"], {})

    # 等价性检验：按 seed 配对 TOST
    equivalence = {}
    if sesoi is not None:
        for cond_a, cond_b in equiv_to_run:
            if cond_a not in paired_data or cond_b not in paired_data:
                continue
            a_values, b_values, common_seeds, missing_seeds = _align_by_seed(
                paired_data, cond_a, cond_b
            )
            if len(a_values) < 2:
                continue
            eq = tost_equivalence(a_values, b_values, sesoi)
            eq["common_seeds"] = list(common_seeds)
            eq["missing_seeds"] = list(missing_seeds)
            eq["n_missing"] = len(missing_seeds)
            equivalence[f"{cond_a}-{cond_b}"] = eq

    # 决策规则
    decisions = _apply_decision_rules(descriptive, contrast_results)

    return {
        "task": task_name,
        "descriptive": descriptive,
        "n_seeds": n_seeds,
        "contrasts": contrast_results,
        "equivalence": equivalence,
        "decision_rules": decisions,
    }


def _apply_decision_rules(descriptive: dict, contrasts: list[dict]) -> dict:
    """应用设计文档 §13 的结果决策规则。

    设计文档 §13：
      - F0 > RA 且 F0 > RW：支持具体 evolved topology 的独立贡献
      - F0 > RA 但 F0 不优于 RW：支持度分布或粗结构贡献
      - M4 获得 E 大部分增益且处于预设等价边界内
      - E > M4 > F0：支持分层增益
    """
    decisions = {}

    def _mean(cond):
        return descriptive.get(cond, {}).get("mean", 0)

    def _sig(contrast_name):
        for c in contrasts:
            if c.get("contrast") == contrast_name:
                return c.get("holm", {}).get("reject_h0", False)
        return False

    f0_ra_sig = _sig("F0-RA")
    f0_rw_sig = _sig("F0-RW")

    decisions["F0_better_than_RA"] = f0_ra_sig and (_mean("F0") > _mean("RA"))
    decisions["F0_better_than_RW"] = f0_rw_sig and (_mean("F0") > _mean("RW"))

    if decisions["F0_better_than_RA"] and decisions["F0_better_than_RW"]:
        decisions["topology_contribution"] = "specific_wiring"
    elif decisions["F0_better_than_RA"] and not decisions["F0_better_than_RW"]:
        decisions["topology_contribution"] = "degree_distribution"
    else:
        decisions["topology_contribution"] = "not_supported"

    # 分层增益检查
    decisions["E_better_than_M4"] = _sig("E-N") or _mean("E") > _mean("M4")
    decisions["M4_better_than_F0"] = _sig("M4-G1") or _mean("M4") > _mean("F0")
    decisions["hierarchical_gains"] = (
        decisions["E_better_than_M4"] and decisions["M4_better_than_F0"]
    )

    return decisions


# ── 从结果文件加载 topology-level means（旧 API）─────────────────────

def load_topology_level_means(
    task_name: str,
    results_dir: str | Path,
) -> dict[str, list[float]]:
    """从 Phase 8 的 gain_result JSON 文件加载 topology-level condition means。

    设计文档 §12：每个 topology 内先计算相同 test episodes 上的条件均值。
    支持文件在 results_dir/ 下或 results_dir/{task_name}/ 下。

    注意：返回 dict[str, list[float]]，列表顺序由 glob 决定，不安全。
    v4 起推荐用 load_paired_condition_data。
    """
    results_dir = Path(results_dir)
    condition_means = {}

    pattern = f"gain_result_{task_name}_seed*.json"
    files = sorted(results_dir.glob(pattern))
    if not files:
        task_subdir = results_dir / task_name
        if task_subdir.exists():
            files = sorted(task_subdir.glob(pattern))
    if not files:
        files = sorted(results_dir.rglob(pattern))

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)

        conditions = data.get("conditions", {})
        for cond, cond_data in conditions.items():
            if "test" in cond_data:
                if cond not in condition_means:
                    condition_means[cond] = []
                condition_means[cond].append(cond_data["test"]["raw_return_mean"])

    return condition_means


# ── v3/v4 paired loader（按 evolution_seed 严格对齐）─────────────────

def _find_files_for_patterns(
    base_dir: Path,
    task_name: str,
    prefixes: Iterable[str],
) -> list[Path]:
    """在 base_dir（及其 {task_name} 子目录）下查找所有匹配
    '{prefix}_{task}_seed*.json' 的文件，按文件名排序去重。
    """
    collected: list[Path] = []
    seen: set[Path] = set()

    search_dirs = [base_dir]
    task_subdir = base_dir / task_name
    if task_subdir.exists():
        search_dirs.append(task_subdir)

    for prefix in prefixes:
        pattern = f"{prefix}_{task_name}_seed*.json"
        for d in search_dirs:
            for fpath in sorted(d.glob(pattern)):
                rp = fpath.resolve()
                if rp not in seen:
                    seen.add(rp)
                    collected.append(fpath)
    # 若目录内均无匹配，再递归
    if not collected:
        for prefix in prefixes:
            pattern = f"{prefix}_{task_name}_seed*.json"
            for fpath in sorted(base_dir.rglob(pattern)):
                rp = fpath.resolve()
                if rp not in seen:
                    seen.add(rp)
                    collected.append(fpath)
    return sorted(collected, key=lambda p: p.name)


def _extract_seed_from_filename(fpath: Path, task_name: str) -> int | None:
    """从文件名提取 evolution_seed。

    支持：
      gain_result_CartPole_seed3000.json     -> 3000
      v3_gain_result_CartPole_seed3000.json  -> 3000
      result_CartPole_seed3000.json          -> 3000
      v3_result_CartPole_seed3000.json       -> 3000
    若无法解析则尝试读取 JSON 内部的 evolution_seed 字段。
    """
    name = fpath.stem
    marker = "_seed"
    idx = name.find(marker)
    if idx >= 0:
        tail = name[idx + len(marker):]
        # 取连续数字
        digits = ""
        for ch in tail:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            return int(digits)
    # fallback: 读 JSON
    try:
        with open(fpath) as f:
            data = json.load(f)
        if "evolution_seed" in data:
            return int(data["evolution_seed"])
    except Exception:
        pass
    return None


def load_paired_condition_data(
    task_name: str,
    results_dir: str | Path,
    topology_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, dict[int, float]]:
    """v3/v4 paired loader：按 evolution_seed 严格对齐每个条件的 raw_return_mean。

    设计文档 §12 + protocol_v4_mechanism.json §statistics.paired_connection:
    'by_evolution_seed'。

    查找文件：
      - gain 文件：同时匹配 `gain_result_{task}_seed*.json`
                   和 `v3_gain_result_{task}_seed*.json`
      - topology 文件（仅当 topology_dir 提供）：匹配
        `v3_result_{task}_seed*.json` 和 `result_{task}_seed*.json`
        （用于 RA/RW baseline）

    解析每个文件：
      - evolution_seed 从文件名解析，失败时退化为 JSON 内部字段
      - 每个 condition 的 test.raw_return_mean

    Args:
        task_name:    任务名（如 'CartPole'）
        results_dir:  gain_result 文件目录
        topology_dir: 可选，Phase 7 result 文件目录（RA/RW baselines）
        verbose:      是否打印缺失 seed/condition 诊断

    Returns:
        {condition: {evolution_seed: value}}
        - RA / RW 仅当 topology_dir 提供且文件存在时出现
        - 若同一 seed 在多个文件出现，后加载的覆盖前者（并打印警告）

    Reports:
        通过 print 显式报告：
          - 每条件找到的文件数与 seed 数
          - 没有任何文件的提示
          - seed 跨条件缺失情况
    """
    results_dir = Path(results_dir)
    paired: dict[str, dict[int, float]] = {}

    # ── 1. 加载 gain result 文件 ──
    gain_files = _find_files_for_patterns(
        results_dir, task_name,
        prefixes=["gain_result", "v3_gain_result"],
    )

    if not gain_files:
        if verbose:
            print(f"  [load_paired:{task_name}] "
                  f"no gain_result_* files under {results_dir}")
    else:
        if verbose:
            print(f"  [load_paired:{task_name}] "
                  f"found {len(gain_files)} gain files")

        for fpath in gain_files:
            try:
                with open(fpath) as f:
                    data = json.load(f)
            except Exception as e:
                if verbose:
                    print(f"    WARN: failed to read {fpath.name}: {e}")
                continue

            seed = _extract_seed_from_filename(fpath, task_name)
            if seed is None:
                seed = data.get("evolution_seed")
            if seed is None:
                if verbose:
                    print(f"    WARN: cannot determine seed for {fpath.name}, skip")
                continue
            seed = int(seed)

            conditions = data.get("conditions", {})
            for cond, cond_data in conditions.items():
                if not isinstance(cond_data, dict):
                    continue
                test_block = cond_data.get("test")
                if test_block is None or "raw_return_mean" not in test_block:
                    continue
                value = float(test_block["raw_return_mean"])

                cond_map = paired.setdefault(cond, {})
                if seed in cond_map and verbose:
                    print(f"    WARN: {cond} seed={seed} duplicate "
                          f"(existing={cond_map[seed]:.3f}, new={value:.3f}), overwriting")
                cond_map[seed] = value

    # ── 2. 加载 topology result 文件（RA/RW baselines） ──
    if topology_dir is not None:
        topology_dir = Path(topology_dir)
        topo_files = _find_files_for_patterns(
            topology_dir, task_name,
            prefixes=["v3_result", "result"],
        )

        if topo_files:
            if verbose:
                print(f"  [load_paired:{task_name}] "
                      f"found {len(topo_files)} topology files for RA/RW")

            for fpath in topo_files:
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                except Exception as e:
                    if verbose:
                        print(f"    WARN: failed to read {fpath.name}: {e}")
                    continue

                seed = _extract_seed_from_filename(fpath, task_name)
                if seed is None:
                    seed = data.get("evolution_seed")
                if seed is None:
                    continue
                seed = int(seed)

                # RA / RW：从 result 文件直接取（mean across replicates 已计算）
                ra_val = data.get("ra_validation_mean_of_means")
                rw_val = data.get("rw_validation_mean_of_means")

                if ra_val is not None:
                    cond_map = paired.setdefault("RA", {})
                    cond_map[seed] = float(ra_val)
                if rw_val is not None:
                    cond_map = paired.setdefault("RW", {})
                    cond_map[seed] = float(rw_val)
        else:
            if verbose:
                print(f"  [load_paired:{task_name}] "
                      f"no result_* files under {topology_dir} (RA/RW not loaded)")

    # ── 3. 报告 seed 跨条件缺失 ──
    if verbose and paired:
        all_seeds_per_cond = {c: set(s.keys()) for c, s in paired.items()}
        conditions_present = sorted(all_seeds_per_cond.keys())
        universe = set()
        for s in all_seeds_per_cond.values():
            universe |= s

        for cond in conditions_present:
            missing = sorted(universe - all_seeds_per_cond[cond])
            n_have = len(all_seeds_per_cond[cond])
            if missing:
                print(f"    {cond}: {n_have} seeds, "
                      f"missing {len(missing)} -> {missing}")
            else:
                print(f"    {cond}: {n_have} seeds (complete)")

    return paired


# ── 从 Phase 7 拓扑结果加载 RA/RW 数据（旧 API） ─────────────────────

def load_ra_rw_means(
    task_name: str,
    topology_dir: str | Path,
) -> dict[str, list[float]]:
    """从 Phase 7 的 result JSON 文件加载 RA/RW topology-level means（旧 API）。

    返回 {'RA': [ra_mean_per_topology], 'RW': [rw_mean_per_topology]}
    每个 topology 内先对 5 个 replicate 取平均（结果文件已写入 *_mean_of_means）。

    注意：返回 list 顺序由 glob 决定。v4 起推荐用 load_paired_condition_data
    并通过 topology_dir 参数直接获得配对 RA/RW。
    """
    topology_dir = Path(topology_dir)
    task_subdir = topology_dir / task_name
    search_dir = task_subdir if task_subdir.exists() else topology_dir

    files = sorted(search_dir.glob(f"result_{task_name}_seed*.json"))
    if not files:
        files = sorted(search_dir.rglob(f"result_{task_name}_seed*.json"))

    ra_means = []
    rw_means = []

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        if "ra_validation_mean_of_means" in data:
            ra_means.append(data["ra_validation_mean_of_means"])
        if "rw_validation_mean_of_means" in data:
            rw_means.append(data["rw_validation_mean_of_means"])

    return {"RA": ra_means, "RW": rw_means}


# ── 预设 SESOI（pilot 后冻结）────────────────────────────────────────

# SESOI = Smallest Effect Size of Interest
# 设计文档 §12：等价性或非劣效性必须在 pilot 后、确认性运行前定义 SESOI
# 以下为四任务的预设 SESOI（raw return 单位），基于 pilot 结果的标准差设定
# 与 protocol_v4_mechanism.json §SESOI 一致
TASK_SESOI = {
    "CartPole": 30.0,       # 满分 500，SESOI=30 约为 6%
    "MountainCar": 15.0,    # 范围约 -200 到 -110，SESOI=15
    "Acrobot": 10.0,        # 范围约 -500 到 -80，SESOI=10
    "LunarLander": 30.0,    # 范围约 -200 到 -40，SESOI=30
}


# ── 跨任务分层模型 ────────────────────────────────────────────────────

def cross_task_hierarchical_model(
    task_data: dict[str, dict[str, list[float]]],
    contrast: tuple[str, str] | None = None,
) -> dict:
    """跨任务分层模型估计总体自由度效应。

    设计文档 §12：同时使用跨任务分层模型估计总体自由度效应，
    但不替代任务级结果。

    用 mixed-effects model 估计总体效应：
      diff_{t,s} = β_0 + u_t + ε_{t,s}
    其中 diff_{t,s} 是任务 t 中 topology s 的配对差值，
    β_0 是总体固定效应，u_t 是任务级随机效应。

    如果 statsmodels 可用，使用 MixedLM；否则退化为随机效应 meta-analysis。

    Args:
        task_data: {task_name: {condition: [per-topology means]}}
        contrast:  要分析的配对，如 ("F0", "RA")；None 则分析全部主要对比

    Returns:
        {contrast_name: {fixed_effect, fixed_se, random_var, n_tasks, n_total}}
    """
    contrasts_to_analyze = PRIMARY_CONTRASTS if contrast is None else [contrast]
    results = {}

    for cond_a, cond_b in contrasts_to_analyze:
        all_diffs = []
        task_groups = []

        for task_name, conds in task_data.items():
            if cond_a not in conds or cond_b not in conds:
                continue

            a = np.array(conds[cond_a])
            b = np.array(conds[cond_b])
            n = min(len(a), len(b))
            if n < 1:
                continue

            diffs = a[:n] - b[:n]
            all_diffs.extend(diffs.tolist())
            task_groups.extend([task_name] * n)

        if len(all_diffs) < 3 or len(set(task_groups)) < 2:
            results[f"{cond_a}-{cond_b}"] = {
                "error": "insufficient data for hierarchical model",
                "n_tasks": len(set(task_groups)),
                "n_total": len(all_diffs),
            }
            continue

        diffs_arr = np.array(all_diffs)
        groups_arr = np.array(task_groups)
        unique_tasks = sorted(set(task_groups))

        try:
            import pandas as pd
            from statsmodels.formula.api import mixedlm

            df = pd.DataFrame({
                "diff": diffs_arr,
                "task": groups_arr,
            })

            model = mixedlm("diff ~ 1", df, groups=df["task"])
            fit = model.fit(method="lbfgs", maxiter=200, disp=False)

            fixed_effect = float(fit.fe_params.iloc[0])
            fixed_se = float(fit.bse_fe.iloc[0])
            random_var = float(fit.cov_re.iloc[0, 0]) if hasattr(fit, 'cov_re') else 0.0
            residual_var = float(fit.scale)

            results[f"{cond_a}-{cond_b}"] = {
                "method": "MixedLM",
                "fixed_effect": fixed_effect,
                "fixed_se": fixed_se,
                "fixed_z": fixed_effect / fixed_se if fixed_se > 0 else 0.0,
                "fixed_p": float(2 * stats.norm.sf(abs(fixed_effect / fixed_se))) if fixed_se > 0 else 1.0,
                "random_task_var": random_var,
                "residual_var": residual_var,
                "n_tasks": len(unique_tasks),
                "n_total": len(all_diffs),
            }

        except (ImportError, Exception) as e:
            task_means = []
            task_vars = []
            for task_name in unique_tasks:
                mask = groups_arr == task_name
                t_diffs = diffs_arr[mask]
                if len(t_diffs) > 0:
                    task_means.append(t_diffs.mean())
                    task_vars.append(t_diffs.var(ddof=1) / len(t_diffs) if len(t_diffs) > 1 else 0)

            task_means = np.array(task_means)
            task_vars = np.array(task_vars)

            w = np.where(task_vars > 0, 1.0 / task_vars, 1.0)
            Q = np.sum(w * task_means ** 2) - np.sum(w * task_means) ** 2 / np.sum(w)
            df_q = len(task_means) - 1
            tau2 = max(0, (Q - df_q) / (np.sum(w) - np.sum(w ** 2) / np.sum(w))) if np.sum(w) > np.sum(w ** 2) / np.sum(w) else 0

            w_star = np.where(task_vars + tau2 > 0, 1.0 / (task_vars + tau2), 1.0)
            pooled_effect = np.sum(w_star * task_means) / np.sum(w_star)
            pooled_se = np.sqrt(1.0 / np.sum(w_star))

            results[f"{cond_a}-{cond_b}"] = {
                "method": "DerSimonian-Laird_meta_analysis",
                "fallback_reason": str(e)[:100] if isinstance(e, Exception) else "statsmodels unavailable",
                "fixed_effect": float(pooled_effect),
                "fixed_se": float(pooled_se),
                "fixed_z": float(pooled_effect / pooled_se) if pooled_se > 0 else 0.0,
                "fixed_p": float(2 * stats.norm.sf(abs(pooled_effect / pooled_se))) if pooled_se > 0 else 1.0,
                "tau2_between_task_var": float(tau2),
                "n_tasks": len(unique_tasks),
                "n_total": len(all_diffs),
            }

    return results


def run_full_analysis(
    tasks: list[str] | None = None,
    results_dir: str | Path = "results_v2/confirmatory_gains",
    topology_dir: str | Path | None = None,
    output_dir: str | Path = "results_v2/analysis",
    sesoi: float | dict | None = None,
    use_paired_loader: bool = True,
) -> dict:
    """运行完整的统计分析。

    设计文档 §12：
      - topology-level paired contrasts
      - bootstrap 95% CI、配对效应量、配对置换检验
      - Holm 多重比较校正
      - 预设 SESOI 的 equivalence/non-inferiority 分析

    v3 修复：
      - 加载 RA/RW 数据（从 topology_dir）并纳入 F0-RA, F0-RW 对比
      - 使用预设 TASK_SESOI（如 sesoi=None）
      - 支持 sesoi 为 dict（每任务不同）

    v4 修复（use_paired_loader=True 默认开启）：
      - 用 load_paired_condition_data 按 evolution_seed 严格配对
      - analyze_task_paired + 配对 TOST + 精确 sign-flip + 单调 Holm
    """
    if tasks is None:
        from experiments_v2.core.protocol import TASK_SPECS
        tasks = list(TASK_SPECS.keys())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_analyses = {}
    all_task_means = {}  # 用于跨任务分层模型（仍按 list 形式喂入）

    for task_name in tasks:
        print(f"\n[analysis] {task_name}")

        # SESOI 选择
        task_sesoi = None
        if isinstance(sesoi, dict):
            task_sesoi = sesoi.get(task_name)
        elif isinstance(sesoi, (int, float)):
            task_sesoi = sesoi
        elif sesoi is None:
            task_sesoi = TASK_SESOI.get(task_name)

        if use_paired_loader:
            # ── v4 路径：按 seed 配对 ──
            paired = load_paired_condition_data(
                task_name, results_dir, topology_dir=topology_dir
            )
            if not paired:
                print(f"  No data found, skipping.")
                continue

            analysis = analyze_task_paired(
                task_name, paired, sesoi=task_sesoi
            )

            # 同时为跨任务分层模型构造 list 形式
            task_means_list = {
                c: [v for _, v in sorted(seed_map.items())]
                for c, seed_map in paired.items()
            }
            all_task_means[task_name] = task_means_list
        else:
            # ── 旧路径（向后兼容） ──
            means = load_topology_level_means(task_name, results_dir)
            if topology_dir is not None:
                ra_rw = load_ra_rw_means(task_name, topology_dir)
                if ra_rw["RA"]:
                    means["RA"] = ra_rw["RA"]
                if ra_rw["RW"]:
                    means["RW"] = ra_rw["RW"]
            if not means:
                print(f"  No data found, skipping.")
                continue

            all_task_means[task_name] = means
            analysis = analyze_task(task_name, means, sesoi=task_sesoi)

        all_analyses[task_name] = analysis

        # 打印主要结果
        print(f"  Descriptive:")
        for cond, st in analysis["descriptive"].items():
            print(f"    {cond}: {st['mean']:.1f} ± {st['std']:.1f} "
                  f"(CI [{st['bootstrap_ci_95'][0]:.1f}, {st['bootstrap_ci_95'][1]:.1f}])")

        print(f"  Decision: topology_contribution="
              f"{analysis['decision_rules']['topology_contribution']}")

    # 跨任务分层模型（设计文档 §12）
    if len(all_task_means) >= 2:
        print(f"\n[analysis] Cross-task hierarchical model")
        hierarchical = cross_task_hierarchical_model(all_task_means)
        all_analyses["_cross_task_hierarchical"] = hierarchical
        for contrast_name, result in hierarchical.items():
            if "fixed_effect" in result:
                print(f"  {contrast_name}: pooled={result['fixed_effect']:.1f} "
                      f"± {result['fixed_se']:.1f} (p={result['fixed_p']:.4f}, "
                      f"method={result['method']})")

    # 保存
    with open(output_dir / "full_analysis.json", "w", encoding="utf-8") as f:
        json.dump(all_analyses, f, indent=2, ensure_ascii=False, default=str)

    return all_analyses


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 11 统计分析 (v4 mechanism)")
    parser.add_argument("--results-dir", type=str,
                        default="results_v2/confirmatory_gains")
    parser.add_argument("--topology-dir", type=str, default=None,
                        help="Phase 7 result 目录（RA/RW baselines）")
    parser.add_argument("--output-dir", type=str, default="results_v2/analysis")
    parser.add_argument("--sesoi", type=float, default=None,
                        help="预设的 SESOI（等价性边界）")
    parser.add_argument("--legacy-loader", action="store_true",
                        help="使用旧版列表顺序 loader（默认使用 v4 paired loader）")
    args = parser.parse_args()

    run_full_analysis(
        results_dir=args.results_dir,
        topology_dir=args.topology_dir,
        output_dir=args.output_dir,
        sesoi=args.sesoi,
        use_paired_loader=not args.legacy_loader,
    )
