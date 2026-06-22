"""
批量实验运行器
按论文 Section 结构组织，运行所有实验脚本并生成进度报告。

用法:
  python run_all.py                     # 运行全部
  python run_all.py --group sec1 sec2   # 只运行 sec1、sec2
  python run_all.py --dry-run           # 只显示计划，不运行
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# ── 路径设置 ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # experiments/
CODE_DIR = BASE_DIR.parent                         # code/
sys.path.insert(0, str(CODE_DIR))

GLOBAL_RESULTS = BASE_DIR / 'results'

# ── 实验分组定义 ──────────────────────────────────────────────────────────

EXPERIMENT_GROUPS = {
    'sec1': {
        'name': '存在性证据',
        'name_en': 'Existence Evidence',
        'paper_section': 'Section 3.1',
        'dir': 'sec1_existence',
        'experiments': [
            {'id': '1.1', 'script': 'exp_multitask.py',
             'name': '跨任务进化 (CartPole/MountainCar/Acrobot)', 'timeout': 7200},
            {'id': '1.2', 'script': 'exp_multiseed.py',
             'name': 'CartPole 20种子可复现性', 'timeout': 14400},
            {'id': '1.3', 'script': 'exp_multiseed_expansion.py',
             'name': '5任务×5种子扩展验证', 'timeout': 14400},
            {'id': '1.4', 'script': 'exp_motif_2node.py',
             'name': '2节点连接 motif z-score', 'timeout': 300},
            {'id': '1.5', 'script': 'exp_motif_3node.py',
             'name': '3节点 triadic motif 分析', 'timeout': 300},
            {'id': '1.6', 'script': 'exp_connection_ablation.py',
             'name': 'E→E/E→I/I→E/I→I 系统性消融', 'timeout': 1800},
            {'id': '1.7', 'script': 'exp_cross_task_fingerprint.py',
             'name': '跨任务19维结构特征分类', 'timeout': 600},
            {'id': '1.8', 'script': 'exp_baseline_ppo.py',
             'name': 'PPO 基线 (10种子)', 'timeout': 3600},
            {'id': '1.9', 'script': 'exp_baseline_neat.py',
             'name': 'NEAT 基线 (10种子)', 'timeout': 3600},
            {'id': '1.10', 'script': 'exp_baseline_comparison.py',
             'name': '三方对比汇总', 'timeout': 600},
        ],
    },
    'sec2': {
        'name': '机制分析',
        'name_en': 'Mechanism Analysis',
        'paper_section': 'Section 3.2',
        'dir': 'sec2_mechanism',
        'experiments': [
            {'id': '2.1', 'script': 'exp_tsteps_ablation.py',
             'name': 'T_steps 连续消融曲线', 'timeout': 1800},
            {'id': '2.2', 'script': 'exp_membrane_traces.py',
             'name': '膜电位轨迹可视化', 'timeout': 300},
            {'id': '2.3', 'script': 'exp_trace_erasure.py',
             'name': '中间步膜电位重置', 'timeout': 300},
            {'id': '2.4', 'script': 'exp_input_decoding.py',
             'name': 'Ridge 回归解码输入历史', 'timeout': 600},
            {'id': '2.5', 'script': 'exp_spike_timing.py',
             'name': '输出 spike 时序分析', 'timeout': 600},
        ],
    },
    'sec4': {
        'name': '环境适应',
        'name_en': 'Environmental Adaptation',
        'paper_section': 'Section 3.4',
        'dir': 'sec4_adaptation',
        'experiments': [
            {'id': '4.1', 'script': 'exp_param_boundary.py',
             'name': '重力/杆长/管道速度参数边界扫描', 'timeout': 1800},
            {'id': '4.2', 'script': 'exp_adaptation_compare.py',
             'name': '5种适应策略对比', 'timeout': 3600},
        ],
    },
    'sec5': {
        'name': 'MuJoCo 连续控制',
        'name_en': 'MuJoCo Continuous Control',
        'paper_section': 'Section 3.5',
        'dir': 'sec5_mujoco',
        'experiments': [
            {'id': '5.1', 'script': 'exp_pendulum.py',
             'name': 'Pendulum-v1 连续控制', 'timeout': 3600},
            {'id': '5.2', 'script': 'exp_hopper_evolution.py',
             'name': 'Hopper-v4 拓扑进化', 'timeout': 7200},
            {'id': '5.3', 'script': 'exp_hopper_analysis.py',
             'name': 'Hopper-v4 深度分析', 'timeout': 3600},
            {'id': '5.4', 'script': 'exp_halfcheetah_boundary.py',
             'name': 'HalfCheetah-v4 边界验证', 'timeout': 3600},
        ],
    },
    'supp': {
        'name': '补充实验',
        'name_en': 'Supplementary',
        'paper_section': 'Supplementary',
        'dir': 'supplementary',
        'experiments': [
            {'id': 'S1', 'script': 'exp_eval_robustness.py',
             'name': '评估轮数对 GA 过拟合的影响', 'timeout': 1800},
            {'id': 'S2', 'script': 'exp_selective_ablation.py',
             'name': '神经元/回路选择性消融', 'timeout': 600},
            {'id': 'S3', 'script': 'exp_angular_velocity.py',
             'name': '输入维度屏蔽测试', 'timeout': 600},
        ],
    },
}


def ensure_results_dir():
    GLOBAL_RESULTS.mkdir(parents=True, exist_ok=True)


def run_experiment(exp, group_dir):
    """运行单个实验，返回结果字典"""
    exp_id = exp['id']
    exp_name = exp['name']
    script_path = group_dir / exp['script']

    result = {
        'id': exp_id,
        'name': exp_name,
        'script': exp['script'],
        'status': 'pending',
        'start_time': None,
        'end_time': None,
        'elapsed_sec': 0,
        'error': None,
    }

    print(f"\n  [{exp_id}] {exp_name}")
    print(f"    脚本: {script_path}")

    if not script_path.exists():
        result['status'] = 'skipped'
        result['error'] = f'脚本不存在'
        print(f"    [跳过] {result['error']}")
        return result

    result['start_time'] = datetime.now().isoformat()
    t0 = time.time()
    timeout_limit = exp.get('timeout', 7200)

    try:
        proc = subprocess.Popen(
            [sys.executable, '-u', str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(group_dir),
            env={**os.environ, 'PYTHONPATH': str(CODE_DIR)},
        )

        # 实时流式输出
        last_lines = []
        for line in proc.stdout:
            print(line, end='')
            last_lines.append(line.rstrip())
            if len(last_lines) > 10:
                last_lines.pop(0)

        proc.wait(timeout=timeout_limit)
        elapsed = time.time() - t0

        if proc.returncode == 0:
            result['status'] = 'success'
            print(f"    [完成] {elapsed:.0f}s")
        else:
            result['status'] = 'failed'
            result['error'] = '\n'.join(last_lines[-5:])
            print(f"    [失败] (exit={proc.returncode})")

    except subprocess.TimeoutExpired:
        proc.kill()
        result['status'] = 'timeout'
        result['error'] = f'超时 (>{timeout_limit}s)'
        print(f"    [超时] {result['error']}")

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        print(f"    [异常] {e}")

    result['end_time'] = datetime.now().isoformat()
    result['elapsed_sec'] = time.time() - t0

    return result


def generate_report(all_results, output_path):
    """生成 Markdown 实验报告"""
    lines = []
    lines.append(f"# 实验运行报告\n")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    total = 0
    success = 0
    failed = 0
    skipped = 0
    total_time = 0

    status_icon = {
        'success': '[OK]',
        'failed': '[FAIL]',
        'timeout': '[TIMEOUT]',
        'skipped': '[SKIP]',
        'error': '[ERR]',
    }

    for gk, group_results in all_results.items():
        gi = EXPERIMENT_GROUPS[gk]
        lines.append(f"\n## Group {gk}: {gi['name']} ({gi['paper_section']})\n")
        lines.append("| 编号 | 实验 | 状态 | 耗时 | 备注 |")
        lines.append("|------|------|------|------|------|")

        for r in group_results:
            icon = status_icon.get(r['status'], '[?]')
            error_note = ''
            if r.get('error'):
                error_note = r['error'][:80].replace('\n', ' ')
            elapsed_str = f"{r['elapsed_sec']:.0f}s" if r['elapsed_sec'] > 0 else '-'
            lines.append(f"| {r['id']} | {r['name']} | {icon} | {elapsed_str} | {error_note} |")

            total += 1
            if r['status'] == 'success':
                success += 1
            elif r['status'] in ('failed', 'error', 'timeout'):
                failed += 1
            else:
                skipped += 1
            total_time += r['elapsed_sec']

    lines.append(f"\n## 汇总\n")
    lines.append(f"- 总实验数: {total}")
    lines.append(f"- 成功: {success}")
    lines.append(f"- 失败: {failed}")
    lines.append(f"- 跳过: {skipped}")
    lines.append(f"- 总耗时: {total_time:.0f}s ({total_time/60:.1f}min)")
    if total > 0:
        lines.append(f"- 成功率: {success/total*100:.1f}%")

    report_text = '\n'.join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n报告: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='批量实验运行器')
    parser.add_argument('--group', nargs='+',
                        default=list(EXPERIMENT_GROUPS.keys()),
                        help='实验组 (sec1 sec2 sec4 sec5 supp)')
    parser.add_argument('--dry-run', action='store_true',
                        help='只显示计划')
    args = parser.parse_args()

    print("=" * 60)
    print("  批量实验运行器")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 显示计划
    total_exps = 0
    for gk in args.group:
        gi = EXPERIMENT_GROUPS[gk]
        n = len(gi['experiments'])
        total_exps += n
        print(f"\n  {gi['paper_section']} {gi['name']} ({n} 个实验)")
        for exp in gi['experiments']:
            print(f"    [{exp['id']}] {exp['name']}")

    print(f"\n  共 {total_exps} 个实验")

    if args.dry_run:
        print("\n[dry-run] 不运行")
        return

    # 运行
    ensure_results_dir()
    all_results = {}
    t_start = time.time()

    for gk in args.group:
        gi = EXPERIMENT_GROUPS[gk]
        group_dir = BASE_DIR / gi['dir']

        print(f"\n{'=' * 60}")
        print(f"  {gi['paper_section']}: {gi['name']}")
        print(f"{'=' * 60}")

        group_results = []
        for exp in gi['experiments']:
            r = run_experiment(exp, group_dir)
            group_results.append(r)
        all_results[gk] = group_results

    t_total = time.time() - t_start

    # 保存报告
    report_path = GLOBAL_RESULTS / 'experiment_report.md'
    generate_report(all_results, report_path)

    # 保存 JSON
    json_path = GLOBAL_RESULTS / 'experiment_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    # 汇总
    total = sum(len(v) for v in all_results.values())
    ok = sum(1 for v in all_results.values() for r in v if r['status'] == 'success')
    fail = sum(1 for v in all_results.values() for r in v if r['status'] in ('failed', 'error', 'timeout'))
    skip = sum(1 for v in all_results.values() for r in v if r['status'] == 'skipped')

    print(f"\n{'=' * 60}")
    print(f"  完成! {t_total:.0f}s ({t_total/60:.1f}min)")
    print(f"  成功={ok}  失败={fail}  跳过={skip}  总计={total}")
    print(f"  报告: {report_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
