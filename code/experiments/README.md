# 实验代码说明

本目录按论文章节组织实验代码，对应论文 "Recurrent Connection Topology Encodes Reactive Control through Propagation Dynamics in Fixed-Weight Spiking Neural Networks" 的 Results 部分。

## 目录结构

```
code/
├── src/                          ← 核心 SNN 实现
│   ├── snn.py                    ← SNNController, WeightedSNNController
│   ├── evolution.py              ← GA 进化算法（锦标赛选择 + 精英保留 + 均匀交叉 + 位翻转变异）
│   ├── flappy_env_parametric.py  ← FlappyBird 环境（参数化）
│   └── __init__.py
├── experiments/
│   ├── sec1_existence/           ← Section 3.1: 存在性证据
│   │   ├── snake/snake_env.py    ← Snake 自定义环境
│   │   └── ...
│   ├── sec1_coevolution/         ← Section 3.1.5: 拓扑-权重共进化对照
│   ├── sec2_mechanism/           ← Section 3.2: 机制分析
│   ├── sec4_adaptation/          ← Section 3.4: 环境适应
│   ├── sec5_mujoco/              ← Section 3.5: MuJoCo 连续控制
│   ├── supplementary/            ← 补充实验
│   ├── utils/                    ← 工具脚本
│   └── results/                  ← 实验输出（运行后生成）
├── figures_arxiv/                ← 论文图片（PDF + PNG）
└── requirements.txt
```

## 快速开始

### 依赖安装

```bash
pip install numpy gymnasium scipy scikit-learn matplotlib
# 可选（仅 NEAT 基线和 MuJoCo 实验需要）:
pip install neat-python stable-baselines3 mujoco
```

### 运行实验

```bash
# 单个实验：跨任务存在性（5 任务 × 10 seed）
cd code/experiments/sec1_existence
python exp_multitask.py                           # 批量运行
python exp_multitask.py --task CartPole --seed 0  # 单次运行（适合并行）

# 单个实验：拓扑-权重共进化（TO/TG/TI 三组 × 10 seed）
cd code/experiments/sec1_coevolution
python exp_coevolution_to_tg_ti.py                # 批量运行（支持 --resume）
python run_one_seed.py --mode TO --seed 0         # 单次运行

# 从已有数据直接生成论文图表（无需重新跑实验）
cd code/experiments
python utils/reproduce_all_figures.py

# 批量运行所有实验
cd code/experiments
python utils/run_all.py
```

### 并发运行（服务器）

```bash
# 跨任务实验：50 个进程并发
cd code/experiments/sec1_existence
for task in CartPole MountainCar Acrobot FlappyBird Snake; do
  for seed in $(seq 0 9); do
    python exp_multitask.py --task $task --seed $seed &
  done
done

# 共进化实验：30 个进程并发
cd code/experiments/sec1_coevolution
for mode in TO TG TI; do
  for seed in $(seq 0 9); do
    python run_one_seed.py --mode $mode --seed $seed &
  done
done
```

## 实验文件与论文章节对应表

### sec1_existence/ — 存在性证据 (Section 3.1)

| 文件 | 论文位置 | 内容 | 对应图表 | Seeds |
|------|---------|------|---------|-------|
| `exp_multitask.py` | 3.1.1 | 5 任务跨任务进化（CartPole/MountainCar/Acrobot/FlappyBird/Snake） | Table 2, Fig 1 | 10/任务 |
| `exp_multiseed.py` | 3.1.2 | CartPole 20 种子可重复性验证 | Fig 10a | 20 |
| `exp_multiseed_expansion.py` | 3.1.2 | 5 任务 × 5 种子扩展验证 | Fig 10b-c | 5/任务 |
| `exp_motif_2node.py` | 3.1.3 | 2 节点连接 motif z-score 分析 | — | — |
| `exp_motif_3node.py` | 3.1.3 | 3 节点 triadic motif 分析 | — | — |
| `exp_connection_ablation.py` | 3.1.4 | E→E/E→I/I→E/I→I 系统性消融 | Fig 7, Table 4 | — |
| `exp_cross_task_fingerprint.py` | 3.1.8 | 跨任务结构特征分类（KNN/Mantel/t-SNE） | Fig 13 | 10/任务 |
| `exp_baseline_ppo.py` | 3.1.7 | PPO 基线 | Fig 12 | 10 |
| `exp_baseline_neat.py` | 3.1.7 | NEAT 基线 | Fig 12 | 10 |
| `exp_baseline_comparison.py` | 3.1.7 | 三方对比汇总 | Fig 12, Table 7 | — |

### sec1_coevolution/ — 拓扑-权重共进化 (Section 3.1.5)

| 文件 | 内容 | 对应图表 |
|------|------|---------|
| `exp_coevolution_to_tg_ti.py` | TO/TG/TI 三模式对照，ANOVA 统计检验 | Fig 8, Table 5 |
| `run_one_seed.py` | 单 seed 运行（支持并行） | — |

### sec2_mechanism/ — 机制分析 (Section 3.2)

| 文件 | 论文位置 | 内容 | 对应图表 |
|------|---------|------|---------|
| `exp_tsteps_ablation.py` | 3.2.1 | T_steps 连续消融曲线 (1→30) | Fig 2, Table 3 |
| `exp_membrane_traces.py` | 3.2.2 | CartPole 隐藏层膜电位轨迹可视化 | Fig 3 |
| `exp_trace_erasure.py` | 3.2.2 | 中间步膜电位重置不影响功能 | — |
| `exp_input_decoding.py` | 3.2.3 | Ridge 回归解码输入历史 (R²) | Fig 4 |
| `exp_spike_timing.py` | 3.2.4 | 历史条件调制输出 spike 时序 | Fig 6 |

### sec4_adaptation/ — 环境适应 (Section 3.4)

| 文件 | 论文位置 | 内容 | 对应图表 |
|------|---------|------|---------|
| `exp_param_boundary.py` | 3.4 | 重力/杆长/管道速度参数边界扫描 | Fig 11 |
| `exp_adaptation_compare.py` | 3.4 | 5 种适应策略对比（zero-shot/ws/T_steps/PPO/re-evolution） | Fig 9 |

### sec5_mujoco/ — MuJoCo 连续控制 (Section 3.5)

| 文件 | 论文位置 | 内容 | 对应图表 |
|------|---------|------|---------|
| `exp_pendulum.py` | 3.5 | Pendulum-v1 64 神经元连续控制 | Fig 14d |
| `exp_hopper_evolution.py` | 3.5 | Hopper-v4 拓扑进化 | Fig 14a-b |
| `exp_hopper_analysis.py` | 3.5 | Hopper-v4 深度分析（gravity perturbation 等） | Fig 14c |
| `exp_halfcheetah_boundary.py` | 3.5 | HalfCheetah-v4 边界验证 | — |

### supplementary/ — 补充实验

| 文件 | 内容 | 备注 |
|------|------|------|
| `exp_eval_robustness.py` | 评估轮数对 GA 过拟合的影响 | Section 4, Limitation 3 |
| `exp_selective_ablation.py` | 神经元/回路选择性消融 | 探索性实验 |
| `exp_angular_velocity.py` | 输入维度屏蔽测试 | 探索性实验 |

## 统一参数规范

### SNN 动力学参数（所有实验共享）

| 参数 | 值 | 说明 |
|------|-----|------|
| tau | 5.0 | 膜时间常数 |
| threshold | 0.8 | 发放阈值 |
| dt | 1.0 | 时间步长 |
| weight_scale | 0.6 | 全局权重幅值 |
| INPUT_SCALE | 4.0 | 输入缩放因子 |

### 网络拓扑参数（各任务）

| 参数 | CartPole | MountainCar | Acrobot | FlappyBird | Snake | Hopper | Pendulum |
|------|----------|-------------|---------|------------|-------|--------|----------|
| n_total | 128 | 64 | 128 | 64 | 80 | 64 | 64 |
| n_input | 8 | 4 | 12 | 8 | 16 | 22 | 6 |
| n_output | 2 | 3 | 3 | 2 | 4 | 6 | 2 |
| T_steps | 20 | 20 | 20 | 20 | 20 | 15 | 15 |
| sparsity | 0.10 | 0.12 | 0.10 | 0.10 | 0.10 | 0.15 | 0.10 |
| 抑制性比例 | 30% | 30% | 30% | 30% | 30% | 30% | 30% |

### GA 进化参数（共进化实验 & 跨任务实验）

| 参数 | 值 | 说明 |
|------|-----|------|
| pop_size | 80 | 种群规模 |
| n_elite | 16 | 精英保留（前 20%） |
| tournament_size | 3 | 锦标赛选择 |
| crossover_rate | 0.5 | 均匀交叉概率 |
| mutation_rate | 0.03 | 位翻转基础变异率 |
| stagnation_limit | 10 | 停滞代数阈值 |
| mut_rate_cap | 0.10 | 自适应变异率上限 |
| n_eval_episodes | 10-20 | 每代评估局数（固定种子） |
| solve_threshold | 任务相关 | 通关判定阈值 |

## 输出格式

### 进化实验输出

```
results/cross_task/
├── {Task}_seed{N}.npz          ← 最优拓扑 + 进化曲线（adj, types, best_hist, mean_hist）
├── result_{Task}_{N}.json      ← 指标（含 20 局严格评估）
└── summary.csv                 ← 汇总表

results/coevolution/
├── result_{MODE}_{SEED}.json   ← 单 seed 结果
└── coevolution_results.json    ← 合并结果 + ANOVA
```

### JSON 结果格式

```json
{
  "task": "CartPole",
  "seed": 0,
  "generations": 94,
  "best_fitness": 500.0,
  "solved": true,
  "n_connections": 6866,
  "elapsed_sec": 123.4,
  "strict_20ep": {
    "mean": 498.9,
    "std": 6.2,
    "min": 478.0,
    "max": 500.0
  }
}
```

## 注意事项

1. 所有实验脚本通过 `sys.path.insert(0, ...)` 引用 `code/src/`，不要修改相对路径
2. 进化实验支持断点续跑（检查输出文件是否存在，已完成的自动跳过）
3. 评估使用固定种子确保确定性：`eval_seeds = [seed * 1000 + ep for ep in range(n_eval_episodes)]`
4. 严格评估（strict evaluation）使用独立种子：`seed = 20000 + ep`
5. `sparsity` 参数是初始种群密度（10-15%），最终进化密度在 16-41%
