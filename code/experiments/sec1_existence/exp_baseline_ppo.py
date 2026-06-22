"""
实验 D1: MLP+PPO 独立基线
目标：建立标准深度强化学习的性能天花板，用于与结构动力智能对比。

环境：gymnasium.CartPole-v1
方法：PPO with MLP policy [128, 128]
评估：10 个随机种子，每种子训练 200k timesteps
"""
import os
import sys
import csv
import json
import time
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 依赖检查 ──────────────────────────────────────────────────────
try:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback
    import gymnasium as gym
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请安装: pip install stable-baselines3 gymnasium")
    sys.exit(1)

# ── 配置 ──────────────────────────────────────────────────────────
PPO_CONFIG = {
    'policy': 'MlpPolicy',
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'verbose': 0,
    'policy_kwargs': dict(net_arch=[128, 128])
}

N_SEEDS = 10
TOTAL_TIMESTEPS = 200_000
EVAL_FREQ = 5_000   # global timesteps between evaluations (n_envs=1, so this is exact)
N_EVAL_EPISODES = 20
EVAL_EPISODES_CALLBACK = 5  # episodes for periodic callback eval (lighter than final)
ENV_NAME = 'CartPole-v1'
N_ENVS = 1  # use 1 env so eval_freq maps 1:1 to global steps
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'ppo_baseline')

# 中文支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ── 辅助函数 ──────────────────────────────────────────────────────

def count_mlp_params(input_dim, hidden_dims, output_dim):
    """计算 MLP 总参数量（含偏置）"""
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    total = 0
    for i in range(len(dims) - 1):
        total += dims[i] * dims[i + 1]      # 权重
        total += dims[i + 1]                 # 偏置
    return total


def evaluate_policy_manual(model, env_name='CartPole-v1', n_episodes=20, seed=0):
    """
    用训练好的模型跑 n_episodes 严格评估。
    每局最多 500 步，返回每局步数列表。
    """
    env = gym.make(env_name)
    episode_lengths = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        steps = 0
        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        episode_lengths.append(steps)
    
    env.close()
    return np.array(episode_lengths)


def train_ppo_single_seed(seed, total_timesteps=200_000):
    """
    训练单个种子的 PPO，返回训练历史和最终评估结果字典。
    """
    print(f"\n[Seed {seed + 1}/{N_SEEDS}] Training PPO on {ENV_NAME}...")
    t0 = time.time()
    
    # 创建向量化训练环境
    vec_env = make_vec_env(ENV_NAME, n_envs=N_ENVS, seed=seed)
    
    # 创建独立评估环境（用于 EvalCallback）
    eval_env = make_vec_env(ENV_NAME, n_envs=1, seed=seed + 1000)
    
    # 构建模型
    model = PPO(env=vec_env, seed=seed, device='cpu', **PPO_CONFIG)
    
    # 获取实际模型参数量（仅在 seed==0 时打印，避免重复）
    total_params = sum(p.numel() for p in model.policy.parameters())
    
    # EvalCallback：每 EVAL_FREQ global timesteps 评估一次
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None,
        log_path=None,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES_CALLBACK,
        deterministic=True,
        render=False,
        verbose=0,
    )
    
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    
    # 提取训练曲线
    training_curve = []
    if hasattr(eval_callback, 'evaluations_timesteps') and eval_callback.evaluations_timesteps:
        for t, r in zip(eval_callback.evaluations_timesteps, eval_callback.evaluations_results):
            # r 是 n_eval_episodes 次的 episode reward，取均值
            training_curve.append((int(t), float(np.mean(r))))
    
    # 训练结束后严格评估 20 局
    final_lengths = evaluate_policy_manual(model, env_name=ENV_NAME, n_episodes=N_EVAL_EPISODES, seed=seed + 2000)
    final_mean = float(np.mean(final_lengths))
    final_std = float(np.std(final_lengths))
    final_min = int(np.min(final_lengths))
    final_max = int(np.max(final_lengths))
    
    # 计算达到 450 步所需的 timesteps
    steps_to_450 = None
    for t, r in training_curve:
        if r >= 450:
            steps_to_450 = int(t)
            break
    if steps_to_450 is None:
        # 若训练曲线中未记录到，检查最终 20 局评估
        if final_mean >= 450:
            steps_to_450 = total_timesteps
        else:
            steps_to_450 = f">{total_timesteps}"
    
    elapsed = time.time() - t0
    
    print(f"[Seed {seed + 1}/{N_SEEDS}] Final eval: {final_mean:.1f} ± {final_std:.1f} (min={final_min}, max={final_max})")
    print(f"[Seed {seed + 1}/{N_SEEDS}] Steps to 450: {steps_to_450}")
    
    vec_env.close()
    eval_env.close()
    
    return {
        'seed': seed,
        'training_curve': training_curve,
        'final_lengths': final_lengths,
        'final_mean': final_mean,
        'final_std': final_std,
        'final_min': final_min,
        'final_max': final_max,
        'steps_to_450': steps_to_450,
        'elapsed_sec': elapsed,
        'total_params': total_params,
    }


# ── 可视化与保存 ──────────────────────────────────────────────────

def save_results(all_results, all_curves):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 统一使用第一个种子的 total_params（所有种子相同）
    total_params = all_results[0]['total_params']
    
    # ── JSON (raw data, including training curves) ──────────
    json_path = os.path.join(RESULTS_DIR, 'ppo_training_curves.json')
    json_data = []
    for r, curve in zip(all_results, all_curves):
        json_data.append({
            'seed': r['seed'],
            'training_curve': curve,
            'final_mean': r['final_mean'],
            'final_std': r['final_std'],
            'steps_to_450': r['steps_to_450'],
        })
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON saved: {json_path}")

    # ── CSV ───────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, 'ppo_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'final_mean', 'final_std', 'min', 'max', 'steps_to_450', 'total_params'])
        for r in all_results:
            writer.writerow([
                r['seed'],
                f"{r['final_mean']:.2f}",
                f"{r['final_std']:.2f}",
                r['final_min'],
                r['final_max'],
                r['steps_to_450'],
                total_params,
            ])
    print(f"\nCSV saved: {csv_path}")
    
    # ── 训练曲线图 ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for r, curve in zip(all_results, all_curves):
        if curve:
            ts, rs = zip(*curve)
            ax.plot(ts, rs, alpha=0.7, lw=1.5, label=f"Seed {r['seed']}")
    ax.axhline(450, color='red', linestyle='--', alpha=0.5, label='目标 450')
    ax.axhline(500, color='green', linestyle='--', alpha=0.5, label='满分 500')
    ax.set_xlabel('训练步数 (timesteps)')
    ax.set_ylabel('评估平均回报')
    ax.set_title('PPO 基线训练曲线 (CartPole-v1, 10 seeds)')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig_path_curve = os.path.join(RESULTS_DIR, 'ppo_training_curves.png')
    plt.savefig(fig_path_curve, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {fig_path_curve}")
    
    # ── 最终评估箱线图 ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    all_lengths = [r['final_lengths'] for r in all_results]
    bp = ax.boxplot(all_lengths, labels=[f"S{r['seed']}" for r in all_results], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    ax.axhline(450, color='red', linestyle='--', alpha=0.5, label='目标 450')
    ax.axhline(500, color='green', linestyle='--', alpha=0.5, label='满分 500')
    ax.set_xlabel('随机种子')
    ax.set_ylabel('单局步数')
    ax.set_title('PPO 基线最终评估 (每种子 20 局)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig_path_box = os.path.join(RESULTS_DIR, 'ppo_final_eval.png')
    plt.savefig(fig_path_box, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {fig_path_box}")
    
    # ── 文本摘要 ──────────────────────────────────────────────
    final_means = [r['final_mean'] for r in all_results]
    final_stds_seed = [r['final_std'] for r in all_results]
    steps_450_vals = [r['steps_to_450'] for r in all_results if isinstance(r['steps_to_450'], int)]
    
    summary_lines = [
        "=" * 70,
        "                     PPO 基线实验摘要",
        "=" * 70,
        f"环境         : {ENV_NAME}",
        f"网络结构     : MLP [128, 128]",
        f"总参数量     : {total_params}",
        f"种子数       : {N_SEEDS}",
        f"训练步数     : {TOTAL_TIMESTEPS}",
        "",
        f"最终 20 局性能 (10 种子均值 ± std): {np.mean(final_means):.1f} ± {np.std(final_means):.1f}",
        f"最终性能范围 : {np.min(final_means):.1f} / {np.max(final_means):.1f}",
        f"训练稳定性 (10 种子最终性能方差) : {np.var(final_means):.2f}",
        "",
    ]
    
    if steps_450_vals:
        summary_lines.append(f"达到 450 步所需 timesteps 中位数 : {int(np.median(steps_450_vals))}")
        summary_lines.append(f"达到 450 步范围 : {np.min(steps_450_vals)} / {np.max(steps_450_vals)}")
    else:
        summary_lines.append("达到 450 步所需 timesteps 中位数 : N/A (均未达到)")
    
    summary_lines.extend([
        "",
        "─" * 70,
        "每种子详细结果:",
        f"{'Seed':>5} {'FinalMean':>10} {'FinalStd':>9} {'Min':>5} {'Max':>5} {'StepsTo450':>12}",
        "-" * 60,
    ])
    for r in all_results:
        st = str(r['steps_to_450'])
        summary_lines.append(
            f"{r['seed']:5d} {r['final_mean']:10.1f} {r['final_std']:9.1f} "
            f"{r['final_min']:5d} {r['final_max']:5d} {st:>12}"
        )
    summary_lines.append("=" * 70)
    
    summary_text = "\n".join(summary_lines)
    txt_path = os.path.join(RESULTS_DIR, 'ppo_results_summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"Summary saved: {txt_path}")
    
    # 控制台输出
    print("\n" + summary_text)


# ── 主实验 ────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  实验 D1: MLP+PPO 独立基线")
    print("=" * 70)
    print(f"环境: {ENV_NAME}")
    print(f"网络: MLP [128, 128]")
    print(f"种子: 0–{N_SEEDS - 1}")
    print(f"训练: {TOTAL_TIMESTEPS} timesteps")
    print("=" * 70)
    
    all_results = []
    all_training_curves = []
    
    for seed in range(N_SEEDS):
        result = train_ppo_single_seed(seed, total_timesteps=TOTAL_TIMESTEPS)
        all_results.append(result)
        all_training_curves.append(result['training_curve'])
    
    # 汇总统计与保存
    save_results(all_results, all_training_curves)
    
    print("\nDone.")


if __name__ == '__main__':
    main()
