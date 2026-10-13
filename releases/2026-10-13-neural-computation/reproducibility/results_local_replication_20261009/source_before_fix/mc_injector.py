# -*- coding: utf-8 -*-
"""mc2026 核心注入仪器：J^HO 在线双向交换（P2）。

recipient 决定环境动作；donor shadow 在线接收同一 obs、同步 reset。
每内部步：native J^HO = (W@spikes)[-n_out:]（掩码禁止 input→output，
故天然等于 HO 分区贡献）；注入 J*_ho = (1-α)J_rec_ho + αJ_donor_ho，
以 v[-n_out:] += J*_ho - J_native_ho 加到输出膜电位（阈值前）。

α=0 / mode=none 时跳过校正行 → 与原生 forward 位级等价（QC 断言）。
不钳制 v⁻/脉冲/动作/回报（设计 §7）。

测量链：keep_trace=True 时逐步记录 j_native/j_donor/j_star/v_minus/
spikes(packbits)；每环境步记录 evidence/action/margin/tie/return。

对照模式：
  none           原生
  exact/self_exact  donor=配对 F0 在线 J（双向：rescue/destroy）
  energy_random   每内部步与 exact correction 同 L2、方向随机
  time_shuffle    donor J 轨迹按冻结块长打乱时间顺序（步内等能）
  channel_permute 只置换输出通道身份
  action_only     donor 动作标签 one-hot，逐环境步与 exact 等能
  foreign         donor=同任务另一 F0（配对冻结于 controls_mc.json）
"""
import hashlib

import numpy as np


def derive_seed(*parts):
    s = "mc2026|" + "|".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16) % (2**32)


def j_ho(ctrl, spikes):
    """hidden→output 净突触电流 (n_out,)。input→output 被掩码禁止。"""
    if not spikes.any():
        return np.zeros(ctrl.n_out)
    return (ctrl.W[-ctrl.n_out:] @ spikes).astype(np.float64, copy=False)


def _pack_spikes(spikes):
    return [int(x) for x in np.packbits(spikes.astype(np.uint8)).tolist()]


def _margin_tie(activity):
    s = np.sort(activity)[::-1]
    if len(s) >= 2:
        margin = float(s[0] - s[1])
        tie = bool(abs(s[0] - s[1]) < 1e-12)
    else:
        margin = float(s[0]); tie = False
    return margin, tie


class JExchanger:
    """一次环境步的 J 交换（donor 先行）。

    donor shadow 先跑完整 T 步，记录 J_donor 轨迹 (T, n_out)；
    recipient 后跑 T 步，按 mode 注入（exact/time_shuffle/channel_permute/
    action_only/foreign/energy_random）。recipient 决定动作。
    """

    def __init__(self, ctrl_rec, ctrl_don, alpha=1.0, mode="none",
                 subset_mask=None, rng=None, block_len=4, channel_perm=None,
                 keep_trace=False, shuffle_perm=None):
        self.rec = ctrl_rec
        self.don = ctrl_don
        self.alpha = float(alpha)
        self.mode = mode
        self.n_out = ctrl_rec.n_out
        self.subset_mask = (np.ones(self.n_out, dtype=bool)
                            if subset_mask is None
                            else np.asarray(subset_mask, dtype=bool))
        self.rng = rng if rng is not None else np.random.RandomState(0)
        self.block_len = block_len
        self.channel_perm = channel_perm
        self.shuffle_perm = shuffle_perm   # time_shuffle 块置换（T 序列）
        self.keep_trace = keep_trace
        self.trace_steps = []       # 保留接口兼容（主运行不再用）
        self.env_step_records = []  # 每环境步聚合记录
        self._randn = None
        # episode 级累积器（keep_trace）
        self._j_star_sum = 0.0
        self._j_don_sum = 0.0
        self._corr_exact_sum = 0.0
        self._corr_inj_sum = 0.0
        self._vminus_sum = 0.0
        self._n_internal = 0

    def _donor_traj(self, obs):
        """donor 完整 T 步：返回 (J_traj (T,n_out), accum_d (n_out,))。"""
        self.don._I_ext = self.don.encoder.encode(obs)
        v_d, sp_d = self.don.v, np.zeros(self.don.n)
        accum_d = np.zeros(self.don.n_out)
        T = self.don.T
        J_traj = np.zeros((T, self.n_out))
        for k in range(T):
            v_d *= self.don._leak
            if sp_d.any():
                v_d += self.don.W @ sp_d
            v_d[:self.don.n_in] += self.don._I_ext * self.don.dt / self.don.tau
            J_traj[k] = j_ho(self.don, sp_d) if sp_d.any() else 0.0
            fired_d = v_d >= self.don.threshold
            sp_d = fired_d.astype(np.float64)
            v_d[fired_d] = 0.0
            accum_d += sp_d[-self.don.n_out:]
        self.don.v = v_d
        return J_traj, accum_d

    def forward(self, obs):
        J_traj, accum_d = self._donor_traj(obs)
        T = self.rec.T
        if self.mode == "time_shuffle" and self.shuffle_perm is not None:
            J_use = J_traj[self.shuffle_perm]
        else:
            J_use = J_traj
        # recipient 推进
        self.rec._I_ext = self.rec.encoder.encode(obs)
        v, sp = self.rec.v, np.zeros(self.rec.n)
        accum = np.zeros(self.rec.n_out)
        for k in range(T):
            v *= self.rec._leak
            j_native = np.zeros(self.n_out)
            if sp.any():
                v += self.rec.W @ sp
                j_native = j_ho(self.rec, sp)
            v[:self.rec.n_in] += self.rec._I_ext * self.rec.dt / self.rec.tau
            # --- J 交换注入（α>0 且 mode != none）---
            if self.alpha > 0.0 and self.mode != "none":
                j_donor = J_use[k]
                j_exact = j_donor - j_native
                j_star = self._mix(j_native, j_donor, j_exact, k, accum_d)
                corr = (j_star - j_native)
                corr[~self.subset_mask] = 0.0
                v[-self.n_out:] += corr
            else:
                j_donor = np.zeros(self.n_out)
                j_star = np.zeros(self.n_out)
            v_minus = v[-self.n_out:].copy()
            fired = v >= self.rec.threshold
            sp = fired.astype(np.float64)
            v[fired] = 0.0
            accum += sp[-self.rec.n_out:]
            # 逐步累积标量（不存列表，避免膨胀）
            if self.keep_trace:
                self._j_star_sum += float(np.linalg.norm(j_star))
                self._j_don_sum += float(np.linalg.norm(j_donor))
                self._corr_exact_sum += float(np.linalg.norm(j_donor - j_native))
                self._corr_inj_sum += float(np.linalg.norm(j_star - j_native))
                self._vminus_sum += float(np.linalg.norm(v_minus))
                self._n_internal += 1
        action = int(np.argmax(accum))
        margin, tie = _margin_tie(accum)
        # 仅 keep_trace 时记录每环境步的轻量聚合（action/margin/tie，不含 evidence）
        if self.keep_trace:
            self.env_step_records.append({
                "action": action, "margin": float(margin), "tie": bool(tie),
            })
        self.rec.v = v
        return action, accum

    def _mix(self, j_native, j_donor, j_exact, k, accum_d):
        if self.mode in ("exact", "self_exact"):
            return (1.0 - self.alpha) * j_native + self.alpha * j_donor
        if self.mode == "energy_random":
            e = np.linalg.norm(j_exact)
            if e <= 0.0:
                return j_native
            if self._randn is None:
                self._randn = self.rng.randn(self.n_out)
            r = self._randn / np.linalg.norm(self._randn) * e
            return j_native + self.alpha * r
        if self.mode == "time_shuffle":
            # 块置换已在 forward() 中应用于 J_traj（shuffle_perm）
            return (1.0 - self.alpha) * j_native + self.alpha * j_donor
        if self.mode == "channel_permute":
            jp = j_donor[self.channel_perm]
            return (1.0 - self.alpha) * j_native + self.alpha * jp
        if self.mode == "action_only":
            # 逐步等能：注入量 = α * unit_action * ||j_donor - j_native||
            # tie 规则（协议 §8.1 v3）：平局取 argmax 最低索引（与网络读出 §2.1 一致），
            # 每步都注入——此前 v2"平局取零"使平局步（~48%）不注入、能量仅 ~52%，已废弃
            a = np.zeros(self.n_out)
            if accum_d is not None and accum_d.any():
                lab = int(np.argmax(accum_d))
                a[lab] = 1.0
            e_norm = np.linalg.norm(j_exact)
            return j_native + self.alpha * a * e_norm
        if self.mode == "action_centered":
            # centered/push-pull 动作标签：u_a = (e_a - 1/m)/||..||，逐步等能
            # tie 规则（协议 §8.1 v3）：平局取 argmax 最低索引（与网络读出一致），每步注入
            m = self.n_out
            u = np.zeros(self.n_out)
            if accum_d is not None and accum_d.any():
                lab = int(np.argmax(accum_d))
                e = np.zeros(self.n_out)
                e[lab] = 1.0
                u = e - 1.0 / m
                un = np.linalg.norm(u)
                if un > 0:
                    u = u / un
            e_norm = np.linalg.norm(j_exact)
            return j_native + self.alpha * u * e_norm
        if self.mode == "foreign":
            return (1.0 - self.alpha) * j_native + self.alpha * j_donor
        return j_native


def run_cell(rec, don, env, eps, alpha, mode, rng_seed, subset_mask=None,
             block_len=4, channel_perm=None, keep_trace=False,
             shuffle_perm=None):
    """跑一个 cell（固定 alpha/mode/subset）：eps 个 episode，返回行列表。"""
    rng = np.random.RandomState(rng_seed)
    ex = JExchanger(rec, don, alpha=alpha, mode=mode, subset_mask=subset_mask,
                    rng=rng, block_len=block_len, channel_perm=channel_perm,
                    keep_trace=keep_trace, shuffle_perm=shuffle_perm)
    rows = []
    for ep in eps:
        obs, _ = env.reset(seed=ep)
        rec.reset()
        don.reset()
        # 重置 episode 级累积器
        ex._j_star_sum = ex._j_don_sum = 0.0
        ex._corr_exact_sum = ex._corr_inj_sum = 0.0
        ex._vminus_sum = 0.0
        ex._n_internal = 0
        ex.env_step_records = []
        total = 0.0
        n_steps = 0
        for _ in range(500):
            action, accum = ex.forward(obs)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            n_steps += 1
            if term or trunc:
                break
        row = {"ep": int(ep), "return": float(total), "n_steps": n_steps}
        if keep_trace and ex._n_internal > 0:
            ni = ex._n_internal
            row["trace_agg"] = {
                "j_star_norm_mean": ex._j_star_sum / ni,
                "j_don_norm_mean": ex._j_don_sum / ni,
                "j_corr_exact_norm_mean": ex._corr_exact_sum / ni,
                "j_corr_inj_norm_mean": ex._corr_inj_sum / ni,
                "vminus_norm_mean": ex._vminus_sum / ni,
                "n_internal": ni,
            }
        rows.append(row)
    return rows
