"""Headless 评估: 加载 walk 的 checkpoint, 用**受控指令**实测策略到底会什么。

为什么需要它(而不是只看 tfevents 曲线):
  曲线里的 Episode_Reward/* 是"每步值 × (实际回合时长/20s)" —— 回合早早终止时会被压小
  十几倍; 而 air_time / 跟踪误差这类量从奖励值反推还夹着别的分量。想回答
  "它到底会不会抬脚、跟不跟得上指令", 得让策略在受控条件下跑一遍, 直接量。

用法:
    # 单个 checkpoint
    .venv/bin/python scripts/eval_walk.py logs/rsl_rl/microduck-walk/<run>/model_499.pt
    # 多个 checkpoint 横向对比(同一套指令网格)
    .venv/bin/python scripts/eval_walk.py <run>/model_0.pt <run>/model_100.pt <run>/model_499.pt

每个指令配置跑 --steps 步, 打印:
  速度跟踪误差 / 竖直乱窜 / 倾角 / 躯干高度 / 抬脚情况(摆动时长+峰值离地) / 回合终止原因。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.rl.runner import MjlabOnPolicyRunner

from mjlab_microduck.tasks import microduck_walk_env_cfg
from mjlab_microduck.tasks.microduck_walk_cfg import (
    _FEET_CONTACT_SENSOR,
    _foot_clearance_heights,
    _posture_gate,
)

TASK = "Mjlab-Velocity-Flat-MicroDuck"
# 评估用指令网格: (vx, wz)。留 vx=0 是为了看"能不能原地站住"。
COMMAND_GRID = [(0.0, 0.0), (0.15, 0.0), (0.30, 0.0), (0.15, 0.3), (0.15, -0.3)]
AIR_WINDOW = (0.125, 0.300)  # air_time 奖励窗口, 用来判断"步子迈得像不像"


def build_env(num_envs: int, vx: float, wz: float, track_std: float | None = None) -> RslRlVecEnvWrapper:
    """固定指令 + 关掉随机推的评估环境(保留终止和重置, 好统计回合结局)。"""
    cfg = microduck_walk_env_cfg()
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = 4.0          # 一次尝试最多 4s(200 步), 方便多攒几个回合
    cfg.events.pop("push_robot", None)   # 隔离变量: 不要随机扰动
    twist = cfg.commands["twist"]
    twist.rel_standing_envs = 0.0        # 不许有环境被强行置零指令
    twist.resampling_time_range = (1e9, 1e9)  # 指令全程不变
    twist.ranges.lin_vel_x = (vx, vx)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (wz, wz)
    cfg.curriculum = {}                  # 评估不跑课程
    if track_std is not None:            # 标定用: 临时改跟踪核宽度
        cfg.rewards["track_lin_vel"].params["std"] = track_std
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    return RslRlVecEnvWrapper(env)


def load_policy(venv: RslRlVecEnvWrapper, checkpoint: str):
    runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
    # 构造成员时会 print 整个网络结构, 评估时把它静音(否则输出被淹没)
    with contextlib.redirect_stdout(io.StringIO()):
        runner = runner_cls(venv, asdict(load_rl_cfg(TASK)), device="cpu")
        runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location="cpu")
        policy = runner.get_inference_policy(device="cpu")
    return policy


def rollout(venv: RslRlVecEnvWrapper, policy, steps: int, vx: float, wz: float) -> dict:
    env = venv.unwrapped
    robot = env.scene["robot"]
    contact = env.scene[_FEET_CONTACT_SENSOR]
    clearance_cfg = env.reward_manager.get_term_cfg("foot_clearance").params["asset_cfg"]
    term_names = list(env.termination_manager.active_terms)

    obs = venv.get_observations()
    prev_found = None
    swing_peak = torch.zeros(env.num_envs, 2)
    acc = {k: 0.0 for k in (
        "vx_err", "vy_abs", "vz_abs", "tilt", "trunk_z", "foot_air_frac",
        "in_window_frac", "cleared_frac", "n_steps", "gate", "vx_body", "vy_body", "wz_body",
    )}
    reward_terms = list(env.reward_manager.active_terms)
    term_acc = {n: 0.0 for n in reward_terms}
    landings: list[float] = []      # 每次落脚的"上一段摆动时长"
    peaks: list[float] = []         # 每次落脚的"摆动期峰值离地高度"
    causes = {n: 0 for n in term_names}
    episodes = 0

    for _ in range(steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, _, dones, _ = venv.step(actions)

        found = (contact.data.found > 0)                       # [B, 2] 接触
        air = contact.data.current_air_time                     # [B, 2] 已离地时长
        heights = _foot_clearance_heights(env, clearance_cfg)   # [B, 2] 脚底最低点
        vel_b = robot.data.root_link_lin_vel_b
        g = robot.data.projected_gravity_b

        acc["vx_err"] += float((vel_b[:, 0] - vx).abs().mean())
        acc["vy_abs"] += float(vel_b[:, 1].abs().mean())
        acc["vz_abs"] += float(vel_b[:, 2].abs().mean())
        acc["tilt"] += float(torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1, 1))).mean())
        acc["trunk_z"] += float(robot.data.root_link_pos_w[:, 2].mean())
        acc["foot_air_frac"] += float((~found).float().mean())
        acc["in_window_frac"] += float(((air > AIR_WINDOW[0]) & (air < AIR_WINDOW[1])).float().mean())
        acc["cleared_frac"] += float((heights > 0.01).float().mean())
        acc["gate"] += float(_posture_gate(env).mean())  # 1=还靠腿站着, 0=躺/趴
        acc["vx_body"] += float(vel_b[:, 0].mean())        # 有符号: 正=往指令的 +x 走
        acc["vy_body"] += float(vel_b[:, 1].mean())        # 有符号: 非零=横着走
        acc["wz_body"] += float(robot.data.root_link_ang_vel_b[:, 2].mean())
        acc["n_steps"] += 1
        # mjlab 在每个 compute() 里把"每步加权值(未乘 dt)"写进 _step_reward:
        # env 返回给 RL 的奖励 = Σ 这些值 × dt, 所以这里就是"每步真实量级"
        step_terms = env.reward_manager._step_reward.mean(dim=0)
        for i, n in enumerate(reward_terms):
            term_acc[n] += float(step_terms[i])

        swing_peak = torch.where(found, swing_peak, torch.maximum(swing_peak, heights))
        if prev_found is not None:
            landing = found & (~prev_found)                     # 刚落脚
            if landing.any():
                for b, f in landing.nonzero().tolist():
                    if air[b, f] >= 0:  # 落地瞬间 current_air_time 已清零, 用 last_air_time
                        pass
                last = contact.data.last_air_time
                for b, f in landing.nonzero().tolist():
                    landings.append(float(last[b, f]))
                    peaks.append(float(swing_peak[b, f]))
                    swing_peak[b, f] = 0.0
        prev_found = found

        if dones.any():
            episodes += int(dones.sum())
            for n in term_names:
                t = env.termination_manager.get_term(n)
                if t is not None:
                    causes[n] += int((t & dones).sum())
            swing_peak[dones] = 0.0

    n = max(acc["n_steps"], 1)
    out = {k: v / n for k, v in acc.items() if k != "n_steps"}
    out["episodes"] = episodes
    out["steps_per_episode"] = steps * env.num_envs / max(episodes, 1)
    out["landings"] = len(landings)
    out["mean_swing_s"] = sum(landings) / len(landings) if landings else 0.0
    out["in_window_pct"] = (sum(1 for x in landings if AIR_WINDOW[0] < x < AIR_WINDOW[1])
                            / len(landings) * 100 if landings else 0.0)
    out["peak_clearance_mm"] = (sum(peaks) / len(peaks) * 1000) if peaks else 0.0
    out["causes"] = causes
    out["terms"] = {k: v / n for k, v in term_acc.items()}
    out["total"] = sum(out["terms"].values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoints", nargs="+", help="一个或多个 model_*.pt")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400, help="每个指令配置跑多少步")
    ap.add_argument("--max-configs", type=int, default=len(COMMAND_GRID),
                    help="只跑指令网格的前 N 个(快速对比用)")
    ap.add_argument("--track-std", type=float, default=None,
                    help="临时覆盖跟踪核 std(标定用)")
    ap.add_argument("--vx", type=float, default=None,
                    help="只测这个前进速度(默认跑整个指令网格)")
    ap.add_argument("--terms", action="store_true",
                    help="额外打印逐项奖励(每步真实量级)+ 合计 —— 用来给旧策略在**新奖励**下重新打分")
    args = ap.parse_args()

    for ckpt in args.checkpoints:
        name = Path(ckpt).name
        print(f"\n{'=' * 100}\n{name}\n{'=' * 100}")
        print(f"{'指令(vx,wz)':>14} {'vx实体':>7} {'vy实体':>7} {'vx误差':>7} {'|vz|':>6} {'倾角':>6} {'躯干z':>7} "
              f"{'姿态门':>7} {'离地%':>6} {'摆动s':>6} {'峰值mm':>7} "
              f"{'步/回合':>8} {'回合':>5}  终止原因")
        rows = []
        grid = [(args.vx, 0.0)] if args.vx is not None else COMMAND_GRID[: args.max_configs]
        for vx, wz in grid:
            venv = build_env(args.num_envs, vx, wz, track_std=args.track_std)
            policy = load_policy(venv, ckpt)
            r = rollout(venv, policy, args.steps, vx, wz)
            causes = " ".join(f"{k}={v}" for k, v in r["causes"].items() if v > 0) or "-"
            print(f"{f'({vx:+.2f},{wz:+.2f})':>14} {r['vx_body']:>7.3f} {r['vy_body']:>7.3f} "
                  f"{r['vx_err']:>7.3f} {r['vz_abs']:>6.3f} {r['tilt']:>5.1f}° {r['trunk_z']:>7.4f} {r['gate']:>7.3f} "
                  f"{r['foot_air_frac'] * 100:>5.1f}% "
                  f"{r['mean_swing_s']:>6.3f} {r['peak_clearance_mm']:>7.1f} "
                  f"{r['steps_per_episode']:>8.1f} {r['episodes']:>5d}  {causes}")
            if args.terms:
                t = r["terms"]
                pos = sum(v for k, v in t.items() if v > 0)
                neg = sum(v for k, v in t.items() if v <= 0)
                print(f"{'':>14} 逐项: 合计 {r['total']:+.3f} | 正 {pos:+.3f} | 负 {neg:+.3f}")
                print(f"{'':>14}   " + "  ".join(
                    f"{k}={v:+.2f}" for k, v in sorted(t.items(), key=lambda kv: -abs(kv[1]))
                    if abs(v) > 1e-4))
                rows.append((vx, wz, r))
        if args.terms and rows:
            def avg(key: str) -> float:
                return sum(r["terms"][key] for _, _, r in rows) / len(rows)

            tot = sum(r["total"] for _, _, r in rows) / len(rows)
            gate = sum(r["gate"] for _, _, r in rows) / len(rows)
            air = sum(r["foot_air_frac"] for _, _, r in rows) / len(rows)
            sw = sum(r["mean_swing_s"] for _, _, r in rows) / len(rows)
            print(f"{'>>> 全网格平均':>14} 合计 {tot:+.3f} | track_lin {avg('track_lin_vel'):+.3f} "
                  f"air_time {avg('air_time'):+.3f} upright {avg('upright'):+.3f} | "
                  f"action_rate {avg('action_rate'):+.3f} | 姿态门 {gate:.3f} 离地 {air * 100:.1f}% "
                  f"摆动 {sw * 1000:.0f}ms")
    print(
        "\n列怎么读:\n"
        "  vx/vy实体= 实际机体系速度的有符号均值: vx 该接近指令值, vy 该接近 0(非零=横着走)\n"
        "  vx误差   = |实际机体系 vx - 指令| 的平均值(越小越跟得上; 站着不动时≈|指令|)\n"
        "  姿态门   = 「还靠腿站着吗」的软门(躯干高度 × 直立程度, 见 _posture_gate): 1=站/走, 0=躺/趴。\n"
        "             这个值 <0.5 说明它在爬/躺 —— 那种姿势下技能类奖励会被乘成 0, 学不到东西\n"
        "  离地%    = 有脚离地的步数占比;  窗口% = current_air_time 落在奖励窗口的占比\n"
        "  摆动s    = 每次摆动(离地到落地)的平均时长;  落地窗口% = 摆动时长落在窗口内的比例\n"
        "  峰值mm   = 每次摆动中脚底最低点的峰值高度(这才是'抬脚抬多高')\n"
        "  步/回合  = 平均活了多少步(50Hz), 停在哪个终止条件上(见末列)"
    )


if __name__ == "__main__":
    main()
