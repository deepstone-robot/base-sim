"""在**我们的环境**里跑上游训练好的策略(../microduck/policies/*.onnx), 当作参照/教师。

为什么值得做(第七轮的核心思路):
  短测已经证明"奖励地形是对的, 卡住的是探索" —— 从"站住"出发, 任何单步扰动都会倒,
  而走路需要多关节协调的连续动作, iid 噪声蒙不到。绕开这个瓶颈最可靠的办法不是再调权重,
  而是**给策略一个能走的参照**。而上游恰好有一个训练好的走路策略(真机在用的那个 ONNX)。

  接口核对(读 ONNX metadata):
    obs[1,61] = base_ang_vel(3) + projected_gravity(3) + joint_pos(14) + joint_vel(14)
                + actions(14) + twist(3) + head_pose(4) + body_pose(6)
    action_scale = 1.0, default_joint_pos 与我们的站姿一致, 关节名顺序一致, 归一化已烘进图。
  我们的 env: 同一个 walk 模型 / 同样的关节位置动作语义(scale=1.0 + use_default_offset) / 50Hz。

用法:
    .venv/bin/python scripts/eval_upstream_policy.py alpha_walking.onnx --vx 0.15
    .venv/bin/python scripts/eval_upstream_policy.py alpha_stand.onnx alpha_sitstand.onnx --vx 0.15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from mjlab.envs import ManagerBasedRlEnv

from mjlab_microduck.tasks import microduck_walk_env_cfg
from mjlab_microduck.tasks.microduck_walk_cfg import (
    _FEET_CONTACT_SENSOR,
    _foot_clearance_heights,
    _posture_gate,
)

_POLICY_DIR = Path("/Users/junbo.dai/projects/microduck/policies")


def build_obs(env: ManagerBasedRlEnv, last_action: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
    """按上游 metadata 的顺序拼 61 维观测: 3+3+14+14+14+3+4+6 = 61。"""
    d = env.scene["robot"].data
    b = cmd.shape[0]
    return torch.cat(
        [
            d.root_link_ang_vel_b,                      # 3  base_ang_vel
            d.projected_gravity_b,                      # 3  projected_gravity
            d.joint_pos - d.default_joint_pos,          # 14 joint_pos(相对默认)
            d.joint_vel,                                # 14 joint_vel
            last_action,                                # 14 actions
            cmd,                                        # 3  twist
            torch.zeros(b, 4),                          # 4  head_pose 命令(零=名义头姿)
            torch.zeros(b, 6),                          # 6  body_pose 命令
        ],
        dim=-1,
    )


def onnx_actions(sess: ort.InferenceSession, obs: torch.Tensor) -> torch.Tensor:
    """上游导出的图是 batch=1, 所以逐环境喂(MLP 很小, 开销可忽略)。"""
    out = [sess.run(["actions"], {"obs": obs[i : i + 1].cpu().numpy().astype(np.float32)})[0]
           for i in range(obs.shape[0])]
    return torch.tensor(np.concatenate(out, axis=0))


def make_env(num_envs: int, vx: float, steps_length: float = 4.0, upstream_sim: bool = False,
             track_std: float | None = None) -> ManagerBasedRlEnv:
    cfg = microduck_walk_env_cfg()
    if track_std is not None:      # 标定用: 临时改跟踪核宽度
        cfg.rewards["track_lin_vel"].params["std"] = track_std
    if upstream_sim:
        # 上游 velocity 任务的仿真设置(他们没覆盖 mjlab velocity 基类的默认值):
        #   timestep 0.005 × decimation 4 = 50Hz(控制率一样), njmax 1500, iterations/ls_iterations
        from mjlab.sim import MujocoCfg, SimulationCfg
        cfg.sim = SimulationCfg(
            nconmax=35, njmax=1500, contact_sensor_maxmatch=64,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        )
        cfg.decimation = 4
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = steps_length
    cfg.events.pop("push_robot", None)
    twist = cfg.commands["twist"]
    twist.rel_standing_envs = 0.0
    twist.resampling_time_range = (1e9, 1e9)
    twist.ranges.lin_vel_x = (vx, vx)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    cfg.curriculum = {}
    return ManagerBasedRlEnv(cfg=cfg, device="cpu")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("policies", nargs="+", help="onnx 文件名(在 ../microduck/policies/ 下)或绝对路径")
    ap.add_argument("--vx", type=float, default=0.15, help="指令前进速度")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--track-std", type=float, default=None, help="临时覆盖跟踪核 std")
    ap.add_argument("--terms", action="store_true", help="打印逐项奖励")
    ap.add_argument("--upstream-sim", action="store_true",
                    help="用上游的仿真设置(timestep 0.005/decimation 4/njmax 1500/iterations 10)")
    args = ap.parse_args()

    for name in args.policies:
        path = Path(name) if Path(name).exists() else _POLICY_DIR / name
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        env = make_env(args.num_envs, args.vx, upstream_sim=args.upstream_sim,
                       track_std=args.track_std)
        robot = env.scene["robot"]
        contact = env.scene[_FEET_CONTACT_SENSOR]
        clearance_cfg = env.reward_manager.get_term_cfg("foot_clearance").params["asset_cfg"]

        last_action = torch.zeros(args.num_envs, 14)
        acc = dict(vx=0.0, vy=0.0, vz=0.0, tilt=0.0, z=0.0, gate=0.0, air=0.0, n=0, rew=0.0)
        reward_terms = list(env.reward_manager.active_terms)
        term_acc = {t: 0.0 for t in reward_terms}
        swings: list[float] = []
        peaks: list[float] = []
        causes: dict[str, int] = {}
        episodes = 0
        prev_found = None
        swing_peak = torch.zeros(args.num_envs, 2)

        for _ in range(args.steps):
            cmd = env.command_manager.get_command("twist")
            obs = build_obs(env, last_action, cmd)
            action = onnx_actions(sess, obs)  # [B,14], scale=1.0(与我们的动作语义一致)
            last_action = action.clone()
            _, rew, _, _, _ = env.step(action)   # (obs, rew, terminated, truncated, extras)
            acc["rew"] += float(rew.mean()) / env.step_dt   # 换算成"每步真实量级"(与 eval_walk 同口径)
            step_terms = env.reward_manager._step_reward.mean(dim=0)
            for i, t in enumerate(reward_terms):
                term_acc[t] += float(step_terms[i])

            found = contact.data.found > 0
            heights = _foot_clearance_heights(env, clearance_cfg)
            vel = robot.data.root_link_lin_vel_b
            g = robot.data.projected_gravity_b
            acc["vx"] += float(vel[:, 0].mean()); acc["vy"] += float(vel[:, 1].mean())
            acc["vz"] += float(vel[:, 2].abs().mean())
            acc["tilt"] += float(torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1, 1))).mean())
            acc["z"] += float(robot.data.root_link_pos_w[:, 2].mean())
            acc["gate"] += float(_posture_gate(env).mean())
            acc["air"] += float((~found).float().mean())
            acc["n"] += 1
            swing_peak = torch.where(found, swing_peak, torch.maximum(swing_peak, heights))
            if prev_found is not None:
                landing = found & (~prev_found)
                last_air = contact.data.last_air_time
                for b, f in landing.nonzero().tolist():
                    swings.append(float(last_air[b, f]))
                    peaks.append(float(swing_peak[b, f]))
                    swing_peak[b, f] = 0.0
            prev_found = found
            done = env.termination_manager.dones
            if bool(done.any()):
                episodes += int(done.sum())
                for t in env.termination_manager.active_terms:
                    v = env.termination_manager.get_term(t)
                    if v is not None:
                        causes[t] = causes.get(t, 0) + int((v & done).sum())
                swing_peak[done] = 0.0

        n = max(acc.pop("n"), 1)
        rew_per_step = acc.pop("rew") / n
        print(f"\n=== {path.name}  指令 vx={args.vx} ===")
        print(f"  vx实体 {acc['vx'] / n:+.3f} (目标 {args.vx:+.2f})   vy实体 {acc['vy'] / n:+.3f}   |vz| {acc['vz'] / n:.3f}")
        print(f"  躯干z {acc['z'] / n:.4f}   倾角 {acc['tilt'] / n:5.1f}°   姿态门 {acc['gate'] / n:.3f}")
        print(f"  有脚离地 {acc['air'] / n * 100:5.1f}%   摆动次数 {len(swings)}   "
              f"平均摆动 {np.mean(swings) * 1000 if swings else 0:5.0f}ms   峰值离地 "
              f"{np.mean(peaks) * 1000 if peaks else 0:5.1f}mm")
        if args.terms:
            print("  逐项: " + "  ".join(
                f"{k}={v / n:+.2f}" for k, v in sorted(term_acc.items(), key=lambda kv: -abs(kv[1])) if abs(v / n) > 1e-4))
        print(f"  在我们奖励下: {rew_per_step:+.3f} 分/步   (我们最好的策略: 站住 +3.43 / 溜+摔 +2.92 / 真步态估算 ≈6)")
        print(f"  回合数 {episodes}   平均存活 {args.steps * args.num_envs / max(episodes, 1):.1f} 步   "
              f"终止: {causes if causes else '-'}")


if __name__ == "__main__":
    main()
