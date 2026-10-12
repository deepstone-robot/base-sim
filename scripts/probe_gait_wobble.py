"""量"摇头晃脑": 走起来以后躯干/偏航方向的周期性摆动到底有多大、多快。

只用一次前向 rollout 的时间序列, 不训练。想回答三件事:

1. 摆动幅度 —— 偏航角速度 RMS、横滚角速度 RMS、侧向速度 RMS、横滚/偏航角峰峰值。
   (判断"晃"是不是真的偏离了"稳定行走"这个量级)
2. 摆动频率 —— 对偏航角速度做 FFT 取主频, 再和落脚频率(步频)比。
   主频 ≈ 步频 -> 是步态自带的左右重心转移(正常);
   主频 ≈ 步频/2 或其它 -> 是极限环/共振, 属于控制问题。
3. 摆动与"要倒"的关系 —— 峰值倾斜角、倾斜角是否随时间发散。

用法:
    python scripts/probe_gait_wobble.py model_3250.pt --vx 0.15 --steps 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_walk import _FEET_CONTACT_SENSOR, build_env, load_policy  # noqa: E402

DT = 0.02  # 控制周期, 50 Hz
_UPSTREAM_POLICY_DIR = Path("/Users/junbo.dai/projects/microduck/policies")


def load_onnx_policy(path: Path, env, vx: float, wz: float):
    """把上游导出的 onnx 包成和我们的策略一样的 callable(喂 61 维上游观测)。

    对照组用: 上游 alpha_walking 是"已证明能走"的策略, 拿它的晃动幅度当参照,
    才能判断我们的 ±13° 是这只机器人的天生步态, 还是我们没压住。
    """
    import numpy as np
    import onnxruntime as ort

    from eval_upstream_policy import build_obs, onnx_actions

    if not path.exists():
        path = _UPSTREAM_POLICY_DIR / path
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    state = {"last_action": torch.zeros(env.num_envs, 14)}
    cmd = torch.tensor([[vx, 0.0, wz]] * env.num_envs)

    def policy(obs):  # noqa: ARG001  onnx 用的是自己拼的 61 维观测
        full = build_obs(env, state["last_action"], cmd)
        act = onnx_actions(sess, full)
        state["last_action"] = act
        return act

    return policy


def quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    """四元数 (w,x,y,z) -> 世界系偏航角。"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def dominant_freq(series: torch.Tensor, dt: float) -> float:
    """对 [T] 的一维序列(已去均值)做 FFT, 返回主频 (Hz)。"""
    n = series.numel()
    if n < 8:
        return 0.0
    s = series - series.mean()
    window = torch.hann_window(n)
    spec = torch.fft.rfft(s * window).abs()
    freqs = torch.fft.rfftfreq(n, d=dt)
    spec[0] = 0.0                     # 去直流
    if spec.numel() < 2 or spec[1:].max() <= 0:
        return 0.0
    k = int(spec[1:].argmax()) + 1    # 跳过直流
    return float(freqs[k])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", type=str, help="策略 checkpoint 路径")
    ap.add_argument("--vx", type=float, default=0.15, help="指令前进速度 (m/s)")
    ap.add_argument("--wz", type=float, default=0.0, help="指令偏航角速度 (rad/s)")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=500, help="rollout 步数 (500 步 = 10 s)")
    ap.add_argument("--freeze-head", action="store_true",
                    help="把头/脖子 4 个关节的动作通道置零(= 位置目标锁在名义姿势), 做零成本 A/B")
    args = ap.parse_args()

    # 回合长度必须在构造前给: 否则序列会被"每 4 s 全体重置"打断
    venv = build_env(args.num_envs, args.vx, args.wz,
                     episode_length_s=(args.steps + 5) * DT)
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.suffix == ".onnx" or (not ckpt_path.exists()
                                       and (_UPSTREAM_POLICY_DIR / ckpt_path).exists()):
        policy = load_onnx_policy(ckpt_path, venv.unwrapped, args.vx, args.wz)
    else:
        policy = load_policy(venv, args.checkpoint)
    env = venv.unwrapped
    robot = env.scene["robot"]
    contact = env.scene[_FEET_CONTACT_SENSOR]

    # 头/脖子 4 个关节在动作向量里的下标: 实测(单点脉冲法)=关节下标, 即 5,6,7,8
    head_ids = [robot.joint_names.index(n) for n in ("neck_pitch", "head_pitch", "head_yaw", "head_roll")]
    if args.freeze_head:
        print(f"  [冻头] 动作通道 {head_ids} 置零(位置目标 = 名义姿势)")
    obs = venv.get_observations()
    log: dict[str, list[torch.Tensor]] = {
        k: [] for k in ("wz", "wy", "wx", "vy", "tilt", "yaw", "found", "pos", "vel_w")
    }
    for _ in range(args.steps):
        with torch.no_grad():
            actions = policy(obs)
        if args.freeze_head:
            actions = actions.clone()
            actions[:, head_ids] = 0.0
        obs, _, dones, _ = venv.step(actions)
        w = robot.data.root_link_ang_vel_b
        v = robot.data.root_link_lin_vel_b
        g = robot.data.projected_gravity_b
        log["wz"].append(w[:, 2].clone())
        log["wy"].append(w[:, 1].clone())
        log["wx"].append(w[:, 0].clone())
        log["vy"].append(v[:, 1].clone())
        log["tilt"].append(torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1, 1))).clone())
        log["yaw"].append(quat_to_yaw(robot.data.root_link_quat_w).clone())
        log["found"].append((contact.data.found > 0).clone())
        log["pos"].append(robot.data.root_link_pos_w[:, :2].clone())   # 世界系 xy
        log["vel_w"].append(robot.data.root_link_lin_vel_w[:, :2].clone())
        if dones.any():
            print(f"  [warn] 第 {len(log['wz'])} 步有 {int(dones.sum())} 个环境终止, 时间序列会被重置打断")

    T = torch.stack(log["wz"])              # [T, B]
    yaw = torch.stack(log["yaw"])           # [T, B]
    tilt = torch.stack(log["tilt"])         # [T, B]
    vy = torch.stack(log["vy"])
    found = torch.stack(log["found"])       # [T, B, 2]
    n = T.shape[0]
    dur = n * DT

    # 步频: 每只脚"由支撑转摆动"的次数 / (2 * 时长) -> 单腿一个完整周期的频率
    lift = (found[:-1] & ~found[1:]).float().sum(dim=0).sum(dim=1)  # [B] 总抬脚次数
    step_freq = (lift / dur / 2.0)          # 两条腿交替, 除以 2 = 步频(Hz)

    # 偏航角速度主频(去均值后)
    wz_freq = torch.tensor([dominant_freq(T[:, b], DT) for b in range(T.shape[1])])

    def rms(x: torch.Tensor) -> torch.Tensor:
        return x.pow(2).mean(dim=0).sqrt()   # [B]

    src = "上游 onnx(对照组)" if ckpt_path.suffix == ".onnx" else "本仓库 checkpoint"
    if args.freeze_head:
        src += " / 冻头"
    print(f"\n=== 摇头晃脑量化: {ckpt_path.name} [{src}]  指令 vx={args.vx} wz={args.wz} ===")
    print(f"时长 {dur:.1f} s ({n} 步), 环境数 {T.shape[1]}\n")
    print(f"{'指标':<26}{'均值':>10}{'最大':>10}{'单位':>8}")
    rows = [
        ("偏航角速度 RMS |wz|", rms(T), "rad/s"),
        ("横滚角速度 RMS |wx|", rms(torch.stack(log["wx"])), "rad/s"),
        ("俯仰角速度 RMS |wy|", rms(torch.stack(log["wy"])), "rad/s"),
        ("侧向速度 RMS |vy|", rms(vy), "m/s"),
    ]
    for name, val, unit in rows:
        print(f"{name:<26}{float(val.mean()):>10.4f}{float(val.max()):>10.4f}{unit:>8}")

    # 姿态角: 用倾斜角的时间序列算峰峰值和标准差
    tilt_pp = (tilt.max(dim=0).values - tilt.min(dim=0).values)
    print(f"{'倾斜角 均值':<24}{float(tilt.mean()):>10.3f}{float(tilt.mean(dim=0).max()):>10.3f}{'deg':>8}")
    print(f"{'倾斜角 峰峰值':<24}{float(tilt_pp.mean()):>10.3f}{float(tilt_pp.max()):>10.3f}{'deg':>8}")
    print(f"{'倾斜角 标准差':<24}{float(tilt.std(dim=0).mean()):>10.3f}{float(tilt.std(dim=0).max()):>10.3f}{'deg':>8}")
    # 偏航角漂移(不是摆动, 是"走歪"): 首末之差
    # 偏航角会绕着 ±180° 穿, 必须解缠再统计, 否则峰峰值会被算成 2pi 或莫名很小
    yaw_unwrapped = torch.rad2deg(torch.cumsum(
        torch.cat([yaw[:1], torch.atan2(torch.sin(yaw[1:] - yaw[:-1]),
                                        torch.cos(yaw[1:] - yaw[:-1]))]), dim=0))
    yaw_pp = yaw_unwrapped.max(dim=0).values - yaw_unwrapped.min(dim=0).values
    yaw_drift = (yaw_unwrapped[-1] - yaw_unwrapped[0]).abs()
    print(f"{'偏航角 峰峰值(摆动+走歪)':<21}{float(yaw_pp.mean()):>10.3f}{float(yaw_pp.max()):>10.3f}{'deg':>8}")
    print(f"{'偏航角 漂移|末-初|':<23}{float(yaw_drift.mean()):>10.3f}{float(yaw_drift.max()):>10.3f}{'deg':>8}")
    # 把"走歪"(线性趋势)从"摆动"(残差)里剥出来: 否则两者混在一起看不出谁是问题
    t = torch.arange(n, dtype=yaw_unwrapped.dtype).unsqueeze(1)          # [T,1]
    t_c = t - t.mean()
    slope = (t_c * (yaw_unwrapped - yaw_unwrapped.mean(dim=0))).sum(dim=0) / t_c.pow(2).sum()
    resid = yaw_unwrapped - (yaw_unwrapped.mean(dim=0) + slope * t_c)
    signed_drift = yaw_unwrapped[-1] - yaw_unwrapped[0]
    same_way = float((signed_drift.sign() == signed_drift.sign().mode().values).float().mean())
    print(f"{'走歪(=趋势) 速率':<22}{float(slope.mean()):>10.3f}{float(slope.abs().max()):>10.3f}{'deg/s':>8}")
    print(f"{'  -> 转弯半径 |v/w|':<22}"
          f"{float((abs(args.vx) / slope.abs().clamp(min=1e-6) * torch.pi / 180).mean()):>10.2f}"
          f"{float((abs(args.vx) / slope.abs().clamp(min=1e-6) * torch.pi / 180).max()):>10.2f}{'m':>8}")
    print(f"{'  -> 同向比例':<24}{same_way * 100:>9.0f}%{'':>10}"
          "  (100% = 所有环境都往同一侧拐, 50% = 每局随机挑一侧 -> 对称性被打破)")
    print(f"{'摆动(去趋势残差) 峰峰值':<19}"
          f"{float((resid.max(dim=0).values - resid.min(dim=0).values).mean()):>10.3f}"
          f"{float((resid.max(dim=0).values - resid.min(dim=0).values).max()):>10.3f}{'deg':>8}")
    print(f"{'  -> 半幅(±N)':<23}"
          f"{float((resid.max(dim=0).values - resid.min(dim=0).values).mean() / 2):>10.3f}"
          f"{float((resid.max(dim=0).values - resid.min(dim=0).values).max() / 2):>10.3f}{'deg':>8}")

    print(f"\n{'摆动频率':<26}{'均值':>10}{'最大':>10}{'单位':>8}")
    print(f"{'偏航角速度 主频':<24}{float(wz_freq.mean()):>10.3f}{float(wz_freq.max()):>10.3f}{'Hz':>8}")
    print(f"{'落脚步频':<26}{float(step_freq.mean()):>10.3f}{float(step_freq.max()):>10.3f}{'Hz':>8}")
    ratio = wz_freq.mean() / step_freq.mean().clamp(min=1e-6)
    print(f"{'主频 / 步频':<26}{float(ratio):>10.3f}{'':>10}")

    # 原始序列自检: RMS 和"角度峰峰值"如果不自洽, 说明摆动里混着高频抖动
    print("\n=== 原始序列自检 (env 0, 前 24 步 wz, rad/s) ===")
    print("  " + " ".join(f"{float(x):+.2f}" for x in T[:24, 0]))
    flips = ((T[:-1, 0] * T[1:, 0]) < 0).float().mean()
    print(f"  相邻步符号翻转比例 {float(flips):.2f}"
          "  (≈0.5 = 随机抖动/每步换向; ≈0.1 = 平滑的低频摆动)")
    # 用积分反推: 若 wz 真是频率 f 的正弦, 偏航角峰峰值应该是 2*A/(2*pi*f)
    amp = rms(T).mean() * (2 ** 0.5)
    implied_pp = torch.rad2deg(2 * amp / (2 * torch.pi * wz_freq.mean()))  # 正弦假设下的峰峰值
    resid_pp = (resid.max(dim=0).values - resid.min(dim=0).values).mean()
    print(f"  由 |wz| RMS 反推的偏航角峰峰值 {float(implied_pp):.2f} deg"
          f"  vs 去趋势残差峰峰值 {float(resid_pp):.2f} deg"
          f"  -> 比值 {float(implied_pp / resid_pp.clamp(min=1e-6)):.2f}x"
          "  (≈1 = 摆动就是这条 wz 积出来的, 自洽)")

    # ---- 路径几何: 躯干扭动 ≠ 路径拐弯, 这两件事必须分开 ----
    pos = torch.stack(log["pos"])            # [T, B, 2]
    net = (pos[-1] - pos[0]).norm(dim=-1)                       # 净位移
    seg = (pos[1:] - pos[:-1]).norm(dim=-1).sum(dim=0)          # 走过的路程
    straight = net / seg.clamp(min=1e-9)                        # 1.0 = 直线
    vel_w = torch.stack(log["vel_w"])        # [T, B, 2]
    bearing = torch.rad2deg(torch.atan2(vel_w[..., 1], vel_w[..., 0]))
    d_bearing = torch.atan2(torch.sin(bearing[1:] - bearing[:-1]),
                            torch.cos(bearing[1:] - bearing[:-1]))
    # 解缠: u[0]=bearing[0], u[k]=bearing[0]+sum(d[0..k-1])。写错成 cumsum([b0, b0+d...]) 会把 b0 累加 T 次
    bearing_u = torch.cat([bearing[:1], bearing[:1] + torch.cumsum(d_bearing, dim=0)], dim=0)
    t = torch.arange(n, dtype=bearing_u.dtype).unsqueeze(1)
    t_c = t - t.mean()
    vel_slope = (t_c * (bearing_u - bearing_u.mean(dim=0))).sum(dim=0) / t_c.pow(2).sum()
    speed = vel_w.norm(dim=-1).mean()
    print(f"\n{'路径几何':<26}{'均值':>10}{'最大':>10}{'单位':>8}")
    print(f"{'速度方向漂移(路径拐弯)':<21}{float(vel_slope.mean() * 50):>10.3f}"
          f"{float(vel_slope.abs().max() * 50):>10.3f}{'deg/s':>8}")
    print(f"{'  -> 拐弯半径 |v/w|':<22}"
          f"{float((speed / (vel_slope.mean().abs() * 50 * torch.pi / 180).clamp(min=1e-9))):>10.2f}"
          f"{float((speed / (vel_slope.abs().max() * 50 * torch.pi / 180).clamp(min=1e-9))):>10.2f}{'m':>8}")
    print(f"{'路径直线度(净/路程)':<21}{float(straight.mean()):>10.3f}{float(straight.max()):>10.3f}{'':>8}"
          "  (1.0=笔直, 0.5 以下=明显绕圈)")
    print(f"{'平均速度 |v|':<26}{float(speed):>10.3f}{'':>10}{'m/s':>8}")

    # ---- 扭头是"发力机制"还是"抖"? 看 wz 和左右脚载荷的相位关系 ----
    # s = 右脚在摆动(+1) - 左脚在摆动(-1): 载荷偏到哪只脚上。
    # 若 wz 与 s 在某个固定滞后上强相关 -> 躯干扭动是"倒重心/甩腿"的手段(有功能);
    # 若各滞后都不相关 -> 只是抖, 可以直接压掉。
    in_air = ~found                                       # [T, B, 2] True=这只脚离地
    s_load = in_air[..., 1].float() - in_air[..., 0].float()   # [T, B]
    print("\n=== 扭动 vs 左右脚载荷 的互相关 (找最佳滞后) ===")
    best_lag, best_r = [], []
    for b in range(T.shape[1]):
        x = T[:, b] - T[:, b].mean()
        y = s_load[:, b] - s_load[:, b].mean()
        if float(y.std()) < 1e-6 or float(x.std()) < 1e-6:
            best_lag.append(0); best_r.append(0.0); continue
        rs = []
        for lag in range(-8, 9):                           # ±8 步 = ±160ms
            ys = torch.roll(y, lag)
            rs.append(float((x * ys).mean() / (x.std() * ys.std())))
        k = int(torch.tensor(rs).abs().argmax())
        best_lag.append(k - 8); best_r.append(rs[k])
    bl = torch.tensor(best_lag, dtype=torch.float32)
    br = torch.tensor(best_r)
    print(f"  最佳滞后 {float(bl.mean()):+.1f} 步 (负=扭动先于抬脚)   峰值相关 |r| {float(br.abs().mean()):.2f}"
          f"   分环境 |r|: " + " ".join(f"{abs(float(v)):.2f}" for v in br))
    print("  读法: |r|>0.5 且滞后稳定 -> 扭动是这只机器人的倒重心机制, 别硬压;")
    print("        |r|<0.3 或滞后每局乱跳 -> 只是抖动, 可以用奖励压掉")

    print("\n=== 怎么读 ===")
    print("  主频/步频 ≈ 1.0   : 摆动就是左右重心转移, 正常步态, 不用管")
    print("  主频/步频 ≈ 0.5    : 每两步晃一次 -> 极限环, 该用奖励压")
    print("  主频/步频 远离整数 : 躯干共振/控制带宽不够 -> 查 action_rate / body_ang_vel")
    print(f"  侧向速度 RMS 与前进速度之比: {float((rms(vy).mean() / max(abs(args.vx), 1e-6)) * 100):.1f}%"
          "  (经验线: < 25% 算走得干净)")


if __name__ == "__main__":
    main()
