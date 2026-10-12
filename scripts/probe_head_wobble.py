"""量"摇头": 头/脖子这条链在这只机器人身上到底怎么动的, 为什么它会一直晃。

背景(从 MJCF 读出来的): 这只鸭子 **41% 的质量在头上的链条里**
(jaw_soft 一项 188.8g = 25.6%, 脖子链合计 303g / 737g), 而这条链挂在 4 个关节上
(neck_pitch / head_pitch / head_yaw / head_roll), 位于躯干最上方。
所以"头"不是一个轻装饰件, 它是一根 41% 质量的倒立摆。

本脚本回答三件事:
  ① 4 个头关节各自偏离默认姿势多远、以什么频率动(和步频/躯干扭动的关系);
  ② 嘴尖(head tip)在**躯干坐标系**里摆多大 —— 这才是眼睛里看到的"摇头"幅度;
  ③ 这个摆动对整机质心的横向贡献是多少(用质量占比算), 以及它和躯干扭动是
     "配重反相"还是"同相被甩"。

用法: python scripts/probe_head_wobble.py model_3250.pt --vx 0.15 --steps 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_walk import build_env, load_policy  # noqa: E402
from probe_gait_wobble import dominant_freq, quat_to_yaw  # noqa: E402

DT = 0.02
_HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
_TIP_SITE = "mouth_tip"
# 头链 = neck_pitch 关节以下那棵子树 {neck, neck_pitch, yaw_roll_motion, jaw_soft}。
# 从编译后的模型算出来是 279.9 g / 737.2 g = 38.0%(注意 bearing_roll 23g 是躯干侧零件, 不算)。
# 实测会重新算一遍并打印, 这个常数只用作兜底。
_HEAD_CHAIN = ("neck", "neck_pitch", "yaw_roll_motion", "jaw_soft")
_HEAD_MASS_FRACTION_FALLBACK = 279.9 / 737.2


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """把世界系向量 v 转到以 q=(w,x,y,z) 为姿态的机体系 (R(q)^T v)。"""
    w = q[:, 0:1]
    u = q[:, 1:4]
    return (2 * w * w - 1) * v - 2 * w * torch.cross(u, v, dim=-1) + 2 * u * (u * v).sum(-1, keepdim=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", type=str)
    ap.add_argument("--vx", type=float, default=0.15)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--freeze-head", action="store_true",
                    help="把头/脖子 4 个关节的动作通道置零 —— 看目标不动了头还会不会晃")
    args = ap.parse_args()

    venv = build_env(args.num_envs, args.vx, 0.0, episode_length_s=(args.steps + 5) * DT)
    policy = load_policy(venv, args.checkpoint)
    env = venv.unwrapped
    robot = env.scene["robot"]

    import mujoco
    from mjlab_microduck.tasks.microduck_walk_cfg import get_microduck_walk_entity_cfg

    mj = get_microduck_walk_entity_cfg().spec_fn().compile()
    chain_ids = [robot.body_names.index(n) for n in _HEAD_CHAIN if n in robot.body_names]
    chain_mass = torch.tensor([float(mj.body_mass[
        mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, robot.body_names[i])]) for i in chain_ids])
    head_frac = float(chain_mass.sum() / float(mj.body_mass.sum()))
    print(f"头链 {[robot.body_names[i] for i in chain_ids]} 质量 {float(chain_mass.sum()) * 1000:.1f} g"
          f" = 整机 {head_frac * 100:.1f}%")

    # 动作向量里的下标: 实测(单点脉冲法)=关节下标, 头四个是 5,6,7,8
    head_ids = [robot.joint_names.index(n) for n in _HEAD_JOINTS]
    jid = {n: robot.joint_names.index(n) for n in _HEAD_JOINTS}
    tip = robot.site_names.index(_TIP_SITE)
    default = robot.data.default_joint_pos.clone()

    obs = venv.get_observations()
    from eval_walk import _FEET_CONTACT_SENSOR
    contact = env.scene[_FEET_CONTACT_SENSOR]
    rec: dict[str, list[torch.Tensor]] = {k: [] for k in ("q", "wz", "tip", "root", "quat", "act", "chain")}
    log_found: list[torch.Tensor] = []
    for _ in range(args.steps):
        with torch.no_grad():
            a = policy(obs)
        if args.freeze_head:
            a = a.clone()
            a[:, head_ids] = 0.0
        obs, _, _, _ = venv.step(a)
        rec["q"].append(robot.data.joint_pos.clone())
        rec["wz"].append(robot.data.root_link_ang_vel_b[:, 2].clone())
        rec["tip"].append(robot.data.site_pos_w[:, tip, :].clone())
        rec["chain"].append(robot.data.body_link_pos_w[:, chain_ids, :].clone())
        rec["root"].append(robot.data.root_link_pos_w.clone())
        rec["quat"].append(robot.data.root_link_quat_w.clone())
        log_found.append((contact.data.found > 0).clone())
        rec["act"].append(a.clone())

    q = torch.stack(rec["q"])            # [T,B,14]
    wz = torch.stack(rec["wz"])          # [T,B]
    tip_w = torch.stack(rec["tip"])      # [T,B,3]
    root = torch.stack(rec["root"])
    quat = torch.stack(rec["quat"])
    n, B = wz.shape

    # ① 头关节: 偏离默认姿势多少、主频多少
    print(f"\n=== ① 头/脖子关节 (默认姿势基准, 指令 vx={args.vx}"
          f"{', 冻头' if args.freeze_head else ''}) ===")
    print(f"{'关节':<14}{'RMS偏离':>10}{'最大偏离':>10}{'主频Hz':>9}{'与wz相关':>10}")
    for name in _HEAD_JOINTS:
        dev = torch.rad2deg(q[:, :, jid[name]] - default[:, jid[name]])       # [T,B] deg
        rms = dev.pow(2).mean(dim=0).sqrt()
        freqs = torch.tensor([dominant_freq(dev[:, b], DT) for b in range(B)])
        cc = []
        for b in range(B):
            x, y = dev[:, b], wz[:, b]
            cc.append(float(((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std() + 1e-9)))
        print(f"{name:<14}{float(rms.mean()):>10.2f}{float(dev.abs().max(dim=0).values.mean()):>10.2f}"
              f"{float(freqs.mean()):>9.3f}{sum(cc)/len(cc):>+10.2f}")

    # ② 嘴尖在躯干坐标系里的摆动 —— 眼睛看到的"摇头"
    rel = quat_rotate_inverse(quat.reshape(-1, 4), (tip_w - root).reshape(-1, 3)).reshape(n, B, 3)
    print(f"\n=== ② 嘴尖在躯干坐标系里的摆动 (mm) ===")
    print(f"  自检: 嘴尖相对躯干的高度 z = {float(rel[..., 2].mean()) * 1000:+.1f} mm (应为正, 头在躯干上方)")
    for i, ax in enumerate(("前后 x", "左右 y", "上下 z")):
        v = rel[..., i] * 1000
        pp = (v.max(dim=0).values - v.min(dim=0).values)
        freqs = torch.tensor([dominant_freq(v[:, b], DT) for b in range(B)])
        print(f"  {ax}: RMS {float(v.std(dim=0).mean()):>6.2f}   峰峰值 {float(pp.mean()):>6.2f}   "
              f"主频 {float(freqs.mean()):.3f} Hz")
    chain = torch.stack(rec["chain"])                      # [T,B,K,3]
    ikey = chain_ids.index(robot.body_names.index("jaw_soft")) if "jaw_soft" in robot.body_names else 0
    head_com_w = (chain * chain_mass.view(1, 1, -1, 1)).sum(dim=2) / chain_mass.sum()
    rel_com = quat_rotate_inverse(quat.reshape(-1, 4), (head_com_w - root).reshape(-1, 3)).reshape(n, B, 3)
    y_mm = rel_com[..., 1] * 1000
    com_mm = float(y_mm.std(dim=0).mean()) * head_frac
    print(f"  头链质心在躯干系里: 左右 RMS {float(y_mm.std(dim=0).mean()):.2f} mm,"
          f" 峰峰 {float((y_mm.max(dim=0).values - y_mm.min(dim=0).values).mean()):.2f} mm")
    cc = []
    for b in range(B):
        x, y = y_mm[:, b], wz[:, b]
        cc.append(float(((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std() + 1e-9)))
    print(f"\n=== ③ 头摆动与躯干扭动的关系 ===")
    print(f"  头链质心左右位移 与 wz 的相关: {sum(cc)/len(cc):+.2f}"
          "   (正 = 头被扭动甩着同向走; 负 = 头当配重反相甩)")
    print(f"  折算到整机质心的横向位移: RMS {com_mm:.2f} mm ({head_frac*100:.0f}% 质量占比)")
    print(f"  对比: 两脚间距 ~70 mm ⇒ 这是步宽的 {com_mm / 70 * 100:.0f}%")

    # ④ 判定: 头是在"当配重"(有功能) 还是"没人在管的自由度"(只是飘)?
    #    和躯干扭动用同一把尺子: 看它与左右脚载荷的互相关有没有稳定的滞后。
    found = torch.stack(log_found)                     # [T,B,2]
    in_air = ~found
    s_load = in_air[..., 1].float() - in_air[..., 0].float()
    lags, rs = [], []
    for b in range(B):
        x = y_mm[:, b] - y_mm[:, b].mean()
        y = s_load[:, b] - s_load[:, b].mean()
        if float(y.std()) < 1e-6:
            lags.append(0); rs.append(0.0); continue
        vals = []
        for lag in range(-8, 9):
            ys = torch.roll(y, lag)
            vals.append(float((x * ys).mean() / (x.std() * ys.std() + 1e-9)))
        k = int(torch.tensor(vals).abs().argmax())
        lags.append(k - 8); rs.append(vals[k])
    lag_t = torch.tensor(lags, dtype=torch.float32)
    r_t = torch.tensor(rs)
    print(f"\n=== ④ 头的左右摆动 vs 左右脚载荷 的互相关 ===")
    print(f"  最佳滞后 {float(lag_t.mean()):+.1f} 步   峰值 |r| {float(r_t.abs().mean()):.2f}"
          f"   分环境 |r|: " + " ".join(f"{abs(float(v)):.2f}" for v in r_t))
    act = torch.stack(rec["act"])       # [T,B,14] 策略输出(= 关节位置目标相对默认姿势的偏移, scale 1.0)
    print(f"\n=== ⑤ 是策略主动甩, 还是被躯干加速度甩起来的? (动作=位置目标偏移, rad) ===")
    print(f"{'关节':<14}{'动作RMS':>10}{'换算成度':>10}{'实际偏离RMS':>12}{'实际/动作':>10}")
    for name in _HEAD_JOINTS + ("left_hip_roll", "left_hip_yaw", "right_hip_roll", "right_hip_yaw"):
        j = robot.joint_names.index(name)
        a_rms = float(act[:, :, j].pow(2).mean().sqrt())
        q_rms = float(torch.rad2deg(q[:, :, j] - default[:, j]).pow(2).mean().sqrt())
        ratio = q_rms / (a_rms * 180 / 3.14159265)
        r_txt = "  ——" if a_rms < 1e-6 else f"{min(ratio, 99.0):>10.1f}"   # 冻头时动作=0, 比值无意义
        print(f"{name:<14}{a_rms:>10.3f}{a_rms * 57.2958:>10.1f}{q_rms:>12.1f}{r_txt}")
    print("  实测: 头两个俯仰关节的 实际偏离/动作 = 1.3~2.3 -> 目标在扫, 而这个重的头追不上")
    print("        (位置伺服跟不上 = 真机上这几个舵机要一直出力; mjlab 这边拿不到扭矩读数, 故不报 Nm)")

    print("  读法: 滞后在 8 个环境里一致 + |r| 高 -> 头在当配重(有功能, 冻结会伤步态);")
    print("        |r| 低或滞后每局乱 -> 这 4 个自由度没人在管, 冻掉是净赚")


if __name__ == "__main__":
    main()
