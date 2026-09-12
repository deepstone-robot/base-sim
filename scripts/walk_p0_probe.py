"""行走任务 P0/P1 修正的「量化探针」 —— 把每条改动背后的数字打印出来, 不用信口头结论。

用法:
    .venv/bin/python scripts/walk_p0_probe.py            # 默认 4 envs × 200 步
    .venv/bin/python scripts/walk_p0_probe.py --steps 600

九段, 每段对应一条"靠实测定下来"的结论(详见 microduck_walk_cfg.py 顶部修正记录):
  ① 探针几何: 脚底贴地时转动脚踝, 四版离地高度探针各漂多少
     → 量化 P0④(单点 11~14mm vs 采样点 min ~0.2mm; 顺带证明上游射线在平地也没用)
  ② 动作尺度: 抬脚需要的膝屈, 在 scale=0.5/1.0 下各要策略输出多大、命中率多少 → P0⑤
  ②b 站着不动的捷径: 逐档算"站着不动能白拿多少跟踪奖励" → P2⑧(500 轮失败的主因)
  ②c 姿态门标定: 五种实测姿势下"还靠腿站着吗"的门值 → 第三轮(躺平薅分)的堵口
  ②d 方向进度门 + 站姿基准: 原地蹦该拿 0; 姿态罚的基准必须用实测站姿 → P2⑮⑰
  ③ upright: 不同倾角下线性版 vs 高斯版的每步得分 → P0③
  ④ 步态奖励质量: 一个步态周期内 air_time 旧写法(落脚一次性) vs 新写法(每步付) → P0①
  ⑤ action_rate 权重阶梯 + 「按正质量份额对齐而不是照抄数字」的算式 → P1⑥
  --- 以下需要建环境(CPU) ---
  ⑥ 环境内健全性检查: 接触相里脚底最低点探针是否读出 ~0
  ⑦ body_ang_vel / angular_momentum 两项动态正则的实测量级 → P1⑦
     (结论: 前者平时 0.2%、出事才咬; 后者只有 1e-7 量级 = 等于没接)
  ⑧ 观测契约: actor/critic 维度与槽位语义, 以及 walk↔stand 的续训陷阱 → P0②
  ⑨ 奖励体检: 逐项打印加权后的奖励量级与符号(AGENTS 那条"每个罚项必须 ≤0")

CPU 秒级可跑(首次会编译 warp CPU 内核, 1~2 分钟)。
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import torch

from mjlab.envs import ManagerBasedRlEnv

from mjlab_microduck.tasks import _STAND_JOINT_POS, microduck_walk_env_cfg

_REPO = Path(__file__).resolve().parents[1]
_FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")
_FOOT_SITES = ("left_foot", "right_foot")
_STAND_JOINT_NAMES = tuple(_STAND_JOINT_POS)
_MODEL_PATH = _REPO / "assets" / "microduck" / "robot_allcollisions.xml"


def _h(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------- ① 探针几何
def _geom_min_z(model: mujoco.MjModel, data: mujoco.MjData, gid: int) -> float:
    """某个 mesh geom 在世界系里的最低点(≈ 脚底离地高度)。"""
    mesh_id = model.geom_dataid[gid]
    adr = model.mesh_vertadr[mesh_id]
    verts = model.mesh_vert[adr : adr + model.mesh_vertnum[mesh_id]]
    rot = data.geom_xmat[gid].reshape(3, 3)
    world = verts @ rot.T + data.geom_xpos[gid]
    return float(world[:, 2].min())


def probe_geometry() -> None:
    _h("① 探针几何: 脚底贴地、只转脚踝, 各探针各漂多少")
    model = mujoco.MjModel.from_xml_path(str(_MODEL_PATH))
    data = mujoco.MjData(model)
    left_ankle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_ankle")
    ankle_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ankle_left")
    left_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    left_foot_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, _FOOT_GEOMS[0])

    def fk(ankle_delta: float):
        data.qpos[:] = 0.0
        data.qpos[2] = 0.12
        data.qpos[3] = 1.0
        for name, val in _STAND_JOINT_POS.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[jid]] = val
        data.qpos[model.jnt_qposadr[left_ankle]] += ankle_delta
        # 把根部高度挪一下, 让左脚脚底正好贴地(模拟"支撑相"), 迭代几次收敛
        for _ in range(3):
            mujoco.mj_forward(model, data)
            data.qpos[2] -= _geom_min_z(model, data, left_foot_geom)
        mujoco.mj_forward(model, data)

        # 模拟射线传感器的 ring 采样: 射线是**竖直**的(ray_alignment="yaw"), 所以
        # 采样点只在**水平面内**偏置 —— 不能直接在 site 的局部 XY 平面上撒点
        # (那样会跟着脚踝倾斜, 算出来是个假的漂移)。取 site 的朝向投影到水平面。
        axis = data.site_xmat[left_site].reshape(3, 3)[:, 0].copy()
        axis[2] = 0.0
        norm = float((axis @ axis) ** 0.5)
        if norm < 1e-9:  # 退化成竖直轴, 退回世界 X
            axis = __import__("numpy").array([1.0, 0.0, 0.0])
        else:
            axis = axis / norm
        radius = 0.04  # 与 RingPatternCfg.single_ring(radius=0.04, num_samples=2) 一致
        ring_pts = [
            data.site_xpos[left_site],
            data.site_xpos[left_site] + radius * axis,
            data.site_xpos[left_site] - radius * axis,
        ]
        ring_min = min(p[2] for p in ring_pts)
        return (
            float(data.site_xpos[left_site][2]),
            float(data.xpos[ankle_body][2]),
            _geom_min_z(model, data, left_foot_geom),
            float(ring_min),
        )

    from mjlab_microduck.tasks.microduck_walk_cfg import _LEFT_SOLE_POINTS

    print("脚底始终贴地(真值列恒 0), 只把脚踝转开:")
    print(f"{'踝角偏移':>9} {'真值:脚底最低点':>15} {'第1版:脚踝body':>14} {'第2版:脚底site':>14} "
          f"{'第3版:上游射线':>14} {'现用:采样点min':>14}")
    rows = []
    for deg in (-20, -10, 0, 10, 20, 30):
        site_z, body_z, sole_z, ring_z = fk(math.radians(deg))
        # 现用探针: 用同一套点表 + FK 算(与奖励里 _foot_clearance_heights 同算法)
        left_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ankle_left")
        rot, pos = data.xmat[left_body].reshape(3, 3), data.xpos[left_body]
        pts_min = min(((rot @ p) + pos)[2] for p in _LEFT_SOLE_POINTS)
        rows.append((deg, site_z, body_z, sole_z, ring_z, float(pts_min)))
        print(f"{deg:>9} {sole_z:>15.5f} {body_z:>14.5f} {site_z:>14.5f} {ring_z:>14.5f} "
              f"{pts_min:>14.5f}")

    d0, d1 = rows[2], rows[-1]
    names = ("脚踝 body", "脚底 site", "上游射线(平地)", "采样点 min(现用)")
    print(f"\n脚踝转 {d1[0] - d0[0]}°(走路 toe-off/heel-strike 的典型量级)时, 各版探针漂移:")
    for name, i in zip(names, (2, 1, 4, 5)):
        print(f"  {name:<18} {(d1[i] - d0[i]) * 1000:+6.1f} mm")
    print(
        "\n结论(这条是实测纠正了我三次猜测, 值得记下来):\n"
        "  ① 脚踝 body / 脚底 site 都是**刚体上的一个点**, 脚一翻就跟着转 —— 谁都不能用\n"
        "     (我原以为换成 site 就好了, 实测 site 漂得更多: +14.1mm vs +10.9mm);\n"
        "  ② 上游那套射线传感器读数**正好等于 site 高度**: 射线是竖直向下的, ring 只在\n"
        "     水平面里偏置 ⇒ 三条射线的起点 z 相同 ⇒ reduction=min 退化成 site 自己的\n"
        "     高度。那套 ring 只在**粗糙地形**上有意义(地形有起伏), 平地上救不了脚踝翻转;\n"
        "  ③ 真正要的是'脚底区域最低那一点', 所以对一片点取 min —— 漂移从 11~14mm 降到 ~2mm。"
    )


def probe_clearance_in_env(env: ManagerBasedRlEnv) -> None:
    """环境里的健全性检查: 奖励用的"脚底最低点"探针, 接触时是否读出 ~0。

    注意它**不能**用来回答"探针会不会随脚踝翻转漂" —— 静态站立时鸭子会自己重新
    平衡(身体跟着脚踝的转动调整), 所以两个探针在这里都看不出漂。漂移要用 ① 那种
    纯运动学扫描来量: 把"脚底贴地"这个条件钉死, 再看探针读数怎么变。
    """
    _h("⑥ 环境内健全性检查: 脚底最低点探针在接触相是否读出 ~0")
    from mjlab_microduck.tasks.microduck_walk_cfg import _foot_clearance_heights

    robot = env.scene["robot"]
    asset_cfg = env.reward_manager.get_term_cfg("foot_clearance").params["asset_cfg"]
    site_ids = list(robot.find_sites(_FOOT_SITES)[0])
    left_ankle_idx = robot.find_joints(("left_ankle",))[0][0]

    def hold_left_ankle(theta: float) -> tuple[float, float, float]:
        """给左踝下目标角 theta, 等它稳住, 回来读三个数。"""
        for _ in range(40):
            act = torch.zeros(env.num_envs, 14)
            act[:, left_ankle_idx] = theta  # scale=1.0 → 目标角 = 默认 + theta
            env.step(act)
        heights = _foot_clearance_heights(env, asset_cfg)[0]
        site_z = float(robot.data.site_pos_w[0, site_ids[0], 2])
        return float(heights[0]), site_z, float(heights[1])

    print("注意: 鸭子靠右腿站着, 左踝被强行转开 —— 左脚的脚底可能部分仍在接触地面。")
    print(f"{'左踝目标偏移':>12} {'脚底最低点(左脚)':>16} {'左脚site z':>12} {'脚底最低点(右脚)':>16}  说明")
    for deg in (0, 5, 10, 20, 30):
        h_left, site_z, h_right = hold_left_ankle(math.radians(deg))
        note = "左脚仍是接触相" if h_left < 0.005 else "左脚确实离地了"
        print(f"{deg:>10}° {h_left:>14.5f} {site_z:>12.5f} {h_right:>14.5f}  {note}")
    print(
        "\n结论: 接触相里新探针读出 ~0(数值在 ±0.5mm 内晃, 是接触求解器的柔度), 说明\n"
        "      它在真实仿真里量的是'脚底离地多少', 而不是别的什么东西。\n"
        "      '脚踝翻转会不会把读数带偏'这个问题由 ① 回答(那里把脚底贴地钉死了, 所以\n"
        "      探针读数的任何变化都只能来自脚踝翻转): 单点偏 11~14mm, 采样点 min 偏 ~0.2mm。\n"
        "'摆动脚抬到 2cm' 这句话, 只有用后者量才说得通。"
    )


# ---------------------------------------------------------------- ② 动作尺度
def action_scale_table() -> None:
    _h("② 动作尺度: 抬脚需要的膝屈, 策略要输出多大才能做到")

    def tail_prob(x: float) -> float:
        """P(|a| >= x), a ~ N(0, 1) —— 高斯策略里"能采到这个幅度"的概率。"""
        return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))))

    print("站立时 left_knee = -0.0049rad; 抬脚需要的膝屈(相对站姿):")
    print(f"{'膝屈 rad':>10} {'scale=0.5 需输出':>18} {'命中率':>9} {'scale=1.0 需输出':>18} {'命中率':>9}")
    for flex in (0.3, 0.5, 0.7, 0.9):
        need5, need1 = flex / 0.5, flex / 1.0
        print(f"{flex:>10.1f} {need5:>18.2f} {tail_prob(need5):>8.1%} {need1:>18.2f} {tail_prob(need1):>8.1%}")
    print(
        "\n结论: 不是'到不了'(高斯无截断, 3σ 也能到), 而是'典型采样点落不落在有用的\n"
        "      区域'。scale=0.5 时抬脚要 1.0~1.8σ, 采样命中率个位数到 20%;\n"
        "      scale=1.0 时只要 <1σ, 学习信号从一开始就稠密。"
    )


# ------------------------------------------------- ②b 站着不动的捷径(500轮教训)
def do_nothing_shortcut() -> None:
    _h("②b 「站着不动」能白拿多少跟踪奖励 —— 500 轮跑出\"学会站着不动\"的根因")
    from mjlab_microduck.tasks.microduck_walk_cfg import _VELOCITY_STAGES

    print("站在不动时 track_lin_vel 的高斯核 = exp(-cmd²/std²), 逐档取指令范围均值:")
    for std in (0.25, 0.10):
        print(f"\n  std={std}(权重 2.0 → 站着不动每步拿多少):")
        for stage in _VELOCITY_STAGES:
            lo, hi = stage["lin_vel_x"]
            if lo == hi:
                continue
            n = 400
            vs = [lo + (hi - lo) * i / n for i in range(n + 1)]
            shortcut = sum(math.exp(-(v**2) / std**2) for v in vs) / len(vs)
            good = math.exp(-(0.04**2) / std**2)
            print(f"    第{_VELOCITY_STAGES.index(stage) + 1}档 step>={stage['step']:>4} "
                  f"vx∈({lo:+.2f},{hi:+.2f}) 均值{sum(abs(v) for v in vs) / len(vs):.3f}: "
                  f"站着不动 {shortcut:5.1%}  好步态(误差0.04) {good:5.1%}  "
                  f"学习信号 {2.0 * (good - shortcut):+.2f} 分/步")
    print(
        "\n结论(这就是那次 500 轮失败的主因):\n"
        "  历史值(改之前): 第1档是 vx∈(0,0.15)、均值只有 0.075, 配 std=0.25 → 站着不动\n"
        "  白拿 90%, 而好步态只多拿 0.06 分/步。上面两张表是新配置下的重算。\n"
        "  策略的理性选择就是: 站住, 拿满 upright(2.0) + 90% 的 track_lin(1.8), 什么都不做。\n"
        "  上游 std=0.316 是配它们 ±0.4m/s 的指令尺度(站着只能拿 20~40%); 我们的指令尺度小\n"
        "  4 倍, 却照抄了同一个 std。改成 0.10 + 第1档下限抬到 0.10 之后: 站着只剩 ~10%, \n"
        "  学习信号从 0.06 放大到 ~1.5 分/步。\n"
        "  教训: std ≈ \"你还在乎的误差\", 要跟**自己的指令尺度**成比例, 不能跨任务照抄。"
    )


# --------------------------------------------- ②c 姿态门(躺平薅分的堵口)
def posture_gate_table(env: ManagerBasedRlEnv) -> None:
    _h("②c 姿态门标定 —— 「还靠腿站着吗」(第三轮 250 轮实测: 鸭子躺平薅 +1.8/步)")
    from mjlab.utils.lab_api.math import quat_from_angle_axis

    from mjlab_microduck.tasks.microduck_walk_cfg import (
        _GATE_COS_HIGH,
        _GATE_COS_LOW,
        _GATE_Z_HIGH,
        _GATE_Z_LOW,
        _posture_gate,
    )

    robot = env.scene["robot"]
    print(f"门的构成: 高度斜坡 z∈[{_GATE_Z_LOW}, {_GATE_Z_HIGH}] × "
          f"直立斜坡 cos(倾角)∈[{_GATE_COS_LOW}, {_GATE_COS_HIGH}]")
    print(f"\n{'姿势(实测量级)':>22} {'躯干z':>8} {'倾角':>7} {'门':>7}  说明")

    def gate(z, tilt):
        quat = quat_from_angle_axis(
            torch.tensor([math.radians(tilt)]), torch.tensor([[0.0, 1.0, 0.0]])
        )
        root = torch.zeros(env.num_envs, 13)
        root[:, 2], root[:, 3:7] = z, quat[0]
        robot.write_root_state_to_sim(root)
        env.sim.forward()
        return float(_posture_gate(env).mean())

    for name, z, tilt, note in (
        ("躺平(250 轮新策略)", 0.041, 50.0, "技能类奖励全被乘掉 → 薅不到了"),
        ("坐姿(SIT 关键帧)", 0.070, 10.0, "只拿到 ~20%, 想起来有梯度"),
        ("蹲低但直立", 0.075, 5.0, "斜坡中间"),
        ("走路(旧 250 轮策略)", 0.095, 12.0, "不误伤"),
        ("站立(旧 499 轮策略)", 0.117, 2.0, "满分"),
    ):
        print(f"{name:>22} {z:>8.3f} {tilt:>6.1f}° {gate(z, tilt):>7.3f}  {note}")
    print(
        "\n为什么必须有它(250 轮实测): 鸭子躺在地上(躯干 0.041、倾角 50°)把 air_time 从\n"
        "  0.08/步薅到 0.76~1.43/步(腿举在空中永远满足摆动窗口)、track_lin 还白拿 0.66\n"
        "  (零指令时不动就是完美跟踪), 唯一罚它的 upright 只有 0.06(而不是 2.0) → 净 +1.8/步。\n"
        "  上游 playbook 原话: Never gate a positive reward on being in a bad state — the\n"
        "  policy parks in the cheapest qualifying pose and farms it.\n"
        "  配套还有 fallen_too_long 终止: 倒地超 1 秒才终止(留恢复样本), 站起来一步清零。"
    )


# ------------------------------------- ②d 方向门 + 站姿基准(第三/四轮的两个薅分点)
def progress_gate_and_stance(env: ManagerBasedRlEnv) -> None:
    _h("②d 方向进度门 + 站姿基准 —— 「原地蹦」和「拿站不到的姿势当基准」两个坑")
    from mjlab_microduck.tasks.microduck_walk_cfg import (
        _LEG_JOINT_ORDER,
        _STANCE_OFFSET,
        _STD_STANCE,
        _progress_gate,
        _std_for,
    )

    cmd = env.command_manager.get_term("twist")
    robot = env.scene["robot"]

    def gate(vx: float, vy: float, cmd_vx: float = 0.2) -> float:
        cmd.vel_command_b[:] = 0.0
        cmd.vel_command_b[:, 0] = cmd_vx
        root = torch.zeros(env.num_envs, 13)
        root[:, 2], root[:, 3] = 0.117, 1.0
        root[:, 7], root[:, 8] = vx, vy
        robot.write_root_state_to_sim(root)
        env.sim.forward()
        return float(_progress_gate(env).mean())

    print("air_time 的方向进度门(指令 vx=0.2; 第三轮鸭子就是靠「原地蹦」薅到 2.04/步的):")
    for name, vx, vy in (("站着不动", 0.0, 0.0), ("横着走", 0.0, 0.2),
                         ("原地蹦(实测 0.07)", 0.07, 0.05), ("往反方向走", -0.1, 0.0),
                         ("走了 1/4", 0.05, 0.0), ("走了一半", 0.10, 0.0),
                         ("按指令走", 0.20, 0.0)):
        print(f"  {name:>18}: 门 = {gate(vx, vy):.3f}")

    print("\n姿态罚的基准 = 实测站姿(不是 XML 默认姿势)。实测量级:")
    for _ in range(6):
        env.step(torch.zeros(env.num_envs, 14))
    dev = robot.data.joint_pos[0] - robot.data.default_joint_pos[0]
    print(f"  {'关节':>18} {'相对XML默认':>12} {'实测偏移(在用)':>15} {'std':>6}")
    for n in _LEG_JOINT_ORDER:
        i = robot.joint_names.index(n)
        print(f"  {n:>18} {float(dev[i]):>+12.3f} {_STANCE_OFFSET[n]:>+15.3f} "
              f"{_std_for(n, _STD_STANCE):>6.2f}")
    print(
        "\n结论: 鸭子站稳时踝关节比 XML 默认转了 ~17.6° —— 拿 XML 默认当基准 = 按\n"
        "      '它站不到的姿势'扣分(旧 joint_dev 就是这么错的)。换成实测基准后,\n"
        "      自然站姿的 cost ≈ 0, 而叉腿 8.6° 罚到 >1(权重 -0.2 → -0.36/步)。"
    )


# ---------------------------------------------------------------- ③ upright
def upright_table() -> None:
    _h("③ upright: 倾角 vs 每步得分(线性版 vs 高斯版)")
    std = math.sqrt(0.05)
    print(f"{'倾角 deg':>9} {'旧: 0.5*cos':>13} {'旧: 相对直立掉分':>17} "
          f"{'新: 2*exp(-s²/std²)':>20} {'新: 相对直立掉分':>17} {'倍率':>7}")
    for deg in (0, 5, 10, 20, 30):
        th = math.radians(deg)
        old = 0.5 * math.cos(th)
        new = 2.0 * math.exp(-(math.sin(th) ** 2) / std**2)
        d_old, d_new = 0.5 - old, 2.0 - new
        ratio = (d_new / d_old) if d_old > 1e-12 else float("nan")
        print(f"{deg:>9} {old:>13.4f} {d_old:>17.4f} {new:>20.4f} {d_new:>17.4f} {ratio:>7.0f}x")
    print(
        "\n结论: 高斯版在 10° 处就比线性版硬 ~140 倍。旧版太软, 而整套奖励里没有\n"
        "      第二项在罚'静态倾角'(track_ang_vel 罚的是角速度), 所以'歪着身子蹭'\n"
        "      在旧配置里几乎免费。"
    )


# ---------------------------------------------------------------- ④ air_time
def air_time_table() -> None:
    _h("④ 步态奖励质量: 一个步态周期里 air_time 付多少钱")
    from mjlab_microduck.tasks.microduck_walk_cfg import _AIR_WINDOW_MIN_STAGES

    step_dt, weight_new, weight_old = 0.02, 3.0, 0.5
    lo, hi = _AIR_WINDOW_MIN_STAGES[0][1], 0.300   # 起步档窗口(课程会把它抬到 0.125)
    print("窗口下限的课程(起步 0.03 → 抬到 0.125; 依据: 实测鸭子一开始只能摆动 20~30ms):")
    for step, w in _AIR_WINDOW_MIN_STAGES:
        print(f"   step >= {step:>6}  →  threshold_min = {w:.3f}")
    print("\n公式:")
    print("  旧(落脚一次性): 周期内 = 2 只脚 × 1 次 × weight_old")
    print("  新(每步付):     周期内 = 2 只脚 × clamp(T_air-lo,0,hi-lo)/step_dt × weight_new")
    print(f"\n{'摆动时长 T_air':>15} {'旧 分/周期':>11} {'新 分/周期':>11} {'倍率':>7} {'新 分/步':>10}")
    for t_air in (0.15, 0.20, 0.25, 0.30):
        old = 2 * 1 * weight_old
        new = 2 * (min(t_air, hi) - lo) / step_dt * weight_new
        cycle_s = 0.7  # 一个典型步态周期(两条腿各摆一次)
        print(f"{t_air:>13.2f}s {old:>11.1f} {new:>11.1f} {new / old:>6.0f}x "
              f"{new / (cycle_s / step_dt):>10.2f}")
    print(
        "\n对照: track_lin_vel 满分是 2.0 分/步。\n"
        "结论: 旧写法的步态奖励(0.5 分/周期 ≈ 0.014 分/步)相对于速度跟踪等于不存在,\n"
        "      于是'前倾蹭地'和'迈步'竞争时, 策略没有任何理由选后者。新写法把它抬到\n"
        "      与速度跟踪同一量级(0.3~1.5 分/步), 步态才第一次成为有回报的行为。"
    )


# ---------------------------------------------------------------- ⑤⑥ 需要环境
def obs_contract(env: ManagerBasedRlEnv) -> None:
    _h("⑧ 观测契约: actor / critic 的维度与尾部槽位语义")
    actor = env.observation_manager.active_terms["actor"]
    critic = env.observation_manager.active_terms["critic"]
    print(f"actor ({len(actor)} 项): {list(actor)}")
    print(f"critic ({len(critic)} 项, 多出来的就是特权信息): "
          f"{[t for t in critic if t not in actor]}")
    obs = env.observation_manager.compute()
    print(f"\n实测维度: actor = {tuple(obs['actor'].shape)}  critic = {tuple(obs['critic'].shape)}")
    print(
        "\n⚠️ walk 的 actor 现在恰好也是 51 维, 和站立任务一样 —— 但尾部 3 维语义不同:\n"
        "     站立 actor[48:51] = 实测机体线速度(base_lin_vel, 真机给不出)\n"
        "     walk   actor[48:51] = 速度指令(command)\n"
        "   形状对得上 ⇒ `--agent.resume` 加载站立 checkpoint 不会报错, 但策略会把\n"
        "   '指令'当成'实测速度'读 → 静默错训。要续训就先做 W4(三任务统一契约)。"
    )


def reward_audit(env: ManagerBasedRlEnv, steps: int, seed: int = 0) -> None:
    _h(f"⑨ 奖励体检: {steps} 步逐项加权值(负权重项必须 ≤ 0)")
    torch.manual_seed(seed)
    totals = {name: 0.0 for name in env.reward_manager.active_terms}
    weighted = {name: 0.0 for name in env.reward_manager.active_terms}
    num_envs = env.num_envs
    for i in range(steps):
        # 前半段零动作(站着), 后半段小随机动作(扰动一下, 让接触状态变化起来)
        if i < steps // 2:
            action = torch.zeros(num_envs, 14)
        else:
            action = 0.3 * torch.randn(num_envs, 14)
        env.step(action)
        for name in totals:
            cfg = env.reward_manager.get_term_cfg(name)
            value = cfg.func(env, **cfg.params)
            totals[name] += float(value.mean())
            weighted[name] += float(value.mean()) * cfg.weight

    print(f"{'奖励项':>18} {'权重':>8} {'原始值/步':>12} {'加权值/步':>12}  符号检查")
    bad = []
    for name in totals:
        cfg = env.reward_manager.get_term_cfg(name)
        raw = totals[name] / steps
        w = weighted[name] / steps
        flag = ""
        if cfg.weight < 0 and w > 1e-9:
            flag = "  ❌ 罚项为正(符号反了!)"
            bad.append(name)
        elif cfg.weight < 0:
            flag = "  ✅ ≤0"
        print(f"{name:>18} {cfg.weight:>8.3f} {raw:>12.5f} {w:>12.5f}{flag}")
    print(f"\n正质量合计/步 ≈ {sum(v for n, v in weighted.items() if env.reward_manager.get_term_cfg(n).weight > 0) / steps:.3f}")
    print(f"负质量合计/步 ≈ {sum(v for n, v in weighted.items() if env.reward_manager.get_term_cfg(n).weight < 0) / steps:.3f}")
    if bad:
        print(f"❌ 有罚项符号反了: {bad}")
    else:
        print("✅ 所有负权重项都 ≤ 0(AGENTS 那条 'infallible check' 通过)")
    print(
        "\n注意: 这是**随机/零动作**下的量级, 不是学会走之后的量级 —— 它的用途是确认\n"
        "      '每一项都在动、符号对、量级可见', 以及给训练时的曲线一个静止基线。\n"
        "      真正的判断要看 GPU 训练里 Episode_Reward/<term> 有没有朝想要的方向走。"
    )


# ---------------------------------------------------------------- ⑦ P1 三项
def p1_action_rate_curriculum() -> None:
    _h("⑤ P1: action_rate 的权重阶梯(以及「为什么不照抄上游 -1.0」)")
    from mjlab_microduck.tasks.microduck_walk_cfg import _ACTION_RATE_WEIGHT_STAGES

    print("阶梯(单位: 策略步; 达到某档就取该档权重, 阶梯不插值):")
    for step, w in _ACTION_RATE_WEIGHT_STAGES:
        print(f"   step >= {step:>6}  →  {w:+.2f}")
    print("\n⚠️ 这一档上一版是 -0.05 → -0.55, 依据 500 轮实测重算了: 上一版把 raw(=14 维")
    print("   动作变化平方和)估成 ~1.2(用 0.3×randn 探针量的), 真实策略 raw 是 5~25 →")
    print("   权重 -0.30 时每步 -1.66~-3.0, 占全部罚项 85%, 而它恰好在鸭子尝试迈步的")
    print("   窗口(125~250 轮)加压 → 实测'有脚离地'从 32% 掉到 0.9%(策略学会别动)。")
    # "占正质量的份额"对照: 上游正质量按它们自己的权重表算, 本项目按当前配置算
    up_pos = 2.0 + 2.0 + 2.0 + 1.5 + 1.0 + 2.0   # track_lin/ang + upright + air_time + pose + head_pose
    mine_pos = 2.0 + 1.0 + 2.0 + 1.5             # track_lin + track_ang + upright + air_time
    up_final, mine_final = -1.0, _ACTION_RATE_WEIGHT_STAGES[-1][1]
    print(f"\n上游: 正质量 ≈ {up_pos:.2f}/步, 平滑税 {up_final:+.2f} → 占 {abs(up_final) / up_pos:.1%}")
    print(f"本项目: 正质量 ≈ {mine_pos:.2f}/步, 平滑税 {mine_final:+.2f} → 占 {abs(mine_final) / mine_pos:.1%}")
    print(
        "\n结论: 份额对齐只是第一层; 第二层必须按**实测的每步量级**换算成权重。"
        "\n      实测末轮 action_rate raw≈5.5 → 权重 "
        f"{mine_final:+.2f} 只占正质量 {abs(mine_final) * 5.5 / mine_pos:.1%}(目标 3~9%)。"
    )


def p1_dynamic_regularizers(env: ManagerBasedRlEnv, steps: int = 120) -> None:
    _h("⑦ P1: body_ang_vel / angular_momentum 两项动态正则的实测量级")
    robot = env.scene["robot"]
    w_zero, w_rand, l_zero, l_rand = [], [], [], []
    for i in range(steps):
        if i < steps // 2:
            env.step(torch.zeros(env.num_envs, 14))
            w_zero.append(float((robot.data.body_link_ang_vel_w[:, 0, :2] ** 2).sum(-1).mean()))
            l_zero.append(float(env.scene["robot/root_angmom"].data.norm(dim=-1).mean()))
        else:
            env.step(0.5 * torch.randn(env.num_envs, 14))
            w_rand.append(float((robot.data.body_link_ang_vel_w[:, 0, :2] ** 2).sum(-1).mean()))
            l_rand.append(float(env.scene["robot/root_angmom"].data.norm(dim=-1).mean()))
    term_w = env.reward_manager.get_term_cfg("body_ang_vel")
    term_l = env.reward_manager.get_term_cfg("angular_momentum")
    print(f"{'量':>22} {'零动作(中位)':>14} {'随机乱蹬(中位)':>16} {'权重':>8} {'加权/步':>12}")
    for name, a, b, term in (
        ("躯干 |ω_xy|²", w_zero, w_rand, term_w),
        ("整机 |L| (kg·m²/s)", l_zero, l_rand, term_l),
    ):
        ma, mb = float(torch.tensor(a).median()), float(torch.tensor(b).median())
        print(f"{name:>22} {ma:>14.5f} {mb:>16.5f} {term.weight:>8.3f} "
              f"{abs(term.weight) * mb ** 2 if 'L' in name else abs(term.weight) * mb:>12.6f}")
    print(
        "\n结论(两条, 第二条是「量了才知道」的):\n"
        "  · body_ang_vel: 正常走路 ~0.2% 正质量, 踉跄时才涨到几个百分点 —— 平时不干扰、\n"
        "    出事才咬, 符合 motion-blocker 该有的样子(权重别再加了, 加大会让策略变保守)。\n"
        "  · angular_momentum: |L| 只有 0.003~0.008(鸭子太小, L ∝ 质量×尺度²), 平方后乘 -0.01\n"
        "    是 1e-7 量级 —— **等于没接**。上游同权重在本机上也是这个量级。留着只为对齐上游\n"
        "    和那条 Metrics/angular_momentum_mean 曲线; 想要它真有影响, 得先处理量纲。\n"
        "    教训: 加正则前先量一下它到底有多大, 别用'加了正则所以更稳'骗自己。"
    )


def main() -> None:  # noqa: D103
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    probe_geometry()
    action_scale_table()
    do_nothing_shortcut()
    upright_table()
    air_time_table()
    p1_action_rate_curriculum()

    cfg = microduck_walk_env_cfg()
    cfg.scene.num_envs = args.envs
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    probe_clearance_in_env(env)
    progress_gate_and_stance(env)
    posture_gate_table(env)
    p1_dynamic_regularizers(env, max(40, args.steps // 2))
    obs_contract(env)
    reward_audit(env, args.steps)


if __name__ == "__main__":
    main()
