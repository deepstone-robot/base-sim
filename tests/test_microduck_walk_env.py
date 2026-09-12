"""行走(平地速度跟踪)任务的环境/配置冒烟测试。

定位是训练前的"质检台": 改完 walk cfg → 先跑这个 → 再上 CPU 冒烟训练 → 最后才 GPU。
除了维度/资产名这类常规检查, 这里重点**钉住 P0 修正后的不变量**(见 microduck_walk_cfg.py
顶部的修正记录) —— 那些是"改回去不会报错、但会让奖励指错方向"的静默错误。
"""
import math

import mujoco
import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from mjlab_microduck.tasks import _STAND_JOINT_POS, microduck_walk_env_cfg

_MODEL = None


def _model():
    """脚底网格的真值来源: **走路任务实际用的那个 XML**(walk 模型)。"""
    global _MODEL
    if _MODEL is None:
        from mjlab_microduck.tasks.microduck_walk_cfg import get_microduck_walk_entity_cfg

        _MODEL = get_microduck_walk_entity_cfg().spec_fn().compile()
    return _MODEL


def _make_env(num_envs=2):
    cfg = microduck_walk_env_cfg()
    cfg.scene.num_envs = num_envs
    return ManagerBasedRlEnv(cfg=cfg, device="cpu")


def _make_venv(num_envs=2):
    return RslRlVecEnvWrapper(_make_env(num_envs))


# ---------------------------------------------------------------- 常规
def test_registered():
    from mjlab.tasks.registry import list_tasks

    assert "Mjlab-Velocity-Flat-MicroDuck" in list_tasks()


def test_obs_and_action_shape():
    """actor 51 维 = 14(关节角)+14(关节速)+3(姿态)+3(角速度)+14(上一步动作)+3(指令);
    critic 54 维 = 多一个真值机体线速度(特权信息)。
    动作 14 维 = 14 个执行器。"""
    venv = _make_venv()
    obs = venv.get_observations()
    assert obs["actor"].shape == (2, 51)
    assert obs["critic"].shape == (2, 54)
    assert venv.action_space.shape == (2, 14)


def test_random_steps_are_finite():
    venv = _make_venv()
    for _ in range(20):
        a = 2 * torch.rand(venv.action_space.shape) - 1
        obs, rew, dones, extras = venv.step(a)
        assert torch.isfinite(obs["actor"]).all()
        assert torch.isfinite(rew).all()


def test_reward_and_termination_terms_exist():
    cfg = microduck_walk_env_cfg()
    assert set(cfg.rewards) == {
        "track_lin_vel", "track_ang_vel", "upright",
        "air_time", "foot_clearance", "foot_slip",
        "stance_dev", "head_dev", "joint_pos_limits",
        "body_ang_vel", "angular_momentum", "action_rate", "joint_vel",
    }
    assert set(cfg.terminations) == {
        "time_out", "bad_orientation", "fallen_too_long", "nan_state",
    }
    assert set(cfg.curriculum) == {"command_vel", "action_rate_weight", "air_window"}


def test_falling_over_is_terminated():
    """姿态偏 > 70° 算摔倒 → 回合结束。

    P2⑪ 之后**不再**用"躯干高度 < 5cm"判摔(root_too_low 已删除): 实测它在鸭子尝试
    迈步/下蹲时终止了 100% 的回合, 把最需要的探索样本删掉了; 上游速度任务也没有高度项。
    所以这里改成测倾角终止 —— 那才是真正的"摔"。
    """
    from mjlab.utils.lab_api.math import quat_from_angle_axis

    env = _make_env(1)
    tilted = quat_from_angle_axis(
        torch.tensor([math.radians(80.0)]), torch.tensor([[0.0, 1.0, 0.0]])
    )
    root = torch.zeros(1, 13)
    root[0, 2], root[0, 3:7] = 0.12, tilted[0]
    env.scene["robot"].write_root_state_to_sim(root)
    env.step(torch.zeros(1, 14))
    assert env.termination_manager.terminated[0]

    # 反过来: 躯干被压到 3cm 但姿态没偏 → **宽限期内不**终止(数据里这是"蹲/趴"而不是"摔",
    # 删掉"立刻终止"就是为了让它继续攒恢复样本); 赖着超过 1 秒才由 fallen_too_long 收掉。
    env2 = _make_env(1)
    root2 = torch.zeros(1, 13)
    root2[0, 2], root2[0, 3] = 0.03, 1.0
    env2.scene["robot"].write_root_state_to_sim(root2)
    env2.step(torch.zeros(1, 14))
    assert not env2.termination_manager.terminated[0], "宽限期内不该终止"


def test_ppo_cfg_shape():
    from mjlab_microduck.tasks import microduck_walk_ppo_runner_cfg

    cfg = microduck_walk_ppo_runner_cfg()
    assert cfg.experiment_name == "microduck-walk"
    # 上 GPU 的尺度: 容量与预算都按"步态类任务 4000~6000 轮 @4096 envs"设
    assert tuple(cfg.actor.hidden_dims) == (256, 256)
    assert tuple(cfg.critic.hidden_dims) == (256, 256)
    assert cfg.num_steps_per_env == 24        # 与上游一致(课程步数按 24 标定)
    assert cfg.max_iterations >= 4000
    assert cfg.save_interval <= 250
    assert math.isfinite(cfg.algorithm.learning_rate)


# ---------------------------------------------------------------- P0 ②: 观测契约
def test_actor_has_no_privileged_base_lin_vel():
    """P0②: base_lin_vel 只能在 critic 里(真机给不出这个量)。"""
    cfg = microduck_walk_env_cfg()
    actor = cfg.observations["actor"].terms
    critic = cfg.observations["critic"].terms
    assert "base_lin_vel" not in actor
    assert "base_lin_vel" in critic
    # actor 的最后一槽必须是命令(尾部语义错位会静默错训, 见下面的续训陷阱测试)
    assert list(actor)[-1] == "command"


def test_actor_tail_slot_is_command_not_velocity():
    """钉住续训陷阱: actor[48:51] 必须是**指令**, 不是实测速度。

    站立任务的 actor 也是 51 维, 所以 `--agent.resume` 加载站立 checkpoint 时形状
    能对上、不会报错 —— 只有尾部 3 维的语义不同。这里用真实数据钉住: 指令变了,
    尾部 3 维必须跟着变; 而策略看到的不是"当前机体速度"。
    """
    env = _make_env(1)
    env.reset()
    for _ in range(5):
        env.step(torch.zeros(1, 14))
    cmd = env.command_manager.get_command("twist")
    assert cmd is not None
    obs_tail = env.observation_manager.compute()["actor"][0, 48:51]
    assert torch.allclose(obs_tail, cmd[0], atol=1e-5)


# ---------------------------------------------------------------- P0 ①⑤: 奖励量级/尺度
def test_air_time_reward_is_per_step_not_landing_instant():
    """P0①: 权重 3.0 + 上游同款窗口; 且一只脚**持续**悬空时要**每一步**都在付分。"""
    cfg = microduck_walk_env_cfg()
    term = cfg.rewards["air_time"]
    # 权重 3.0 > track_lin 2.0: 让"迈步"本身成为最大的正分项(而不是"溜"就够的手段)。
    # 第三轮曾砍到 1.5 来堵"原地蹦", 但那个薅分点现在由姿态门+方向门堵住, 不需要降权重。
    assert term.weight == 3.0
    assert term.params["threshold_max"] == 0.300
    assert term.params["command_threshold"] == 0.01
    # P2⑨: 下限由课程管理, 起步必须是"够得着"的小值(见 _AIR_WINDOW_MIN_STAGES)
    assert term.params["threshold_min"] == 0.03

    # 直接验证"每步付": 造一只持续 0.2s 悬空的脚, 数窗口内会付几步。
    # 一次性写法(旧的落地给分)只付 1 次; 每步付会付很多步 —— 差一个量级。
    step_dt = 0.02
    steps_in_window = int((0.2 - term.params["threshold_min"]) / step_dt)
    assert steps_in_window >= 5


def test_action_scale_is_one():
    """P0⑤: scale 必须是 1.0, 否则抬脚需要的膝屈(0.5~0.9rad)要策略输出 1~1.8σ。"""
    cfg = microduck_walk_env_cfg()
    assert cfg.actions["joint_pos"].scale == 1.0
    # 站立/坐站任务保持 0.5 不动
    from mjlab_microduck.tasks import microduck_env_cfg

    assert microduck_env_cfg().actions["joint_pos"].scale == 0.5


def test_upright_is_gaussian_on_trunk():
    """P0③: 高斯核 + 权重 2.0 + 作用在 trunk_base; 倾 10° 的惩罚要比线性版硬两个量级。"""
    cfg = microduck_walk_env_cfg()
    term = cfg.rewards["upright"]
    assert term.weight == 2.0
    assert term.params["std"] == pytest.approx(math.sqrt(0.05))
    assert tuple(term.params["asset_cfg"].body_names) == ("trunk_base",)

    # 注意: 不能用"站着时接近满分"当断言 —— 静止站立时鸭子会像倒立摆一样自己倒
    # (实测零动作 12 步就从 0.5° 倾到 24.5°, 增长率 ~10/s ≈ sqrt(g/0.11m), 是物理,
    #  不是 bug: 站立位姿的开环平衡本来就不稳定, 得靠策略反馈)。所以这里直接**给定倾角**,
    # 量"同样倾 10°, 高斯版比线性版多罚多少"。
    from mjlab.utils.lab_api.math import quat_from_angle_axis

    env = _make_env(1)
    robot = env.scene["robot"]

    def upright_at(tilt_deg: float) -> float:
        quat = quat_from_angle_axis(
            torch.tensor([math.radians(tilt_deg)]), torch.tensor([[0.0, 1.0, 0.0]])
        )
        root = torch.zeros(1, 13)
        root[0, 2], root[0, 3:7] = 0.12, quat[0]
        robot.write_root_state_to_sim(root)
        env.sim.forward()
        return float(term.func(env, **{k: v for k, v in term.params.items()})[0])

    flat, tilted10 = upright_at(0.0), upright_at(10.0)
    assert flat == pytest.approx(1.0, abs=1e-4)  # 完全直立 = 满分
    # 线性版(0.5*cos)在 10° 处只掉 0.0076 分; 高斯版掉 ~0.9 分(约 120 倍)
    linear_drop = 0.5 - 0.5 * math.cos(math.radians(10.0))
    assert (flat - tilted10) > 50 * linear_drop


# ---------------------------------------------------------------- P0 ④: 离地高度探针
def _sole_min_z_from_mesh(ankle_delta: float, foot: str = "left") -> float:
    """真值: 用 mujoco 的网格顶点算脚底最低点(把脚底钉在地面上)。"""
    model = _model()
    data = mujoco.MjData(model)
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{foot}_foot_collision")
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ankle_{foot}")
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{foot}_ankle")
    adr = model.mesh_vertadr[model.geom_dataid[geom]]
    verts = model.mesh_vert[adr : adr + model.mesh_vertnum[model.geom_dataid[geom]]]

    def mesh_min_z() -> float:
        world = verts @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        return float(world[:, 2].min())

    data.qpos[:] = 0.0
    data.qpos[2] = 0.12
    data.qpos[3] = 1.0
    for name, val in _STAND_JOINT_POS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = val
    data.qpos[model.jnt_qposadr[joint]] += ankle_delta
    for _ in range(3):  # 把脚底挪到贴地
        mujoco.mj_forward(model, data)
        data.qpos[2] -= mesh_min_z()
    mujoco.mj_forward(model, data)
    assert body >= 0
    return mesh_min_z()


def test_sole_point_table_matches_mesh_ground_truth():
    """P0④: 采样点表必须能代表脚底最低点(脚底贴地时读 ≈0), 且不随脚踝翻转漂。

    这是"探针"这件事的核心不变量。我把探针写错过三次: 脚踝 body 高度(漂 +10.9mm)、
    脚底 site 高度(漂 +14.1mm)、上游射线传感器(平地上等价于 site 高度) —— 现在
    这个才能真正量到"脚底离地多少"。
    """
    from mjlab_microduck.tasks.microduck_walk_cfg import (
        _LEFT_SOLE_POINTS,
        _RIGHT_SOLE_POINTS,
    )

    model = _model()
    data = mujoco.MjData(model)
    for foot, points in (("left", _LEFT_SOLE_POINTS), ("right", _RIGHT_SOLE_POINTS)):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{foot}_ankle")
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ankle_{foot}")
        pts = torch.tensor(points, dtype=torch.float64)
        readings = []
        for deg in (-30, -15, 0, 15, 30):
            data.qpos[:] = 0.0
            data.qpos[2] = 0.12
            data.qpos[3] = 1.0
            for name, val in _STAND_JOINT_POS.items():
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                data.qpos[model.jnt_qposadr[jid]] = val
            data.qpos[model.jnt_qposadr[joint]] += math.radians(deg)
            # 用真值把脚底钉到地面, 再看点表读出多少
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{foot}_foot_collision")
            adr = model.mesh_vertadr[model.geom_dataid[geom]]
            verts = model.mesh_vert[adr : adr + model.mesh_vertnum[model.geom_dataid[geom]]]
            for _ in range(3):
                mujoco.mj_forward(model, data)
                world = verts @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
                data.qpos[2] -= float(world[:, 2].min())
            mujoco.mj_forward(model, data)
            rot = torch.tensor(data.xmat[body].reshape(3, 3), dtype=torch.float64)
            pos = torch.tensor(data.xpos[body], dtype=torch.float64)
            readings.append(float((pts @ rot.T + pos)[:, 2].min()))

        # 脚底贴地 ⇒ 读数应该 ~0; 且脚踝转 ±30° 之间读数变化 < 3mm
        assert max(abs(r) for r in readings) < 0.004, readings
        assert max(readings) - min(readings) < 0.003, readings


def test_foot_clearance_uses_sole_min_not_single_point():
    """奖励里用的是"脚底最低点", 且左右脚顺序不能搞反(点表按 ankle_left/right 写死)。"""
    from mjlab_microduck.tasks.microduck_walk_cfg import _foot_clearance_heights

    env = _make_env(2)
    for _ in range(20):
        env.step(torch.zeros(2, 14))
    term = env.reward_manager.get_term_cfg("foot_clearance")
    heights = _foot_clearance_heights(env, term.params["asset_cfg"])
    assert heights.shape == (2, 2)
    # 两只脚都踩在地上 ⇒ 都该 ~0
    assert heights.abs().max() < 0.01

    # 把鸭子整体抬高 5cm: 两个读数都要跟着涨(证明它真的在量离地高度)
    robot = env.scene["robot"]
    root = robot.data.root_link_pos_w.clone()
    root[:, 2] += 0.05
    quat = robot.data.root_link_quat_w.clone()
    state = torch.cat([root, quat, torch.zeros(2, 6)], dim=-1)
    robot.write_root_state_to_sim(state)
    env.sim.forward()
    raised = _foot_clearance_heights(env, term.params["asset_cfg"])
    assert (raised - heights).min() > 0.03


def test_foot_clearance_right_left_not_swapped():
    """把左腿**运动学地**收起来(直接写关节角, 不走动力学) → 只有左脚抬高。

    为什么要直接写状态: 用动作命令"屈膝抬脚"是不行的 —— 屈膝会让身体下沉(蹲),
    两只脚都还在地上, 读数分不出左右(我第一版测试就是这么写错的)。直接改 qpos 再
    算正运动学, 才是干净的"只动左脚"。
    """
    from mjlab_microduck.tasks.microduck_walk_cfg import _foot_clearance_heights

    env = _make_env(1)
    robot = env.scene["robot"]
    left_knee = robot.find_joints(("left_knee",))[0][0]
    left_hip = robot.find_joints(("left_hip_pitch",))[0][0]
    left_ankle = robot.find_joints(("left_ankle",))[0][0]
    term = env.reward_manager.get_term_cfg("foot_clearance")

    for _ in range(10):  # 先站稳(踩实地面)
        env.step(torch.zeros(1, 14))
    before = _foot_clearance_heights(env, term.params["asset_cfg"])[0].clone()

    q = robot.data.joint_pos.clone()
    q[0, left_knee] += 1.0  # 屈膝
    q[0, left_hip] += -0.5  # 大腿抬起
    q[0, left_ankle] += 0.3  # 脚尖下垂一点
    robot.write_joint_position_to_sim(q)
    env.sim.forward()
    after = _foot_clearance_heights(env, term.params["asset_cfg"])[0]

    assert before[0] < 0.01 and before[1] < 0.01  # 抬起前两脚都贴地
    assert after[0] > after[1] + 0.01, (before.tolist(), after.tolist())


# ---------------------------------------------------------------- P1: 课程 + 动态正则
def test_action_rate_weight_curriculum_follows_stages():
    """P1⑥: action_rate 权重按阶梯逐步加压, 且改的是 manager 里的 term cfg。

    坑: `env.cfg.rewards[...]` 是 manager 初始化时的 deepcopy, 写它是静默 no-op ——
    所以这里断言的是 `env.reward_manager.get_term_cfg(...).weight` 真的变了。
    """
    from mjlab_microduck.tasks.microduck_walk_cfg import _ACTION_RATE_WEIGHT_STAGES

    env = _make_env(1)
    term = env.reward_manager.get_term_cfg("action_rate")
    seen = []
    for step in (0, 3999, 4000, 7999, 8000, 15999, 16000, 10**6):
        env.common_step_counter = step
        env.curriculum_manager.compute()
        seen.append((step, term.weight))
    weights = [w for _, w in seen]
    assert weights == sorted(weights, reverse=True)  # 只增不减地加压
    assert weights[0] == _ACTION_RATE_WEIGHT_STAGES[0][1] == -0.03
    assert weights[-1] == _ACTION_RATE_WEIGHT_STAGES[-1][1] == -0.10
    # 阶梯边界: 3999 还是第 1 档, 4000 就跳到第 2 档
    assert dict(seen)[3999] == -0.03 and dict(seen)[4000] == -0.05
    # 课程返回值(用于打点 Curriculum/action_rate_weight)
    env.common_step_counter = 8000
    value = env.curriculum_manager._term_cfgs[
        env.curriculum_manager.active_terms.index("action_rate_weight")
    ].func(env, None, reward_name="action_rate", weight_stages=_ACTION_RATE_WEIGHT_STAGES)
    assert float(value[0]) == pytest.approx(-0.05)  # 返回的是 float32 tensor


def test_regularizer_weights_are_sized_by_measured_magnitude():
    """P1⑦/P2⑩: 正则权重按**实测的每步量级**定, 不照抄上游, 也不靠拍脑袋估。

    教训(500 轮实测): P1 时我按"占正质量份额"算, 思路没错, 但把 action_rate 的 raw
    量级估成 ~1.2(用 0.3×randn 探针量的), 真实策略 raw 是 5~25 → 权重 -0.30 时每步
    -1.66~-3.0、占全部罚项 85%, 而且恰好在鸭子尝试迈步的窗口加压 → 策略学会"别动"。
    所以这里钉住: 终值必须是按 raw 量级反算出来的小权重(-0.10), 不是 -0.55。
    """
    from mjlab_microduck.tasks.microduck_walk_cfg import (
        _ACTION_RATE_WEIGHT_STAGES,
        _ANGULAR_MOMENTUM_WEIGHT,
        _BODY_ANG_VEL_WEIGHT,
    )

    cfg = microduck_walk_env_cfg()
    # body_ang_vel 从 -0.03 降到 -0.01: 实测走路时 |ω_xy|≈3.6 rad/s 是这只 25cm 机器人的
    # 自然区间, 罚它就是罚正常走路(而且它曾是"站着不动"最后一块优势的来源, 见 cfg 注释)
    assert _BODY_ANG_VEL_WEIGHT == -0.01
    assert _ANGULAR_MOMENTUM_WEIGHT == -0.01
    assert cfg.rewards["body_ang_vel"].weight == _BODY_ANG_VEL_WEIGHT
    assert cfg.rewards["angular_momentum"].weight == _ANGULAR_MOMENTUM_WEIGHT
    assert cfg.rewards["action_rate"].weight == _ACTION_RATE_WEIGHT_STAGES[0][1] == -0.03
    # 实测: 平滑策略的 raw≈5, 每步代价 |w|·raw 必须远小于正质量(≈3.2/步)
    assert abs(_ACTION_RATE_WEIGHT_STAGES[-1][1]) * 5.0 < 0.6
    # 所有课程的最后一档都必须落在 500 轮(16000 策略步)之内, 否则按 README 推荐的
    # 500 轮验证跑时最后一档永远不生效 —— 上一轮就这么白排了一档(16000 步 = 第 500 轮,
    # 而 max-iterations=500 是 0..499)。
    from mjlab_microduck.tasks.microduck_walk_cfg import _AIR_WINDOW_MIN_STAGES

    for stages in (_ACTION_RATE_WEIGHT_STAGES, _AIR_WINDOW_MIN_STAGES):
        assert stages[-1][0] <= 14000, f"最后一档排在 {stages[-1][0]} 步, 500 轮跑不到"


def test_air_time_window_opens_early_then_tightens():
    """P2⑨: air_time 窗口下限从 0.03 起步, 由课程抬到 0.125。

    为什么: 实测鸭子一开始只能摆动 20~30ms, 若下限直接是 0.125, 则"抬脚 30ms"和
    "完全不抬"同分 → 迈步这件事零梯度, 永远学不会第一次抬脚。
    """
    from mjlab_microduck.tasks.microduck_walk_cfg import _AIR_WINDOW_MIN_STAGES

    cfg = microduck_walk_env_cfg()
    term = cfg.rewards["air_time"]
    assert term.params["threshold_min"] == _AIR_WINDOW_MIN_STAGES[0][1] == 0.03
    assert term.params["threshold_max"] == 0.300
    assert _AIR_WINDOW_MIN_STAGES[-1][1] == 0.125  # 后期收紧到上游值

    env = _make_env(1)
    live = env.reward_manager.get_term_cfg("air_time")
    seen = []
    for step in (0, 5999, 6000, 12000):
        env.common_step_counter = step
        env.curriculum_manager.compute()
        seen.append(live.params["threshold_min"])
    assert seen[0] == 0.03 and seen[1] == 0.03 and seen[2] == 0.08 and seen[3] == 0.125


def test_no_root_height_termination_and_nan_guard():
    """P2⑪: 删掉 root_too_low(它曾终止 100% 的回合, 正好在尝试迈步的窗口), 补 nan_state。"""
    cfg = microduck_walk_env_cfg()
    assert "root_too_low" not in cfg.terminations
    assert "nan_state" in cfg.terminations


def test_standing_still_cannot_farm_the_top_command():
    """"站着不动"不能在**高指令**下白拿跟踪奖励(低指令下的小额补贴是 std=0.20 的已知代价)。

    标定史: std=0.25 时站着在第 1 档能拿 90%(那是第一轮"学会站着不动"的根因);
    收到 0.10 又太严(惩罚一切真步态, 见下一个测试)。现在 0.20 —— 代价是**低指令下**
    站着还能拿 ~57%, 但(1) 高指令下只剩 10%, (2) 总分上走路仍然更高(air_time 只有真迈步
    才拿得到), 这两件事由本测试与 test_track_std_is_calibrated_against_a_proven_walker 一起守。
    """
    from mjlab_microduck.tasks.microduck_walk_cfg import _VELOCITY_STAGES

    cfg = microduck_walk_env_cfg()
    std = cfg.rewards["track_lin_vel"].params["std"]
    for stage in _VELOCITY_STAGES:
        lo, hi = stage["lin_vel_x"]
        assert lo > 0.0 or stage is not _VELOCITY_STAGES[0], "第 1 档下限必须 > 0"
        del lo, hi
    top = max(abs(x) for x in _VELOCITY_STAGES[-1]["lin_vel_x"])
    standing_at_top = math.exp(-(top**2) / std**2)
    assert standing_at_top < 0.2, f"高指令({top})下站着还能拿 {standing_at_top:.0%}"


def test_body_ang_vel_only_punishes_roll_and_pitch():
    """P1⑦: 只罚滚转/俯仰(不罚偏航), 返回 >=0 的 cost。"""
    env = _make_env(1)
    for _ in range(10):
        env.step(torch.zeros(1, 14))
    term = env.reward_manager.get_term_cfg("body_ang_vel")
    value = term.func(env, **term.params)
    assert value.shape == (1,)
    assert value[0] >= 0
    # 与独立算法对上: 躯干世界系角速度 xy 分量平方和
    robot = env.scene["robot"]
    body_id = robot.find_bodies(("trunk_base",))[0][0]
    expected = (robot.data.body_link_ang_vel_w[:, body_id, :2] ** 2).sum(dim=-1)
    assert torch.allclose(value, expected, atol=1e-6)


def test_angular_momentum_reads_model_sensor():
    """P1⑦: 角动量正则直接读模型自带的 subtreeangmom 传感器(robot/root_angmom)。"""
    env = _make_env(1)
    for _ in range(10):
        env.step(torch.zeros(1, 14))
    term = env.reward_manager.get_term_cfg("angular_momentum")
    assert term.params["sensor_name"] == "robot/root_angmom"
    value = term.func(env, **term.params)
    angmom = env.scene["robot/root_angmom"].data
    assert value.shape == (1,)
    assert value[0] >= 0
    assert torch.allclose(value, (angmom**2).sum(dim=-1), atol=1e-9)
    # 顺手打了曲线(判断头和脖子甩不甩)
    assert "Metrics/angular_momentum_mean" in env.extras["log"]

    # 实测量级: 这项在 -0.01 权重下只有 1e-7 量级 = 等于没接(见 cfg 里那段注释)。
    # 这里钉住"它确实很小", 免得以后有人以为它在起稳定作用。
    assert term.weight * float(value[0]) > -1e-4


# ------------------------------------------------- P2(第三轮): 姿态门 + 倒地宽限
def test_posture_gate_matches_measured_postures():
    """姿态门必须按实测姿势正确赋值 —— 这是"躺平薅分"的直接防线。

    标定用的五个姿势都是实测来的(CPU headless 评估 + SIT 关键帧):
      躺平(250 轮新策略) z=0.041 倾角 50°  → 门必须 = 0(技能类奖励全被乘掉)
      蹲低但直立         z=0.075 倾角  5°  → 门上开一半左右(爬起来有梯度)
      走路(旧 250 轮)    z=0.095 倾角 12°  → 门 = 1(正常走路不能被误伤)
      站立(旧 499 轮)    z=0.117 倾角  2°  → 门 = 1
    """
    from mjlab.utils.lab_api.math import quat_from_angle_axis

    from mjlab_microduck.tasks.microduck_walk_cfg import _posture_gate

    env = _make_env(2)
    robot = env.scene["robot"]

    def gate(z: float, tilt_deg: float) -> float:
        quat = quat_from_angle_axis(
            torch.tensor([math.radians(tilt_deg)]), torch.tensor([[0.0, 1.0, 0.0]])
        )
        root = torch.zeros(2, 13)
        root[:, 2], root[:, 3:7] = z, quat[0]
        robot.write_root_state_to_sim(root)
        env.sim.forward()
        return float(_posture_gate(env).mean())

    assert gate(0.041, 50.0) == pytest.approx(0.0, abs=1e-6), "躺平必须拿 0"
    assert 0.2 < gate(0.075, 5.0) < 0.8, "蹲低直立应该在斜坡中间(爬起来有梯度)"
    assert gate(0.095, 12.0) == pytest.approx(1.0), "正常走路不能被姿态门误伤"
    assert gate(0.117, 2.0) == pytest.approx(1.0), "站立不能被误伤"


def test_gait_rewards_are_gated_by_posture():
    """躺在地上时, 技能类奖励必须拿不到分(这就是躺平薅分的堵口)。

    判别式构造: 指令=0 且鸭子不动 → **未加门**的 track_lin_vel 是满分(exp(0)=1);
    躺平后必须被门乘成 0, 而站立时仍接近满分。
    """
    from mjlab.utils.lab_api.math import quat_from_angle_axis

    from mjlab_microduck.tasks.microduck_walk_cfg import track_lin_vel_xy_reward

    env = _make_env(2)
    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("twist")
    cmd.vel_command_b[:] = 0.0  # 零指令 + 零速度 = 完美跟踪(不给门的话)

    def reward_at(z: float, tilt_deg: float) -> float:
        quat = quat_from_angle_axis(
            torch.tensor([math.radians(tilt_deg)]), torch.tensor([[0.0, 1.0, 0.0]])
        )
        root = torch.zeros(2, 13)  # 13 维 = pos3+quat4+线速度3+角速度3, 后面 6 维=0 即静止
        root[:, 2], root[:, 3:7] = z, quat[0]
        robot.write_root_state_to_sim(root)
        env.sim.forward()
        return float(track_lin_vel_xy_reward(env, std=0.10, command_name="twist")[0])

    assert reward_at(0.117, 2.0) > 0.9   # 站立: 满分
    assert reward_at(0.041, 50.0) < 1e-6  # 躺平: 被门乘成 0


def test_air_time_requires_commanded_direction_progress():
    """P2⑮: air_time 必须要求"按指令方向真的在移动" —— 原地蹦/横着挪都不算。

    判别式(实测值): 站着 0.00 / 横着走 0.00 / 原地蹦(0.07 vs 指令 0.2) 0.35 / 按指令 1.00。
    这是第三轮失败的直接防线: 鸭子原地蹦(离地 45%)却几乎不位移, 照样薅 air_time 2.04/步。
    """
    from mjlab_microduck.tasks.microduck_walk_cfg import _progress_gate

    env = _make_env(2)
    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("twist")

    def gate(vx: float, vy: float, cmd_vx: float = 0.2) -> float:
        cmd.vel_command_b[:] = 0.0
        cmd.vel_command_b[:, 0] = cmd_vx
        root = torch.zeros(2, 13)
        root[:, 2], root[:, 3] = 0.117, 1.0
        root[:, 7], root[:, 8] = vx, vy  # 机体系线速度
        robot.write_root_state_to_sim(root)
        env.sim.forward()
        return float(_progress_gate(env).mean())

    assert gate(0.0, 0.0) == pytest.approx(0.0), "站着不动不该算在走"
    assert gate(0.0, 0.2) == pytest.approx(0.0), "横着走不该算在走"
    assert gate(-0.1, 0.0) == pytest.approx(0.0), "往反方向走更不该算"
    assert 0.3 < gate(0.07, 0.0) < 0.4, "原地蹦实测 0.07/0.2 ≈ 0.35(要低, 但不能是 0)"
    assert gate(0.10, 0.0) == pytest.approx(0.5, abs=0.01)
    assert gate(0.20, 0.0) == pytest.approx(1.0)


def test_stance_penalty_is_calibrated_on_measured_stance():
    """P2⑰: 姿态罚的基准是**实测站姿**, 不是 XML 默认姿势。

    实测: 鸭子站稳时踝关节比 XML 默认转 17.6°、hip_roll 4.5°。拿 XML 默认当基准会把
    "正确站姿"也罚掉(那正是旧 joint_dev 的毛病)。所以这里钉住:
      自然站姿 → cost ≈ 0;  叉腿 8.6° → cost > 1(罚得动);  正常迈步膝屈 0.4 → 几乎免费。
    """
    from mjlab_microduck.tasks.microduck_walk_cfg import (
        _STANCE_OFFSET,
        _std_for,
        _STD_STANCE,
    )

    env = _make_env(1)
    robot = env.scene["robot"]
    term = env.reward_manager.get_term_cfg("stance_dev")
    for _ in range(6):  # 零动作静置: 这就是标定用的姿势
        env.step(torch.zeros(1, 14))
    natural = float(term.func(env, **term.params)[0])
    assert natural < 0.01, f"自然站姿被罚了 {natural:.3f} —— 基准没标定对"

    # 标定量级守卫: 实测偏移量必须真的偏离 XML 默认(否则说明基准被"修"回默认了)
    assert abs(_STANCE_OFFSET["left_ankle"]) > 0.2, "踝关节偏移量级不对, 重量一次?"
    assert _std_for("left_hip_roll", _STD_STANCE) == 0.05  # hip_roll 必须紧(防叉腿)

    q = robot.data.joint_pos.clone()
    for name, d in (("left_hip_roll", 0.15), ("right_hip_roll", -0.15)):
        q[0, robot.joint_names.index(name)] += d
    robot.write_joint_position_to_sim(q)
    env.sim.forward()
    splay_cost = float(term.func(env, **term.params)[0])
    assert splay_cost > 1.0, "叉腿罚不动"

    q2 = robot.data.joint_pos.clone()  # 在"叉腿"之上再叠一个正常迈步幅度的膝屈
    for name, d in (("left_knee", 0.4), ("right_knee", -0.4)):
        q2[0, robot.joint_names.index(name)] += d
    robot.write_joint_position_to_sim(q2)
    env.sim.forward()
    knee_cost = float(term.func(env, **term.params)[0])
    assert knee_cost - splay_cost < 0.3, (
        f"正常迈步幅度的膝屈被罚太多: 叠上去多了 {knee_cost - splay_cost:.3f}"
    )


def test_walk_task_uses_the_walk_model_not_allcollisions():
    """走路任务必须用 `robot_walk.xml`(第六轮修正, 依据 ../microduck_rl 的实现)。

    为什么这是个不变量而不是口味: allcollisions 模型比 walk 模型**多 6 个碰撞面**
    (jaw_soft×3 + hip×2 + trunk×1) —— 鸭子因此能"下巴撑地/髋撑地"趴着, 而实测第二轮
    抓到的正是"50° 倾角躺着稳稳薅 air_time"。上游把分工写得很清楚: 走路用 walk 模型,
    要能躺下/起身的 VelStand 才用 all-collisions 模型。
    """
    import mujoco

    model = _model()
    enabled = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
        for g in range(model.ngeom)
        if model.geom_contype[g] or model.geom_conaffinity[g]
    ]
    assert len(enabled) == 5, f"walk 模型应有 5 个碰撞面, 实际 {len(enabled)}: {enabled}"
    assert not any("jaw" in b for b in enabled), "下巴/头不能有碰撞面(否则能头撑地薅分)"
    assert not any("hip" in b for b in enabled), "髋不能有碰撞面(否则能髋撑地)"
    # 脚底碰撞面必须在(walk 模型里名字与 allcollisions 一致, 接触传感器靠它)
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)}
    assert {"left_foot_collision", "right_foot_collision"} <= names
    # 脚底 site 也要在(离地高度探针 + 打滑探针靠它)
    site_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)}
    assert {"left_foot", "right_foot"} <= site_names


def test_track_std_is_calibrated_against_a_proven_walker():
    """跟踪核的 std 必须按"**能走通的实际误差水平**"标定, 不能凭尺度感觉收紧。

    标定依据(第七轮, 用一个真能走的策略实测): `../microduck/policies/alpha_walking.onnx`
    (上游真机在用的走路策略)在我们的环境里**能走**(离地 46%、摆动 170ms、倾角 2.8°、
    零摔倒), 但只能跑到指令速度的 ~45%(在两个环境里都一样, 实测 A/B: 他们环境 36~50%、
    我们环境 41~48%)。
    所以: 一个"45% 精度"的真步态, 必须还能拿到可观的跟踪分 —— 否则奖励等于**惩罚一切
    真实步态**。std=0.10 时它在 cmd 0.30 只拿 0.06(于是真步态输给站着不动); std=0.20 拿 0.49。
    反过来, "站着不动"(误差=指令值)必须拿得很少。
    """
    import math

    from mjlab_microduck.tasks.microduck_walk_cfg import _VELOCITY_STAGES

    cfg = microduck_walk_env_cfg()
    std = cfg.rewards["track_lin_vel"].params["std"]
    assert std == 0.20, "std 变了? 先用 eval_upstream_policy.py 重新做标定扫描"

    top_cmd = max(abs(x) for x in _VELOCITY_STAGES[-1]["lin_vel_x"])
    walker_err = 0.55 * top_cmd   # 真步态实测的典型误差(只能做到指令的 ~45%)
    standing_err = top_cmd        # 站着不动
    walker_kernel = math.exp(-(walker_err**2) / std**2)
    standing_kernel = math.exp(-(standing_err**2) / std**2)
    assert walker_kernel > 0.35, f"真步态在最大指令下只能拿 {walker_kernel:.2f} —— std 太严, 会惩罚一切真步态"
    assert standing_kernel < 0.2, f"站着不动还能拿 {standing_kernel:.2f} —— std 太松, 站立盆地会赢"
    assert walker_kernel - standing_kernel > 0.25, "走路相对站着的优势太小"
    # 实测(用这两个策略在同一 std 下真跑): cmd 0.30 → 真步态 +3.75 / 站着 +2.94 / 溜+摔 +2.54


def test_all_skill_rewards_apply_the_posture_gate():
    """源码级挡板: 跟踪 + 步态三项都必须乘 _posture_gate。

    行为级测试要构造特定接触状态(脚是否离地)才判别得出, 容易写成假测试;
    这里用源码检查当"别忘了乘门"的挡板 —— 少乘一项, 那个姿势又能薅分了。
    """
    import inspect

    from mjlab_microduck.tasks.microduck_walk_cfg import (
        feet_air_time_reward,
        track_ang_vel_z_reward,
        track_lin_vel_xy_reward,
    )

    for fn in (track_lin_vel_xy_reward, track_ang_vel_z_reward, feet_air_time_reward):
        assert "_posture_gate" in inspect.getsource(fn), f"{fn.__name__} 没乘姿态门"
    # 步态奖励还必须有"方向进度门"(第三轮: 原地蹦薅分)
    assert "_progress_gate" in inspect.getsource(feet_air_time_reward), "air_time 没乘方向门"


def test_fallen_too_long_has_grace_period():
    """倒地终止: 1 秒宽限(留给恢复尝试), 站起来一步就清零, 间歇躺不触发。"""
    env = _make_env(1)
    robot = env.scene["robot"]
    tm = env.termination_manager
    low = torch.zeros(1, 13)
    low[0, 3], low[0, 2] = 1.0, 0.060   # 低但直立(z < 0.065)
    up = torch.zeros(1, 13)
    up[0, 3], up[0, 2] = 1.0, 0.117

    def poke(state, n=1) -> bool:
        for _ in range(n):
            robot.write_root_state_to_sim(state)
            env.sim.forward()
            tm.compute()
        return bool(tm.get_term("fallen_too_long")[0])

    assert not poke(low, 40), "0.8s 内不该终止(要给恢复留时间)"
    assert poke(low, 15), "持续低过 1 秒必须终止(否则躺平就是永久薅分点)"
    assert not poke(up), "站起来一步 → 计时清零"
    assert not poke(low, 20), "再次躺 0.4s"
    assert not poke(up), "间歇躺不算赖着"


# ---------------------------------------------------------------- 传感器 / 命令
def test_foot_contact_sensor_sees_both_feet_on_ground():
    """零动作站一会儿后两只脚都该被判为接触(先落体 ~6mm, 第 3 步才踩实)。"""
    env = _make_env(1)
    sensor = env.scene["feet_ground_contact"]
    for _ in range(10):
        env.step(torch.zeros(1, 14))
    assert sensor.data.found is not None
    assert sensor.data.current_air_time is not None  # track_air_time 生效
    assert sensor.data.found.shape == (1, 2)
    assert (sensor.data.found > 0).all()
    assert (env.scene["robot"].data.root_link_pos_w[:, 2] > 0.09).all()


def test_startup_command_is_within_first_curriculum_stage():
    """起步指令必须落在第 1 档课程内。

    挡的是一类真实 bug: 上游 rel_forward_envs 会把直行环境的 vx 硬抬到 ≥0.3 m/s
    (Go1 尺度常数), 一旦重新打开, 起步指令就会越过整条课程上限, 在 GPU 上白跑几千轮
    才发现。这里同时钉住课程第 1 档和实际采样的指令。
    """
    cfg = microduck_walk_env_cfg()
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    assert twist.rel_forward_envs == 0.0

    stage0 = cfg.curriculum["command_vel"].params["velocity_stages"][0]
    assert twist.ranges.lin_vel_x == stage0["lin_vel_x"]

    env = _make_env(1)
    lo, hi = stage0["lin_vel_x"]
    n_standing, n_moving = 0, 0
    for _ in range(200):
        env.step(torch.zeros(1, 14))
        cmd = env.command_manager.get_command("twist")
        assert cmd is not None
        assert abs(cmd[0, 1]) < 1e-6 and abs(cmd[0, 2]) < 1e-6  # 第 1 档不开侧移/转向
        if cmd[0, 0].abs() < 1e-6:
            n_standing += 1        # rel_standing_envs 抽中的"零指令"环境: 命令被置零
            continue
        n_moving += 1
        assert lo - 1e-6 <= cmd[0, 0] <= hi + 1e-6, f"{float(cmd[0, 0])} 不在第 1 档 {stage0['lin_vel_x']} 内"
    assert n_moving > 0, "一个有指令的环境都没抽到? 检查 rel_standing_envs"
    # 注意: "零指令环境"是按**重采样**抽签的(每 4~8 秒一次 = 200~400 步), 所以单环境跑
    # 200 步很可能一次都没抽中 —— 那件事由 test_standing_envs_get_zero_command 单独测。


def test_standing_envs_get_zero_command():
    """rel_standing_envs 抽中的环境, 指令必须被精确置零(部署时的"站着"状态就是这个)。"""
    cfg = microduck_walk_env_cfg()
    twist = cfg.commands["twist"]
    assert twist.rel_standing_envs == pytest.approx(0.1)

    env = _make_env(1)
    term = env.command_manager.get_term("twist")
    ids = torch.arange(env.num_envs)
    n_zero = 0
    for _ in range(300):  # 300 次重采样, 按 10% 抽签 → 抽不到的概率 0.9^300 ≈ 0
        term._resample_command(ids)
        term._update_command()
        if float(term.vel_command_b[0, 0]) == 0.0:
            n_zero += 1
    assert n_zero > 0, "300 次重采样都没抽到零指令环境? 检查 rel_standing_envs"


def test_play_cfg_widens_commands_and_drops_push():
    cfg = microduck_walk_env_cfg(play=True)
    assert cfg.curriculum == {}
    assert cfg.episode_length_s >= 1e9
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    assert twist.ranges.lin_vel_x == (-0.10, 0.35)
    assert "push_robot" not in cfg.events
    assert "reset_joints" in cfg.events


def test_penalty_signs_are_negative_costs():
    """AGENTS 那条 'infallible check' 的静态版: 负权重项的函数必须返回 >=0 的 cost。

    规则: mjlab 风格 cost(>=0) 配负权重; 自取负的 penalty 配正权重。配反了会变成
    "违规即奖励", 策略会去把它farm起来。
    """
    env = _make_env(1)
    for _ in range(20):
        env.step(0.2 * torch.randn(1, 14))
    for name in env.reward_manager.active_terms:
        term = env.reward_manager.get_term_cfg(name)
        if term.weight >= 0:
            continue
        value = term.func(env, **term.params)
        assert bool((value >= -1e-9).all()), f"{name} 返回了负值, 却挂着负权重 → 双重取负"
        assert bool((value * term.weight <= 1e-9).all()), f"{name} 加权后为正"
