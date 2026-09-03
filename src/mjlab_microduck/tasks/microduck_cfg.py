from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel, projected_gravity, base_ang_vel, last_action, \
    time_out, JointPositionActionCfg
from mjlab.managers import ObservationTermCfg, ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

_REPO = Path(__file__).resolve().parents[3]
_DUCK_DIR = _REPO / "assets" / "microduck"
_DUCK_XML = _DUCK_DIR / "robot_walk.xml"

_STAND_JOINT_POS = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": -0.0873,
    "left_hip_pitch": -0.4579,
    "left_knee": -0.0049,
    "left_ankle": 0.453,
    "neck_pitch": 0.3491,
    "head_pitch": 0.3491,
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0873,
    "right_hip_pitch": 0.4579,
    "right_knee": 0.0049,
    "right_ankle": -0.453,
}


def get_microduck_entity_cfg() -> EntityCfg:
    """
    把robot_walk.xml 接成mjlab认识的实体。
    """
    return EntityCfg(
        # 模型来源：读取 XML。传函数(lambda)而非直接结果，mjlab 每个 env 各建一份
        spec_fn=lambda: mujoco.MjSpec.from_file(str(_DUCK_XML)),
        # 关键：让 XML 里已有的 14 个位置执行器生效。
        # 正则是"关节名筛选"：排除 free 根关节(它没有执行器)，其余 14 个全部包进来
        articulation=EntityArticulationInfoCfg(
            actuators=(
                XmlActuatorCfg(target_names_expr=(r"(?!.*freejoint).*",)),
            ),
        ),
        # 初始位形 = STAND 站姿：根位置(0,0,0.12) + 关节角
        # 关节速度默认全 0，不用写
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.12),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=_STAND_JOINT_POS,
        ),
    )


def upright_reward(env):
    """想竖直: projected_gravity = 重力在机体坐标系的投影。
        直立时它是 (0,0,-1)，z=-1；取 -z → 直立=+1。"""
    return -env.scene["robot"].data.projected_gravity_b[:, 2]


def joint_dev_reward(env):
    """想贴近默认站姿: 关节离 STAND 越远，扣分越多(负值)。"""
    q = env.scene["robot"].data.joint_pos
    d = env.scene["robot"].data.default_joint_pos
    return -((q - d) ** 2).mean(dim=1)


def fell_term(env):
    """躯干(根部)低于 5cm 视为摔倒 → 终止本回合。"""
    z = env.scene["robot"].data.root_link_pos_w[:, 2]
    return z < 0.05


def microduck_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    actor_terms = {
        "joint_pos": ObservationTermCfg(func=joint_pos_rel),  # 14 维: 关节角(相对默认)
        "joint_vel": ObservationTermCfg(func=joint_vel_rel),  # 14 维: 关节角速度
        "gravity": ObservationTermCfg(func=projected_gravity),  # 3 维: 姿态(靠它站直)
        "ang_vel": ObservationTermCfg(func=base_ang_vel),  # 3 维: 机体角速度
        "actions": ObservationTermCfg(func=last_action),  # 14 维: 上一步动作
    }
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_microduck_entity_cfg()},
            num_envs=1,
            env_spacing=4.0,
        ),
        observations={
            "actor": ObservationGroupCfg(actor_terms, concatenate_terms=True),
            "critic": ObservationGroupCfg(dict(actor_terms), concatenate_terms=True),
        },
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=0.5,
                use_default_offset=True,
            )
        },
        rewards={
            "upright": RewardTermCfg(func=upright_reward, weight=1.0),
            "joint_dev": RewardTermCfg(func=joint_dev_reward, weight=-0.1),
        },
        terminations={
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
            "fell": TerminationTermCfg(func=fell_term),
        },
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01)),
        decimation=2,
        episode_length_s=10.0,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot", body_name="trunk_base",
            distance=1.0, elevation=1.0, azimuth=45.0),
    )
    if play:
        cfg.episode_length_s = 1e10
        cfg.observations["actor"].enable_corruption = False
    return cfg


# ============ ③ RL 训练器配置 (PPO 超参) ============
def microduck_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """rsl_rl 的 PPO 配置：网络结构 + 算法超参 + 训练节奏。"""
    return RslRlOnPolicyRunnerCfg(
        # 网络：actor/critic 都是 64x64 MLP(先小，够站立任务用)
        actor=RslRlModelCfg(
            hidden_dims=(64, 64), activation="elu", obs_normalization=False,
            distribution_cfg={"class_name": "GaussianDistribution",
                              "init_std": 1.0, "std_type": "scalar"},
        ),
        critic=RslRlModelCfg(
            hidden_dims=(64, 64), activation="elu", obs_normalization=False,
        ),
        # PPO 算法超参(先照抄，之后调)
        algorithm=RslRlPpoAlgorithmCfg(
            learning_rate=1.0e-3, schedule="adaptive",   # 自适应退火 lr
            gamma=0.99, lam=0.95,                        # 折扣/GAE
            entropy_coef=0.01,                           # 探索
            clip_param=0.2,                              # PPO clip
            num_learning_epochs=5, num_mini_batches=4,
            value_loss_coef=1.0, use_clipped_value_loss=True,
            desired_kl=0.01, max_grad_norm=1.0,
        ),
        # 训练节奏 & 存档
        experiment_name="microduck",   # 决定 logs/rsl_rl/<name>/ 目录名
        num_steps_per_env=32,
        save_interval=200,             # 每 200 轮存一次 checkpoint
        max_iterations=1000,           # 默认上限，CPU 冒烟时用 CLI 覆盖小值
    )
