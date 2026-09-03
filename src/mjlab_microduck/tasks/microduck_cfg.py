from pathlib import Path

import mujoco
from bam.mjlab import BamActuatorCfg, bam_init
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel, projected_gravity, base_ang_vel, last_action, \
    time_out, JointPositionActionCfg, push_by_setting_velocity, reset_root_state_uniform, base_lin_vel, \
    reset_joints_by_offset
from mjlab.managers import ObservationTermCfg, ObservationGroupCfg, EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

_REPO = Path(__file__).resolve().parents[3]
_DUCK_DIR = _REPO / "assets" / "microduck"
_DUCK_XML = _DUCK_DIR / "robot_allcollisions.xml"

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

_FULL_COLLISION = CollisionCfg(
    geom_names_expr=[".*_collision"],
    condim={r"^(left|right)_foot_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (1.0,)},
)


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
                # XmlActuatorCfg(target_names_expr=(r"(?!.*freejoint).*",)),
                BamActuatorCfg(
                    motor_name="xl330", model="m6",
                    target_names_expr=(r"^(?!passive_).*",),
                    kp_fw=200.0,
                    vin_range=(6.5, 8.2),
                    vin_min=6.0,
                    delay_min_lag=3,
                    delay_max_lag=3,# (ms)
                ),
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
        # 初始位形 = STAND 站姿：根位置(0,0,0.12) + 关节角
        # 关节速度默认全 0，不用写
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.1114),
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
    return ((q - d) ** 2).mean(dim=1)

def base_velocity_penalty(env):
    lin = env.scene["robot"].data.root_link_lin_vel_w
    ang = env.scene["robot"].data.root_link_ang_vel_w
    return (lin ** 2).sum(dim=1) + 0.1 * (ang ** 2).sum(dim=1)

def joint_vel_penalty(env):
    return (env.scene["robot"].data.joint_vel ** 2).mean(dim=1)

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
        "base_lin_vel": ObservationTermCfg(func=base_lin_vel),
    }
    cfg = ManagerBasedRlEnvCfg(
        events={
            "bam_init": EventTermCfg(func=bam_init, mode="startup"),
            # 每回合初始姿态微扰: 稍微歪一点/带一点初速度开始
            # "reset_disturbance": EventTermCfg(
            #     func=reset_root_state_uniform,
            #     mode="reset",
            #     params={
            #         "pose_range": {
            #             "x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0),
            #             "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2),
            #         },
            #         "velocity_range": {
            #             "x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (0.0, 0.0),
            #         },
            #     },
            # ),
            # 训练/播放中每 2~5 秒随机"推一把"(给根节点加冲量)
            # "random_push": EventTermCfg(
            #     func=push_by_setting_velocity,
            #     mode="interval",
            #     interval_range_s=(2.0, 5.0),
            #     params={
            #         "velocity_range": {
            #             "x": (-0.2, 0.2), "y": (-0.3, 0.3), "z": (0.0, 0.0),
            #         },
            #     },
            # ),
            "reset_to_stand": EventTermCfg(
                func=reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (0.0, 0.0),
                    "velocity_range": (0.0, 0.0),
                }
            )
        },
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
            "vel_penalty": RewardTermCfg(func=base_velocity_penalty, weight=-0.2),
            "jvel_penalty": RewardTermCfg(func=joint_vel_penalty, weight=-0.02),
        },
        terminations={
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
            "fell": TerminationTermCfg(func=fell_term),
        },
        # 物理步长 0.002s(和上游一致, 接触+高刚度执行器需要细步长), decimation 10 → 控制频率仍是 50Hz
        sim=SimulationCfg(njmax=300, mujoco=MujocoCfg(timestep=0.002)),
        decimation=10,
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
            hidden_dims=(64, 64), activation="elu", obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution",
                              "init_std": 1.0, "std_type": "scalar"},
        ),
        critic=RslRlModelCfg(
            hidden_dims=(64, 64), activation="elu", obs_normalization=True,
        ),
        # PPO 算法超参(先照抄，之后调)
        algorithm=RslRlPpoAlgorithmCfg(
            learning_rate=1.0e-3, schedule="adaptive",  # 自适应退火 lr
            gamma=0.99, lam=0.95,  # 折扣/GAE
            entropy_coef=0.01,  # 探索
            clip_param=0.2,  # PPO clip
            num_learning_epochs=5, num_mini_batches=4,
            value_loss_coef=1.0, use_clipped_value_loss=True,
            desired_kl=0.01, max_grad_norm=1.0,
        ),
        # 训练节奏 & 存档
        experiment_name="microduck",  # 决定 logs/rsl_rl/<name>/ 目录名
        num_steps_per_env=32,
        save_interval=200,  # 每 200 轮存一次 checkpoint
        max_iterations=1000,  # 默认上限，CPU 冒烟时用 CLI 覆盖小值
    )
