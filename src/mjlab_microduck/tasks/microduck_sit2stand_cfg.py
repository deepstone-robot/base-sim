"""R1 里程碑: 坐(SIT)→站 任务环境。

技能目标: 鸭子从"坐姿"(或站姿)开始, 学会主动站起来并保持站立。
设计要点(为什么这样写):
  - 每个回合从 SIT 或 STAND 随机开始(reset_start 事件), 让"起身"经常被练习;
  - 没有 fell 终止: 摔倒了不重置, 回合继续(否则永远学不会从地上起来);
  - episode 20s: 给足"坐→站→保持"的时间;
  - 奖励沿用站立任务: upright(站起来得分) + 小惩罚(别乱飘/乱抖/离站姿太远)。
"""
from __future__ import annotations

import torch

from bam.mjlab import bam_init
from mjlab.envs import ManagerBasedRlEnvCfg, ManagerBasedRlEnv
from mjlab.envs.mdp import (
    base_ang_vel, base_lin_vel, joint_pos_rel, joint_vel_rel, last_action,
    projected_gravity, time_out,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

# 共享零件: 实体(BAM/碰撞/站姿) + 奖励函数 + PPO 配置, 全部复用站立任务的
from mjlab_microduck.tasks.microduck_cfg import (
    _STAND_JOINT_POS,
    base_velocity_penalty,
    get_microduck_entity_cfg,
    joint_dev_reward,
    joint_vel_penalty,
    microduck_ppo_runner_cfg,
    upright_reward, stand_reward,
)

# SIT 坐姿(取自 scene_walk.xml 的 SIT keyframe, 键顺序与 _STAND_JOINT_POS 一致)
_SIT_JOINT_POS = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": -0.5236,
    "left_knee": 1.0472, "left_ankle": 0.0,
    "neck_pitch": 0.5, "head_pitch": 1.6, "head_yaw": 0.0, "head_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": 0.5236,
    "right_knee": -1.0472, "right_ankle": 0.0,
}

# 站姿时 root 高度(SIT 时更低)
_STAND_ROOT_Z = 0.1114
_SIT_ROOT_Z = 0.07

_ASSET = SceneEntityCfg("robot")


def reset_to_sit_stand(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    sit_fraction: float = 0.7,
    asset_cfg: SceneEntityCfg | None = None,
) -> None:
    """每回合重置: 以 sit_fraction 概率从 SIT 坐姿开始, 否则从 STAND 站姿开始。

    为什么两种都留: 坐姿是练习"起身", 站姿是防止策略忘掉"站住"。
    每个 env 独立抽签 → 并行环境里两种起步都有, 样本更丰富。
    """
    asset_cfg = asset_cfg if asset_cfg is not None else _ASSET
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    dev = asset.data.joint_pos.device

    # 两种姿态的关节目标 + 每 env 抽签 + 少量抖动
    stand = torch.tensor([list(_STAND_JOINT_POS.values())], dtype=torch.float32, device=dev)
    sit = torch.tensor([list(_SIT_JOINT_POS.values())], dtype=torch.float32, device=dev)
    pick = (torch.rand(n, 1, device=dev) < sit_fraction).float()   # 1=坐, 0=站
    joint_pos = torch.lerp(
        stand.expand(n, 14), sit.expand(n, 14), pick
    ) + (torch.rand(n, 14, device=dev) * 2 - 1) * 0.02

    # 写入仿真: 关节位置/速度 + 根部状态(x,y,z, 四元数 wxyz, 线速度, 角速度 = 13维)
    asset.write_joint_position_to_sim(joint_pos, env_ids=env_ids)
    asset.write_joint_velocity_to_sim(torch.zeros_like(joint_pos), env_ids=env_ids)

    root = torch.zeros(n, 13, device=dev)
    root[:, 2] = torch.lerp(
        torch.full((n, 1), _STAND_ROOT_Z, device=dev),
        torch.full((n, 1), _SIT_ROOT_Z, device=dev),
        pick,
    )[:, 0]
    root[:, 3] = 1.0  # 四元数 w=1 → 直立
    asset.write_root_state_to_sim(root, env_ids=env_ids)


def microduck_sitstand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """坐→站 任务环境配置。"""
    actor_terms = {
        "joint_pos": ObservationTermCfg(func=joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=joint_vel_rel),
        "gravity": ObservationTermCfg(func=projected_gravity),
        "ang_vel": ObservationTermCfg(func=base_ang_vel),
        "actions": ObservationTermCfg(func=last_action),
        "base_lin_vel": ObservationTermCfg(func=base_lin_vel),
    }
    cfg = ManagerBasedRlEnvCfg(
        events={
            "bam_init": EventTermCfg(func=bam_init, mode="startup"),
            "reset_start": EventTermCfg(
                func=reset_to_sit_stand,
                mode="reset",
                params={"sit_fraction": 0.7},
            ),
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
            "stand": RewardTermCfg(func=stand_reward, weight=1.0),
            "joint_dev": RewardTermCfg(func=joint_dev_reward, weight=-0.02),
            "vel_penalty": RewardTermCfg(func=base_velocity_penalty, weight=-0.2),
            "jvel_penalty": RewardTermCfg(func=joint_vel_penalty, weight=-0.02),
        },
        terminations={
            # 注意: 只有 time_out, 没有 fell —— 摔倒了继续, 策略才学得会爬起来
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
        },
        sim=SimulationCfg(njmax=300, mujoco=MujocoCfg(timestep=0.002)),
        decimation=10,
        episode_length_s=20.0,   # 给"坐→站→保持"留足时间
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot", body_name="trunk_base",
            distance=1.0, elevation=1.0, azimuth=45.0),
    )
    if play:
        cfg.episode_length_s = 1e10
        cfg.observations["actor"].enable_corruption = False
    return cfg


def microduck_sit2stand_ppo_runner_cfg():
    """复用站立任务的 PPO 配置, 只改实验名(日志目录分开)。"""
    base = microduck_ppo_runner_cfg()
    base.experiment_name = "microduck-sitstand"
    return base
