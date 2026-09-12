"""MicroDuck 任务注册包：import 本包 = 任务注册进 mjlab。

mjlab 启动时会自动加载注册在 "mjlab.tasks" entry-point 组的包
(见 pyproject.toml 的 [project.entry-points."mjlab.tasks"])，
所以只要 mjlab import 到本包，下面 register_mjlab_task 就会执行。

本文件同时"重新导出" cfg 函数，方便测试直接写:
    from mjlab_microduck.tasks import microduck_env_cfg
"""
from mjlab.tasks.registry import register_mjlab_task

from mjlab_microduck.tasks.microduck_cfg import (
    _STAND_JOINT_POS,
    get_microduck_entity_cfg,
    microduck_env_cfg,
    microduck_ppo_runner_cfg,
)
from mjlab_microduck.tasks.microduck_sit2stand_cfg import (
    microduck_sitstand_env_cfg,
    microduck_sit2stand_ppo_runner_cfg,
)
from mjlab_microduck.tasks.microduck_walk_cfg import (
    microduck_walk_env_cfg,
    microduck_walk_ppo_runner_cfg,
)

register_mjlab_task(
    task_id="Mjlab-Stand-Flat-MicroDuck",   # 命名约定: 技能-地形-机器人
    env_cfg=microduck_env_cfg(),
    play_env_cfg=microduck_env_cfg(play=True),
    rl_cfg=microduck_ppo_runner_cfg(),
)

register_mjlab_task(
    task_id="Mjlab-Sit2Stand-Flat-MicroDuck",   # R1: 坐→站
    env_cfg=microduck_sitstand_env_cfg(),
    play_env_cfg=microduck_sitstand_env_cfg(play=True),
    rl_cfg=microduck_sit2stand_ppo_runner_cfg(),
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",   # W: 平地速度跟踪(行走)
    env_cfg=microduck_walk_env_cfg(),
    play_env_cfg=microduck_walk_env_cfg(play=True),
    rl_cfg=microduck_walk_ppo_runner_cfg(),
)
