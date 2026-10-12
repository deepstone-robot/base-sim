# 站立（Stand）

> 任务 id：`Mjlab-Stand-Flat-MicroDuck` · 代码：`src/mjlab_microduck/tasks/microduck_cfg.py`（233 行）
> 这是本仓库**最小**的一个任务 —— 想理解"一个 mjlab 任务该长什么样"，从这个开始读。

## 目标

学会**站直并保持**。没有速度指令、没有课程，只有"别倒、别乱飘、别乱抖、贴近站姿"四件事。

## 配置一览（实测，非抄注释）

| 块 | 值 |
|---|---|
| 地形 | `TerrainEntityCfg(terrain_type="plane")` 平面 |
| 实体 | `get_microduck_entity_cfg()`（[机器人资产](../robot.md)） |
| `num_envs` / `env_spacing` | 1 / 4.0（`num_envs=1` 写在 cfg 里，训练靠 CLI 覆盖） |
| `timestep` × `decimation` | 0.002 s × 10 = **50 Hz** |
| `njmax` | 300 |
| `episode_length_s` | **10.0 s = 500 控制步** |
| viewer | 跟随 `trunk_base`，`distance=1.0, elevation=1.0, azimuth=45°` |
| 初始位形 | `pos=(0,0,0.1114)`、`rot=(1,0,0,0)`、`joint_pos=_STAND_JOINT_POS` |

### 观测（actor = critic = **51 维**）

| 项 | 来源 | 维度 |
|---|---|---|
| `joint_pos` | `joint_pos_rel`（关节角 − 默认） | 14 |
| `joint_vel` | `joint_vel_rel` | 14 |
| `gravity` | `projected_gravity`（机体系） | 3 |
| `ang_vel` | `base_ang_vel`（机体系） | 3 |
| `actions` | `last_action` | 14 |
| `base_lin_vel` | `root_link_lin_vel_b`（机体系线速度） | 3 |

::: danger 这个任务的两个真机/续训隐患

1. **actor 里有 `base_lin_vel`** —— 真机没有速度传感器 ⇒ 策略不能直接上机，
   要先补状态估计或像行走任务那样把它挪进 critic。
2. **critic 是 actor 的逐项拷贝**（`dict(actor_terms)`）⇒ **没有特权信息**，
   非对称 actor-critic 等于没启用。
3. **它的 actor 恰好也是 51 维**，与行走任务相同，但**尾部 3 维语义不同**
   （站立 = 线速度，行走 = 速度指令）⇒ 用站立 checkpoint `--agent.resume` 续训走路会**静默错训**。
   测试 `test_actor_tail_slot_is_command_not_velocity` 钉住的是行走那一侧。

:::

### 动作

`JointPositionActionCfg(entity_name="robot", actuator_names=(".*",), scale=**0.5**, use_default_offset=True)`。
14 维 = 14 个关节的位置偏移。注意**行走任务用的是 scale=1.0** ——
0.5 在站立任务够用（只需要小幅微调），但抬脚需要的膝屈要 1~1.8σ，学习信号会很稀
（这也是行走任务把它改成 1.0 的原因，见 [失败模式史](../failure-modes.md)）。

### 奖励（4 项）

| 项 | 权重 | 函数 | 作用 |
|---|---|---|---|
| `upright` | **+1.0** | `upright_reward` | `-projected_gravity_b[:,2]`：直立 = +1，躺着 = −1 的方向 |
| `joint_dev` | −0.1 | `joint_dev_reward` | 关节偏离默认站姿的均方误差 |
| `vel_penalty` | −0.2 | `base_velocity_penalty` | 根部**世界系**线速度平方 + 0.1 × 角速度平方（"别乱飘"） |
| `jvel_penalty` | −0.02 | `joint_vel_penalty` | 关节速度均方（"别乱抖"） |

⚠️ 坐标系的坑：**奖励里的速度用世界系**（`root_link_lin_vel_w`），
**观测里的速度用机体系**（`root_link_lin_vel_b`）。站立任务的 reset 会给 ±0.05 rad 的 yaw 扰动，
两者在 yaw ≠ 0 时并不等价。

### 终止（2 项）

| 项 | 条件 |
|---|---|
| `time_out` | 到 10 s 上限（用于 bootstrap） |
| `fell` | `root_link_pos_w[:,2] < 0.05`（躯干低于 5 cm 视为摔倒） |

### 事件（4 项）

| 项 | mode | 作用 |
|---|---|---|
| `bam_init` | startup | 展开 BAM 的 per-world 摩擦字段（不做的话摩擦随机化/写入不按环境生效） |
| `reset_disturbance` | reset | 初始姿态微扰：x/y ±0.02 m、roll/pitch ±0.03 rad、yaw ±0.05 rad、初速度 x/y ±0.05 m/s |
| `random_push` | interval | **每 2~4 s** 随机推一把：速度 x/y ±0.3 m/s、pitch ±0.2 rad |
| `reset_to_stand` | reset | 关节精确回到默认站姿（range 全 0），soft limit factor 0.9 |

### PPO

网络 `(64,64)` + ELU + `obs_normalization`，actor 是高斯分布（`init_std=1.0`，标量 std）；
`lr=1e-3`（adaptive，`desired_kl=0.01`）、`gamma=0.99`、`lam=0.95`、`entropy_coef=0.01`、
`clip_param=0.2`、5 epochs × 4 minibatches、`max_grad_norm=1.0`；
`num_steps_per_env=32`、`save_interval=200`、`max_iterations=1000`、
`experiment_name="microduck"`（决定 `logs/rsl_rl/microduck/<时间戳>_<run-name>/` 目录名）。

## play 与训练的差别

```python
if play:
    cfg.episode_length_s = 1e10                          # 不再超时终止(只剩 fell)
    cfg.observations["actor"].enable_corruption = False  # ← 实测是空操作
```

::: warning `enable_corruption = False` 在当前配置下是空操作

该字段默认就是 `False`，训练配置也没设成 `True`，而且**没有任何观测项配 `noise=`**。
所以这一行看着像"play 时关掉噪声"，实际什么都没关 —— 因为训练时本来就没有噪声。
（行走任务同理；观测噪声要真的加上，见 [sim2real](../sim2real.md)。）

:::

其余一切（奖励、终止、事件里的 `random_push`、动作、物理参数）play 下都不变 ——
**回放时鸭子照样每 2~4 秒被推一把**，这是有意的（看鲁棒性），不是 bug。

## 怎么跑

```bash
# 看图（macOS 桌面窗口必须 mjpython）
uv run mjpython -m mjlab.scripts.play Mjlab-Stand-Flat-MicroDuck --agent zero --viewer native
uv run mjpython -m mjlab.scripts.play Mjlab-Stand-Flat-MicroDuck --agent random --viewer native
uv run mjpython -m mjlab.scripts.play Mjlab-Stand-Flat-MicroDuck --agent trained \
  --viewer native --checkpoint-file logs/rsl_rl/microduck/<时间戳>/model_XXX.pt

# 训练（macOS 只做冒烟，正式训练在 Linux + GPU）
scripts/run_cpu.sh Mjlab-Stand-Flat-MicroDuck --env.scene.num-envs 16 --agent.max-iterations 5
uv run train Mjlab-Stand-Flat-MicroDuck --env.scene.num-envs 4096 --gpu-ids 0
```

零动作（`--agent zero`）时鸭子**不是稳定平衡**：实测约 0.25 s 就从 0.5° 倒到 24°
（倒立摆增长率 ~10/s ≈ `sqrt(g/0.11m)`）。这是物理，不是 bug —— 站稳必须靠反馈。

## 已知的"注释与代码不一致"（这个任务贡献了好几条）

| 位置 | 注释说 | 实际是 |
|---|---|---|
| XML 文件名 | docstring 写"把 `robot_walk.xml` 接成实体" | 加载的是 **`robot_allcollisions.xml`** |
| 初始根高度 | 注释写 `0.12` | 代码用 **`0.1114`**（XML 的 STAND keyframe 才是 0.12） |
| 推力间隔 | 注释写"每 2~5 秒" | 代码是 **2.0~4.0 s** |
| "加冲量" | 注释写"给根节点加冲量" | `push_by_setting_velocity` 是**直接改写根速度**，没有力的过程 |
| `delay_max_lag=3` | 注释写 `# (ms)` | 单位是**物理步** ⇒ 3 × 0.002 = **6 ms**（见 [机器人资产](../robot.md)） |
| "排除 free 根关节" | 注释这么解释正则 | 正则过滤的是 **actuator 名**（`^(?!passive_).*`）；freejoint 本来就没有 actuator |
| `_FULL_COLLISION` | 定义了一套碰撞配置 | **全仓库没人引用**（死代码）；详情见 [机器人资产](../robot.md) |

这类问题不会让训练崩，但会让读代码的人得出错误结论 —— 改代码时顺手修掉。
