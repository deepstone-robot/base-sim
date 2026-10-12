# 任务总览

本仓库注册了三个任务，命名约定是 **`Mjlab-<技能>-<地形>-<机器人>`**：

| | 站立 | 坐→站 | 行走（速度跟踪） |
|---|---|---|---|
| 任务 id | `Mjlab-Stand-Flat-MicroDuck` | `Mjlab-Sit2Stand-Flat-MicroDuck` | `Mjlab-Velocity-Flat-MicroDuck` |
| 代码 | `microduck_cfg.py`（233 行） | `microduck_sit2stand_cfg.py`（165 行） | `microduck_walk_cfg.py`（1406 行） |
| 技能 | 站起来并保持 | 从坐姿/站姿起步，主动起身并保持 | 跟踪 (vx, vy, wz) 速度指令走 |
| 模型 | `robot_allcollisions.xml` | 同 | **`robot_walk.xml`**（碰撞面 11→5） |
| 观测 actor/critic | 51 / **51（无特权）** | 51 / **51（无特权）** | 51 / **54（多 3 维线速度）** |
| actor 含 `base_lin_vel` | **有**（真机给不出，见下） | **有** | 没有（挪到 critic 了） |
| 动作 | `JointPositionAction` scale **0.5** | 同 0.5 | scale **1.0** |
| 奖励项 | 4 | 4 | **13** |
| 终止项 | 2（`time_out` + `fell`） | **1**（只有 `time_out`） | 4 |
| 命令 | 无 | 无 | **`twist`（vx, vy, wz）** |
| 课程 | 无 | 无 | **3 套** |
| 回合长度 | 10 s | 20 s | 20 s |
| PPO 网络 | 64×64 | 64×64（同站立） | **256×256** |
| `experiment_name` | `microduck` | `microduck-sitstand` | `microduck-walk` |
| 状态 | 可用 | 可用 | **已训出能走的策略**（见 [行走](walk.md)） |

**三个任务共享同一份机器人与执行器配置**（`get_microduck_entity_cfg()`）：
同一个 MJCF 加载方式、同一套 BAM 执行器参数（`xl330/m6`、`kp_fw=200`、`vin 6.5~8.2`）、
`timestep=0.002 × decimation=10 = 50 Hz`、`njmax=300`。
行走任务唯一的另外之处是**换模型**（用 `robot_walk.xml` 去掉下巴/髋/躯干碰撞面）和**全套奖励/课程**。

## 该看哪一个

| 你的目的 | 看 |
|---|---|
| 想理解一个 mjlab 任务**最小**该长什么样 | [站立](stand.md)（4 个奖励项、2 个终止项、读 10 分钟就懂） |
| 想看"怎么从地上起来"（无终止 + 混合起步的设计） | [坐站](sit2stand.md) |
| 想改奖励/课程、或者想看"训出来的策略是什么水平" | [行走](walk.md) + [奖励设计原则](../reward-design.md) + [失败模式史](../failure-modes.md) |

## 三个任务共同的一个真机隐患

站立与坐站任务的 **actor 观测里包含 `base_lin_vel`**（机体坐标系线速度），
而真机上没有速度传感器 —— 也就是说**这两个任务的策略在真机上无法直接跑**
（需要先补状态估计，或者像行走任务那样把它挪进 critic）。

行走任务已经把 `base_lin_vel` 挪到 critic、把尾部 3 维改成速度指令槽（见 [架构](../architecture.md)），
但这也带来一个新问题：**walk 与 stand 的 actor 都是 51 维**，
用站立 checkpoint 续训走路时形状能对上、不报错、却会静默错训。
统一三套契约的计划在 [路线图](../roadmap.md) 的 W4。
