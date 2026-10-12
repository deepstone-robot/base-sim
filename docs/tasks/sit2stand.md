# 坐→站（Sit2Stand）

> 任务 id：`Mjlab-Sit2Stand-Flat-MicroDuck` · 代码：`src/mjlab_microduck/tasks/microduck_sit2stand_cfg.py`（165 行）
> 只复用站立任务的零件、改四处：**起步姿势、奖励项、终止、回合长度**。

## 目标与设计意图（模块 docstring 原文）

> 技能目标: 鸭子从"坐姿"(或站姿)开始, 学会主动站起来并保持站立。
> 设计要点(为什么这样写):
> - 每个回合从 SIT 或 STAND 随机开始(reset_start 事件), 让"起身"经常被练习;
> - **没有 fell 终止: 摔倒了不重置, 回合继续**(否则永远学不会从地上起来);
> - episode 20s: 给足"坐→站→保持"的时间;
> - 奖励沿用站立任务: upright(站起来得分) + 小惩罚(别乱飘/乱抖/离站姿太远)。

这四条里最值得学的是第二条：**摔倒不终止** —— 如果摔了就重置，策略永远拿不到"从地上爬起来"的样本。

## 配置一览

除下表外，场景/实体/执行器/仿真/观测/动作与[站立](stand.md)完全相同
（同样 51 维 actor = critic，同样 `scale=0.5`，同样 50 Hz）。

| 块 | 值 |
|---|---|
| `episode_length_s` | **20.0 s = 1000 控制步** |
| 终止 | **只有 `time_out`，没有 `fell`** |
| 事件 | 只有 2 个：`bam_init`(startup) + `reset_start`(reset)。**没有推力事件** |
| `reset_start` 参数 | `sit_fraction=0.7` |
| 奖励 | `stand` +1.0、`joint_dev` **−0.02**、`vel_penalty` −0.2、`jvel_penalty` −0.02 |
| PPO | 完全复用站立（`(64,64)`、32 steps、1000 iters），只把 `experiment_name` 改成 `microduck-sitstand` |

### `reset_to_sit_stand` 怎么起步

每个环境**独立抽签**（`sit_fraction=0.7`）：

- 70% 从 **SIT** 起步、30% 从 **STAND** 起步（站姿起步是为了"防止策略忘掉站住"）；
- 关节目标 = `lerp(STAND, SIT, pick)`，再叠 **±0.02 rad 均匀抖动**；关节速度写 0；
- 根部 13 维状态：z = `lerp(0.1114, 0.07, pick)`、四元数 `w=1`（直立、无倾斜）。

SIT 坐姿不是手编的，而是取自 `scene_walk.xml` 的 **SIT keyframe**（14 个值逐个核对一致，
root z = 0.07 与代码里的 `_SIT_ROOT_Z` 一致）。

### 奖励

| 项 | 权重 | 作用 |
|---|---|---|
| `stand` | **+1.0** | `stand_reward` = 竖直程度 × 高度因子；高度因子 = `((z − 0.085)/0.03).clamp(0,1)`（0.085 m 得 0，≥0.115 m 满分） |
| `joint_dev` | −0.02 | 关节偏离默认站姿的均方误差（比站立任务**松 5 倍**：起身必须大幅偏离站姿） |
| `vel_penalty` | −0.2 | 世界系根速度平方（"别乱飘"） |
| `jvel_penalty` | −0.02 | 关节速度均方（"别乱抖"） |

::: warning 高度基准 0.085 与真实坐姿 0.07 不匹配 —— 起步阶段没有梯度

代码注释把 0.085 写成"≈坐"，但**实测坐姿 root z 是 0.07**（SIT keyframe 与 `_SIT_ROOT_Z` 都是 0.07）。
于是从 SIT 起步时 `(0.07 − 0.085).clamp(min=0) = 0` ⇒ **高度因子恒为 0**，
要长到 0.085 m 以上才开始拿分。也就是说**起身的前 15 mm 完全没有奖励梯度**
（`stand` 是唯一带有"起身"信号的正项，它在这段区间里对动作的偏导为 0）。
这和当年"`air_time` 窗口下限太高 ⇒ 迈步零梯度"是同一类问题（见 [失败模式史](../failure-modes.md)）。
**待验证**：还没量过训练是否因此卡住；如果要修，把 0.085 改成实测坐姿高度（0.07）即可，
但要注意别让"半个坐姿"也拿分。

:::

### 为什么可以没有 `fell`

站立任务用 `root z < 0.05` 终止（防止"躺着薅分"）。坐站任务**故意不这么做**：
它要学的东西恰恰是"从低处起来"，终止会把最有价值的样本砍掉。

代价是**躺着也能拿到时间**（只要 `time_out` 不触发）—— 靠的是奖励结构本身：
`stand` 需要"又直又高"才给分，躺着高度因子为 0；`vel_penalty`/`jvel_penalty` 只罚乱动，
不罚静止。所以理论上"躺着不动"拿到的是 0 分，而不是负分。
如果要严谨地排除"赖着不动"，可以加一个与行走任务同款的 `fallen_too_long`（倒地超时终止），
但要保留足够长的宽限期，别把恢复尝试砍掉。

## play 与训练的差别

同站立：`episode_length_s = 1e10`、`actor.enable_corruption = False`（同样是空操作）。
**回放时没有随机推**（这个任务本来就没有推力事件），所以回放看到的是"干净"的起身过程。

## 怎么跑

```bash
uv run mjpython -m mjlab.scripts.play Mjlab-Sit2Stand-Flat-MicroDuck --agent trained \
  --viewer native --checkpoint-file logs/rsl_rl/microduck-sitstand/<时间戳>/model_XXX.pt

scripts/run_cpu.sh Mjlab-Sit2Stand-Flat-MicroDuck --env.scene.num-envs 16 --agent.max-iterations 5
uv run train Mjlab-Sit2Stand-Flat-MicroDuck --env.scene.num-envs 4096 --gpu-ids 0
```

## 已知问题

| 位置 | 问题 |
|---|---|
| `stand_reward` 的高度基准 | 0.085 ≠ 实测坐姿 0.07 ⇒ 起步 15 mm 无梯度（见上） |
| `upright_reward` | 从站立任务**导入但没用到**（这个任务用的是 `stand_reward`） |
| actor 含 `base_lin_vel` | 同站立：真机拿不到 ⇒ 不能直接上机 |
| critic = actor 拷贝 | 同上：没有特权信息 |
| 无 `metrics` | 没有"离站姿多远 / 起身用了多久"这类曲线，只能靠 `Episode_Reward/stand` 看趋势 |
