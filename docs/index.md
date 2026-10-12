# base-sim · MicroDuck

MicroDuck 是一只**约 25 cm、737 g 的双足小机器人**。本仓库是它在
[mjlab](https://github.com/mujocolab/mjlab)（MuJoCo-Warp 物理 + RSL-RL 的 PPO）上的强化学习任务集合：
把机器人的 MJCF、三个技能任务（站立 / 坐站 / 行走）、奖励设计、评估探针和测试放在一起，
目标是"**训出来的策略能上真机**"。

::: warning 这份文档的写法

文档里出现的**每一个数字都来自本仓库的实测脚本**（`scripts/`），并且尽量写清"怎么测出来的、
测的时候什么条件下"。凡是没测过的推断，都标成"**待验证**"。这是本项目的习惯 —— 见
[奖励设计原则](reward-design.md) 和 [失败模式史](failure-modes.md)：这个仓库里绝大多数
返工，都是因为"配置能跑通、数字没人看"。

:::

## 30 秒上手

```bash
# 1. 装依赖（首次, aarch64 上给足超时）
uv sync

# 2. 改完代码先跑测试: 秒级, 钉住维度/资产名/课程边界这类静默错误
uv run --with pytest pytest tests/ -q

# 3. 看鸭子动起来（macOS 桌面窗口必须走 mjpython；网页查看器用普通 python）
uv run mjpython -m mjlab.scripts.play Mjlab-Stand-Flat-MicroDuck --agent zero --viewer native
uv run python    -m mjlab.scripts.play Mjlab-Velocity-Flat-MicroDuck \
  --agent trained --checkpoint-file model_3250.pt --num-envs 4 --viewer viser   # 浏览器开 :8080

# 4. 训练（macOS 只能冒烟；正式训练在 Linux + GPU）
scripts/run_cpu.sh Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 16 --agent.max-iterations 5
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 --gpu-ids 0
```

## 文档地图

| 你想干什么 | 看哪一页 |
|---|---|
| 弄清"一个环境由哪些部件组成、一次 step 里发生了什么" | [架构](architecture.md) |
| 知道三个任务分别是什么、谁学会了、观测/奖励/终止长什么样 | [任务总览](tasks/index.md) |
| 改行走任务的奖励或课程前，先看它为什么长这样 | [行走任务](tasks/walk.md) → [奖励设计原则](reward-design.md) |
| 想知道"哪些坑已经踩过、别再踩" | [失败模式史](failure-modes.md) |
| 换模型 / 查关节顺序 / 查质量分布 | [机器人资产](robot.md) |
| 训练命令、CLI 覆盖、wandb、CPU/GPU 差别 | [训练](training.md) |
| 量一个 checkpoint 到底走得好不好（跟踪误差、步态、摇头） | [评估与探针](evaluation.md) |
| 上真机前要加哪些随机化，哪些随机化**在这套栈里其实是空操作** | [sim2real 与域随机化](sim2real.md) |
| 测试都钉住了什么、怎么写新测试 | [测试](testing.md) |
| 下一步做什么 | [路线图](roadmap.md) |
| 为什么用 VitePress 而不是别的 | [文档工具说明](tooling.md) |

## 仓库结构（一句话版）

```
src/mjlab_microduck/tasks/   三个任务的 env/RL 配置 —— 本仓库的核心(行走那份 1400 行)
assets/microduck/            机器人 MJCF(多套变体) + 网格/惯性参数
scripts/                     评估与探针: 量步态、量摇头、量随机化是否真的生效
tests/                       质检台: 钉住"改回去不报错但会指错方向"的静默错误
logs/rsl_rl/microduck-*/     每次训练一个目录(model_XXX.pt 检查点)
```

## 当前状态（诚实版）

| 任务 | 任务 id | 状态 |
|---|---|---|
| 站立 | `Mjlab-Stand-Flat-MicroDuck` | 可用 |
| 坐→站 | `Mjlab-Sit2Stand-Flat-MicroDuck` | 可用 |
| 平地速度跟踪（行走） | `Mjlab-Velocity-Flat-MicroDuck` | **已训出能走的策略**：跟踪误差 0.018~0.023 m/s、倾角 <2°、路径直线度 0.95~0.98、零摔倒 |
| sim2real 加固 | — | 随机化清单已逐项实测（见 [sim2real](sim2real.md)），**尚未合入训练配置** |
| ONNX 导出 / 真机部署 | — | 未做 |
