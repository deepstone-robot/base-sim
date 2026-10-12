# 训练

## 三条路径（成本递增，按顺序走）

| 路径 | 命令 | 时间 | 回答什么 |
|---|---|---|---|
| 1. 测试 | `uv run --with pytest pytest tests/ -q` | **秒级** | 改坏了维度/资产名/课程边界没有（[测试](testing.md)） |
| 2. headless 评估 | `uv run python scripts/eval_walk.py <ckpt>` | **分钟级** | 某个 checkpoint 到底会不会走（[评估与探针](evaluation.md)） |
| 3. 训练 | `uv run train <task> ...` | **小时级** | 学不学得会 |

**别跳步**。这个仓库的每一轮返工，几乎都能被前一步挡住
（见 [失败模式史](failure-modes.md)：其中至少四轮是"改完奖励直接上 GPU，事后才发现排序都没翻"）。

## 环境

```bash
uv sync                      # 首次；aarch64 上可能需要 UV_HTTP_TIMEOUT=600
uv run --with pytest pytest  tests/ -q
```

- **训练需要 Linux + NVIDIA GPU**（MuJoCo-Warp CUDA）。macOS 只能评估/冒烟/看画面。
- `pyproject.toml` 里含 aarch64(CUDA) 的 torch 源配置。
- 首次跑新模型时 MuJoCo-Warp 要编译内核，**1~2 分钟**属正常。

## 训练入口

```bash
# CPU 冒烟（验证链路：能建环境、能步进、能存 checkpoint）
scripts/run_cpu.sh Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 16 --agent.max-iterations 5

# GPU 全量
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 6000
```

入口是 `mjlab/scripts/train.py`。**CLI 覆盖规则**：`TrainConfig` 是个 dataclass，
任何字段都能用点号覆盖，例如：

| 想改什么 | 写法 |
|---|---|
| 并行环境数 | `--env.scene.num-envs 4096` |
| 训练轮数 | `--agent.max-iterations 6000` |
| 实验目录名 | `--agent.run-name walk-p2` |
| 用哪几张卡 | `--gpu-ids 0 1` |
| 关掉 wandb | `WANDB_MODE=disabled`（环境变量） |

::: danger `--gpu-ids` 只接受三种形态（踩过）

```
--gpu-ids 0 1      ✅ 列表
--gpu-ids all      ✅ 全部
--gpu-ids None     ✅ 显式空
--gpu-ids 0        ❌ Error parsing --gpu-ids: no type matching {None}|[INT,...]|{all} could be instantiated from ['0']
```
省略该参数 = 用 GPU 0。给单个裸整数会直接报错 —— 因为它期望的是**列表**。

:::

## 日志与 checkpoint

```
logs/rsl_rl/<experiment_name>/<时间戳>_<run-name>/
    model_0.pt  model_250.pt  ...  model_5999.pt   # save_interval 决定间隔
```

`experiment_name` 由各任务的 runner cfg 决定：行走 `microduck-walk`、站立 `microduck`、
坐站 `microduck-sitstand`。所以**不同任务的日志天然分开**。

本地目录 `logs/` 已在 `.gitignore` 里（训练产物不入库）；仓库根目录那个 `/model_*.pt`
也被忽略（游离的 checkpoint 不该进 git）。

## 时间基准（用真实的数来规划预算）

| 环境 | 机器 | 实测 |
|---|---|---|
| 16 envs | Mac CPU | ~2.3 s/iteration |
| 32 envs | Mac CPU | ~3.06 s/iteration |
| 64 envs | Mac CPU | ~4.80 s/iteration |
| 4096 envs | Linux GPU | 6000 轮可跑完（用户实跑） |

::: warning 课程推进只跟 iteration 数有关，与 `num_envs` 无关

`common_step_counter` 每次 `env.step()` 只 +1（所有环境一起走）。`num_steps_per_env=24` 时，
**无论 16 还是 4096 个环境，1 个 iteration 都只让课程前进 24 步**。
所以"课程最后一档在 14000 步"意味着 **≈583 个 iteration**，
而 `max_iterations=6000` 时它当然会到；但如果做 500 轮短测，**那一档永远不生效**。
规划短测/长测时先算这个换算。

:::

## 不要开 `--video`

在训练里开录制，代价是**每个策略步渲染一帧**：实测 250 轮约 **23 分钟** 的额外开销。
要看动作就用 `play` 回放 checkpoint（[评估与探针](evaluation.md)），别在训练时录。

## 在 Linux 容器上跑（踩过的准备步骤）

```bash
# 1. 装 uv（容器里通常没有）
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. 同步依赖（aarch64 上给足超时）
UV_HTTP_TIMEOUT=600 uv sync

# 3. 跑（wandb 关掉，避免没登录时卡住）
WANDB_MODE=disabled uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 4096 --gpu-ids 0
```

确认真的用上了 GPU：启动日志里应有 `Training with: device=cuda:0`。
另外 MuJoCo-Warp 的编译缓存在某些沙箱里会失败（`os.mkdir` 被拒），
此时用 `WARP_CACHE_PATH=$TMPDIR/warp-cache` 指到可写目录。

## 推荐的训练节奏（这个仓库的规矩）

1. **先短测 250~500 轮**（GPU 上十几分钟），用 `scripts/eval_walk.py` 量行为，
   判据写清楚再决定要不要全量。历史上 500 轮 CPU 短测就抓出了"学会站着不动"。
2. 改奖励/课程之后，**先做零成本重打分**（旧 checkpoint + 新奖励 + `--terms`），
   看行为排序有没有按预期翻转 —— 没翻就别上 GPU。
3. 全量训练完，**立刻用同一套 headless 脚本复测行为**，不要只看曲线。

一条经验判据（行走任务，来自多轮实测）：短测后要看到
**"有脚离地 >20%、摆动时长 >50 ms、cmd=0.15 时 vx 误差 <0.12、终止里没有奇怪项"**，
才值得投入全量训练。

## 训练规则

跑通一个里程碑就 commit（`git add -A && git commit`），保持每一步可复现。
外部可见的动作（push / 部署 / 发消息）一律先问人。
