# MicroDuck RL（mjlab-microduck）

基于 [mjlab](https://github.com/mujocolab/mjlab)（Isaac Lab 风格 API + MuJoCo-Warp）的 MicroDuck 双足机器人强化学习仓库，PPO 训练由 rsl_rl 提供。当前进度：**MicroDuck 最小站立任务已注册并可构建/训练（CPU 冒烟级）**。

> 原名 base-sim，项目/包已改名 `mjlab-microduck` / `mjlab_microduck`（与上游 microduck_rl 对齐）。

## 目录结构

```
src/mjlab_microduck/
├── __init__.py
└── tasks/
    ├── __init__.py            # 任务注册(import 本包即注册 Mjlab-Stand-Flat-MicroDuck)
    └── microduck_cfg.py       # ①实体配置 + ②环境配置(五件套) + ③PPO runner 配置
assets/microduck/              # 鸭子机器人 MJCF 模型(STL 网格 + scene/keyframe)
scripts/
├── run_cpu.sh                 # CPU 冒烟训练包装(mac/linux 各自渲染后端)
├── view_microduck.py          # 交互窗口看鸭子(macOS 需 mjpython)
└── render_microduck.py        # 离屏渲染 PNG/MP4(不依赖窗口)
tests/                         # CPU 冒烟测试(纯配置/env 正确性)
logs/rsl_rl/<experiment>/…     # 训练产物(运行时生成, git 忽略)
```

## 环境准备

```bash
# 首次/改名后(会联网拉 BAM 的 git 依赖)
uv sync

# 确认任务已注册(应看到 Mjlab-Stand-Flat-MicroDuck)
uv run list-envs

# 跑测试(6 passed; 首次会编译 warp CPU 内核, 约 1~2 分钟)
uv run --with pytest pytest tests/ -q
```

## 命令速查

### 看模型与画面

| 目的 | 命令 |
|---|---|
| 交互窗口看鸭子(macOS 必须 mjpython) | `uv run mjpython scripts/view_microduck.py [STAND\|SIT\|FOLD\|INIT]` |
| 离屏渲染成 mp4/png(不依赖窗口) | `uv run python scripts/render_microduck.py move`（或 `stand`），输出到 `renders/` |
| 官方 demo(G1 人形翻跟头, 预训练推理) | `uv run demo` → 浏览器开 viser 地址(默认 http://127.0.0.1:8080) |

### play（推理：加载/指定策略看鸭子动，不训练）

```bash
# 随机策略: 鸭子被随机信号推着扭(未训练时的"环境检查")
uv run mjpython -m mjlab.scripts.play Mjlab-Stand-Flat-MicroDuck --agent random --viewer native

# 零动作: 应保持 STAND 站姿不动
uv run mjpython -m mjlab.scripts.play Mjlab-Stand-Flat-MicroDuck --agent zero --viewer native

# 回放训练好的 checkpoint(推理自己的成果)
uv run mjpython -m mjlab.scripts.play Mjlab-Stand-Flat-MicroDuck --agent trained \
  --viewer native --checkpoint-file logs/rsl_rl/microduck/<时间戳>/model_XXX.pt
```

> macOS 提示：窗口类画面必须走 `mjpython`；`play` 的 `--viewer auto` 在 mac 上会选 viser 网页，想要桌面窗口请显式 `--viewer native`。

### 训练

训练入口是 `uv run train <task>`（即 `mjlab/scripts/train.py`），CLI 字段由 `TrainConfig`（env / agent / video / gpu-ids / wandb-* / enable-nan-guard 等）暴露，所有 `dataclass` 字段都可通过点号 CLI 覆盖。

```bash
# CPU 冒烟(几十秒~几分钟): 验证"训练链路能跑通", 不产出可用策略
scripts/run_cpu.sh Mjlab-Stand-Flat-MicroDuck \
  --env.scene.num-envs 16 --agent.max-iterations 100

# CPU 冒烟 + 训练中自动录像(每 50 轮录一段, 看画面)
scripts/run_cpu.sh Mjlab-Stand-Flat-MicroDuck \
  --env.scene.num-envs 16 --agent.max-iterations 300 \
  --video True --video-interval 50 --video-length 120
```

### GPU 训练（真学习 — Linux + NVIDIA）

CPU 冒烟只验证链路，不学策略；"学没学会"必须靠 GPU 全量训练观察。

**前置条件**

| 项 | 要求 |
|---|---|
| OS | Linux（macOS 不支持训练；本机用 `scripts/run_cpu.sh` 仅做冒烟） |
| GPU | NVIDIA 独显 ≥ 1 张，显存按并行度定（`num-envs=4096` 通常要 16~24 GiB） |
| 驱动 / CUDA | 主机装有与 `warp-lang==1.12.0` 兼容的 CUDA runtime（参考官方 warp 的 CUDA 支持矩阵） |
| Python | 3.12.x（见 `pyproject.toml` 的 `<3.13` 上界） |

> `pyproject.toml` 已按平台路由 torch 源：x86_64 Linux 直接走 PyPI（自带 CUDA 运行时依赖），linux-aarch64（DGX Spark / GB10 等 PyPI 只有 CPU 轮的机器）改走 `cu129` 索引，无需手动改 source。

**一次性确认 CUDA 链路通畅**

```bash
uv sync                                                 # 拉好依赖
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
# 期望: True (>=1) — False/0 就是 wheel 没匹配上, 看 pyproject.toml 的 torch source 注释

uv run list-envs                                        # 应看到 Mjlab-Stand-Flat-MicroDuck
```

**单卡训练（最常用）**

```bash
# 默认就用第 0 张卡; num-envs=4096 是站立任务的经验起点
uv run train Mjlab-Stand-Flat-MicroDuck --env.scene.num-envs 4096

# 显式指定某张卡 (默认 --gpu-ids [0])
uv run train Mjlab-Stand-Flat-MicroDuck --gpu-ids 1

# 加长训练 + 改名 run, 方便回放时对得上
uv run train Mjlab-Stand-Flat-MicroDuck \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 5000 \
  --agent.save-interval 500 \
  --agent.run-name stand_v1
```

启动后日志里看到 `Training with: device=cuda:0` 即代表跑在 GPU 上；如果打到 `cpu` 说明 `CUDA_VISIBLE_DEVICES` 没设上。

**多卡训练（torchrunx 自动起 N 个 worker）**

```bash
# 自动用上所有可见 GPU (物理 GPU 个数 = torchrunx worker 数)
uv run train Mjlab-Stand-Flat-MicroDuck --gpu-ids all --env.scene.num-envs 4096

# 指定几张
uv run train Mjlab-Stand-Flat-MicroDuck --gpu-ids 0 1 2 3 --env.scene.num-envs 8192

# 关闭 torchrunx 自己的 log 子目录
uv run train Mjlab-Stand-Flat-MicroDuck --gpu-ids all --torchrunx-log-dir ""
```

torchrunx 子进程的 stdout 会写到 `logs/rsl_rl/<experiment>/<run>/torchrunx/` 下。

**从 checkpoint 续训**

```bash
# 从本地某个 run 的某个 checkpoint 接上继续学
uv run train Mjlab-Stand-Flat-MicroDuck \
  --agent.resume True \
  --agent.load-run 2026-09-04_15-12-33_stand_v1 \
  --agent.load-checkpoint model_1000.pt

# 从 W&B 上的 run 拉 checkpoint 续训 (需 wandb 已登录)
uv run train Mjlab-Stand-Flat-MicroDuck \
  --agent.resume True \
  --wandb-run-path <entity>/<project>/<run_id> \
  --wandb-checkpoint-name model_2000.pt
```

**后台运行（nohup，适合 SSH / 容器断开后继续）**

```bash
# 在仓库根目录执行; 训练日志重定向到 logs/, 记下 PID 方便以后停
export WANDB_MODE=disabled

cd /data/base-sim
nohup uv run train Mjlab-Stand-Flat-MicroDuck \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 300 \
  --agent.save-interval 100 \
  --agent.run-name push-p1 \
  > logs/train_push-p1.log 2>&1 &
echo $! > logs/train_push-p1.pid

# 看进度(终端实时打印每轮指标)
tail -f logs/train_push-p1.log

# 停止
kill $(cat logs/train_push-p1.pid)

# run-name 语义: 日志目录 = logs/rsl_rl/<experiment>/<时间戳>_<run-name>
# 起名建议带"内容+难度+版本", 如 push-x015-v1 / push-p2-med-v1, 方便回放与续训时定位
```

**训练时录像 + NaN 防护**

```bash
# 每 200 轮录 200 步一段 (默认只在 rank 0 录, 多卡下不会重复写)
uv run train Mjlab-Stand-Flat-MicroDuck \
  --video True --video-interval 200 --video-length 200

# 物理仿真出 NaN 时立刻停掉并 dump 现场, 排查奖励 / 关节爆掉
uv run train Mjlab-Stand-Flat-MicroDuck --enable-nan-guard True
```

**训练产物**

`logs/rsl_rl/microduck/<时间戳>[_run_name]/` 下：

```
model_*.pt            # 周期性 checkpoint (默认 save-interval=200)
params/env.yaml       # 训练用的 env 配置快照
params/agent.yaml     # 训练用的 PPO runner 配置快照
videos/train/         # --video 开启时的 mp4 片段
events.out.tfevents.* # TensorBoard 日志
config.json           # git 哈希 + 完整配置 (复现用)
```

### 看训练曲线（TensorBoard）

```bash
.venv/bin/tensorboard --logdir logs/rsl_rl    # 浏览器开 http://127.0.0.1:6006
```

常看曲线：`Train/mean_reward`、`Train/mean_episode_length`、`Episode_Reward/upright`、`Loss/*`、`Perf/*`。

## 任务代码怎么改

- **调超参/并行数**：CLI 点号覆盖，如 `--env.scene.num-envs 4096`、`--agent.max-iterations 5000`、`--agent.save-interval 500`（配置即 dataclass，路径即字段）。
- **改奖励/观测/终止**：编辑 `microduck_cfg.py` 里的 `actor_terms`、`rewards`、`terminations`（每个函数是 `(env) -> tensor`），改完跑测试 + CPU 冒烟再上 GPU。
- **加新任务**：复制 `microduck_ppo_runner_cfg` 风格新建 cfg → 在 `tasks/__init__.py` 里 `register_mjlab_task`，id 命名 `Mjlab-<技能>-<地形>-<机器人>`（如上游 `Mjlab-Velocity-Flat-MicroDuck`）。
- **环境"五件套"心智模型**：`scene`(世界) / `observations`(看到啥) / `actions`(控制啥) / `rewards`(啥算好) / `terminations`(何时结束)。

## 平台与注意事项

- **训练需要 NVIDIA GPU + Linux**（MuJoCo-Warp）；本仓库含 aarch64(CUDA) torch 源配置，见 `pyproject.toml` 注释。macOS 仅用于评估/冒烟/看画面。
- CPU 冒烟 = 验证链路正确（环境能建、能步进、出 checkpoint），**不代表策略学会了**；"学没学会"只看 GPU 全量训练后的曲线与回放。
- 测试是训练前的"质检台"：改代码 → `pytest` 秒级检查 → CPU 冒烟分钟级 → GPU 训练小时级。
- 首次跑新模型时 MuJoCo-Warp 会编译 CPU 内核，耗时 1~2 分钟，属正常。
- 本仓库规则：改完跑通一个里程碑就 `git add -A && git commit`，保持可复现。

## 路线图

- [x] L1 模型侦察 + 实体配置(①) + 配置测试
- [x] 可视化：交互窗口 / 离屏渲染 / 官方 demo 对比
- [x] L2 完整环境配置(② 五件套) + env 冒烟测试
- [x] L3-① 注册 `Mjlab-Stand-Flat-MicroDuck`（entry-point 自动发现）
- [ ] L3-② play 验收画面 / CPU 录像冒烟
- [ ] L4 GPU 全量训练（站立 → 行走），配合 TensorBoard 观察收敛
- [ ] L5 回放验收 → ONNX 导出 → 真机部署（上游: export.py / infer_policy.py）
