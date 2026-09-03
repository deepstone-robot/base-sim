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

```bash
# CPU 冒烟(几十秒~几分钟): 验证"训练链路能跑通", 不产出可用策略
scripts/run_cpu.sh Mjlab-Stand-Flat-MicroDuck \
  --env.scene.num-envs 16 --agent.max-iterations 100

# CPU 冒烟 + 训练中自动录像(每 50 轮录一段, 看画面)
scripts/run_cpu.sh Mjlab-Stand-Flat-MicroDuck \
  --env.scene.num-envs 16 --agent.max-iterations 300 \
  --video True --video-interval 50 --video-length 120

# GPU 全量训练(真学习; 需要 NVIDIA GPU, macOS 不支持训练)
uv run train Mjlab-Stand-Flat-MicroDuck --env.scene.num-envs 4096
```

训练产物：`logs/rsl_rl/microduck/<时间戳>/`（checkpoint `model_*.pt`、TensorBoard 事件、解析后配置 `params/*.yaml`、录像 `videos/train/`）。

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
