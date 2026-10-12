# MicroDuck RL（mjlab-microduck）

基于 [mjlab](https://github.com/mujocolab/mjlab)（Isaac Lab 风格 API + MuJoCo-Warp）的 MicroDuck 双足机器人强化学习仓库，PPO 训练由 rsl_rl 提供。当前进度：**三个任务已注册并可构建/训练（CPU 冒烟级）** —— 站立 `Mjlab-Stand-Flat-MicroDuck`、坐→站 `Mjlab-Sit2Stand-Flat-MicroDuck`、行走(平地速度跟踪) `Mjlab-Velocity-Flat-MicroDuck`。

> 原名 base-sim，项目/包已改名 `mjlab-microduck` / `mjlab_microduck`（与上游 microduck_rl 对齐）。

## 📖 文档站

**完整文档在 [`docs/`](docs/index.md)**（VitePress，部署到 GitHub Pages：`Settings → Pages → Source` 选 **GitHub Actions** 后，push 到 `main` 自动发布）：

| 想看什么 | 去哪 |
|---|---|
| 门面与 30 秒上手 | [`docs/index.md`](docs/index.md) |
| 一个环境由哪些部件组成、一次 `step()` 里发生什么 | [`docs/architecture.md`](docs/architecture.md) |
| 三个任务分别是什么、谁学会了 | [`docs/tasks/`](docs/tasks/index.md) |
| 改奖励之前必读（设计原则 + 失败模式史） | [`docs/reward-design.md`](docs/reward-design.md) · [`docs/failure-modes.md`](docs/failure-modes.md) |
| 机器人资产 / 关节序 / 质量分布 / 执行器 | [`docs/robot.md`](docs/robot.md) |
| 训练、评估与探针 | [`docs/training.md`](docs/training.md) · [`docs/evaluation.md`](docs/evaluation.md) |
| 上真机前要加的随机化（哪些在这套栈里其实是空操作） | [`docs/sim2real.md`](docs/sim2real.md) |
| 测试都钉住了什么 | [`docs/testing.md`](docs/testing.md) |
| 下一步做什么 | [`docs/roadmap.md`](docs/roadmap.md) |

本地预览：`npm ci && npm run docs:dev`（地址是 `http://localhost:5173/base-sim/`，带 `base` 前缀是正常的）；`npm run docs:build` 就是 CI 用的检查 —— **内链写错会直接构建失败**，和原来的 `mkdocs build --strict` 一样。

## 目录结构

```
src/mjlab_microduck/
├── __init__.py
└── tasks/
    ├── __init__.py                 # 任务注册(import 本包即注册下面三个任务)
    ├── microduck_cfg.py            # ①实体配置 + ②站立环境配置(五件套) + ③PPO runner 配置(共享零件都在这)
    ├── microduck_sit2stand_cfg.py  # 坐→站环境配置
    └── microduck_walk_cfg.py       # 行走环境配置(速度指令 + 步态奖励 + 课程)
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

# 确认任务已注册(应看到 Stand / Sit2Stand / Velocity 三个 MicroDuck 任务)
uv run list-envs

# 跑测试(15 passed; 首次会编译 warp CPU 内核, 约 1~2 分钟)
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

### 行走任务（`Mjlab-Velocity-Flat-MicroDuck`）

任务目标：听懂速度指令 `(vx, vy, wz)` 在平地上走，全程不摔。设计意图、奖励权重、课程表都在 `src/mjlab_microduck/tasks/microduck_walk_cfg.py` 的注释里（每个奖励项都写了"为什么这么定"）。

| 项 | 设定 |
|---|---|
| actor 观测 (51 维) | 关节角 14 + 关节角速度 14 + 姿态 3 + 机体角速度 3 + 上一步动作 14 + **速度指令 3**（没有机体线速度 —— 真机给不出这个量） |
| critic 观测 (54 维) | 上面 51 维 + 真值机体线速度 3（特权信息，只用来算价值函数，不参与部署） |
| 动作 (14 维) | 14 个舵机关节的目标角（相对默认站姿，scale 1.0） |
| 指令 | 每 4~8 秒重采；起步档 `vx∈(0.10, 0.20)` `vy=0` `wz=0`（**下限不为 0**：否则"站着不动"白拿跟踪分，见下），10% 的环境拿零指令（顺带保持站立） |
| 课程 | 三条：`command_vel` 指令范围 0→1500→4000→8000 步（最终 `vx∈(-0.10,0.35)` `vy∈(-0.05,0.05)` `wz∈(-0.7,0.7)`）；`air_window` 步态窗口下限 0.03→0.08→0.125；`action_rate_weight` 平滑税 -0.03→-0.05→-0.08→-0.10 |
| 奖励 | `track_lin_vel`(+2.0, std=**0.10**) `track_ang_vel`(+1.0, std=0.5) `upright`(+2.0 高斯) `air_time`(**+1.5**, 每步付, 窗口 **0.03**~0.300 起步) / `foot_clearance`(-2.0) `foot_slip`(-0.1) `stance_dev`(-0.2, 基准=**实测站姿**) `head_dev`(-0.2) `body_ang_vel`(-0.01) `angular_momentum`(-0.01) `action_rate`(**-0.03 → -0.10 课程**) `joint_vel`(-0.001) `joint_pos_limits`(-1.0) |
| 离地高度探针 | 从脚底网格抽 53/54 个采样点，按脚踝位姿转到世界系取**最低 z**（脚踝翻转时漂 ≤2mm；单点探针会漂 11~14mm） |
| 动态正则权重 | 两轮实测才定下来：先按"占正质量份额"缩放（-0.03/-0.01），再用**新奖励给旧 checkpoint 重新打分**发现 `body_ang_vel` 是"站着不动"最后一块优势（走路时 |ω_xy|≈3.6 rad/s 正是这只 25cm 机器人的自然区间，罚它就是罚正常走路）⇒ 降到 **-0.01**。`action_rate` 同理：按**实测 raw（5~25）**换算成 -0.03→-0.10，而不是按份额算出的 -0.55。⚠️ `angular_momentum` 实测只有 **1e-7** 量级（L∝质量×尺度²）—— 等于没接，留着只为对齐上游 + 打那条曲线 |
| 终止 | `time_out`（20s）/ 姿态偏 > 70° / **`fallen_too_long`**（倒地超 1 秒才终止，留恢复样本）/ `nan_state`。**没有"立刻"的高度终止**：实测那会砍掉 100% 的尝试回合 |
| 步态信号 | 两只脚 vs 地形的接触传感器（`feet_ground_contact`）：`air_time`、`foot_slip`、`foot_clearance` 都靠它 |
| std 标定 | `track_lin_vel` 的 `std=0.20`：**按"能走通的实际误差"标定**（上游真机在用的 `alpha_walking.onnx` 在我们的环境里能走——离地 46%/摆动 170ms/倾角 2.8°/零摔倒——但只跑到指令速度的 ~45%，两个环境实测一致）。std=0.10 会惩罚一切真步态（真步态 − 站着 = **−0.31~−0.55**）；0.20 时 **+0.35~+0.81**（走 > 站 > 溜） |
| 机器人模型 | 走路任务用 **`robot_walk.xml`**（5 个碰撞面：trunk+腿×2+脚×2），**不用**站立里程碑继承的 `robot_allcollisions.xml`（11 个，多出 jaw_soft×3 + hip×2 + trunk×1 → 鸭子能下巴/髋撑地趴着薅分）。上游同样分工：walk 用 walk 模型，能躺下/起身的 VelStand 才用 all-collisions |
| 姿态门 | `_posture_gate`（躯干高度 × 直立程度的软门，实测标定 0.065→0.090m / cos 0.60→0.90）乘在 `track_lin_vel`/`track_ang_vel`/`air_time` 上：躺平得 0、坐姿 ~20%、走/站得 1。没有它就会"躺平薅分"（实测 +1.8/步） |
| 方向门 | `_progress_gate`（指令方向上的速度 / 指令大小）乘在 `air_time` 上：站着 0、横着走 0、原地蹦 0.35、按指令走 1。没有它就会"原地蹦着薅步态奖励"（实测 2.04/步，是跟踪分的 25 倍） |
| ~~跟踪门~~ | （第七轮**已移除**：拿真步态标定发现它把"速度差 50%"平方级罚两次，导致真步态输给站着不动）原设计：`air_time` 还要乘**跟踪核**（与 `track_lin_vel` 同一个核）：只要求"有前进速度"时策略会退化成"匀速行进器"（不管指令要 0/0.15/0.30 都冲 ~0.26m/s 并摔倒）。实测超速 1/3.4、乱冲 1/13、零指令 0 |
| 频率 | 物理 0.002s × decimation 10 = 控制 50Hz（与站立/坐站任务一致，便于互相初始化） |

```bash
# 零成本"奖励排序"测量: 用**新奖励**给**旧 checkpoint** 重新打分
#   —— 不用训练就能回答"新奖励是否把'会走'排在'站着不动'之上"(排序翻不过来就别急着大规模跑)
uv run python scripts/eval_walk.py <旧run>/model_250.pt <旧run>/model_499.pt --terms

# headless 受控指令评估: 给任意 checkpoint 实测"到底会不会走"(而不是从曲线反推)
#   逐指令打印: 速度误差 / 倾角 / 躯干高度 / 有脚离地占比 / 摆动时长 / 峰值离地 / 终止原因
uv run python scripts/eval_walk.py logs/rsl_rl/microduck-walk/<run>/model_499.pt
uv run python scripts/eval_walk.py <run>/model_250.pt <run>/model_499.pt   # 横向对比轮次

# 量化探针(CPU 秒级, 不用训练): 把 P0/P1/P2 每条改动背后的数字打印出来
#   ① 离地高度探针漂移(四版对比) ② 动作尺度 vs 抬脚命中率 ③ upright 线性/高斯对比
#   ④ 一个步态周期的步态奖励质量 ⑤ actor/critic 维度与续训陷阱 ⑥ 逐项奖励量级+罚项符号
uv run python scripts/walk_p0_probe.py --envs 4 --steps 200

# 看环境（未训练: 随机策略, 鸭子会乱蹬）
uv run mjpython -m mjlab.scripts.play Mjlab-Velocity-Flat-MicroDuck --agent random --viewer native

# 看环境（零动作: 名义站姿, 但注意它**不是**稳定平衡 —— 实测零动作下鸭子约 0.25s
# 就从 0.5° 倒到 24°(倒立摆增长率 ~10/s ≈ sqrt(g/0.11m)), 这是物理不是 bug;
# 站立位姿得靠策略反馈才站得住。所以零动作只用来确认"环境建起来了、脚踩在地上"。)
uv run mjpython -m mjlab.scripts.play Mjlab-Velocity-Flat-MicroDuck --agent zero --viewer native

# 手动摇杆: 想看"给它一个速度指令它怎么反应", 得用网页查看器 —— 速度指令项的
# 滑条只挂在 viser 的 GUI 上(viewer/viser/viewer.py 里 create_gui 的调用点),
# 原生窗口没有。起来后浏览器开提示的 http://127.0.0.1:8080, 左侧 Commands 面板就是摇杆。
uv run python -m mjlab.scripts.play Mjlab-Velocity-Flat-MicroDuck --agent zero --viewer viser

# 回放训练好的行走策略（play 模式已自动用最宽指令范围、且不再随机推它）
uv run mjpython -m mjlab.scripts.play Mjlab-Velocity-Flat-MicroDuck --agent trained \
  --viewer native --checkpoint-file logs/rsl_rl/microduck-walk/<时间戳>/model_XXX.pt

# 一次看 4 只鸭子各跑各的（网页查看器, 不用窗口）
#   --num-envs 4 = 4 份独立物理, 每只采样到**不同**的指令 —— 实测 4 只 vx 分别是
#                  -0.03/-0.08/+0.33/+0.13, 所以它们真的各跑各的, 不是同一只的复制品
#   play 模式把 env_spacing 从 4m 压到 1.2m, 4 只才进得了同一个画面
#   viser 的 "Environment → Hide others" 默认**不勾选** = 同时画所有环境;
#   勾上就只画 "Select" 滑块选中的那一只
uv run python -m mjlab.scripts.play Mjlab-Velocity-Flat-MicroDuck --agent trained \
  --checkpoint-file model_3250.pt --num-envs 4 --viewer viser

# CPU 冒烟: 验证"行走任务训练链路能跑通"(约 10s/5 轮)
scripts/run_cpu.sh Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 16 --agent.max-iterations 5

# GPU 全量训练（Linux + NVIDIA）: 走路比站立难, 并行数别低于 4096, 轮数给足
uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 4096 --agent.max-iterations 5000 \
  --agent.save-interval 500 --agent.run-name walk-p1
```

**⚠️ 续训陷阱（P0② 之后才出现的）**：删掉 actor 里的 `base_lin_vel` 之后，行走的 actor 恰好也是 **51 维**，和站立任务一样 —— 但尾部 3 维语义不同：站立 `actor[48:51]` 是**实测机体线速度**，行走是**速度指令**。形状对得上 ⇒ `--agent.resume` 加载站立 checkpoint **不会报错**，策略却会把"指令"当成"实测速度"读 → 静默错训（比直接崩更坏）。要续训得先做 W4：把站立/坐站的 `base_lin_vel` 也挪到 critic，三个任务统一成"48 维本体感觉 + 尾部命令槽"。`tests/test_microduck_walk_env.py` 里有测试钉住行走的尾部 3 维就是命令槽。

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

# 指定显卡: **不传 = 用第 0 张**(默认 [0]); 多卡空格分隔; 全用可用 all
uv run train Mjlab-Stand-Flat-MicroDuck --gpu-ids 0 1        # 0、1 两张
uv run train Mjlab-Stand-Flat-MicroDuck --gpu-ids all        # 所有可见卡
# ⚠️ 别写 `--gpu-ids 1` 这种"单个数字"的形式: 该字段类型是 list[int]|Literal["all"]|None,
#    在真实 CLI(跟 task 的 union 解析一起)下会报
#      "Error parsing --gpu-ids: ... Value 0 (type int) does not match expected type list[int]"
#    (2026-09-12 实测踩到)。单卡就干脆不传这个参数。

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
行走任务额外看：`Episode_Reward/track_lin_vel`（跟不跟得上指令）、`Episode_Reward/air_time`（步态有没有长出来 —— 这条现在是主要塑形项）、`Curriculum/command_vel/lin_vel_x_max`（课程有没有推进）、`Metrics/twist/error_vel_xy`（速度误差）、`Episode_Termination/bad_orientation` 与 `root_too_low`（摔没摔）、`Metrics/air_time_mean`（摆动相时长）。

## 任务代码怎么改

- **调超参/并行数**：CLI 点号覆盖，如 `--env.scene.num-envs 4096`、`--agent.max-iterations 5000`、`--agent.save-interval 500`（配置即 dataclass，路径即字段）。
- **调速度指令范围**：改 `microduck_walk_cfg.py` 里的 `_VELOCITY_STAGES`。注意 `commands_vel` 课程每一步都会把指令范围改写成课程表里的值（已实测：`--env.commands.twist.ranges.lin_vel_x` 这种命令行覆盖只影响"第一次抽样"，之后每次重采都听课程的），所以想临时放宽就改课程表，别指望 CLI。
- **改奖励/观测/终止**：站立/坐站改 `microduck_cfg.py` 与 `microduck_sit2stand_cfg.py`，走路改 `microduck_walk_cfg.py`（每个奖励函数是 `(env) -> tensor`，额外参数写在 `RewardTermCfg(params=...)`），改完跑测试 + CPU 冒烟再上 GPU。
- **加新任务**：照 `microduck_walk_cfg.py` 的样子新建 cfg（实体/PPO 配置直接复用 `microduck_cfg.py` 的共享零件）→ 在 `tasks/__init__.py` 里 `register_mjlab_task`，id 命名 `Mjlab-<技能>-<地形>-<机器人>`。
- **环境"五件套"心智模型**：`scene`(世界) / `observations`(看到啥) / `actions`(控制啥) / `rewards`(啥算好) / `terminations`(何时结束)；速度类任务再多一件 `commands`(让它听指令)。
- **改完怎么判断"没写错"**：`pytest tests/` 秒级看维度/资产名/终止是否对（walk 的测试专门钉住"起步指令必须在课程第 1 档内"这类静默错误）→ `scripts/run_cpu.sh <task>` 分钟级看链路 → GPU 才谈学没学会。

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
- [x] R1 注册 `Mjlab-Sit2Stand-Flat-MicroDuck`（坐→站）
- [x] W1 注册 `Mjlab-Velocity-Flat-MicroDuck`（平地速度跟踪行走）+ 接触传感器步态奖励 + 指令课程 + 测试/CPU 冒烟
- [x] W1.5 对照上游 `../microduck_rl` 修 P0 五项（见 `microduck_walk_cfg.py` 顶部"P0 修正记录"）：air_time 改成每步付 + 权重 3.0、actor 去掉 `base_lin_vel`、upright 换高斯核权重 2.0、离地高度探针改用脚底采样点取 min、动作 scale 1.0。配套量化探针 `scripts/walk_p0_probe.py`
- [x] W1.6 修 P1 三项：`action_rate` 加权重课程、补 `body_ang_vel`(-0.03) 与 `angular_momentum`(-0.01)（后者实测为 1e-7 量级的"死项"，已在注释里写明）
- [x] W1.7 **500 轮 CPU 训练 + 数据复盘**（`logs/rsl_rl/microduck-walk/2026-09-10_16-56-41_walk-mac-500`）：趋势是"学会了站着不动"（250 轮时还在迈步、离地 32%，499 轮变成完美静止、指 0.15m/s 时一动不动）。根因三条：① `track_lin_vel` 的 std 是给 4 倍大指令尺度配的 → 站着白拿 71~90%；② `air_time` 窗口下限 0.125 是它当时摆动时长(20~30ms)的 4~6 倍 → 迈步零梯度；③ P1 的 `action_rate` 权重把 raw 量级估成 ~1.2（真实 5~25）→ 权重 -0.30 时占全部罚项 85%，且恰好在尝试迈步的窗口加压。配套新增 `scripts/eval_walk.py`
- [x] W2 修 P2 五项（针对上面三条根因）：track_lin std 0.25→0.10 + 指令第 1 档下限 0→0.10；air_time 窗口 0.125→0.03 起步 + 课程抬到 0.125；action_rate 权重阶梯重算为 -0.03→-0.10；删 `root_too_low` + 补 `nan_state`；track_ang_vel 0.5→1.0/std 0.5
- [x] W2.11 **第七轮：用"真步态"标定奖励**（发现 `../microduck/policies/alpha_walking.onnx` 是上游训练好的走路策略，接口与我们的 env 完全兼容：61 维 obs / action_scale 1.0 / 同一站姿与关节顺序）。做了三件事：① 写 `scripts/eval_upstream_policy.py` 在我们的环境里跑它——**它能走**（离地 46%、摆动 170ms、倾角 2.8°、零摔倒），证明我们的 env 能承载一个真走路策略；② A/B 对照（同一策略在**他们**环境里也只能跑到指令的 36~50%）→ 证明那是他们策略的精度，不是我们环境的问题；③ 用它逐项标定奖励，发现两处校准错误：**`air_time` 上的"跟踪核"门控**把速度误差平方级罚了两次（真步态输给站着不动：cmd 0.15 −0.31 / cmd 0.30 −0.55），**std=0.10 比真步态的实际精度还严**。修法：去掉跟踪核因子 + std 0.10→0.20 ⇒ 现在 **走 > 站 > 溜** 在每个指令下都成立（+0.35~+0.81）
- [x] W2.10 **第六轮：换回 walk 模型 + 对齐上游实现**（读 `../microduck_rl` 发现：他们的 cfg 文档里写着我踩过的**同一批坑** —— *"ANY positive reward for BEING in a fallen-ish state gets farmed"*、*"lying on its trunk can rhythmically tap its feet through the swing window"*、*"the run-3 avoidance/freeze mechanism"*、*"head-tripod at ~55°"*；他们的解法是结构性的：`air_time` 倒地时归零、potential-based `upright_progress`/`height_progress`、`joint_torque_rate_l2` 防抖、Phase-1 保留摔倒终止、逆向课程出生点、走路数据份额 ≥55%）。本轮先做最便宜且最可能一次性消掉一整类退化解的那条：**走路任务换 `robot_walk.xml`**（碰撞面 11→5，下巴/髋不能撑地），并在新模型上复核站姿基准（最大差 1.1°，旧表通用）
- [x] W2.9 **第五轮：修"匀速行进器"**（250 轮短测实测：`air_time` 权重还回 3.0 后第一次出现真正的摆动相 —— 离地 46%、摆动 121ms，但策略变成"不论指令都以 ~0.26m/s 前进"，`track_lin` 从 1.14 掉到 0.19，连零指令时也走并摔倒）。修法：`air_time` 再乘**跟踪核**（超速 1/3.4、乱冲 1/13、零指令 0）；`bad_orientation` 70°→45°
- [x] W2.8 **第四轮：修"原地蹦薅分"**（250 轮短测实测：离地 45%、摆动 121ms、竖向 |vz|=0.11，但机体系 vx 只有 0.07 —— 几乎不位移，`air_time` 却拿到 2.04/步 = 跟踪分的 25 倍）。修法：`air_time` 加**方向进度门**（原地/横着挪 = 0）+ 权重大幅下调 3.0→1.5；姿态项重做 —— 删掉以 **XML 默认姿势**为基准的 `joint_dev`（实测鸭子站稳时踝关节就差 17.6°，等于按它站不到的姿势扣分），换成以**实测站姿**为基准的 `stance_dev`(-0.2)。净效果（零训练成本重打分）：蹦跳 +2.76 → **+0.18**，站立 +3.52 → **+2.84**
- [x] W2.7 **第三轮：修"躺平薅分"**（250 轮短测实测：躯干 0.041m、倾角 50°、有脚离地 92%、全回合 time_out，`air_time` 被薅到 0.76~1.43/步）。根因是 W2 的两处改动叠加（删高度终止 + 窗口下限 0.03）。修法：新增 `_posture_gate` 姿态门（乘在三个技能类奖励上）+ `fallen_too_long`（倒地 1 秒宽限后终止）。修后用同一策略复测：合计 **+1.81 → +0.05**，终止原因全变 `fallen_too_long`
- [x] W2.0 **零成本奖励排序测量**（`eval_walk.py --terms`）：把旧 checkpoint 放进新奖励打分。第一版结果**排序还是平局**（vx=0.15：走路 +2.88 / 站着 +2.82），拆项发现 `body_ang_vel` 一项就占走路罚项的 -0.426 ⇒ 降到 -0.01 后**排序翻转**（+3.17 vs +2.98，vx=0.30 也翻），按训练配比加权 +3.20 vs +3.15。以后每次改奖励都先做这一步再谈大规模训练
- [x] W2.13 **"摇头"的第二层：头/脖子链（38% 质量在 4 个没人管的自由度上）**：`scripts/probe_head_wobble.py` 实测 —— 头链 = `neck_pitch` 以下子树 {neck, neck_pitch, yaw_roll_motion, jaw_soft} = **279.9 g / 737.2 g = 38.0%**（jaw_soft 一项 188.8 g = 25.6%）；4 个头关节偏离默认姿势 RMS 9.8~**37.2°**（head_yaw 峰值 57°）且**全部锁在步频 1.700 Hz**；嘴尖在**躯干系**里左右 RMS 33.9 mm / **峰峰 94 mm**（这就是眼睛看到的摇头），与左右脚载荷 **|r|=0.92、8/8 环境一致**；但头链质心只动 7.0 mm ⇒ 整机质心仅 2.7 mm = 两脚间距的 **4%** ⇒ **是空转不是配重**（真正的倒重心是躯干扭动，领先抬脚 60~100 ms）。奖励上唯一管头的 `head_dev`(-0.2) 实测只值 **-0.04/步**（air_time +2.51/upstream…），14 个动作通道里有 4 个只值 1%。注意 `head_camera`/`tof`/`head_imu` 都挂在这条链上 ⇒ 相机输出在晃动里没有意义。三条修法（冻结 / 给头下命令 / 加惩罚）写在 cfg 顶部 P3 记录里；测试 `test_head_chain_carries_38_percent_of_the_mass` 钉住这个质量前提。**零成本冻结 A/B（同 checkpoint 只把动作通道 5,6,7,8 置零）**：vx=0.15 倾角 1.47°→**4.58°**、侧向占比 21.4%→**59.5%**、直线度 0.949→**0.663**；vx=0.30 倾角 1.82°→**4.51°**、侧向 12.5%→**30.4%**⇒ **现有策略是靠甩头稳住的（头 = 反作用质量），不能直接冻**；注意头链质心只走 2.7mm（步宽 4%）⇒ 起作用的是**角动量**不是位置。因此真机头舵机的 kp/延迟必须与 BAM 配置核对，否则是 sim2real 地雷（比"好不好看"重要）
- [x] W2.12 **"摇头晃脑"的定量判定（第 3250 轮 checkpoint）**：新增 `scripts/probe_gait_wobble.py`（一次 rollout 提时间序列，不提训练）。实测躯干偏航摆动 **±12.4°**、wz RMS 1.86 rad/s（峰值 ≈240°/s），主频 1.625 Hz 与落脚步频 1.750 Hz 重合（相位锁在步态上），侧向速度只占前进速度的 12~21%，**路径直线度 0.95~0.97（没绕圈）**。关键判据：把 wz 与"左右脚载荷"做互相关，8/8 环境都在滞后 **−3~−5 步** 给出 |r|=0.60~0.67 ⇒ 扭动**先于**抬脚 ⇒ 那是这只 25cm 机器人的**倒重心机制**，不是没压住的抖动；奖励账上它也只用 0.49/步（track_ang_vel 1.0 里的一半）换 +2.51(air_time)+1.90(track_lin)。**结论：不改奖励去压它**；要让偏航轴听话，正路是给偏航轴下命令（turn-in-place 桶），见 cfg 顶部"P3 实测记录"
- [ ] L4 GPU 全量训练（站立 → 行走），配合 TensorBoard 观察收敛
- [ ] W2.1 **短期测量（250 轮 ≈ 13 分钟 @32 envs）**：判据 ① `Metrics/air_time_mean` 上行且 >0.05；② `eval_walk.py` 里"有脚离地%">20%、摆动时长>50ms、cmd=0.15 时 vx 误差<0.12；③ 终止里不再有 root_too_low。达标才进 W2.5 大规模训练
- [ ] W2.5 GPU 全量训练行走策略（Linux + 4096 envs，3000+ 轮）
- [ ] W2.6 仍未做（优先级靠后，理由：不针对实测失败模式，且"步态还没出现就加 DR 只会让信号更难"）：`pose` 换分档高斯形态、turn-in-place 命令桶、观测噪声/延迟/编码器偏置、域随机化、PPO 容量与轮数预算、ONNX 导出 runner、self-collision 惩罚 + 换 `robot_walk.xml`
- [ ] W3 行走健壮性：域随机化(foot_friction / encoder_bias / base_com)、推挤加强、粗糙地形 / 坡度
- [ ] W4 统一观测接口：把三个任务的观测维度对齐（站立/坐站补上恒零指令项），让站立策略能直接续训走路
- [ ] L5 回放验收 → ONNX 导出 → 真机部署（上游: export.py / infer_policy.py）
