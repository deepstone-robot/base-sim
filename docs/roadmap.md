# 路线图

完整的逐轮记录在仓库根的 `README.md`（含每轮的数字与根因）；
这一页是**压缩视图 + 下一步与理由**。

## 已完成

| 阶段 | 内容 | 证据 / 产物 |
|---|---|---|
| L1 | 模型侦察 + 实体配置（MJCF 接入、BAM 执行器、站姿） | `microduck_cfg.py`、`tests/test_microduck_env_cfg.py` |
| L2 | 站立任务五件套 + 冒烟测试 | 任务 `Mjlab-Stand-Flat-MicroDuck` |
| R1 | 坐→站任务（混合起步 + 无摔倒终止） | 任务 `Mjlab-Sit2Stand-Flat-MicroDuck` |
| W1 | 行走任务骨架：接触传感器步态奖励 + 指令课程 | 任务 `Mjlab-Velocity-Flat-MicroDuck` |
| W1.5~W2 | 对照上游修 P0/P1/P2 共十余项 | 见 [失败模式史](failure-modes.md) 各轮 |
| W2.0 | **零成本奖励重打分**方法确立（改奖励前的固定流程） | `eval_walk.py --terms` |
| W2.7~W2.11 | 七轮迭代修掉四种退化解（站着不动 / 躺平 / 原地蹦 / 匀速行进器） | [失败模式史](failure-modes.md) |
| W2.11 | 用上游真步态 ONNX 标定奖励（发现两处校准错误） | `eval_upstream_policy.py` |
| W2.12 | "摇头晃脑"定量化：判定为**倒重心机制**，不改 | `probe_gait_wobble.py` |
| W2.13 | "摇头"第二层：头链 38% 质量 + 零成本冻结 A/B | `probe_head_wobble.py` |
| W2.14 | **sim2real 随机化逐项生效性判决**（7 项 + 3 个静默 no-op 陷阱） | `probe_sim2real_dr.py` |
| — | 多只鸭子回放（`--num-envs 4`）+ play 布局调整 | [行走](tasks/walk.md) |
| — | 本套文档站（VitePress + GitHub Pages） | `docs/.vitepress/config.mts`、`.github/workflows/docs.yml` |

**当前行走策略的水平**：0.30 m/s 指令下跟踪误差 0.020 m/s、倾角 2.0°、52% 有脚离地、
零摔倒；跟踪精度已经超过上游真机在用的 ONNX 教师（后者只做到指令的 ~45%）。

## 下一步（按建议顺序）

### 1. sim2real 加固 —— **当前最大的缺口**

随机化清单已经逐项验证过（哪些生效、哪些是静默 no-op 都写在 [sim2real](sim2real.md)），
但**一行都还没进训练配置**：训练侧目前没有任何观测噪声和延迟，推挤只有 ±0.2 m/s。

要做的事（都是已验证可用的原语）：

- 第一批（mjlab stock 原语，不需要自定义代码）：脚底摩擦 `(0.7, 1.3)`、编码器偏置 ±0.015
  （**必须配 actor `biased=True`**）、躯干质心 ±3 mm、头部质心 ±3 mm、质量+惯量 ±5%、
  观测噪声、观测延迟、推挤 ±0.3 m/s。
- 第二批（需要自定义代码 + 单独验证）：**BAM 关节摩擦**（上游的 `FrictionDRBamActuator` 子类，
  我们的 bam 有 `_compute_friction_budget` 所以可搬，但因为"BAM 每步重写 `dof_*`"必须先实测）。
- 不做：转子惯量（我们这版 bam 会覆盖）、kp/kd 与关节阻尼随机化（上游已主动关掉）。

**为什么先做这个**：真机上第一个会崩的不是"走路好不好看"，是摩擦/延迟/质量偏差没被训练覆盖。

### 2. 导出与上机

- 走 ONNX 导出（当前任务用默认 runner，没有自动导出），核对**部署侧观测契约**：
  51 维、`[48:51]` 是**指令槽**而不是实测速度。
- 写一个"30 秒脚本化指令序列 + 摔倒检测"的上机脚本，先在真机上开环跑通链路。
- ⚠️ 站立/坐站任务的 actor 含 `base_lin_vel`（真机给不出），**这两个任务不能直接上机**；
  行走任务已经把它挪到 critic，可以直接导出。

### 3. 统一三任务观测契约（W4）

现在 walk 与 stand 的 actor **都是 51 维但尾部语义不同** ⇒ `--agent.resume` 不报错却会静默错训。
统一成"48 维本体感觉 + 尾部命令槽"之后：站立策略可以直接续训走路，跨任务迁移也会干净。
（顺带能把站立/坐站的 critic 真正变成特权组 —— 它们现在只是 actor 的拷贝。）

### 4. 让"摇头"变成有用的动作（产品决策，不是 bug 修复）

只有当**要用电台上的相机/IMU**时才需要做：移植上游的 `head_pose`（4 维命令 + 跟踪 + bias 惩罚），
把头从"自由维度"变成"有正事的自由度"。
现状：头以 1.7 Hz、峰峰 94 mm、±37° 在晃，`head_camera`/`tof`/`head_imu` 的输出在晃动里没有意义。
另外**真机头舵机的 kp/延迟必须与 BAM 配置核对** —— 策略靠甩头平衡，这个对不上就是 sim2real 地雷。

### 5. 其他（优先级更低）

| 项 | 说明 |
|---|---|
| 偏航轴下命令（turn-in-place 桶） | 让"该转的时候转"变成被跟踪的目标，而不是永远为 0 的误差项 |
| 指令范围外推 | 现在只训过 0~0.35 前进；负速度 / 大转向 / 侧移幅度都没验证 |
| 粗糙地形 / 坡度 | 现在只有平面；上游那套射线高度扫描在这个地形上才有意义 |
| 自碰撞惩罚 | `robot_walk.xml` 已经把危险碰撞面去掉了，但策略仍可能自穿 |
| Phase-1 式摔倒终止 + 逆向课程出生点 | 上游的做法：先学站、再学恢复，出生点从"接近成功"逆向铺开 |
| backlash 链路接通 | 3 个 `*_backlash.xml` 与 `add_backlash.py` 都好了，但 mjlab 侧的 `BacklashEncoderBamActuator` 还不存在（[机器人资产](robot.md)） |
| `error_vel_yaw` 改名/换定义 | 它测的是躯干偏航误差（被甩头放大），不是路径误差；现在的名字会误导 |

## 已知的文档/代码不一致（待收拾）

| 位置 | 问题 |
|---|---|
| `microduck_cfg.py` | `delay_max_lag` 注释写 `# (ms)`，实际单位是**物理步**（3 步 = 6 ms） |
| `microduck_cfg.py` | docstring 写"把 `robot_walk.xml` 接成实体"，实际加载 `robot_allcollisions.xml` |
| `microduck_cfg.py` | 初始根高度注释 0.12 vs 代码 0.1114；推力间隔注释"2~5 秒" vs 代码 2.0~4.0 |
| `microduck_cfg.py` | `_FULL_COLLISION` 定义了但**全仓库无人引用**（死代码，且启用它会静默降低非脚底 geom 的 condim） |
| `microduck_sit2stand_cfg.py` | `stand_reward` 的高度基准 0.085 ≠ 实测坐姿 0.07 ⇒ 起步 15 mm 无梯度 |
| `probe_sim2real_dr.py` | `spread` 用 `and` 短路当守卫（碰巧能跑） |
| `eval_walk.py` | 落脚分支里有一小段死代码 |
| `tests/test_microduck_walk_env.py` | 两个 docstring 里的数字按旧指令上限 0.30 写的，现值是 0.35（断言没问题，注释要改） |
