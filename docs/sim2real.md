# sim2real 与域随机化

仿真里走得好 ≠ 真机上走得动。真机上一样的东西和仿真不一样，主要差在**这几处**：
脚底摩擦、编码器零位、质心/质量、关节摩擦与阻尼、执行器增益与延迟、观测噪声与延迟、外界扰动。
这一页记录的是：**哪些随机化项在这套栈（mjlab 1.3.0 + 我们装的 `bam`）里真的生效**，
以及在什么条件下会"配置能跑通、训练不报错、但什么都没随机化"。

::: danger 先读这一条

上游 `microduck_rl` 的注释里记着一串 **silent no-op** 教训，共同点是
**配置能跑通、训练不报错、但一个数字都没变**：

- stock `dr.dof_frictionloss` 在 BAM 下是 no-op（BAM 把 `dof_frictionloss` 置零了）
- 直接写 `body_mass` / `body_inertia` 在 mjlab 1.3.0 下不按环境展开（会塌成一个共享值）
- 写 `site_quat` 做 IMU 安装误差是 no-op（那些观测项根本不读 `site_quat`）
- `joint_pos_rel` 默认**忽略** encoder bias（要 `biased=True` 才读 `joint_pos_biased`）

所以本项目对随机化项的态度是：**每一项都要有一条能测出"sim 真的变了"的证据**，
那就是 `scripts/probe_sim2real_dr.py`。

:::

## 判决表（2026-09-12 实测，mjlab 1.3.0 + 本仓库装的 bam，16 环境）

用 `python scripts/probe_sim2real_dr.py` 复现。判决口径：
**生效 = 字段确实按环境采样且被消费；no-op = 配置能跑但什么都没变。**

| 项 | 判决 | 证据 |
|---|---|---|
| ① 脚底摩擦 `dr.geom_friction` | **生效** | 两脚 0.74~1.26（目标 0.7~1.3），环境间跨度 0.52，`shared_random=True` 确实让两脚同值 |
| ② 编码器偏置 `dr.encoder_bias` | **生效** | 字段按环境采样（±0.015 rad），与 actor 观测之差 1.5e-8 |
| ③ 质心偏移 `dr.body_ipos` | **生效** | 偏移 ≤0.0030 m，连续 5 次 reset **不累积**，环境间 σ=0.0017 m |
| ④ 质量+惯量 `dr.pseudo_inertia` | **生效** | 躯干质量 190.7~208.4 g（跨度 8.9%），惯量同步变 |
| ⑤ 转子惯量 `dr.joint_armature` | **no-op** | 倍数恒为 1.000，跑一步后仍是 1.000 ⇒ **BAM 每步重写 `dof_armature`，把随机值覆盖了** |
| ⑥ 观测噪声 `Unoise` | **生效** | actor−critic 同槽相减 σ=0.1441，理论 0.25/√3=0.1443 |
| ⑦ 观测延迟 `delay_min/max_lag` | **生效** | 延迟 1 时观测=上一步状态（0.0163 < 0.0615）；延迟 0 时=本步状态（0.0000 < 0.0788） |
| ⑧ BAM 关节摩擦 | **待验证** | 需要 `FrictionDRBamActuator` 子类；见下文"第二批" |

### 三个陷阱，逐条说清后果

**陷阱 1：`geom_names=()` 是静默 no-op。**
上游模板里 `dr.geom_friction` 的 `asset_cfg` 写的是 `SceneEntityCfg("robot", geom_names=())` 加一句
`# Set per-robot`。实测：**留空会改动 0/76 个 geom** —— 不报错、不警告，脚底摩擦一点没随机化。
必须按模型里的真实名字填（本模型是 `left_foot_collision` / `right_foot_collision`）。

**陷阱 2：编码器偏置不设 `biased=True` 就完全没用。**
实测：加了 `dr.encoder_bias` 事件、偏置也确实采样了（±0.015），但用默认的 `joint_pos_rel` 时
`|actor观测 − 真实关节角| = 0.000e+00` —— **策略一个数字都看不到**。
设 `biased=True`（actor 组）后观测与 `encoder_bias` 之差 1.5e-8（相等）。
critic 组保持 `biased=False`，让它保有特权信息。
测试 `test_encoder_bias_dr_only_reaches_the_actor_when_biased_is_set` 钉住了这件事。

**陷阱 3：reset 模式的事件在 env 构造完成时还没生效。**
实测：用 `mode="reset"` 的 `dr.body_ipos`，构造完立刻读字段是 **0**；显式 `env.reset()` 一次之后才变成
≤0.003 m。含义是每个环境的**第一条回合**跑的是名义参数（之后按回合重采样）。
如果你要的是"整轮训练都固定一组随机参数"，用 `mode="startup"`（例如 `foot_friction`、`pseudo_inertia`）。

## 上游现用的参数（照抄对象）

`microduck_rl` 的 `microduck_velocity_env_cfg.py` 里这些是**审计后**的值，注意和 stock 默认的差别：

| 项 | 上游值 | 备注 |
|---|---|---|
| 脚底摩擦 | `(0.7, 1.3)`，startup，abs，shared | 从 stock `(0.3, 1.2)` 收窄成"防滑脚垫" |
| 编码器偏置 | `±0.015 rad`（±0.86°/关节，每环境常数） | 必须配 `biased=True` |
| 躯干质心 | `add ±0.003 m` | 起步值，课程里再往上抬 |
| **头部质心** | `add ±0.003 m`，作用于头链 5 个 body | 对我们的意义更大：头链占整机 **38%** 质量，见 [机器人资产](robot.md) |
| 质量 + 惯量 | `(0.95, 1.05)`，`pseudo_inertia`，startup | `alpha_range = ln(range)/2`；实测质量与惯量各 ±5% |
| 转子惯量 | `(0.9, 1.1)` | 上游说在 BAM 下生效；**我们实测是 no-op**（见判决表 ⑤） |
| 推挤 | `±0.3 m/s`，每 3~6 s | 上游注释：±0.5 的加性踢比最大步速还大，会训出"永久紧张的恢复步态" |
| 观测噪声 | ang_vel ±0.03 / gravity ±0.01 / joint_pos ±0.001 / joint_vel ±0.25 | 全部是**审计后大幅收窄**的值（stock 是 ±0.2/±0.15/±0.05/±2.0），理由是"真实 dxl IMU 路径很快，±20 ms 包络" |
| 观测延迟 | ang_vel 0~1、gravity 0~1、joint_vel **1~1（=20 ms）** | 延迟单位是控制步 |
| IMU 安装误差 | 随机轴 ε ≤ 6° | 必须在**观测层**做（写 `site_quat` 是 no-op） |

**上游主动关掉的项**（照抄这个决定，别自作聪明打开）：

- `ENABLE_KP_RANDOMIZATION = False  # Was True`
- `ENABLE_KD_RANDOMIZATION = False  # Was True`
- `ENABLE_JOINT_DAMPING_RANDOMIZATION = False`

**我们当前训练配置的真实状态**：训练侧**没有任何观测噪声和延迟**（`enable_corruption=False`
只出现在 play 分支，是对的；训练侧从来没设过 `noise=`）。推挤只有 ±0.2 m/s。
也就是说：**上真机前这一页列的东西，一行都还没进训练配置。**

## 加固清单（分批）

### 第一批：用 mjlab stock 原语，不需要写自定义代码

| 项 | 原语 | 参数 |
|---|---|---|
| 脚底摩擦 | `dr.geom_friction`（startup, abs, shared） | `(0.7, 1.3)`，`geom_names` 必须按模型填 |
| 编码器偏置 | `dr.encoder_bias`（startup）+ actor `biased=True` | `±0.015` |
| 躯干质心 | `dr.body_com_offset`（reset, add） | `±0.003 m` |
| 头部质心 | `dr.body_ipos`（reset, add） | `±0.003 m`，头链 4 个 body |
| 质量 + 惯量 | `dr.pseudo_inertia`（startup） | `alpha_range = (ln0.95/2, ln1.05/2)` |
| 观测噪声 | `Unoise` + `enable_corruption=True`（仅 actor） | 见上表 |
| 观测延迟 | `term.delay_min_lag / delay_max_lag`（**是 term 的字段，不是 params**） | 见上表 |
| 推挤 | `push_by_setting_velocity`（interval 3~6 s） | `±0.3 m/s` |

### 第二批：需要自定义代码，且必须先验证

**BAM 下的关节摩擦**。MuJoCo 的 `dof_frictionloss` 被 BAM 置零 ⇒ stock `dr.dof_frictionloss` 是 no-op。
上游的解法是 `FrictionDRBamActuator`（35 行子类，重写 `_compute_friction_budget` 乘一个
per-env `friction_scale`）+ `randomize_bam_friction` 事件 + `expand_bam_friction_fields`
（注册 per-world 展开）。**好消息**：我们装的 bam 里
`BamActuator._compute_friction_budget` 与 `kp_scale` **都在**，所以那个子类理论上可直接搬。
**但**判决表 ⑤ 已经证明"BAM 每步会重写 `dof_*` 字段"，所以这条必须单独实测：
同一个动作序列下把 `friction_scale` 设成 0.2 / 2.0，看关节轨迹是否不同（相同 = no-op）。

### 不做的项

- **转子惯量**：我们这版 bam 会覆盖，做了也没用（要么改 BAM 侧，要么放弃）。
- **kp/kd / 关节阻尼随机化**：上游关掉了（`Was True`），照抄关掉的决定。
- **IMU 安装误差**：等要用电台上的相机/IMU 做实际任务时再做 —— 现在做只是徒增配置面。

## 怎么确认"加固真的加上了"

```bash
# 逐项判决：每一项都要能说出"开了以后 sim 哪里变了"
uv run python scripts/probe_sim2real_dr.py

# 加固前后用同一个 checkpoint 对比"策略在扰动下还站不站得住"
uv run python scripts/eval_walk.py <new>/model_XXXX.pt --vx 0.15 --terms
```

::: warning 别用"加了随机化所以更稳"说服自己

这个仓库里翻过车的都是这一类判断：`action_rate` 权重 -0.30 是照抄来的、实测占全部罚项 85%，
直接把"尝试迈步"的窗口压死（有脚离地从 32% 掉到 0.9%）；
`angular_momentum` 权重 -0.01 看着像个稳定项，实测量级只有 **1e-7 = 等于没接**。
加任何一项之前，先量它到底有多大、以及它到底有没有接上。

:::