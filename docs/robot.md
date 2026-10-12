# 机器人资产

MicroDuck 的 MJCF 是 **onshape-to-robot 从 CAD 导出的**，加上两份后处理（拼接 `additional_xml`、
可选注入 backlash）。这一页讲清：**哪个任务用哪份模型、两份模型差在哪、关节顺序、质量分布、
脚底几何、执行器、以及编译后到底什么参数在生效。**

## 资产分层

| 层 | 文件 | 说明 |
|---|---|---|
| 机器人模型 | `robot_walk.xml`（428 行）、`robot_allcollisions.xml`（434）、`robot_allcollisions_rollers.xml`（470）、`robot_*_backlash.xml` ×3 | `worldbody` + `asset` + `actuator` + `sensor`，**没有地面、没有 keyframe** |
| 场景 | `scene.xml`、`scene_walk.xml`、`scene_rollers.xml`、`scene_ball.xml`、`scene_*backlash.xml` ×2 | `<include>` 机器人 + 地面/光照/keyframe；只给"看图"类脚本用 |
| 构建碎片 | `joints_properties.xml`、`sensors.xml`、`additional.xml` | **不是可独立加载的 MJCF**（无 `<mujoco>` 根或根是片段），由 `config_mjcf_*.json` 的 `additional_xml` 字符串拼接进机器人 XML |
| 构建脚本/配置 | `add_backlash.py`、`config_mjcf_*.json` ×6 | backlash 注入；onshape-to-robot 的输入配置 |
| 网格 | 47 个 `.stl`（+47 个 `.part`） | 6 个 XML 实际引用 43 个 |

**任务 ↔ 模型（实测加载点）**：

| 任务 | 模型 | 为什么 |
|---|---|---|
| `Mjlab-Velocity-Flat-MicroDuck`（行走） | **`robot_walk.xml`** | 最少碰撞面，防止"躺/趴/头撑地"薅分 |
| `Mjlab-Stand-Flat-MicroDuck` / `Mjlab-Sit2Stand-...` | `robot_allcollisions.xml` | 要能躺下、能撑地起身 |
| —（未接入） | `robot_*_rollers.xml`（脚换成 4 个被动轮）、`robot_*_backlash.xml`（注入齿隙） | 见文末"未接通的部分" |

## 两份模型只差碰撞面

逐行 diff：**body 集合、body 树、joint 集合与顺序、限位、site、sensor、actuator、mesh、
以及全部 inertial 质量完全一致**；连脚底碰撞几何都逐字段相同。差别只有两处：

| | `robot_walk.xml` | `robot_allcollisions.xml` |
|---|---|---|
| 可碰撞 geom 数 | **5** | **11** |
| trunk | 1（`power_support`，self-collision-only） | 2（多 `np_f970`） |
| hip | 0 | 2（`hip_l` / `hip_l_2`） |
| leg | 2（self-collision-only） | 2（改成 `collision`） |
| 脚底 | 2（`left/right_foot_collision`） | 2（完全相同） |
| **`jaw_soft`（头）** | **0** | **3**（`top_head_shell` / `jaw` / `bottom_head_shell`） |

多出来的 6 个碰撞面让鸭子**下巴/髋/躯干能撑地** —— 实测第二轮就是"50° 倾角躺着稳稳拿分"
（上游同样点名过这个坑：*head-tripod at ~55°*）。所以行走任务必须用 walk 模型，
测试 `test_walk_task_uses_the_walk_model_not_allcollisions` 钉住了"恰好 5 个碰撞面、无 jaw/hip"。

几何近似：**没有任何凸包或图元近似**，所有 collision geom 都是 `type="mesh"`，直接复用对应
visual geom 的**同一个 STL**（`pos`/`quat`/`material` 逐字符相同），全库没有一条 collision geom
带 `size`/`scale`/`fromto`。唯一的简化来自导出配置 `simplify_stls: true, max_stl_size: 1.0`。

## 关节顺序（= qpos 顺序，14 个驱动关节）

| # | 名字 | 所属 body | 限位 (rad) |
|---|---|---|---|
| 1 | `left_hip_yaw` | **`yaw2roll`** | −0.436 ~ +0.524 |
| 2 | `left_hip_roll` | `hip_l` | ∓0.384 |
| 3 | `left_hip_pitch` | `upper_leg_left` | ∓1.571 |
| 4 | `left_knee` | `leg` | ∓1.571 |
| 5 | `left_ankle` | `ankle_left` | ∓1.571 |
| 6 | `neck_pitch` | `neck` | −1.571 ~ +1.047 |
| 7 | `head_pitch` | `neck_pitch` | ∓1.571 |
| 8 | `head_yaw` | `yaw_roll_motion` | ∓2.967 |
| 9 | `head_roll` | `jaw_soft` | ∓0.436 |
| 10 | `right_hip_yaw` | **`bearing_roll`** | −0.524 ~ +0.436 |
| 11 | `right_hip_roll` | `hip_l_2` | ∓0.384 |
| 12 | `right_hip_pitch` | `upper_leg_right` | ∓1.571 |
| 13 | `right_knee` | `leg_2` | ∓1.571 |
| 14 | `right_ankle` | `ankle_right` | ∓1.571 |

`nq = 7（freejoint）+ 14 = 21`，`nv = 20`，`nu = 14`（`tests/test_microduck_env_cfg.py` 钉住这三个数）。

::: warning body 命名不对称，写配置时最容易踩

- 承载 `left_hip_yaw` 的 body 叫 **`yaw2roll`**，承载 `right_hip_yaw` 的 body 叫 **`bearing_roll`**
  （不是 `yaw2roll_2`）。注意 `bearing_roll` **不属于头链** —— 它是右髋的 yaw 连杆。
- 右腿是 `hip_l_2` / `upper_leg_right` / `leg_2`，左右命名不统一。
- `head_pitch` 的 body 叫 `neck_pitch`，`head_roll` 的 body 叫 `jaw_soft`（"jaw_soft" 其实是整个头壳）。

:::

### body 树

```
trunk_base                      ← freejoint
├── yaw2roll          → left_hip_yaw
│   └── hip_l         → left_hip_roll
│       └── upper_leg_left → left_hip_pitch
│           └── leg   → left_knee
│               └── ankle_left  → left_ankle
├── neck              → neck_pitch
│   └── neck_pitch    → head_pitch
│       └── yaw_roll_motion → head_yaw
│           └── jaw_soft    → head_roll
└── bearing_roll      → right_hip_yaw
    └── hip_l_2       → right_hip_roll
        └── upper_leg_right → right_hip_pitch
            └── leg_2   → right_knee
                └── ankle_right → right_ankle
```

## 质量分布（**这只鸭子的关键物理事实**）

总质量 **737.2432 g**（两种模型完全相同 —— 碰撞面不贡献质量；
rollers 变体轻 0.41356 g，就是脚部换件的差）。

| body | 质量 (g) | 占比 |
|---|---|---|
| `trunk_base` | 199.2240 | 27.02% |
| **`jaw_soft`（头）** | **188.7660** | **25.60%** |
| `yaw_roll_motion` | 48.6000 | 6.59% |
| `upper_leg_left` / `_right` | 48.2067 each | 6.54% each |
| `neck` | 36.8414 | 5.00% |
| `ankle_left` / `_right` | 30.0246 / 30.0251 | 4.07% each |
| `yaw2roll` / `bearing_roll` | 23.0406 each | 3.13% each |
| `leg` / `leg_2` | 21.5844 each | 2.93% each |
| `hip_l` / `hip_l_2` | 6.1893 each | 0.84% each |
| `neck_pitch` | 5.7200 | 0.78% |

**子树质量**：`trunk_base` 199.22 + 左腿 129.05 + 右腿 129.05 + 头颈 **279.93** = 737.24 g
（四分支校验通过）。也就是说：

> **头/脖子链（`neck` 子树）占整机 37.97% 的质量，挂在 4 个关节上，位于躯干最上方。**

这条物理事实决定了行走时"摇头"的量级（见 [行走任务](tasks/walk.md)），
测试 `test_head_chain_carries_38_percent_of_the_mass` 把"头链成员 + 占比 36~40% + jaw_soft > 20%"
钉成契约 —— **换模型/改质量会让摇头的全部结论作废，必须重测**。

## site 与 sensor

### sites（7 个，6 个 XML 全同）

| site | body | 用途 |
|---|---|---|
| `imu` | `trunk_base` | 躯干 IMU（`orientation`/`angular-velocity`/`imu_*` 都挂它） |
| `left_foot` / `right_foot` | `ankle_left` / `ankle_right` | 脚部参考点（`eval_walk.py`、接触相关脚本用） |
| `head_camera` | `jaw_soft` | 头部相机（同时是一个 `<camera>`） |
| `tof` | `jaw_soft` | 测距 |
| `head_imu` | `jaw_soft` | 头部 IMU |
| `mouth_tip` | `jaw_soft` | **嘴尖** —— `probe_head_wobble.py` 用它量"摇头"摆幅 |

### sensors（6 个，`sensors.xml` 注入）

| sensor | 类型 | 挂载 | 噪声 |
|---|---|---|---|
| `orientation` | `framequat` | site `imu` | 0.001 |
| `angular-velocity` | `gyro` | site `imu` | 0.005 |
| `imu_ang_vel` | `gyro` | site `imu` | — |
| `imu_lin_vel` | `velocimeter` | site `imu` | — |
| `imu_accel` | `accelerometer` | site `imu` | — |
| `root_angmom` | `subtreeangmom` | **body** `trunk_base` | — |

`orientation`/`angular-velocity` 与 `imu_ang_vel` 语义重复（前面带噪声、后面不带），是同一物理量的两套读数。
**没有** jointpos/jointvel/touch/rangefinder 类 sensor —— 脚底接触、离地高度、自碰撞计数都在 mjlab 侧另建。

## 脚底几何与"离地高度"为什么必须采样

两个模型里的脚底 geom 逐字段相同（同一份 STL、同样的 `pos`/`quat`、都没有 `size`/`scale`）：

| 脚 | geom | mesh | pos |
|---|---|---|---|
| 左 | `left_foot_collision` | `sole_left` | `-0.022 0.00622291 -0.0647` |
| 右 | `right_foot_collision` | `sole_right` | `0.022 0.00622291 -0.0647` |

**脚底是曲面，不是一个点。** 实测：脚底贴地时把脚踝从 −30° 转到 +30°，
任何固定单点探针都会漂 **11~14 mm**（脚踝 body 高度 +10.9 mm；脚底 site +14.1 mm，更差）；
从脚底网格顶点按 4 mm 栅格取最低点（**53 点/脚**，右脚 54）则只漂 **≤1.93 mm**。
"摆动脚抬到 2 cm"这句话，只有用后者量才成立。

## 执行器：XML 一层 + mjlab 一层

**XML 侧**：每个机器人 XML 末尾有 14 个 `<position class="chosen_actuator">`（名字与关节同名，顺序同上）。
参数在 `joints_properties.xml` 里，现行 `chosen_actuator` 是：

```
joint    damping=0.053  frictionloss=0.0048  armature=0.0018
position kp=0.55  kv=0.0  forcerange="-0.96 0.96"  ctrlrange="-10.0 10.0"
```

同名类里还留着三代标定痕迹（`chosen_actuator_old` kp=0.52 / `_new` kp=0.386
（注释 *removing contaminated data*）/ `_antoine` kp=0.43），以及 `<!-- 200 kp -->`、`<!-- 125 kp -->`
两条注释 —— 标定过程留在文件里了。

**mjlab 侧**：`BamActuatorCfg` 接管这 14 个执行器，换成真实电机模型
（`microduck_cfg.py`）：

```python
BamActuatorCfg(motor_name="xl330", model="m6",
               target_names_expr=(r"^(?!passive_).*",),   # 排除被动关节
               kp_fw=200.0, vin_range=(6.5, 8.2), vin_min=6.0,
               delay_min_lag=3, delay_max_lag=3)           # 3 个**物理步** = 6 ms（不是 60 ms）
```

含义：**电压随电量波动、位置环增益、以及 6 ms 的执行延迟都建模了**。

::: danger 这个延迟的单位，配置注释写错了（实测纠正）

`microduck_cfg.py` 那行的注释写的是 `# (ms)`，但字段单位是**仿真步**
（bam 的字段说明是 *Minimum command delay in simulation steps*）。
实测：把 `apply_delay` 打桩计数，**一次 `env.step()` 里它被调用 10 次 = `decimation`**
⇒ 每个物理子步推进一格 ⇒ `lag=3` 是 **3 × 0.002 s = 6 ms**。
如果按注释当成毫秒读，会以为有 60 ms 延迟，那是 10 倍误差。
改任何与"延迟"有关的配置（尤其是以后加观测延迟做对比）之前，先记住这个口径。

:::

`kp_fw=200` 对应 XML 里那条 `<!-- 200 kp -->` 注释；`XmlActuatorCfg` 在配置里被注释掉（不用 XML 的 P 控制器）
—— 也就是说 XML 里 `chosen_actuator` 的 `kp="0.55"` 在 BAM 路径下**不决定控制律**，只当模型参数留存。

力限幅实测：`force_limit = max(vin_range) × kt / R = 8.2 × 0.366013 / 2.811392 ≈ **±1.07 N·m**`
（`vin=6.5` 时约 ±0.85 N·m）；BAM 会把 XML 的 `<position>` 执行器转成 motor 并写入这个 forcerange。

## 编译后实际生效的物理参数（实测）

跑一次 `microduck_walk_cfg` 并读编译后的模型，**6 个可碰撞 geom 全部是**：

| geom | body | condim | friction | solref |
|---|---|---|---|---|
| terrain | terrain | 3 | `[1.0, 0.005, 0.0]` | `[0.02, 1.0]` |
| trunk（`power_support`） | `trunk_base` | 3 | `[1.0, 0.005, 0.0]` | 同上 |
| leg / leg_2 | `leg` / `leg_2` | 3 | 同上 | 同上 |
| `left_foot_collision` | `ankle_left` | 3 | 同上 | 同上 |
| `right_foot_collision` | `ankle_right` | 3 | 同上 | 同上 |

两条推论：

1. **摩擦基线是 1.0**。所以 [sim2real](sim2real.md) 里 `geom_friction` 用 `operation="abs"`、
   范围 `(0.7, 1.3)` 等价于"在 1.0 上下 ±30%"，这是它该有的含义。
2. !!! danger "`microduck_cfg.py` 里有一份**定义了但没被引用**的碰撞配置"
    `_FULL_COLLISION = CollisionCfg(geom_names_expr=[".*_collision"], condim={脚底: 3, 其余: 1},
    friction={脚底: 1.0}, ...)` —— 全仓库只有定义处，`EntityCfg(...)` 没有传 `collisions=`，
    所以它**完全没生效**。
    它本来会把 9 个无名 collision geom 的 `condim` 降到 1（脚底保持 3）。
    现在实际生效的是 XML 默认值：**全部 condim=3**，物理上更准（condim=3 才有扭转/滚动摩擦）。
    结论：**别顺手把它接上** —— 那会让躯干/腿的接触从 condim=3 降到 1，是个静默的物理变化。
    要么删掉这段死配置，要么明确写清"只给脚底设 friction，不动 condim"。

## 未接通的部分（避免有人以为它们在工作）

| 东西 | 状态 |
|---|---|
| `robot_walk_backlash.xml` 等 3 个 backlash 模型 + `add_backlash.py` | **只生成到 XML，链路未接通**：3 个 backlash XML 没被任何 task 加载；`add_backlash.py` 注释里提到的 mjlab 侧 `BacklashEncoderBamActuatorCfg` / `joint_pos_rel_backlash` 在全仓库（含已安装的 mjlab/bam）**grep 命中 0 处**。做 `--backlash-deg`（默认 2.0 = 峰峰总间隙，关节 range 取 ±1°）之前要先补 mjlab 侧 |
| `robot_allcollisions_rollers.xml` + 4 个被动轮关节 | 未接入任何任务；无脚底 geom，nq=25；缺 `scene_rollers_backlash.xml` |
| `scene*.xml` 场景文件 | 只给 `scripts/view_microduck.py` 这类"看图"脚本用；任务不用场景文件（任务自带 `TerrainEntityCfg`） |
| `ball.xml`（70 mm/15 g 空心球） | 只被 `scene_ball.xml` include，未接入任务 |
| 4 个 STL（`left/right_upper_leg`、`trunk_shell_left/right`） | 未被任何 XML 引用（在导出配置的 `ignore` 白名单里） |

## 重建资产的流程（改 CAD 之后）

```
config_mjcf_*.json     →  onshape-to-robot（拼接 joints_properties.xml / sensors.xml / additional.xml）
                       →  robot_*.xml
add_backlash.py        →  robot_*_backlash.xml（可选，见上：mjlab 侧还没接）
```

每个机器人 XML 顶部保留了 CAD 溯源注释：walk/allcollisions 家族来自 element
`ef6e972847fec8d82570b35e`，rollers 家族来自 `ed34b749f5a3718f68024fd5`。
导出后还有两处 sed 后处理：`trunk_base` 初始高度定为 `pos="0 0 0.12"`、删掉 `inheritrange="1"`。
