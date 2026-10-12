# 测试

## 定位：训练前的质检台

```bash
uv run --with pytest pytest tests/ -q          # 全部（约 30 秒）
uv run --with pytest pytest tests/test_microduck_walk_env.py -q
```

改完 walk cfg → **先跑这个** → 再上 CPU 冒烟训练 → 最后才 GPU。
它挡的不是"代码写错会崩"（那种错误训练自己会报），而是**静默错误**：

> 改回去不会报错、但会让奖励指错方向、或者在 GPU 上白跑几千轮之后才发现。

三个文件是三层递进：

| 文件 | 规模 | 建环境吗 | 挡什么 |
|---|---|---|---|
| `tests/test_microduck_env_cfg.py` | 34 行 / 4 测试 | 否（只编译 XML） | 模型计数与执行器覆盖：`nq=21 / nv=20 / nu=14`；14 个执行器**恰好**覆盖 14 个舵机关节；STAND 站姿在关节限位内；XML 能编译（路径 + mesh 目录对） |
| `tests/test_microduck_env.py` | 26 行 / 2 测试 | 是 | 站立任务最小冒烟：`obs["actor"]`/`obs["critic"]` 都是 `(2,51)`、动作 `(2,14)`；随机动作 20 步后观测与奖励全 finite |
| `tests/test_microduck_walk_env.py` | 916 行 / 36 测试 | 是 | 行走任务质检台：观测契约、动作尺度、奖励标定、课程边界、资产不变量、终止条件、姿态门、DR 生效性 |

### 构造环境的手法（写新测试时照抄）

`test_microduck_walk_env.py` 顶部有 4 个 helper：

- `_model()`：**模块级缓存**编译"行走任务实际用的那个 XML"（`get_microduck_walk_entity_cfg().spec_fn().compile()`）。
  缓存是必须的 —— 每个测试都重编译会很慢。
- `_make_env(num_envs=2)`：`microduck_walk_env_cfg()` → `ManagerBasedRlEnv(device="cpu")`。
  **它保留课程与随机推**（与评估脚本刻意不同：测试要测"训练时的那份配置"）。
- `_make_venv()`：包一层 `RslRlVecEnvWrapper`，只在需要 `get_observations()` 时用。
- `_sole_min_z_from_mesh(ankle_delta, foot)`：**唯一的独立真值来源** —— 直接用 MuJoCo mesh 顶点
  算脚底最低点，并迭代把脚底"钉到地面"。凡是要验证"离地高度探针对不对"的测试，都拿它当基准。

四种摆姿势/推课程的模式（都比"跑几百步看结果"干净）：

| 模式 | 用法 | 为什么 |
|---|---|---|
| 纯 cfg 断言 | 直接读 `cfg.rewards[...]` / `cfg.terminations` | 不建环境，最快 |
| `write_root_state_to_sim`（13 维） | 给定躯干位姿/速度后 `sim.forward()` | 干净地量"给定姿态下的奖励/门值" |
| `write_joint_position_to_sim` | 直接写关节角，绕开动力学 | 用动作命令"屈膝抬脚"不行 —— 屈膝会让身体下沉、两只脚都还在地上，读数分不出左右。**这条是踩过坑之后写在注释里的** |
| `env.common_step_counter = N` | 直接推进课程 | 不用真跑一万步 |

## 36 个测试按主题分组

### ① 注册与形状冒烟

| 测试 | 钉住的静默错误 |
|---|---|
| `test_registered` | 任务名必须在注册表里 —— 改名/漏注册后训练脚本找不到任务 |
| `test_obs_and_action_shape` | actor 51 / critic 54 / action 14 —— 观测项增删后 PPO 输入层静默错位 |
| `test_random_steps_are_finite` | 动力学爆炸/接触 NaN 被奖励项吞掉 |
| `test_reward_and_termination_terms_exist` | 奖励 13 项、终止 4 项、课程 3 项的**集合必须精确相等** —— 删项/改名后 manager 静默少一项、权重白配 |
| `test_ppo_cfg_shape` | `experiment_name`、`(256,256)`、`num_steps_per_env=24`、`max_iterations≥4000`、`save_interval≤250` —— PPO 预算被改小/网络变窄 |

### ② 观测契约

| 测试 | 钉住的静默错误 |
|---|---|
| `test_actor_has_no_privileged_base_lin_vel` | `base_lin_vel` **只能**在 critic（真机给不出）；且 actor 最后一项必须是 `command` |
| `test_actor_tail_slot_is_command_not_velocity` | 用真实数据验证：指令变了 `actor[48:51]` 必须跟着变 —— **续训陷阱**：walk 与 stand 的 actor 都是 51 维、形状对得上、`--agent.resume` 不报错，但策略会把"指令"当"实测速度"读 |

### ③ 动作与执行器

| 测试 | 钉住的静默错误 |
|---|---|
| `test_action_scale_is_one` | 行走的 `joint_pos.scale == 1.0`，而 stand/sit 必须保持 0.5 —— scale=0.5 时抬脚要策略输出 1~1.8σ，学习信号变稀 |

### ④ 奖励标定与量级

| 测试 | 钉住的静默错误 |
|---|---|
| `test_air_time_reward_is_per_step_not_landing_instant` | 权重 3.0、窗口 `[0.03, 0.300]`、`command_threshold=0.01`；并算出"持续悬空 0.2 s 会付 8 步" —— 退回"落脚一次性给分"写法后步态奖励比速度跟踪低一个量级 |
| `test_upright_is_gaussian_on_trunk` | 权重 2.0、`std=sqrt(0.05)`、作用在 `trunk_base`；给定倾角量出"10° 处高斯版掉分 > 50 × 线性版" —— 退回 `0.5*cos` 线性版 ⇒ 歪着身子蹭几乎免费 |
| `test_action_rate_weight_curriculum_follows_stages` | 阶梯只增不减、首档 -0.03、末档 -0.10、切档边界正确；**并且直接读 `env.reward_manager.get_term_cfg(...).weight`** —— 钉住"写 `env.cfg.rewards[...]` 是静默 no-op（manager 初始化时 deepcopy）" |
| `test_regularizer_weights_are_sized_by_measured_magnitude` | `body_ang_vel`/`angular_momentum` 都是 -0.01、`action_rate` 终值 -0.10（**不是 -0.55**）、两套课程最后一档 ≤14000 步 —— 钉住"按拍脑袋定权重量级"和"最后一档排在 500 轮验证之外，永远不生效" |
| `test_standing_still_cannot_farm_the_top_command` | 最高指令下 `exp(-top²/std²) < 0.2` —— 钉住"std 太松 ⇒ 站立盆地赢" |
| `test_body_ang_vel_only_punishes_roll_and_pitch` | 与独立算法全等（`atol=1e-6`）—— 钉住"把偏航也罚了 / 罚项符号反了" |
| `test_angular_momentum_reads_model_sensor` | 读 `robot/root_angmom` 传感器、与平方和全等；并且断言**它在 -0.01 权重下确实很小**（1e-7 量级）—— 避免有人以为它在起稳定作用 |
| `test_track_std_is_calibrated_against_a_proven_walker` | `std == 0.20`；用"真步态只做到指令 45%"建模型算出 `walker_kernel > 0.35`、`standing_kernel < 0.2` —— 钉住"std 收紧到惩罚一切真实步态" |
| `test_penalty_signs_are_negative_costs` | 遍历所有负权重项，断言值 ≥0 且加权后 ≤0 —— AGENTS 那条"infallible check"的静态版：**符号反了就是"违规即奖励"** |

### ⑤ 离地高度探针 / 脚底几何

| 测试 | 钉住的静默错误 |
|---|---|
| `test_sole_point_table_matches_mesh_ground_truth` | 对 mesh 真值 `max\|r\| < 0.004 m`，且脚踝 ±30° 之间读数跨度 < 0.003 m —— 钉住"探针必须代表'脚底区域最低点'且不随脚踝翻转漂"（注释里记着**写错过三次**：脚踝 body +10.9 mm、脚底 site +14.1 mm、上游射线在平地上退化成 site 高度） |
| `test_foot_clearance_uses_sole_min_not_single_point` | 双脚着地时 `max < 0.01`；整体抬高 5 cm 后读数涨 >0.03 —— 钉住"探针量的是别的东西" |
| `test_foot_clearance_right_left_not_swapped` | 运动学摆左腿后 `after[0] > after[1] + 0.01` —— 钉住左右脚顺序搞反 |

### ⑥ 课程边界

| 测试 | 钉住的静默错误 |
|---|---|
| `test_air_time_window_opens_early_then_tightens` | 起点 0.03、终点 0.125；用 `common_step_counter` 推进后实测序列必须是 `[0.03, 0.03, 0.08, 0.125]` —— 钉住"窗口下限一上来就是 0.125 ⇒ '抬脚 30 ms' 和'完全不抬'同分 ⇒ 迈步零梯度，永远学不会第一次抬脚" |
| `test_startup_command_is_within_first_curriculum_stage` | `rel_forward_envs == 0.0`、起步指令必须落在课程第 1 档内、第 1 档不应有侧移/转向 —— 钉住"上游 `rel_forward_envs` 把直行环境硬抬到 ≥0.3 m/s（Go1 尺度常数），一打开就**越过整条课程上限**，在 GPU 上白跑几千轮才发现" |

### ⑦ 姿态门与"薅分"堵口

| 测试 | 钉住的静默错误 |
|---|---|
| `test_posture_gate_matches_measured_postures` | 四个实测姿势的门值：`(0.041 m, 50°)→0`、`(0.075, 5°)∈(0.2,0.8)`、`(0.095, 12°)→1`、`(0.117, 2°)→1` |
| `test_gait_rewards_are_gated_by_posture` | "指令=0 且鸭子不动"时：站立 >0.9、躺平 <1e-6 —— 钉住"技能类奖励没乘姿态门" |
| `test_air_time_requires_commanded_direction_progress` | 方向进度门六情形：站 0 / 横走 0 / 反向 0 / **原地蹦 ∈(0.3,0.4)** / 走一半 ≈0.5 / 按指令 1.0 —— 钉住"第三轮鸭子原地蹦（离地 45% 却不位移）照样薅 `air_time` 2.04/步" |
| `test_stance_penalty_is_calibrated_on_measured_stance` | 自然站姿 `cost < 0.01`、`\|_STANCE_OFFSET["left_ankle"]\| > 0.2`、叉腿 `cost > 1.0`、再叠正常迈步膝屈增量 <0.3 —— 既钉住"基准必须是实测站姿"，又钉住"别罚得太狠连正常迈步一起罚" |
| `test_all_skill_rewards_apply_the_posture_gate` | **源码级挡板**：`inspect.getsource` 里必须出现 `_posture_gate` / `_progress_gate` —— 钉住"新写的技能奖励忘了乘门"。注释说明了为什么用源码检查而不是行为检查：构造特定接触状态的行为测试容易写成假测试 |

### ⑧ 终止条件

| 测试 | 钉住的静默错误 |
|---|---|
| `test_falling_over_is_terminated` | 倾角 80° 终止；躯干压到 3 cm 但姿态不偏则**宽限期内不终止** —— 把"蹲"误判成"摔"会删掉最需要的恢复样本 |
| `test_no_root_height_termination_and_nan_guard` | `root_too_low` 必须**不在**终止项里（它曾终止 100% 的回合，正好在尝试迈步的窗口），`nan_state` 必须在 |
| `test_fallen_too_long_has_grace_period` | 低姿势连续 40 步（0.8 s）不终止、累计 55 步（1.1 s）必须终止、站起来一步清零、再躺 0.4 s 不触发 —— 1 秒宽限期的时间标定 |

### ⑨ 资产不变量

| 测试 | 钉住的静默错误 |
|---|---|
| `test_walk_task_uses_the_walk_model_not_allcollisions` | walk 模型**恰好 5 个碰撞面**、无 `jaw*`/`hip*`、两个脚底 geom 与 site 都在 —— 换回 allcollisions（多 6 个碰撞面）鸭子就能"下巴/髋撑地"趴着 |
| `test_head_chain_carries_38_percent_of_the_mass` | 头链成员必须是 `[neck, neck_pitch, yaw_roll_motion, jaw_soft]`、占比 36~40%、`jaw_soft > 20%` —— 钉住"头很重"这个**物理前提**；换模型/改质量会让摇头的全部结论作废 |

### ⑩ 传感器 / 命令采样 / play 配置

| 测试 | 钉住的静默错误 |
|---|---|
| `test_foot_contact_sensor_sees_both_feet_on_ground` | 接触传感器生效、`found.shape==(1,2)`、`root z > 0.09` |
| `test_standing_envs_get_zero_command` | `rel_standing_envs ≈ 0.1`，重采样 300 次必须抽到零指令 —— 部署时的"站着"状态失效 |
| `test_play_cfg_widens_commands_and_drops_push` | 课程清空、`episode_length_s ≥ 1e9`、每轴上限 **≥0.1** —— 最后一条钉住 **viser 摇杆 GUI 把滑块 `min` 写死 0.1、拿 `ranges.<轴>[1]` 当初值**，上限小于 0.1 会直接抛 `ValueError`（实测踩过） |

### ⑪ sim2real 随机化的静默 no-op

| 测试 | 钉住的静默错误 |
|---|---|
| `test_encoder_bias_dr_only_reaches_the_actor_when_biased_is_set` | 加 `dr.encoder_bias` + `biased=True` 后 actor 的 `joint_pos` 槽必须等于 `真值 + bias`；**不设 `biased` 时差值是 0.000e+00** —— 配置能跑、训练不报错、偏置也采样了，可策略一个数字都看不到 |

## 怎么加一个新测试

1. **问自己"这个错误会让训练白跑吗"**：如果只是"会崩"，不用写测试；如果是"不崩但指错方向"，就写。
2. **优先钉不变量，而不是钉数值**：例如"负权重项加权后必须 ≤0"（`test_penalty_signs_are_negative_costs`）
   比"某项等于 -0.123"更有价值。
3. **能用源码级挡板就别写行为测试**：`test_all_skill_rewards_apply_the_posture_gate` 用
   `inspect.getsource` 检查函数体里有没有 `_posture_gate`，比构造特定接触状态可靠得多。
4. **要真值就找独立来源**：`_sole_min_z_from_mesh` 是 mesh 顶点算出来的，不依赖被测实现。
5. **在 docstring 里写清"为什么"**：现有的测试几乎每个都写了"这次踩的是什么坑、
   当时的实测数字是多少"。这是这个文件最有价值的部分 —— 它让下一个人**不敢**把它改回去。

::: tip 测试里数字的三种强度

引用测试里的数字时注意区分：
**① 进了断言的阈值**（最硬，例如 `std == 0.20`、碰撞面恰好 5 个、门值表）；
**② 只写在 docstring 里的实测**（例如"上游 ONNX 离地 46%"、"站位时踝关节差 17.6°"）——
它们没被断言保护，可能随代码漂移；
**③ 脚本运行时打印的实测**（例如 `probe_sim2real_dr.py` 的判决表）—— 要复跑才能确认。
站内引用时都标了是哪一类。

:::