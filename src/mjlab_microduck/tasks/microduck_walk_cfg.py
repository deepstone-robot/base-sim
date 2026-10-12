"""W 里程碑: 行走(平地速度跟踪)任务环境 —— 注册为 `Mjlab-Velocity-Flat-MicroDuck`。

技能目标: 鸭子听懂"速度指令", 两条腿交替迈步往前走/转向, 全程不摔。

设计要点(为什么这样写):
  - 指令制(command, 上游 mjlab 速度任务同款接口): 每 4~8 秒重采一次
    (vx, vy, wz), 指令进观测。好处是"走多快/往哪转"变成策略的输入参数,
    以后真机部署直接下发速度就行, 不用重训;
  - 起步只开很小的前进速度(0~0.15 m/s), 用 commands_vel curriculum 在训练中
    逐步放宽到 0~0.35 m/s + 转向/侧移: 鸭子腿长才 ~11cm, 一上来就要它跑是学不会的
    (速度尺度怎么定的见下面 _VELOCITY_STAGES 上方注释);
  - 留 10% 的环境执行"零指令"(rel_standing_envs): 逼策略别忘了站立能力,
    否则"往前走"会退化成"往前扑";
  - 迈步形态靠奖励塑形(接触传感器 + 脚底射线高度), 这是小双足能不能迈开腿的关键:
      air_time  每步都在窗口内就付分(鼓励"保持摆动相"这个持续状态, 详见函数注释);
      clearance 摆动脚要抬到目标高度附近(按脚底区域最低点量), 贴地拖/抬太高都扣分;
      slip      支撑脚接触点的水平速度(打滑)扣分;
  - 终止只要 time_out / 高度过低 / 姿态超限: 平地不需要复杂终止条件;
  - 控制频率 50Hz(timestep 0.002s + decimation 10), 与站立/坐站任务一致。

## P0 修正记录(2026-09-10, 对照 ../microduck_rl 后改的 5 处)

对照上游那份"reward 设计教训"playbook(../microduck_rl/AGENTS.md)发现的问题, 按
"会不会让奖励指错方向"筛出来 5 条先修。每条都能量化, 不是口味问题:

  ① air_time: 原来"落脚那一瞬间给一次分"(0.5 分/周期) → 改成"每步都在窗口内就付"
     (权重 3.0, 窗口 0.125~0.300s → ~54 分/周期, 差 ~50 倍)。
     为什么关键: 唯一强激励只有速度跟踪时, "前倾蹭地"和"迈步"都能压速度误差,
     而前者便宜得多 —— 步态奖励太薄 = 策略去选那条更便宜的路。
  ② actor 观测删掉 base_lin_vel(真机给不出这个量, 上游只留给 critic), 51 维;
     critic 保留它当特权信息(54 维)。原来放在 actor 里 = 训练开卷考试(速度跟踪虚高),
     部署时接口不成立。
  ③ upright: 线性 -g_z(权重 0.5) → 高斯 exp(-Σg_xy²/std²)(std=sqrt(0.05), 权重 2.0)。
     同样倾 0.1rad 的惩罚从 0.0025 分/步 → 0.36 分/步(≈140x)。原版太软, 而整套
     奖励里没有第二项在罚"静态倾角"(track_ang_vel 罚的是角**速**度)。
  ④ 离地高度的探针: 猜错两次才修对 —— ① 脚踝 body 高度(漂 +10.9mm@30°) →
     ② 脚底 site 高度(漂 +14.1mm, 更糟) → ③ 上游那套射线传感器(平地上 min 退化成
     site 高度, 没用) → ④ 最终: 从脚底网格抽 53 个采样点、按脚踝位姿转到世界系取
     最低 z(漂 +1.9mm)。目标 0.02、权重 -2.0。滑动速度的测点同时换成脚底 site
     (接触点速度才是打滑), 权重上游化 -0.1。
  ⑤ 动作 scale 0.5 → 1.0: 抬脚需要的膝屈 ~0.5~0.9rad, 0.5 尺度下要策略输出
     1.0~1.8(≈1~2σ)才摸得到, 1.0 尺度下 <1σ —— 探索能不能覆盖到有用的区域。

## P1 修正记录(同一批对照的后续, 只动"会拖慢收敛"的三项)

  ⑥ action_rate_l2 从**恒定 -0.05** 改成**权重课程** -0.05 → -0.55(见
     _ACTION_RATE_WEIGHT_STAGES)。要点是"按正质量份额"定终值而非照抄上游 -1.0。
  ⑦ 补 body_ang_vel(-0.03) 与 angular_momentum(-0.01) 两项动态正则,
     权重同样按份额缩放; 模型自带的 subtreeangmom 传感器(robot/root_angmom)直接用。

## P2 修正记录(2026-09-10, 依据: 500 轮 CPU 训练 + headless 评估的数据)

那次跑完的趋势是"**学会了站着不动**": 第 250 轮还在尝试迈步(有脚离地 32%),
第 499 轮变成完美静止(离地 0.2%、指 0.15m/s 时一动不动、满回合 4s 不摔)。
三条原因都能复算, 对着改:

  ⑧ track_lin_vel 的 std 0.25 → 0.10: 0.25 是给 4 倍大的指令尺度配的, 我们的指令
     均值只有 0.075~0.147 → 站着不动白拿 71~90%。同时把指令课程第 1 档下限
     从 0 抬到 0.10(零指令场合由 rel_standing_envs=10% 单独覆盖)。
  ⑨ air_time 窗口下限 0.125 → 0.03 起步 + 课程抬上去: 实测一开始只能摆动 20~30ms,
     0.125 的下限让"抬脚"零梯度(抬 30ms 和完全不抬都是 0 分, 鸡生蛋)。
  ⑩ action_rate 权重阶梯重算(-0.05→-0.55 改成 -0.03→-0.10): 上一版把 raw 量级估成
     ~1.2(用 0.3×randn 探针量的), 真实策略 raw 是 5~25 → 权重 -0.30 时每步
     -1.66~-3.0、占全部罚项 85%, 而它恰好在 125~250 轮加压, 把"尝试"税死了。
  ⑪ 终止: 删 root_too_low(第 250 轮它终止了 100% 的回合, 正好在尝试迈步的窗口),
     补 nan_state(上游速度任务也是 time_out / 70° / 出界 / NaN, 没有高度项)。
  ⑫ track_ang_vel 权重 0.5→1.0、std 0.4→0.5(上游 2.0/sqrt(0.5), 取一半)。

## P2 后续修正记录(第三轮, 依据: 250 轮短期测量的实测)

那次短测跑出的是"**躺平薅分**": 250 轮后策略躯干只有 0.041m、倾角 50°、有脚离地 92%、
所有回合 time_out 永不终止 —— 它把 air_time 从 0.08/步薅到 0.76~1.43/步(腿举在空中
永远满足摆动窗口), track_lin 还白拿 0.66, 而唯一罚它的 upright 只有 0.06(而不是 2.0)。
根因是我上一轮的两处改动叠加: ① 删掉高度终止 → 躺下不再是"死"; ② 窗口下限降到 0.03 →
任何姿势下腿一抬就算摆动。上游 playbook 早就写了这条: "Never gate a positive reward on
being in a bad state — the policy parks in the cheapest qualifying pose and farms it."

  ⑬ 新增**姿态门** `_posture_gate`(躯干高度 × 直立程度的软门, 实测标定 0.065→0.090m /
     cos 0.60→0.90), 乘在 track_lin_vel / track_ang_vel / air_time 上:
     躺平得 0、坐姿得 ~20%、走路站立得 1; 是软斜坡(爬起来有递增的分, 有梯度)。
  ⑭ 新增 `fallen_too_long` 终止: 倒地(低于 0.065m 或倾角 > 45°)持续超过 1 秒才终止
     —— 既不砍恢复尝试, 又不让"赖在地上"变成永久薅分点; 重置回站立出生点还能把
     训练样本的组成拉回"站着"。

## P2 后续修正记录(第四轮, 依据: 250 轮短测 + headless 实测)

这轮跑出的是"**原地蹦着薅分**": 最终策略有脚离地 45%、摆动 121ms、竖向 |vz|=0.11,
但机体系 vx 只有 0.07(指令要 0.15~0.30) —— 几乎不位移, 却把 air_time 从 0.13 薅到
**2.04/步**(是 track_lin 的 25 倍, 占它正奖励的 92%)。用户看到的"横着走两步就倒"
就是这个: 步态奖励**与"有没有按指令方向前进"完全无关**, 于是"原地抬脚"成了最优解。
(同一类问题的第三次变体: 第一轮"站着不动"、第二轮"躺平"、第三轮"原地蹦"。)

  ⑮ `air_time` 加**方向进度门** `_progress_gate`: 按"指令方向上的速度 / 指令大小"给分,
     原地或横着挪 ≈0、真按要求走 →1。判别式实测: 站着 0.000 / 横着走 0.000 /
     原地蹦(0.07) 0.350 / 半速 0.500 / 按指令 1.000。
  ⑯ `air_time` 权重 3.0 → **1.5**: 3.0 时它单独就能拿 2.04/步, 塑形项变成了目标本身;
     降到 1.5 后"完全做到"也只有 1.5, 低于速度跟踪的 3.0 —— 排序回到"先会走、再磨步态"。
  ⑰ 姿态项重做: 删掉原来的 `joint_dev`(二次罚, 基准是 **XML 默认姿势**) —— 实测发现
     鸭子**站稳时的踝关节就比 XML 默认转了 17.6°**(hip_roll 4.5°), 拿 XML 默认当基准
     等于按"它站不到的姿势"扣分(上游 playbook 那条"目标要从实际机器人上量"的关节角版本)。
     换成 `stance_deviation_penalty`: 以**实测站姿**为基准、分关节 std(hip_roll 紧 0.05、
     hip_pitch/knee 松 0.4)、低权重 -0.2;**不做成正奖励**(正奖励会给"站着"再添 1 分,
     把站立收益从 3.3 抬到 4.3, 反而更难学会走)。
  ⑱ 这一轮的净效果(用同一批 checkpoint 重新打分, 零训练成本):
     原地蹦 平均 +2.76 → **+0.18**; 站着不动 平均 +3.52 → **+2.84**; 蹦跳已被站姿与
     方向两道门+姿态罚压到"远差于站着", 而"真走起来"的上限 ≈5(跟踪 1.6 + 抬脚 1.2 +
     竖直 1.5 + 姿态 ≈0.8)。排序: 走(≈5) > 站(2.8~3.3) > 蹦(0.2) ≫ 躺(-0.3)。

## P2 后续修正记录(第五轮, 依据: 250 轮短测 + headless 实测)

这轮**第一次出现了真正的进展**: 策略开始**按指令速度前进**(实测机体系 vx = 0.198 / 0.259,
指令 0.15 / 0.30), 零指令时能满回合(4s)站住不动; 三种廉价解(站着不动/躺平/原地蹦)都被
门+权重压住了。剩下的问题换了一个层面 —— 不再"薅奖励", 而是:

  · 它用**"溜"**代替"走": 移动时两脚贴地(离地仅 5.5%、摆动 4ms), 靠前倾+滑行拿到指令
    速度, 然后 0.8 秒倒下(bad_orientation 每回合 1.1~1.5 次)。`foot_slip` 在 0.2m/s 下
    只罚 -0.004/步, 等于没有价格。
  · 根因是**"步态 vs 速度"的价差**: air_time 1.5 < track_lin 2.0 时, "溜"足以满足目标,
    迈步只是手段; 上游反过来(air_time 3.0 > track_lin 2.0), 迈步才是目的本身。
  · 而且这是**策略空间**的局部最优而不是奖励空间的: 实测每步期望"溜" 2.19 < "站" 2.91
    < "走" ≈5 —— 奖励排序早就对了, 是 0.25M 转移的预算不足以让策略自己爬出这个盆地。

  ⑲ `air_time` 权重 1.5 → **3.0**: 第三轮砍它是为了堵"原地蹦", 但真正的堵法是门(姿态+
     方向), 门在了权重就该还回去 —— 让"迈步"成为最大的正分项, 而不是可选项。
  ㉑ `air_time` 再乘**跟踪核**(和 track_lin_vel 同一个核)。只乘"方向进度门"是不够的:
     实测策略退化成"匀速 0.26m/s 的行进器" —— 不管指令要 0 / 0.15 / 0.30, 它都往前冲,
     对任何指令都满门, 于是既不必匹配速度、也不必站得住(连零指令时都走并摔倒, 回合
     从 200 步掉到 45 步)。乘上跟踪核后含义变成"**在把速度走对的前提下**抬脚":
     实测超速 1/3.4、乱冲 1/13、零指令 0。这才让"步态奖励 3.0 > 速度奖励 2.0"变得安全
     (否则它就是在奖励"随便往前走")。

  ⑳ `bad_orientation` 阈值 70° → **45°**: 实测"前倾"策略倒之前只到 13~14°, 过了 25~30°
     就已经救不回来; 70° 时它已在必倒轨迹上白跑 ~0.3s。45° 更早给出干净信号(正常步态
     俯仰远小于 25°)。

⚠️ 记在案但**没改**的两条结构性问题(上游结构相同, 不急着动):
  · `action_rate`(-0.03) 训练时实测收 **-0.847/步** —— 它罚的是**采样噪声**(σ≈0.7~1.0)
    而不是策略抖动, 等于给探索收税(std 250 轮只从 1.0 降到 0.71)。若大规模训练里 std
    掉得太快(探索枯竭), 就降到 -0.01~-0.015, 或改成只罚策略均值的变化。
  · 摔倒没有显式代价: 回合重置会把鸭子放回**站立出生点**(高价值状态), 所以"倒下"只损失
    未来的奖励, 没有一次性惩罚。上游 mdp.py 里有 `is_terminated` / `fallen_state_penalty`
    这组工具(我只用到了 fallen_too_long)。

还没改的(见 README 路线图): turn-in-place
命令桶、观测噪声/延迟/编码器偏置、域随机化、PPO 容量与轮数预算、ONNX 导出 runner、
self-collision 惩罚与 robot_walk.xml 模型。优先级排序的理由: 上面 ⑧⑨⑩ 直接针对
实测到的失败模式, 而 DR/噪声是"学会了再加固"的东西 —— 步态都还没有的时候加 DR
只会让信号更难。

## P3 实测记录(2026-09-12, 依据: GPU 4096 envs 第 3250 轮 checkpoint 的 headless 测量)

现象: 走起来"摇头晃脑"。用 `scripts/probe_gait_wobble.py` 量完的结论是**别改**,
记录在此以免下次又想去压它:

  指令 vx=0.15 / 0.30 时(各 8 环境 × 400 步, 无推扰、指令不变):
    实际速度          0.166 / 0.302 m/s   (跟得上)
    倾角              1.44° / 1.86°       (不晃倒)
    躯干偏航摆动      ±12.4° / ±12.4°     (去趋势残差, 峰峰值 25°/25°)
    偏航角速度        RMS 1.86 / 1.67 rad/s, 原始序列峰值 -4.24 rad/s(≈240°/s)
    摆动主频          1.625 / 1.750 Hz  == 落脚步频(1.750 / 1.875) → 相位锁在步态上
    侧向速度          |vy| 0.032 / 0.038 m/s = 前进速度的 21% / 12.6%
    路径直线度        0.954 / 0.974      (净位移/路程; 没有绕圈)
    相邻步 wz 符号翻转 0.20 / 0.26        (≈0.1 = 平滑低频; 不是每步乱抖)
    自检: 由 |wz| RMS 反推的偏航角峰峰值 29.5° vs 实测残差 27.3° → 1.08x(自洽)

**判定它不是缺陷**: 把 wz 和"左右脚载荷"做互相关, 8 个环境**全部**在
滞后 **-3 ~ -5 步(60~100 ms)** 上给出 |r| = **0.60~0.67** —— 躯干扭动**先于**抬脚,
每局都高度一致。也就是: 扭头是这只 25cm 机器人的"倒重心/甩腿"机制(偏航反作用力
换载荷), 不是控制没压住的抖动。压掉它 = 压掉步态原语本身。
上游 playbook 同一结论: "a 25 cm robot tumbles at 3.5–5.5 rad/s NATURALLY —
put anti-violence pressure on impacts and thrash, not on rotation speed."

奖励账也支持"策略是理性选择而不是做不到"(headless 逐项, vx=0.15):
  合计 +6.528/步 = air_time +2.51  upright +1.97  track_lin +1.90  track_ang_vel +0.51
  负项合计 -0.365(其中 body_ang_vel -0.00, action_rate -0.02)
  即: 摇头最多让它少拿 0.49/步(偏航跟踪那 1.0 权重里的一半), 而它靠这套步态
  赚 +2.51(air_time)+1.90(track_lin)。**要压掉, 得先有比 2.51 更值钱的东西去换。**

如果以后确实要"让偏航轴听话", 唯一正当的做法是给偏航轴**下命令**(上游
`rel_turn_in_place_envs=0.15` 那种转向桶), 让它学会"该转的时候转、不该转的时候
不转"; 而不是继续加 body_ang_vel 的权重(那个项现在是 -0.01 的死项, 见 P2⑦ 的
测量: 当年正是这一项在给"站着不动"发补贴)。

### "摇头"还有第二层: 头/脖子这条链(同一次测量, `scripts/probe_head_wobble.py`)

上面量的是**躯干**扭动。眼睛看到的"摇头"比它大得多, 因为还有一层:

  结构: `neck_pitch` 关节以下那棵子树 = {neck, neck_pitch, yaw_roll_motion, jaw_soft}
        = **279.9 g / 737.2 g = 整机质量的 38.0%**(jaw_soft 一项就 188.8 g = 25.6%),
        由 4 个关节(neck_pitch / head_pitch / head_yaw / head_roll)固定, 位于躯干最上方。
        它不是轻装饰件, 是一根"38% 质量的倒立摆"。(注意 bearing_roll 23g 是躯干侧零件,
        不在头链里; 用子树算质量能避免这个错。)

  动起来(指令 vx=0.15, 8 环境 × 500 步):
    4 个头关节偏离默认姿势: neck_pitch RMS 26.6° / head_pitch 25.0° /
        **head_yaw 37.2°(峰值 57.2°)** / head_roll 9.8° —— **全部锁在步频 1.700 Hz**
    嘴尖(在躯干坐标系里)左右摆动: RMS 33.9 mm, **峰峰值 94 mm**  ← 这就是"摇头"
    与左右脚载荷的互相关: **|r| = 0.92, 8/8 环境一模一样**, 滞后 ≈ 0
    动作(位置目标)本身也在扫: head_yaw 目标 RMS 45.4°, 而实际只到 37.2°;
        neck_pitch 目标 12.4° 实际偏离 26.6°(比值 2.2) ⇒ 位置伺服追不上这个重的头

  但它是**空转, 不是配重**: 头链质心在躯干系里只动 RMS 7.0 mm(峰峰 19.5 mm),
    按 38% 质量折算到整机质心 = **2.7 mm = 两脚间距(70mm)的 4%** —— 太小, 不足以
    承担倒重心; 真正的倒重心是上面那个躯干扭动(领先抬脚 60~100ms, |r| 0.6~0.67)。

  为什么奖励不管它: 唯一碰头的项是 `head_dev`(把 4 个关节拉回默认姿势, 权重 -0.2),
    实测在 vx=0.15 只值 **-0.04 分/步**, 而 air_time +2.51 / upright +1.97 /
    track_lin +1.90。也就是说这 4 个自由度(14 个动作通道里的 4 个)只值目标的 1%。
    上游不是这么做的: 他们给头**下命令**(4 维 `head_pose` + `head_pose_tracking`
    + `head_pose_bias_penalty`), 头因此有正事干(指相机), 而我们的头是自由维度。

  还有个实际后果: `head_camera` / `tof` / `head_imu` 都挂在这条链上 —— 头以
  1.7 Hz、峰峰 94 mm、±37° 偏航在晃, 那些传感器(尤其相机)的输出在晃动里没有意义。

  因此"修摇头"的三条路(按代价):
    ① 冻掉(把 4 个头关节的动作 scale 设 0, 锁名义姿势) —— **零成本 A/B 已做, 结果是"不能直接冻"**:
       拿同一个 checkpoint(`model_3250.pt`)只把动作通道 5,6,7,8 置零(位置目标回名义姿势),
       对比(400 步 × 8 环境):
         vx=0.15: 倾角 1.47° → **4.58°**, 侧向速度占比 21.4% → **59.5%**,
                  路径直线度 0.949 → **0.663**(开始绕圈), 速度 0.165 → 0.144
         vx=0.30: 倾角 1.82° → **4.51°**, 侧向 12.5% → **30.4%**, 直线度 0.983 → **0.900**
       也就是说**现有策略是靠甩头在稳住自己的** —— 头不是装饰, 是配重/反作用质量。
       注意这条实验的边界: 它只能证明"这个训练时带头自由度的策略依赖头", 不能证明
       "从头就冻掉的策略一定更差"(重训可以用别的机制换稳定性, 代价是一轮短训)。
       另外头链质心只走 2.7mm(步宽 4%), 所以起作用的是它的**角动量**(反作用轮),
       不是位置 —— 这解释了为什么这么小的位移能撑住平衡。
       再补一次分解(`probe_head_wobble.py --freeze-head`, 200 步): 把头锁在名义姿势后,
       头链质心的**横向**运动从 7.04mm 掉到 **0.04mm**(几乎全是被"命令"出来的, 不是被动甩),
       而 `neck_pitch` 仍有 **7.2° 的残余偏离** = 位置伺服托不住这个重的头(重力下垂);
       同时"头摆动 vs 左右脚载荷"的相位一致性从 |r|=0.92(8/8 一致)退化成 0.58(0.31~0.90 乱),
       说明那种紧锁的相位关系是控制出来的, 不是机械耦合。
    ② 下命令(上游路线): 移植 `head_pose` 命令 + 跟踪 + bias 惩罚, 头从"自由维度"
       变成"有用的自由度"(稳定相机 / 看向目标)。代码多, 但这是唯一让头**有正事**的做法;
       也是"想用头上的相机做任何事"之前必须做的事。
    ③ 加惩罚(head_dev -0.2 → -1.0~-2.0): 最省事但治标 —— 它没有给头正事, 而且是在
       和"策略拿头当反作用轮"的策略对打, 很可能连同平衡一起压坏。不推荐先走这条。
    ④ 加刚度/延迟对齐: 头部两个俯仰关节的"实际偏离/动作目标"实测是 1.3~2.2 倍(追不上),
       这是位置伺服偏软 + 头很重的组合。真机的头部舵机型号/kp 必须与 BAM 配置
       (`motor_name="xl330"`, 全部 14 个关节同一个配置)核对 —— 策略依赖快速甩头来平衡,
       头部执行器模型错了就是 sim2real 地雷, 这条比"好不好看"重要得多。



## 和上游 `mjlab.tasks.velocity.velocity_env_cfg` 仍有的区别

  上游那套为 Unitree 大狗/人形而写: 粗糙地形 + 射线高度扫描 + 一堆域随机化 +
  质心/角动量/自碰撞奖励。本任务先要"在平地上学会走", 所以只用平面地形 +
  两只脚的接触传感器(平地上脚底 site 的高度就是离地高度, 不需要射线扫描)。
  域随机化(foot_friction / encoder_bias / base_com)留到走路学会之后再逐步加。

## ⚠️ 一个新引入的坑(改 ② 带来的, 必须知道)

删掉 base_lin_vel 之后 walk 的 actor 恰好也是 51 维, **和站立任务一样** ——
但最后 3 维的语义不同:
    站立 actor[48:51] = 实测机体线速度(base_lin_vel)
    walk   actor[48:51] = 速度指令(command)
也就是说 `--agent.resume` 加载站立 checkpoint 时**形状能对上、不会报错**,
策略却会把"指令"当成"实测速度"读 —— 静默错训, 比直接崩更坏。
要么先做 W4(把站立/坐站的 base_lin_vel 也挪到 critic, 三任务统一 48 维本体感觉
+ 尾部命令槽), 要么续训前手工确认这 3 维的语义。tests 里有测试钉住 walk 的
尾部 3 维就是命令槽。
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import torch

from bam.mjlab import bam_init
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
    action_rate_l2,
    bad_orientation,
    base_ang_vel,
    base_lin_vel,
    generated_commands,
    joint_pos_limits,
    joint_pos_rel,
    joint_vel_l2,
    joint_vel_rel,
    last_action,
    projected_gravity,
    nan_detection,
    push_by_setting_velocity,
    reset_joints_by_offset,
    reset_root_state_uniform,
    time_out,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.viewer import ViewerConfig

# 共享零件: 实体(BAM 执行器 + 站姿初值) + PPO 配置
from mjlab_microduck.tasks.microduck_cfg import (
    get_microduck_entity_cfg,
    microduck_ppo_runner_cfg,
)

# ⚠️ 机器人模型: 走路任务必须用 `robot_walk.xml`, **不是**站立里程碑继承来的
# `robot_allcollisions.xml`(第六轮修正, 依据: ../microduck_rl 的实现)。
# 两个模型的运动学/执行器/脚底网格完全一样, 差别只在**碰撞面**:
#     robot_allcollisions.xml: 11 个 → trunk×2, hip×2, leg×2, 脚×2, **jaw_soft×3**
#     robot_walk.xml          :  5 个 → trunk×1, leg×2, 脚×2
# 多出来的那 6 个(下巴/头 3 个 + 髋 2 个 + 躯干 1 个)让鸭子**趴/躺/头撑地时有支撑面**,
# 于是"躺平薅分"在几何上就是可行的 —— 实测第二轮正是 50° 倾角躺着稳稳拿分。
# 上游把这件事分工得很清楚: 走路任务用 walk 模型, 要能躺下/起身的 VelStand 才用
# all-collisions 模型, 而且他们的文档里点名过这个坑
# ("head-tripod at ~55° farming the gated +2·cos(tilt)")。
_REPO = Path(__file__).resolve().parents[3]
_WALK_XML = _REPO / "assets" / "microduck" / "robot_walk.xml"


def get_microduck_walk_entity_cfg() -> EntityCfg:
    """站立任务的实体配置, 但换上 walk 模型(只改碰撞面, 其余完全一致)。"""
    cfg = get_microduck_entity_cfg()
    cfg.spec_fn = lambda: mujoco.MjSpec.from_file(str(_WALK_XML))
    return cfg

# ============ 机器人相关的名字(取自 robot_allcollisions.xml) ============

# 脚所在 body: 脚底采样点(_LEFT/_RIGHT_SOLE_POINTS)就是写在这个 body 系里的
_FOOT_BODY_NAMES = ("ankle_left", "ankle_right")
# 脚底 site: 用来量"接触点的速度"(打滑)
_FOOT_SITE_NAMES = ("left_foot", "right_foot")
# 脚底碰撞几何(站立任务里摩擦/接触优先级也是按这两个名字配的)
_FOOT_GEOM_NAMES = ("left_foot_collision", "right_foot_collision")
# 头/脖子 4 个关节: 走路时别乱甩(重心小, 甩头会带偏平衡)
_HEAD_JOINT_NAMES = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")

_FEET_CONTACT_SENSOR = "feet_ground_contact"

_ASSET_ALL_JOINTS = SceneEntityCfg("robot", joint_names=(".*",))
_ASSET_HEAD_JOINTS = SceneEntityCfg("robot", joint_names=_HEAD_JOINT_NAMES)
_ASSET_FEET_BODIES = SceneEntityCfg("robot", body_names=_FOOT_BODY_NAMES)
_ASSET_FEET_SITES = SceneEntityCfg("robot", site_names=_FOOT_SITE_NAMES)

# 摆动脚的离地目标高度(按"脚底区域最低点"量, 单位 m)。
#
# 这个量怎么测才对 —— 我一共猜错两次, 全靠探针实测纠正过来
# (scripts/walk_p0_probe.py 的 ①; 脚底始终贴地、只把脚踝转开 0°→+30°):
#     脚踝 body 高度         0.0226 → 0.0334   (+10.9mm)  ← 第一版写法
#     脚底 site 高度         0.0000 → 0.0141   (+14.1mm)  ← 以为换 site 就好, 更糟
#     上游射线传感器的读数    与 site 高度**完全相等**        ← 平地上它没用!
#     脚底采样点取 min       0.0000 → 0.0019   (+1.9mm)   ← 现在用的
#   原因: ① 脚踝 body / 脚底 site 都是**刚体上的一个点**, 脚一翻就跟着转, 谁都不行;
#         ② 上游 foot_height_scan 是"竖直向下打射线", ring 只在**水平面**里偏置,
#            所以三条射线的 frame_z 相同, min 退化成 site 自己的高度 —— 那套 ring
#            采样只在**粗糙地形**上有意义, 平地上救不了脚踝翻转;
#         ③ "离地高度"问的是**脚底区域里最低那一点**, 所以必须对一片点取 min。
#   代价: 这套算法假设平地(直接用世界系 z)。以后做粗糙地形要换回
#   TerrainHeightSensorCfg(那时 ring 才有用), 或把地面高度一起算进来。
#   target=0.02 沿用上游数值。
# 脚底"最低点"探针用的采样点(ankle body 系, 左/右各一套)。
#
# 生成规则(可复现, 见 scripts/walk_p0_probe.py 的 ①):
#   取脚底碰撞网格(sole_left/sole_right)的全部顶点 → 变到 ankle body 系 →
#   在 (x, y) 上按 4mm 网格分格, 每格只留最低的那个顶点。
# 为什么不是"取一个点": 见 _SWING_TARGET_HEIGHT 上方那组实测数字 —— 脚底是**曲面**,
#   任何固定单点都会随脚踝翻转漂 11~14mm(脚底贴地时!)。对一片点取 min 才是"脚底最低点"。
# 精度(踝角 -30°~+30° 扫, 脚底贴地时与网格真值的偏差): ≤1.93mm —— 相对 20mm 的目标
#   是 ~10%, 足够; 单点探针在同一条扫描上偏 11~14mm(~70%)。
# 点数 53/脚 是"精度 vs 表长"的折中: 4mm 网格 53 点 → 1.93mm; 5mm 网格 34 点 → 4.68mm;
#   3mm 网格 82 点 → 1.86mm(没便宜可占)。
# 注意: 这套算法假设**平地**(直接用世界系 z 当离地高度)。以后做粗糙地形要换成射线
#   高度传感器(TerrainHeightSensorCfg), 否则坡面上的"离地高度"是错的。
_LEFT_SOLE_POINTS = (  # 53 点/脚, 4mm 网格取每格最低顶点
    (-0.03236, -0.01212, -0.03210),
    (-0.03208, -0.02003, -0.03113),
    (-0.03202, -0.01711, -0.03309),
    (-0.02831, -0.02002, -0.03411),
    (-0.02822, -0.01212, -0.03451),
    (-0.02804, -0.02401, -0.02323),
    (-0.02802, -0.01723, -0.03513),
    (-0.02700, -0.01212, -0.03462),
    (-0.02583, -0.01748, -0.03520),
    (-0.02524, -0.02401, -0.02839),
    (-0.02508, -0.02007, -0.03447),
    (-0.02256, -0.01796, -0.03518),
    (-0.02233, -0.01597, +0.00578),
    (-0.02183, -0.02010, -0.03455),
    (-0.02053, -0.02417, -0.02947),
    (-0.01959, -0.02451, -0.02845),
    (-0.01912, -0.01597, +0.00579),
    (-0.01861, -0.02013, -0.03462),
    (-0.01613, -0.01772, -0.03521),
    (-0.01476, -0.02411, -0.03054),
    (-0.01290, -0.01772, -0.03522),
    (-0.01285, -0.01479, +0.00580),
    (-0.01226, -0.02017, -0.03477),
    (-0.00954, -0.01871, -0.03517),
    (-0.00929, -0.02411, -0.03130),
    (-0.00915, -0.02018, -0.03483),
    (-0.00643, -0.01797, -0.03523),
    (-0.00607, -0.02019, -0.03486),
    (-0.00445, -0.02405, -0.03179),
    (-0.00000, -0.02010, -0.03493),
    (-0.00000, -0.02405, -0.03180),
    (-0.00000, -0.01822, -0.03524),
    (-0.00000, -0.01479, +0.00580),
    (+0.00000, -0.02418, -0.03158),
    (+0.00000, -0.02034, -0.03485),
    (+0.00000, -0.01803, -0.03524),
    (+0.00000, -0.01570, +0.00585),
    (+0.00448, -0.02015, -0.03486),
    (+0.00460, -0.02413, -0.03130),
    (+0.00475, -0.01479, +0.00580),
    (+0.00477, -0.01803, -0.03523),
    (+0.00813, -0.02408, -0.03083),
    (+0.00904, -0.02015, -0.03476),
    (+0.00951, -0.01479, +0.00580),
    (+0.00957, -0.01763, -0.03522),
    (+0.01253, -0.02410, -0.02882),
    (+0.01369, -0.02013, -0.03463),
    (+0.01437, -0.01743, -0.03521),
    (+0.01450, -0.01212, -0.03462),
    (+0.01605, -0.01722, -0.03498),
    (+0.01615, -0.01212, -0.03436),
    (+0.01626, -0.02009, -0.03397),
    (+0.02000, -0.01544, -0.00506),
)
_RIGHT_SOLE_POINTS = (  # 54 点/脚, 4mm 网格取每格最低顶点
    (-0.02000, -0.01212, -0.00008),
    (-0.01626, -0.02009, -0.03397),
    (-0.01615, -0.01212, -0.03436),
    (-0.01605, -0.01722, -0.03498),
    (-0.01450, -0.01212, -0.03462),
    (-0.01437, -0.01743, -0.03521),
    (-0.01369, -0.02013, -0.03463),
    (-0.01253, -0.02410, -0.02882),
    (-0.00957, -0.01763, -0.03522),
    (-0.00951, -0.01479, +0.00580),
    (-0.00904, -0.02015, -0.03476),
    (-0.00813, -0.02408, -0.03083),
    (-0.00477, -0.01803, -0.03523),
    (-0.00475, -0.01479, +0.00580),
    (-0.00460, -0.02413, -0.03130),
    (-0.00448, -0.02015, -0.03486),
    (-0.00299, -0.02414, -0.03146),
    (-0.00000, -0.01570, +0.00585),
    (-0.00000, -0.02107, -0.03455),
    (-0.00000, -0.01803, -0.03524),
    (+0.00000, -0.01479, +0.00580),
    (+0.00000, -0.02007, -0.03494),
    (+0.00000, -0.02405, -0.03180),
    (+0.00000, -0.01822, -0.03524),
    (+0.00445, -0.02405, -0.03179),
    (+0.00607, -0.02019, -0.03486),
    (+0.00643, -0.01797, -0.03523),
    (+0.00915, -0.02018, -0.03483),
    (+0.00929, -0.02411, -0.03130),
    (+0.00954, -0.01523, +0.00584),
    (+0.00954, -0.01871, -0.03517),
    (+0.01226, -0.02017, -0.03477),
    (+0.01290, -0.01772, -0.03522),
    (+0.01476, -0.02411, -0.03054),
    (+0.01591, -0.01596, +0.00581),
    (+0.01613, -0.01772, -0.03521),
    (+0.01861, -0.02013, -0.03462),
    (+0.01912, -0.01597, +0.00579),
    (+0.01959, -0.02451, -0.02845),
    (+0.02053, -0.02417, -0.02947),
    (+0.02183, -0.02010, -0.03455),
    (+0.02233, -0.01597, +0.00578),
    (+0.02256, -0.01796, -0.03518),
    (+0.02508, -0.02007, -0.03447),
    (+0.02524, -0.02401, -0.02839),
    (+0.02583, -0.01748, -0.03520),
    (+0.02700, -0.01212, -0.03462),
    (+0.02802, -0.01723, -0.03513),
    (+0.02804, -0.02401, -0.02323),
    (+0.02822, -0.01212, -0.03451),
    (+0.02831, -0.02002, -0.03411),
    (+0.03202, -0.01711, -0.03309),
    (+0.03208, -0.02003, -0.03113),
    (+0.03236, -0.01212, -0.03210),
)

_SWING_TARGET_HEIGHT = 0.02

# 指令放宽节奏(单位: env 的 common_step_counter, 即"策略步"数).
# 参考: 4096 envs × 32 steps/iter → 8000 步 ≈ 250 轮; CPU 冒烟(16 envs)同理.
# 每个 stage 里没写的字段保持上一档不变。
#
# 速度尺度怎么定的: 鸭子髋高 ~0.11m, 人髋高 ~0.9m 走 1.4m/s ≈ 1.55 腿长/秒,
# 等比换算到鸭子 ≈ 0.17m/s 就是"人的常速走"; 所以起步档 0.15、最终档上限 0.35
# (≈2.2 倍常速, 相当于人小跑)。终点再高就该指望跑步任务了, 不是这条走路任务的事。
# P2 修正(依据: 500 轮实测 —— 策略学会了"站着不动", 见 README 的趋势分析):
# 原来第 1 档是 lin_vel_x=(0.0, 0.15), 均值只有 0.075m/s; 配上 std, 站着不动的策略
# 能白拿 71~90% 的 track_lin_vel(算给 scripts/walk_p0_probe.py 的 ②b)。
# 现在把早期下限抬到 0.10: 指令均值 0.15m/s(≈前面算过的"鸭子人的常速走"0.17),
# 站着不动只能拿 ~10%。"零指令"那种场合由 rel_standing_envs=10% 单独覆盖, 不需要
# 在范围里留 0。
_VELOCITY_STAGES: list[dict] = [
    # 第 1 档: 只往前, 速度是"鸭子常速走"的量级 —— 一开始就必须动起来才有分
    {"step": 0, "lin_vel_x": (0.10, 0.20), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
    # 第 2 档: 提速 + 小幅转向
    {"step": 1500, "lin_vel_x": (0.08, 0.25), "ang_vel_z": (-0.3, 0.3)},
    # 第 3 档: 允许一点点后退 + 转向更快
    {"step": 4000, "lin_vel_x": (0.0, 0.30), "ang_vel_z": (-0.5, 0.5)},
    # 第 4 档: 加上微量侧移, 到这一步才是"全向慢走"
    {"step": 8000, "lin_vel_x": (-0.10, 0.35), "lin_vel_y": (-0.05, 0.05), "ang_vel_z": (-0.7, 0.7)},
]

# play(推理/回放)时用最后一档的最宽范围, 这样能看到策略的极限能力
_PLAY_RANGES = _VELOCITY_STAGES[-1]

# 动作平滑惩罚(action_rate_l2)的权重阶梯。
#
# ⚠️ 这一项上一版估错了量级, 依据 500 轮实测重算(这是本次 P2 最重要的修正之一):
#   上一版用"占正质量的份额"对齐上游(算出 -0.55), 思路没错, 但我把 raw 量级估成了
#   ~1.2 —— 那是用"0.3×randn 的随机动作"探针量出来的。真实训练里策略自己的探索
#   std≈1.0, raw(=14 维动作变化平方和)是 **5~25**。
#   结果: 权重 -0.30 时它每步是 **-1.66 ~ -3.0**, 占全部罚项的 **85%**、和整个正质量
#   同量级; 而它加压的时机(125~250 轮)正好是鸭子尝试迈步的窗口 —— 抬脚天生是抖的,
#   等于给"尝试"收重税, 实测那之后"有脚离地"从 32% 掉到 0.9%(策略学会别动)。
#   上游 playbook 的原话: any attempt-tax active while a hard skill is being explored
#   makes "do nothing" win —— 我用错输入量级, 让它真的发生了。
# 现在的定法: 目标占正质量 ~1%(起步) → ~3%(后期)。正质量约 3.2/步(实测末轮),
#   权重 -0.03 配 raw 5~10 ≈ -0.15~-0.30/步(约 5~9%); 后期策略变平滑后 raw 会降,
#   所以终值放到 -0.10 也只占 ~3%。再大就会重新变成"别动"的税。
# 档位必须落在"计划跑的轮数"之内 —— 上一轮就踩过这个坑: 当时最后一档排在 16000 步,
# 而 16000 步 = 第 500 轮, max-iterations=500 是 0..499 → 最强那档根本没跑到。
# 现在的排法(32 envs, 500 轮 = 16000 步)对应第 125 / 312 / 437 轮, 全部跑得到。
_ACTION_RATE_WEIGHT_STAGES: list[tuple[int, float]] = [
    (0, -0.03),
    (4000, -0.05),
    (10000, -0.08),
    (14000, -0.10),
]

# 步态奖励窗口(air_time)下限的阶梯。
#
# 这是 P2 里对"学不会抬脚"最直接的一刀: 实测 250 轮时鸭子的摆动时长只有 20~30ms,
# 而窗口下限是 125ms —— 抬脚 30ms 和不抬脚一样都是 0 分, **完全没有梯度**(鸡生蛋)。
# 所以从 0.03(30ms 就算数)起步, 等实测摆动时长长上来再逐步抬到上游的 0.125。
# 每档都是"当前策略够得着"的下限(playbook: phase-align every stage with what the
# policy has actually learned)。
# 判断该不该抬的观测: Metrics/air_time_mean 与 eval_walk.py 里的"摆动s"列。
_AIR_WINDOW_MIN_STAGES: list[tuple[int, float]] = [
    (0, 0.03),
    (6000, 0.08),
    (12000, 0.125),
]

# 「还靠腿站着吗」的软门阈值(躯干高度 × 直立程度)。实测标定(CPU, headless 评估):
#     躺平(250 轮后的新策略): 躯干 z=0.041~0.043, 倾角 50°   ← 就是它在薅奖励
#     坐姿(SIT 关键帧):        z≈0.070
#     走路(旧 250 轮策略):     z=0.095~0.100, 倾角 10~13°
#     站立(旧 499 轮策略):     z=0.114~0.117, 倾角 1~3°
# 取 z: 0.065(低于坐姿) → 0.090(低于走路); cos(倾角): 0.60(53°) → 0.90(26°)。
# 这样躺平得 0、坐姿得 ~20%、走路/站立得 1, 中间是线性斜坡(有梯度, 不是硬开关)。
_GATE_Z_LOW, _GATE_Z_HIGH = 0.065, 0.090
_GATE_COS_LOW, _GATE_COS_HIGH = 0.60, 0.90


# 「有没有按指令方向走」的判据下限(m/s): 指令方向上的速度低于它就算"没在走"
_PROGRESS_MIN_CMD = 0.03


def _progress_gate(env: ManagerBasedRlEnv, command_name: str = "twist") -> torch.Tensor:
    """[B] ∈ [0,1]: "是否真的在按指令方向移动" —— 步态奖励的**方向条件**。

    为什么必须有它 —— 第三轮 250 轮实测: 鸭子**原地蹦**(有脚离地 45%、摆动 121ms、
    竖向 |vz|=0.11) 却几乎不位移(机体系 vx=0.07, 而指令要 0.15~0.30), 照样把 air_time
    从 0.13 薅到 **2.04/步**(是 track_lin 的 25 倍)。也就是说: 步态奖励与"有没有前进"
    完全无关, 于是"原地抬脚"成了最优解 —— 用户看到的"横着走两步就倒"就是这个。
    这道门按"指令方向上的速度 / 指令大小"给分: 原地或横着挪 ≈ 0, 真按要求走 → 1。
    注意它是**软的**(线性): 只要往对的方向挪一点就有分, 不是硬开关。
    """
    cmd = env.command_manager.get_command(command_name)
    assert cmd is not None, f"找不到指令项 '{command_name}'"
    lin = cmd[:, :2]
    speed_cmd = torch.norm(lin, dim=1)
    direction = lin / speed_cmd.clamp(min=1e-6).unsqueeze(1)
    vel_xy = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    progress = (vel_xy * direction).sum(dim=1)
    scaled = (progress / speed_cmd.clamp(min=_PROGRESS_MIN_CMD)).clamp(0.0, 1.0)
    return scaled * (speed_cmd > _PROGRESS_MIN_CMD).float()


def _posture_gate(env: ManagerBasedRlEnv) -> torch.Tensor:
    """[B] ∈ [0,1]: "还靠腿站着"的软门(躯干高度 × 直立程度)。

    为什么必须有它 —— 250 轮实测的失败: 鸭子躺在地上(躯干 0.041m、倾角 50°) 把
    air_time 从 0.08/步薅到 **0.76~1.43/步**(腿举在空中, 永远满足 0.03~0.3s 摆动窗口),
    track_lin 还白拿 0.66(零指令时"一动不动"就是完美跟踪), 而唯一罚它的是 upright
    (50° 时 ≈0.06 而不是 2.0) —— 净 +1.8/步, 所有回合 time_out 永不终止。
    也就是说: **正奖励上不设姿态条件, 等于请策略去薅它**(上游 playbook 原话:
    "Never gate a positive reward on being in a bad state — the policy parks in the
    cheapest qualifying pose and farms it.")。
    所以"技能类"正奖励(速度跟踪 + 步态)都乘这道门。upright 不用乘 —— 它自己随倾角归零。
    注意门是**软的**(线性斜坡): 倒下去的过程中"爬起来"能拿到递增的分, 有梯度;
    直接硬 0/1 会让"从地上爬起来"这件事没有信号。
    """
    robot = env.scene["robot"]
    z = robot.data.root_link_pos_w[:, 2]
    cos_tilt = -robot.data.projected_gravity_b[:, 2]  # 直立=1, 躺平=0
    gz = ((z - _GATE_Z_LOW) / (_GATE_Z_HIGH - _GATE_Z_LOW)).clamp(0.0, 1.0)
    gt = ((cos_tilt - _GATE_COS_LOW) / (_GATE_COS_HIGH - _GATE_COS_LOW)).clamp(0.0, 1.0)
    return gz * gt


class fallen_too_long:
    """"躺着/大角度倾倒"持续超过 grace_s 秒 → 终止本回合(上游 mdp.fallen_too_long 同款思路)。

    为什么需要(两轮实测的教训):
      P2 删掉了"躯干<5cm 立刻终止" —— 那是对的, 实测它终止了 100% 的尝试回合, 把最需要的
      探索样本砍光了; 但删干净之后鸭子发现**躺在地上可以永久薅分**(见 _posture_gate 注释),
      250 轮后策略变成躺平。问题不在"终止"本身, 而在"立刻"。
      正解是给宽限期:
        · 倒地 1 秒内不终止 → "挣扎/恢复"的样本照样进训练;
        · 倒地超过 1 秒 → 终止并重置, 而重置会回到**站立出生点**,
          于是训练样本的组成重新以"站着"为主, 而不是全在擦地板。
      阈值(0.065m / 45°)取得比姿态门宽一点: 门是"不给分", 这里只是"别一直赖着"。
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.timer_s = torch.zeros(env.num_envs, device=env.device)
        self.step_dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        z_threshold: float = _GATE_Z_LOW,
        tilt_threshold: float = math.radians(45.0),
        grace_s: float = 1.0,
    ) -> torch.Tensor:
        robot = env.scene["robot"]
        z = robot.data.root_link_pos_w[:, 2]
        tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
        fallen = (z < z_threshold) | (tilt > tilt_threshold)
        self.timer_s = torch.where(
            fallen, self.timer_s + self.step_dt, torch.zeros_like(self.timer_s)
        )
        return self.timer_s > grace_s

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.timer_s[env_ids] = 0.0


# 动态类正则(body_ang_vel / angular_momentum)的权重。
#
# P2⑦ 修正(依据: 用**新奖励**给旧 checkpoint 重新打分, 零训练成本):
#   把旧 run 的 model_250(还在迈步)和 model_499(站着不动)放进新奖励里跑:
#     vx=0.15 时 走路 +2.88 / 站着 +2.82  → 平局, 没翻过来;
#     拆开看: 走路的罚项 -0.615 vs 站着 -0.117, 其中 **body_ang_vel 一项 -0.426**。
#     把这一项扣掉 → 走路 +3.31 vs 站着 +2.89, 走路才赢。
#   也就是说: 站着的最后一块优势, 是这个"角速度税"给的。
#   上游 playbook 原话: "a 25 cm robot tumbles at 3.5–5.5 rad/s NATURALLY —
#   don't impose human-scale speed intuitions via caps; put anti-violence pressure
#   on impacts and thrash (|a_z|, action_rate, support gates), not on rotation speed."
#   实测走路时的 raw(=|ω_xy|²)≈12.8 → |ω_xy|≈3.6 rad/s, 正好落在这只机器人"自然"的
#   区间里 —— 罚它等于罚"正常走路"。所以从 -0.03 降到 -0.01(约 1/3):
#   还留着一点点防"疯狂甩"，但不再是决定"站着还是走"的那一项。
#   防抖/防暴力的压力交给 action_rate(已按实测量级重算)与 |a_z| 类项, 不靠它。
_BODY_ANG_VEL_WEIGHT = -0.01
# angular_momentum 实测是 1e-7 量级的死项(见 angular_momentum_penalty 注释), 保留对齐上游
_ANGULAR_MOMENTUM_WEIGHT = -0.01


# ============ 奖励函数 ============
# 约定: 每个函数 (env) -> [num_envs] 的 tensor, 由 mjlab 按 weight 加权求和。
# 需要的额外参数写在 RewardTermCfg(params=...), 由 RewardManager 注入。


def _walking_mask(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float,
) -> torch.Tensor:
    """"当前真的被要求走"的 0/1 掩码: 线速度或角速度指令超过阈值才算在走路。

    为什么需要它: 那些拿到零指令(站着)的环境不该被"抬脚/滑步"类奖励塑形,
    否则策略会把站立也变成小碎步。
    """
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"找不到指令项 '{command_name}'"
    norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    return (norm > command_threshold).float()


def _feet_in_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """[num_envs, 2] 的 0/1: 第 i 只脚是否与地面接触。"""
    sensor = env.scene[sensor_name]
    found = sensor.data.found
    assert found is not None, f"传感器 '{sensor_name}' 没有 found 字段"
    return (found > 0).float()


def track_lin_vel_xy_reward(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str = "twist",
) -> torch.Tensor:
    """想听话(线速度): 实际机体水平速度越接近指令越好, 同时别上下乱蹦。

    高斯核 exp(-err/std²): err=0 → 1 分, err=std → 0.37 分, 越离谱越接近 0。
    用"机体坐标系"速度(root_link_lin_vel_b), 因为指令也是机体系(前进=+x)。
    """
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"找不到指令项 '{command_name}'"
    vel_b = env.scene["robot"].data.root_link_lin_vel_b
    err = ((command[:, :2] - vel_b[:, :2]) ** 2).sum(dim=1) + vel_b[:, 2] ** 2
    # 乘"还站着吗"的姿态门: 躺着/趴着拖动**不算**在跟踪指令(否则地板上的策略也能拿分)
    return torch.exp(-err / std**2) * _posture_gate(env)


def track_ang_vel_z_reward(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str = "twist",
) -> torch.Tensor:
    """想听话(转向): 偏航角速度接近指令, 同时身体别乱滚(pitch/roll 角速度要小)。"""
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"找不到指令项 '{command_name}'"
    ang_vel_b = env.scene["robot"].data.root_link_ang_vel_b
    err = (command[:, 2] - ang_vel_b[:, 2]) ** 2 + (ang_vel_b[:, :2] ** 2).sum(dim=1)
    return torch.exp(-err / std**2) * _posture_gate(env)


def feet_air_time_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    threshold_min: float = 0.03,   # P2: 由课程逐步抬到 0.125(见 _AIR_WINDOW_MIN_STAGES)
    threshold_max: float = 0.300,
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """想迈步: 每只脚只要"已经离地一段时间且还没落地", 每一步都给分。

    这是 P0 改掉的一处关键量级问题。旧写法(落脚那一瞬间给一次分):
        reward = Σ_feet 1{last_air_time ∈ 窗口} · 1{刚落脚}
      一次摆腿只付 1 次 × 权重 0.5 = **0.5 分/周期**。
    新写法(上游 mjlab mdp.feet_air_time 同款, 每步都在窗口内就付):
        reward = Σ_feet 1{current_air_time ∈ 窗口}
      一只脚在空中待 0.125~0.300s ≈ 9 个控制步, 权重 3.0 → **~27 分/脚/周期**,
      两条腿一个步态周期 ~54 分。两者差约 **50 倍**。

    为什么必须是"持续付"而不是"一次性小费": 一次性的钱太薄, 无法与"前倾蹭地
    也能把速度误差压下去"这条捷径竞争 —— 唯一强激励只剩速度跟踪时, 策略就会
    选那条更便宜的路(滑步/前扑)。持续付等于把"保持摆动相"变成每一步的最优解。
    窗口上限防"一直悬着不落地"(金鸡独立/蹦), 下限防"原地抖腿",
    两条一起把摆动时长塑形到 0.125~0.3s 这个真实的摆动相区间。

    ⚠️ 下限由课程管理(起步 0.03, 见 _AIR_WINDOW_MIN_STAGES): 实测一开始鸭子只能摆动
    20~30ms, 如果下限直接写 0.125, 那"抬脚 30ms"和"完全不抬"都是 0 分 —— 迈步这件事
    在奖励里**没有任何梯度**, 策略永远学不会第一次抬脚。上游那个 0.125 是它们的步态
    已经能摆动那么久之后才收紧的。

    第三个作用: 这条曲线会打点到 Metrics/air_time_mean, 判断"步态有没有长出来"
    就看它 —— 一直是 0 表示两脚都贴地蹭, 长期贴着 0.3 表示想蹦。
    """
    sensor = env.scene[sensor_name]
    current_air_time = sensor.data.current_air_time
    assert current_air_time is not None, f"传感器 '{sensor_name}' 需开 track_air_time=True"
    in_window = ((current_air_time > threshold_min) & (current_air_time < threshold_max)).float()

    # 顺手记一条曲线: "此刻悬在空中的脚平均已经离地多久"。
    # setdefault: 裸 ManagerBasedRlEnv(测试里那种, 没套 RslRlVecEnvWrapper)没有
    # extras["log"], 奖励函数不该因此崩掉。
    in_air = (current_air_time > 0).float()
    env.extras.setdefault("log", {})["Metrics/air_time_mean"] = (
        (current_air_time * in_air).sum() / in_air.sum().clamp(min=1)
    )

    # 三个条件乘在一起(每一条都是实测踩出来的):
    #   _walking_mask  : 被要求走(零指令环境不塑形)
    #   _posture_gate  : 还靠腿站着(躺平举腿不算摆动 —— 第二轮薅分点)
    #   _progress_gate : 真的在按指令方向移动(原地蹦/横着挪不算 —— 第三轮薅分点)
    #
    # ⚠️ 第七轮**去掉了**这里原本乘的"跟踪核"(track_kernel), 依据是"拿一个真步态来标定":
    #   ../microduck 的 alpha_walking.onnx(上游真机在用的走路策略)在我们的环境里能走
    #   (离地 46%、摆动 170ms、倾角 2.8°、零摔倒), 但它只能跑到指令速度的 ~45%;
    #   把它放进我们的奖励里逐项对比发现: 乘了跟踪核之后, 它的 air_time 从 ~2.5 掉到 ~0.4,
    #   于是**真步态输给"站着不动"**(cmd 0.15: +3.91 vs +3.93; cmd 0.30: +3.17 vs +2.91)。
    #   也就是说: 跟踪核把"速度差 50%"平方级罚了两次(一次在 track_lin, 一次在 air_time),
    #   而"步态"这件事**与速度精度是正交的** —— 上游正是靠"宽松的跟踪 + 一大笔 air_time(3.0)
    #   只有真迈步才拿得到"才让走路胜出。方向门(progress)仍然保留: 它只要求"大致达到指令速度",
    #   足以挡住"原地蹦", 但不会因为差 20% 就把步态奖励清零。
    return (
        in_window.sum(dim=1)
        * _walking_mask(env, command_name, command_threshold)
        * _posture_gate(env)
        * _progress_gate(env, command_name)
    )


def upright_gauss_reward(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """想竖直: 身体倾角越小越好, 用高斯核 exp(-Σg_xy²/std²)(上游 mdp.upright 同款)。

    P0 换掉了原来的线性写法. 为什么线性那版不够: 原来 reward = -g_z = cos(倾角),
    权重 0.5, 所以倾 0.1rad 只掉 0.0025 分/步; 高斯版(std=sqrt(0.05), 权重 2.0)
    同样倾 0.1rad 掉 0.36 分/步 —— **相差约 140 倍**。
    走路时"歪着身子蹭"几乎免费, 是因为整套奖励里没有第二项在罚"静态倾角":
    track_ang_vel 里那点 ω_xy² 罚的是角速度不是角度, joint_dev/head_dev 罚的是
    关节角不是躯干倾角。所以这一项必须自己够硬。
    """
    asset = env.scene[asset_cfg.name] if asset_cfg is not None else env.scene["robot"]
    gravity_b = asset.data.projected_gravity_b  # 重力在机体系投影, 直立时 = (0,0,-1)
    xy_sq = (gravity_b[:, :2] ** 2).sum(dim=1)  # = sin²(倾角)
    return torch.exp(-xy_sq / std**2)


def _foot_clearance_heights(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """两只脚的"脚底最低点离地高度" [B, 2]。

    做法: 把 _LEFT/_RIGHT_SOLE_POINTS(从脚底网格抽出的采样点)按脚踝 body 的当前
    位姿转到世界系, 取 z 的最小值。平地上这就是"脚底离地多少"。
    为什么不能只取一个点(脚踝 body 或脚底 site): 见 _SWING_TARGET_HEIGHT 上方实测。
    """
    asset = env.scene[asset_cfg.name]
    # 左/右顺序必须和点表一致; 顺序错了会静默把左右脚搞反, 所以这里钉住
    # (解析后的 body_names 字段可能是 None, 所以拿 body_ids 反查实体自己的名字表)
    resolved = tuple(asset.body_names[i] for i in asset_cfg.body_ids)
    assert resolved == _FOOT_BODY_NAMES, (
        f"点表按 {_FOOT_BODY_NAMES} 的顺序写死, 实际解析成 {resolved}"
    )
    pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # [B, 2, 3]
    quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, 2, 4]
    rot_w = matrix_from_quat(quat_w)  # [B, 2, 3, 3]
    # 两只脚的点数不一样(53 / 54, 各自网格分格不同), 所以分开算再拼:
    # 世界系 z = (body 旋转矩阵的第三行) · (body 系下的点) + body 的 z
    heights = []
    for i, pts in enumerate((_LEFT_SOLE_POINTS, _RIGHT_SOLE_POINTS)):
        points = torch.tensor(pts, dtype=torch.float32, device=env.device)  # [P, 3]
        z = torch.einsum("bk,pk->bp", rot_w[:, i, 2, :], points) + pos_w[:, i, 2:3]
        heights.append(z.min(dim=-1).values)  # [B]
    return torch.stack(heights, dim=1)  # [B, 2]


def feet_clearance_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    vel_asset_cfg: SceneEntityCfg,
    command_name: str = "twist",
    target_height: float = _SWING_TARGET_HEIGHT,
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """想迈得像样: 摆动脚的高度偏离目标越多扣越多, 且按"抬脚快慢"加权。

    高度 = **脚底采样点在世界系里的最低 z**(见 _foot_clearance_heights) ——
    不是脚踝 body 高度, 也不是脚底 site 高度: 那两个都是刚体上的单点, 脚踝一翻
    就跟着漂 11~14mm(脚底明明还贴地)。这条我猜错两次, 最后靠实测定下来。
    速度取脚底 site 的线速度(所以要第二个 asset_cfg: 高度用脚踝 body 系里的采样点,
    速度要问接触点 —— 两个量来自不同的 frame, 不能共用一个 cfg)。
    乘上水平速度的用意: 脚飞快扫过时高度不对最要命(容易绊), 慢慢挪的时候
    高度差一点无所谓 —— 上游 feet_clearance 也是这个思路。
    """
    asset = env.scene[asset_cfg.name]
    in_air = 1.0 - _feet_in_contact(env, sensor_name)  # [B, 2]
    foot_height = _foot_clearance_heights(env, asset_cfg)  # [B, 2]
    foot_vel_xy = torch.norm(
        asset.data.site_lin_vel_w[:, vel_asset_cfg.site_ids, :2], dim=-1
    )  # [B, 2]
    cost = (torch.abs(foot_height - target_height) * foot_vel_xy * in_air).sum(dim=1)
    return cost * _walking_mask(env, command_name, command_threshold)


def feet_slip_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """想抓地: 已经在接触地面的脚, 水平速度越大越像"打滑" → 扣分。

    同样是 P0 的探针修正: 用脚底 site 的速度(而不是脚踝 body 的速度)。
    "脚有没有打滑"问的是**接触点的速度**, 脚踝 body 的速度在脚踝转动时并不等于
    接触点速度(差 ω×r 那部分)。
    """
    asset = env.scene[asset_cfg.name]
    in_contact = _feet_in_contact(env, sensor_name)
    foot_vel_xy_sq = (
        asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2] ** 2
    ).sum(dim=-1)  # [B, 2]
    cost = (in_contact * foot_vel_xy_sq).sum(dim=1)
    return cost * _walking_mask(env, command_name, command_threshold)


# 腿关节姿态罚: 分关节 std(上游 variable_posture 的松紧关系)+ **实测的站姿基准**。
#
# ⚠️ 基准必须用"实测站姿", 不能直接用 XML 默认姿势 —— 这是量出来的(CPU, 零动作静置 6 步):
#     关节            相对 XML 默认的偏差     走路档 std     偏差/std
#     hip_roll        ±0.078 rad (4.5°)      0.05           1.56
#     hip_pitch       ±0.235 rad (13.5°)     0.40           0.59
#     knee            ±0.083 rad             0.40           0.21
#     ankle           ∓0.308 rad (17.6°!)    0.25           1.23
#   也就是说: 鸭子**站稳时的踝关节就比 XML 的 STAND 关键帧转了 17.6°** —— 拿 XML 默认当
#   基准, 等于按"它站不到的姿势"扣分(上游 playbook 里"目标高度必须从实际机器人上量,
#   别跨模型版本沿用"是同一个坑, 换成了关节角)。所以这里存一份实测偏移量。
#   (换模型/换执行器参数后必须重量, 测试里有断言守住量级。)
#   实测: walk 模型与 allcollisions 模型的静置站姿最大只差 1.1°(踝/髋), 所以这张表两个模型通用 ——
#   碰撞面不参与关节静平衡。
_LEG_JOINT_ORDER = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
# 实测站姿相对 default_joint_pos 的偏移(零动作静置 6 步, CPU)
_STANCE_OFFSET = {
    "left_hip_yaw": 0.0, "left_hip_roll": +0.0780, "left_hip_pitch": +0.2354,
    "left_knee": +0.0833, "left_ankle": -0.3083,
    "right_hip_yaw": 0.0, "right_hip_roll": -0.0780, "right_hip_pitch": -0.2356,
    "right_knee": -0.0835, "right_ankle": +0.3081,
}
# 只留一套 std(不做"站/走"分档): 分档会出现反向激励 —— 站立档更紧 → 站着被罚 -0.94、
# 走起来反而只罚 -0.12, 等于"抖一下就能放松姿态约束"。这个项只是"别叉腿/别歪骨盆"的
# 低权重护栏, 不需要随指令变形。松紧关系沿用上游 variable_posture 的走路档。
_STD_STANCE = {
    "hip_yaw": 0.30, "hip_roll": 0.05, "hip_pitch": 0.40, "knee": 0.40, "ankle": 0.25,
}


def _std_for(name: str, table: dict[str, float]) -> float:
    for key, value in table.items():
        if key in name:
            return value
    return 0.3


class stance_deviation_penalty:
    """想别叉腿/别歪骨盆: 腿关节偏离**实测站姿**的归一化平方(低权重, 只管"像样不像样")。

    为什么是"罚"而不是上游那种 +1.0 的正奖励 pose: 正奖励会让"站着不动"多拿 1 分,
    把"站着"的收益从 3.3 抬到 4.3, 反而更难学走路(它的上限也就 5 出头)。做成低权重罚,
    干净的站姿和干净的步态都≈0 代价, 只有"叉腿/歪骨盆"才咬 —— 同一件事, 但不抬高站立盆地。

    形状上仍是"分关节 std 的归一化平方"(而不是对所有关节一视同仁的二次罚):
      hip_roll 很紧(0.05 → 骨盆别歪), hip_pitch/knee 很松(0.4 → 允许正常迈步幅度)。
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset = env.scene[cfg.params["asset_cfg"].name]
        ids, names = asset.find_joints(tuple(_LEG_JOINT_ORDER))
        self.joint_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
        self.offset = torch.tensor(
            [_STANCE_OFFSET[n] for n in names], device=env.device
        )
        self.std = torch.tensor([_std_for(n, _STD_STANCE) for n in names], device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """返回 >= 0 的 cost(配负权重)。基准 = default_joint_pos + 实测站姿偏移。"""
        asset = env.scene[asset_cfg.name]
        dev = (
            asset.data.joint_pos[:, self.joint_ids]
            - asset.data.default_joint_pos[:, self.joint_ids]
            - self.offset
        )
        return ((dev / self.std) ** 2).mean(dim=1)


def joint_dev_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """想别乱来: 关节离"默认站姿"越远扣分越多(走路必须偏离, 所以权重必须很小)。"""
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return ((q - q_default) ** 2).mean(dim=1)


def head_joint_dev_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """想稳: 头/脖子 4 个关节别乱甩(它们对行走没用, 甩起来只会扰动平衡)。"""
    return joint_dev_penalty(env, asset_cfg)


def body_ang_vel_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """想别晃: 躯干的**滚转/俯仰**角速度平方和(不罚偏航)(上游 body_angular_velocity_penalty 同款)。

    用上游的精确取法: 世界系角速度的前两个分量。为什么可以直接用世界系: 偏航是绕 z 的
    旋转, 把 (x,y) 平面映到自身, 所以 ω_x²+ω_y² 在"世界系 xy"和"机体系 xy"下相同
    (躯干接近直立时两者都在描述同一件事)。mjlab 只提供任意 body 的世界系角速度
    (body_link_ang_vel_w), 所以照上游写。
    注意它是"动态动作本身必然需要的量", 属于 motion-blocker, 权重必须很小:
    权重一大, 策略会宁可站住不动(上游 playbook: motion-blocker 要 LOW)。

    实测量级(CPU, 4 envs × 200 步, 躯干 |ω_xy|²): 零动作中位 0.43、随机乱蹬中位 5.66。
    乘上 -0.03 后是 **-0.013 ~ -0.17 分/步** —— 正常走路时只有正质量(≈5.75)的 0.2%,
    踉跄/乱蹬时才涨到 3%。这正是想要的行为: 平时不干扰, 出事才咬。
    """
    asset = env.scene[asset_cfg.name]
    ang_vel_w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
    return (ang_vel_w[:, :2] ** 2).sum(dim=1)


def angular_momentum_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """想别甩: 整机角动量平方(上游 angular_momentum_penalty 同款, 用模型的 subtreeangmom 传感器)。

    对这只鸭子尤其有意义: 头和脖子约占全机质量 40%, 甩头会直接带动平衡。
    同样属于 motion-blocker —— 走路时角动量本来就不会是 0, 所以权重必须小。

    ⚠️ 实测结论(务必先看再信): 这只鸭子太小, |L| 只有 0.003~0.008 kg·m²/s
    (L ∝ 质量×尺度²), 平方之后 ~1e-5, 乘 -0.01 是 **1e-7 分/步** —— 比正质量小
    七个数量级, 也就是说**这个权重下它等于没接**。上游同样权重(-0.02)在本机上
    也是这个量级。保留它的理由只有两条: ① 和上游对齐(以后 A/B 不会因为少了这一项
    而困惑); ② 顺手记 Metrics/angular_momentum_mean 这条曲线(看头和脖子甩不甩)。
    要让它真正有数值影响, 得先解决量纲(例如除以参考惯量 L/(I·ω_ref) 再平方),
    那是另一件事, 不属于本次抓的"会拖慢收敛"问题。别被"加了正则所以更稳"骗了。
    """
    angmom = env.scene[sensor_name].data  # [B, 3]
    angmom_sq = (angmom**2).sum(dim=-1)
    env.extras.setdefault("log", {})["Metrics/angular_momentum_mean"] = angmom_sq.sqrt().mean()
    return angmom_sq


def reward_weight_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    weight_stages: list[tuple[int, float]],
) -> torch.Tensor:
    """按训练步数**分段**改写某项奖励的权重(阶梯, 不插值; 上游 mdp.reward_weight 同款)。

    为什么要课程而不是一步到位: 平滑税属于"尝试税" —— 技能还没探索出来的时候先收
    重税, 会让"站着不动"变成局部最优(上游 playbook 原话: any attempt-tax active
    while a hard skill is being explored makes "do nothing" win)。所以: 起步给很轻,
    等步态长出来再加压。

    两个坑(都写进注释免得下次踩):
      ① 必须改 manager 里的 term cfg。env.cfg 是 manager 初始化时 deepcopy 的,
         写 env.cfg.rewards[...] 是**静默 no-op**;
      ② 一旦挂了课程, 命令行覆盖 `--env.rewards.action_rate.weight` 就只影响"第一次",
         每步都会被课程改写回去 —— 想改就改 _ACTION_RATE_WEIGHT_STAGES。
    """
    del env_ids
    term_cfg = env.reward_manager.get_term_cfg(reward_name)
    for stage_step, stage_weight in weight_stages:
        if env.common_step_counter >= stage_step:
            term_cfg.weight = stage_weight
    return torch.tensor([term_cfg.weight], device=env.device)


# ============ 环境配置 ============

def _twist_command_cfg() -> UniformVelocityCommandCfg:
    """速度指令项: 平地小双足的起步档(其它档位由 curriculum 在训练中放宽)。"""
    stage0 = _VELOCITY_STAGES[0]
    return UniformVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(4.0, 8.0),  # 每 4~8 秒换一次指令
        rel_standing_envs=0.1,  # 10% 的环境拿零指令 → 顺带保持站立能力
        # 注意: 上游的 rel_forward_envs 会把"直行环境"的 vx 强制抬到 ≥0.3 m/s
        # (那是给 Unitree Go1 定的常数), 对鸭子来说 0.3 已经超过整条课程的上限,
        # 起步阶段会直接把策略打死 → 这里关掉。反正前几档 vy/wz 本来就是 0,
        # 前期所有环境天然都是"直着往前走"。
        rel_forward_envs=0.0,
        heading_command=False,  # 不做"朝向目标"闭环, 只要转得动就行
        debug_vis=True,  # 看画面时把指令箭头画出来
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=stage0["lin_vel_x"],
            lin_vel_y=stage0["lin_vel_y"],
            ang_vel_z=stage0["ang_vel_z"],
        ),
    )


def _feet_contact_sensor_cfg() -> ContactSensorCfg:
    """两只脚 vs 地形的接触传感器: air_time / slip / clearance 三个奖励的输入源。"""
    return ContactSensorCfg(
        name=_FEET_CONTACT_SENSOR,
        primary=ContactMatch(mode="geom", pattern=_FOOT_GEOM_NAMES, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,  # 开启"离地时长"统计(compute_first_contact 依赖它)
    )


def reward_param_stages_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    param_name: str,
    stages: list[tuple[int, float]],
) -> torch.Tensor:
    """按训练步数**分段**改写某项奖励的某个标量参数(阶梯, 不插值)。

    和 reward_weight_curriculum 同一套路, 只是改的是 params 里的值(例如 air_time 的
    threshold_min: 奖励函数每步都被 manager 用**当前的** params 调用, 所以改这里立刻生效)。
    同样要改 manager 里的 term cfg —— 写 env.cfg 是静默 no-op。
    """
    del env_ids
    term_cfg = env.reward_manager.get_term_cfg(reward_name)
    for stage_step, stage_value in stages:
        if env.common_step_counter >= stage_step:
            term_cfg.params[param_name] = stage_value
    return torch.tensor([term_cfg.params[param_name]], device=env.device)


def microduck_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """行走(平地速度跟踪)环境配置。"""
    # P0 修正: actor 里**不放** base_lin_vel。
    # 真机(鸭子只有 IMU + 舵机编码器)给不出"机体线速度"这个量, 上游
    # microduck_velocity_env_cfg.py:532 专门把它从 actor 删掉、只留给 critic
    # (注释: privileged information)。放在 actor 里有两个害处:
    #   ① 部署时给不出 → 策略行为直接崩(这不是"鲁棒性差", 是接口不成立);
    #   ② 训练时等于开卷考试: 速度真值直接喂进去, track_lin_vel 会虚高好看,
    #      让人误判"奖励设计没问题"。
    # 删掉之后策略必须从 IMU+编码器+动作历史里自己推断速度 —— 这正是要它学的。
    actor_terms = {
        "joint_pos": ObservationTermCfg(func=joint_pos_rel),      # 14: 关节角(相对默认)
        "joint_vel": ObservationTermCfg(func=joint_vel_rel),      # 14: 关节角速度
        "gravity": ObservationTermCfg(func=projected_gravity),    # 3: 姿态(靠它站直/不翻)
        "ang_vel": ObservationTermCfg(func=base_ang_vel),         # 3: 机体角速度
        "actions": ObservationTermCfg(func=last_action),          # 14: 上一步动作
        "command": ObservationTermCfg(                            # 3: 当前速度指令(vx,vy,wz)
            func=generated_commands, params={"command_name": "twist"}
        ),
    }
    # critic 拿特权信息(只用于算价值函数, 不参与部署): 加回真实机体线速度
    critic_terms = {
        **actor_terms,
        "base_lin_vel": ObservationTermCfg(func=base_lin_vel, scale=1.0),  # 3
    }
    cfg = ManagerBasedRlEnvCfg(
        commands={"twist": _twist_command_cfg()},
        events={
            "bam_init": EventTermCfg(func=bam_init, mode="startup"),
            # 每回合随机朝向 + 位置上一点抖动: 免得策略只学会"朝着 +x 往前扑"
            "reset_root": EventTermCfg(
                func=reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0),
                        "yaw": (-math.pi, math.pi),
                    },
                    "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
                },
            ),
            # 关节回到默认站姿, 只加很小的抖动(初学阶段姿态别太野)
            "reset_joints": EventTermCfg(
                func=reset_joints_by_offset,
                mode="reset",
                params={"position_range": (-0.03, 0.03), "velocity_range": (0.0, 0.0)},
            ),
            # 训练中每 3~6 秒推一把: 抗扰动(推得比站立任务轻, 走路的容错本来就小)
            "push_robot": EventTermCfg(
                func=push_by_setting_velocity,
                mode="interval",
                interval_range_s=(3.0, 6.0),
                params={
                    "velocity_range": {
                        "x": (-0.2, 0.2), "y": (-0.2, 0.2),
                        "pitch": (-0.15, 0.15), "yaw": (-0.2, 0.2),
                    },
                },
            ),
        },
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            sensors=(_feet_contact_sensor_cfg(),),
            entities={"robot": get_microduck_walk_entity_cfg()},
            num_envs=1,
            env_spacing=4.0,
        ),
        observations={
            "actor": ObservationGroupCfg(actor_terms, concatenate_terms=True),
            "critic": ObservationGroupCfg(critic_terms, concatenate_terms=True),
        },
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                # P0 修正: 0.5 → 1.0(与上游 velocity_cfg:271-273 一致)。
                # scale 是"策略输出 1.0 对应多少弧度的关节偏移"。走路真正吃紧的
                # 是**抬脚需要的膝屈**: 约 0.5~0.9rad。
                #   scale=0.5 → 策略得输出 1.0~1.8(≈1~2σ, 探索很难摸到)
                #   scale=1.0 → 只要输出 0.5~0.9(<1σ, 探索轻松覆盖)
                # 不是"能不能到"的问题(高斯没截断, 3σ 也能到), 而是**典型的
                # 采样点落不落在有用的区域**。站着/坐站任务仍然用 0.5, 不动它们。
                scale=1.0,
                use_default_offset=True,
            )
        },
        rewards={
            # —— 主目标: 听速度指令 ——
            # std 的标定史(三次): 0.25(照抄上游的绝对量) → 0.10(按我们的指令尺度收紧,
            # 但**比实际能达到的精度还严**) → **0.20**(第七轮, 用真步态标定)。
            # 为什么 0.10 是错的: 拿上游真机在用的 alpha_walking.onnx 在我们的环境里量,
            # 它在整个指令范围内只能跑到 ~45%(误差 0.07~0.17), 而 std=0.10 意味着
            # "误差 0.17"的奖励是 exp(-0.0289/0.01)=0.06 —— 等于**惩罚一切真实步态**。
            # 0.20 时同样的误差是 exp(-0.0289/0.04)=0.49, 与上游(std=0.316 对 ±0.4 指令,
            # 误差 0.17 拿 0.75)是同一个量级; 而"站着不动"(误差=指令值)仍只拿 10~20%。
            "track_lin_vel": RewardTermCfg(
                func=track_lin_vel_xy_reward, weight=2.0,
                params={"std": 0.20, "command_name": "twist"},
            ),
            # P2: 权重 0.5 → 1.0, std 0.4 → 0.5(上游是 2.0/sqrt(0.5), 但我们的指令
            # 尺度小、早期 wz 多为 0, 取一半避免变成一个拿不到的大项)。
            "track_ang_vel": RewardTermCfg(
                func=track_ang_vel_z_reward, weight=1.0,
                params={"std": 0.5, "command_name": "twist"},
            ),
            # P0 修正: 线性 -g_z(0.5) → 高斯 exp(-Σg_xy²/std²)(2.0)。
            # 同一倾角(0.1rad)的惩罚力度从 0.0025/步 提到 0.36/步(≈140x),
            # 因为整套奖励里没有第二项在罚"静态倾角"。
            "upright": RewardTermCfg(
                func=upright_gauss_reward, weight=2.0,
                params={"std": math.sqrt(0.05),
                        "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
            ),
            # —— 步态塑形(小双足能不能走起来基本靠这几项) ——
            # P0: 权重 0.5 → 3.0, 改成"每步都在窗口内就付"(见 feet_air_time_reward)。
            # P2: 窗口下限 0.125 → 0.03 起步, 由课程抬上去(见 _AIR_WINDOW_MIN_STAGES)——
            #     实测鸭子一开始只能摆动 20~30ms, 0.125 的下限让"抬脚"这件事零梯度。
            # 权重回到 3.0(第五轮): 第三轮曾把它砍到 1.5 来堵"原地蹦"的薅分 —— 但那个
            # 薅分点的真正堵法是**门**(姿态门 + 方向进度门), 不是降权重。门已经在之后,
            # 权重该还回去, 因为"步态 vs 速度"的价差决定了策略把哪件事当目标:
            #   air_time 1.5 < track_lin 2.0 时 ===> **溜**(两脚贴地滑着往前推, 实测离地仅
            #     5.5%、摆动 4ms)就能满足目标, 迈步只值 +1.5, 于是"迈步"不是目的只是手段;
            #   air_time 3.0 > track_lin 2.0 时 ===> 迈步本身就是最大的分项(上游就是这个
            #     关系: air_time 3.0 > track_lin 2.0), 策略才会去解决"怎么把脚抬起来还能不倒"。
            # 实测这门挡得住: 原地蹦(离地 45% 但不位移)的门只有 0.35、躺平的门是 0。
            # 但只到"有前进速度"还不够(第五轮实测): 策略变成了"匀速 0.26m/s 的行进器",
            # 对任何指令都满门 —— 所以再乘**跟踪核**(和 track_lin_vel 同一个核), 要求
            # "在把速度走对的前提下抬脚"。两个目标从"替代"变成"协同": 实测(构造速度对比)
            #   指令 0.15 + 机体 0.26(超速) → 得分降到 1/3.4
            #   指令 0.15 + 机体 0.15(走对) → 满分
            #   指令 0.10 + 机体 0.26(乱冲) → 降到 1/13
            "air_time": RewardTermCfg(
                func=feet_air_time_reward, weight=3.0,
                params={"sensor_name": _FEET_CONTACT_SENSOR, "command_name": "twist",
                        "threshold_min": _AIR_WINDOW_MIN_STAGES[0][1], "threshold_max": 0.300,
                        "command_threshold": 0.01},
            ),
            # P0 修正: 探针从"脚踝 body 高度"改成"脚底采样点最低 z", 目标 0.04 → 0.02。
            # 中间试过"换成脚底 site"和"上游的射线传感器", 实测都不行(前者漂得更多,
            # 后者在平地上等于 site 高度), 详见 _SWING_TARGET_HEIGHT 上方那组数字。
            # 目标沿用上游的 0.02, 权重沿用上游的 -2.0。
            "foot_clearance": RewardTermCfg(
                func=feet_clearance_penalty, weight=-2.0,
                params={"sensor_name": _FEET_CONTACT_SENSOR, "asset_cfg": _ASSET_FEET_BODIES,
                        "vel_asset_cfg": _ASSET_FEET_SITES,
                        "command_name": "twist", "target_height": _SWING_TARGET_HEIGHT,
                        "command_threshold": 0.01},
            ),
            # P0 修正: 速度探针换成脚底 site(接触点的速度才是"打滑"), 权重随之上游化 -0.1
            "foot_slip": RewardTermCfg(
                func=feet_slip_penalty, weight=-0.1,
                params={"sensor_name": _FEET_CONTACT_SENSOR, "asset_cfg": _ASSET_FEET_SITES,
                        "command_name": "twist", "command_threshold": 0.01},
            ),
            # —— 正则项: 别乱抖 / 别撞限位 / 姿态别跑太偏 ——
            # 姿态罚: 以**实测站姿**为基准(原来以 XML 默认姿势为基准 = 按站不到的姿势扣分),
            # 低权重只管"别叉腿/别歪骨盆"。干净站姿≈0 代价, 所以不抬高"站着"的收益。
            "stance_dev": RewardTermCfg(
                func=stance_deviation_penalty, weight=-0.2,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINT_ORDER)},
            ),
            "head_dev": RewardTermCfg(
                func=head_joint_dev_penalty, weight=-0.2, params={"asset_cfg": _ASSET_HEAD_JOINTS},
            ),
            "joint_pos_limits": RewardTermCfg(
                func=joint_pos_limits, weight=-1.0, params={"asset_cfg": _ASSET_ALL_JOINTS},
            ),
            # —— 动态类正则(motion-blocker, 权重必须小): 别乱滚/别乱甩 ——
            # 上游 body_ang_vel -0.05 / angular_momentum -0.02, 按正质量份额缩放(见常量处)
            "body_ang_vel": RewardTermCfg(
                func=body_ang_vel_penalty, weight=_BODY_ANG_VEL_WEIGHT,
                params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
            ),
            "angular_momentum": RewardTermCfg(
                func=angular_momentum_penalty, weight=_ANGULAR_MOMENTUM_WEIGHT,
                params={"sensor_name": "robot/root_angmom"},
            ),
            # 初始值 = 课程第 1 档; 训练中由 _ACTION_RATE_WEIGHT_STAGES 逐步加压到 -0.10
            "action_rate": RewardTermCfg(func=action_rate_l2, weight=_ACTION_RATE_WEIGHT_STAGES[0][1]),
            # 站立任务里 -0.02 的"别抖"惩罚对走路太重(腿必须动), 缩到 1/20
            "joint_vel": RewardTermCfg(
                func=joint_vel_l2, weight=-0.001, params={"asset_cfg": _ASSET_ALL_JOINTS},
            ),
        },
        terminations={
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
            # 姿态偏超过 45° 算摔。为什么不用上游的 70°(第五轮实测): 实测"前倾换速度"
            # 的策略在倒之前倾角只到 13~14°, 一旦过了 25~30° 就已经救不回来了 —— 70° 时
            # 它已经在"注定要倒"的轨迹上白跑 ~0.3s, 那 0.3s 的样本既学不到恢复, 又把
            # "倒下的前一步"和"终止"之间的因果拉远。45° 是个安全线: 正常步态的躯干俯仰
            # 远小于 25°(实测静止 3.3°、移动 13.5°), 45° 只在真失控时出现。
            "bad_orientation": TerminationTermCfg(
                func=bad_orientation,
                params={"limit_angle": math.radians(45.0), "asset_cfg": SceneEntityCfg("robot")},
            ),
            # P2⑪: 删掉"躯干 < 5cm **立刻**终止"(实测它砍掉了 100% 的尝试回合)。
            # P2⑧(第三轮): 但删干净之后鸭子学会了"躺地上永久薅分" ⇒ 补一个带**宽限期**
            # 的版本: 倒地 1 秒内不终止(恢复样本留着), 超过 1 秒才终止并重置回站立出生点。
            "fallen_too_long": TerminationTermCfg(
                func=fallen_too_long,
                params={"z_threshold": _GATE_Z_LOW,
                        "tilt_threshold": math.radians(45.0),
                        "grace_s": 1.0},
            ),
            # 仿真炸掉时不至于把 NaN 当正常数据训下去
            "nan_state": TerminationTermCfg(func=nan_detection),
        },
        curriculum={
            # 指令范围随训练步数逐步放宽(见 _VELOCITY_STAGES)
            "command_vel": CurriculumTermCfg(
                func=velocity_mdp.commands_vel,
                params={"command_name": "twist", "velocity_stages": _VELOCITY_STAGES},
            ),
            # 动作平滑税随步态长出来逐步加压(见 _ACTION_RATE_WEIGHT_STAGES)
            "action_rate_weight": CurriculumTermCfg(
                func=reward_weight_curriculum,
                params={"reward_name": "action_rate",
                        "weight_stages": _ACTION_RATE_WEIGHT_STAGES},
            ),
            # 步态奖励窗口下限随实测摆动时长逐步收紧(见 _AIR_WINDOW_MIN_STAGES)
            "air_window": CurriculumTermCfg(
                func=reward_param_stages_curriculum,
                params={"reward_name": "air_time", "param_name": "threshold_min",
                        "stages": _AIR_WINDOW_MIN_STAGES},
            ),
        },
        # 物理步长/控制频率与站立任务一致(0.002s × 10 = 50Hz)
        sim=SimulationCfg(njmax=300, contact_sensor_maxmatch=64, mujoco=MujocoCfg(timestep=0.002)),
        decimation=10,
        episode_length_s=20.0,  # 20s: 一条指令周期(4~8s)能被完整走几遍
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot", body_name="trunk_base",
            distance=1.0, elevation=1.0, azimuth=45.0),
    )
    if play:
        # 回放/推理: 不终止, 用最宽指令范围看策略上限, 并去掉训练用的"随机推"
        cfg.episode_length_s = 1e9
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        twist = cfg.commands["twist"]
        assert isinstance(twist, UniformVelocityCommandCfg)
        twist.ranges.lin_vel_x = _PLAY_RANGES["lin_vel_x"]
        # ⚠️ 每个轴的**上限必须 ≥0.1**, 否则 viser 摇杆会崩:
        #   mjlab 的 UniformVelocityCommand.create_gui 会建一个 "Max {轴}" 滑块,
        #   它把 min 写死 0.1、而 initial_value = ranges.<轴>[1] —— 我们训练用的
        #   lin_vel_y 上限是 0.05, 于是 viser 报
        #     "initial_value (0.05) is outside of [min, max] = [0.1, 10.0]"
        #   (2026-09-12 实测)。评估时给到 ±0.10 既够用又不会崩(策略在 ±0.05 上训练,
        #   超出部分会被饱和, 不影响看动作)。原生窗口(--viewer native)不走这段 GUI。
        twist.ranges.lin_vel_y = (-0.10, 0.10)
        twist.ranges.ang_vel_z = _PLAY_RANGES["ang_vel_z"]
        # 一次看多只鸭子: 训练用的 env_spacing=4m 会把 N 个环境摆成网格、彼此 4m 远 ——
        # 相机跟着某一只时, 其它几只都在画面外。压到 1.2m(鸭子本体 ~0.25m),
        # `--num-envs 4` 时 4 只都能进同一画面。
        # 环境之间互不碰撞(每个环境是独立的一份物理), 挨得近不会互相干扰;
        # 代价是它们走起来会"穿模"—— 那是各自的世界, 看着好玩而已。
        cfg.scene.env_spacing = 1.2
    return cfg


def microduck_walk_ppo_runner_cfg():
    """走路任务的 PPO 配置: 网络容量与训练预算按 **GPU 尺度** 设置。

    为什么这么定(第八轮, 上 GPU 前):
      · 隐藏层 256×256(站立 64×64 / 之前 128×128): 上游对这只鸭子的走路任务用
        512-256-128, 我们先用中间档; CPU 冒烟时它同样能跑(慢一点)。
      · num_steps_per_env 32 → **24**: 与上游一致(他们的课程步数也是按 24 标定的)。
        注意这会让"课程第几轮触发"变快(1 轮 = 24 策略步): 1500 步 = 62 轮(原来 47)。
      · max_iterations 1000 → **6000**, save_interval 200 → **250**:
        上游经验"步态类任务 4000~6000 轮 @4096 envs"(他们的 README: 4096 envs 下
        1~2 小时出可用步态)。默认 1000 轮等于还没开始就停了。
    """
    base = microduck_ppo_runner_cfg()
    base.experiment_name = "microduck-walk"
    base.actor.hidden_dims = (256, 256)
    base.critic.hidden_dims = (256, 256)
    base.num_steps_per_env = 24
    base.max_iterations = 6000
    base.save_interval = 250
    return base
