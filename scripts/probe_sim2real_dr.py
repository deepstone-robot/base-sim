"""逐个验证 sim2real 随机化项在这套 mjlab 1.3.0 + BAM 下到底生不生效。

为什么需要这个脚本: 上游 `microduck_rl` 的注释里记着一串 **"silent no-op"** 教训 ——
  * stock `dr.dof_frictionloss` 在 BAM 下是 no-op(BAM 把 dof_frictionloss 置零了)
  * 直接写 body_mass/body_inertia 在 mjlab 1.3.0 下不按环境展开(会塌成一个共享值)
  * 写 site_quat 做 IMU 安装误差是 no-op(那些 obs 根本不读 site_quat)
  * `joint_pos_rel` 默认**忽略** encoder bias(要 biased=True 才读 joint_pos_biased)
这些坑的共同点: **配置能跑通、训练不报错、但什么都没随机化**。所以每一项都要实测。

本脚本每一项都给"生效 / no-op"的判决 + 证据数字。只读、不训练、不改任何 cfg 文件。

用法: python scripts/probe_sim2real_dr.py

2026-09-12 实测判决(mjlab 1.3.0 + 我们装的 bam, 16 环境):

  ① 脚底摩擦 dr.geom_friction        生效   两脚 0.74~1.26(目标 0.7~1.3), 环境间跨度 0.52,
                                             shared_random=True 确实让两脚同值
                                             ⚠️ 陷阱: geom_names 留空(上游模板里的占位写法)
                                                会改动 0/76 个 geom = 静默 no-op, 必须填名字
  ② 编码器偏置 dr.encoder_bias       生效   字段按环境采样(±0.015), 且与 actor 观测之差 1.5e-8
                                             ⚠️ 陷阱: 不设 joint_pos_rel(biased=True) 时
                                                |actor观测-真实关节角| = 0.000e+00 = 静默 no-op
  ③ 质心偏移 dr.body_ipos            生效   偏移 ≤0.0030m, 连续 5 次 reset **不累积**(正确),
                                             环境间 σ=0.0017m(说明确实按环境展开)
                                             ⚠️ 注意 reset 模式的事件在**构造完 env 时还没生效**,
                                                要等一次真正的 reset(第一次采样前是 0)
  ④ 质量+惯量 dr.pseudo_inertia      生效   躯干质量 190.7~208.4 g(跨度 8.9%), 惯量同步变
  ⑤ 转子惯量 dr.joint_armature       **no-op** 倍数恒为 1.000; 跑一步之后仍是 1.000
                                             ⇒ **BAM 每步重写 dof_armature, 把 DR 覆盖了**。
                                             上游注释说"BAM 下这条真生效"是在他们打过补丁的
                                             bam 上; 我们这版不生效, 要做得改 BAM 侧(见下)。
  ⑥ 观测噪声 Unoise                  生效   actor-critic 同槽相减 σ=0.1441(理论 0.25/√3=0.1443)
  ⑦ 观测延迟 delay_min/max_lag       生效   延迟 1 时观测 = 上一步状态(0.0163 < 0.0615),
                                             延迟 0 时 = 本步状态(0.0000 < 0.0788)

还没验的一项: **BAM 下的关节摩擦**(上游 `FrictionDRBamActuator` + `randomize_bam_friction`)。
  我们的 bam 里 `BamActuator._compute_friction_budget` 与 `kp_scale` **都在**, 所以上游那个
  35 行的子类理论上可直接搬; 但因为 ⑤ 已经证明"BAM 每步会重写 dof_* 字段", 这条必须单独实测
  (做法: 同一个动作序列下把 friction_scale 设成 0.2 / 2.0, 看关节轨迹是否不同; 相同 = no-op)。
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import mujoco
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.envs.mdp import dr  # noqa: E402
from mjlab.managers import EventTermCfg, SceneEntityCfg  # noqa: E402
from mjlab.utils.noise import UniformNoiseCfg as Unoise  # noqa: E402

from mjlab_microduck.tasks import microduck_walk_env_cfg  # noqa: E402

NUM = 16
FOOT_FRICTION_RANGE = (0.7, 1.3)      # 上游现用值(grippier footpad)
ENCODER_BIAS_RANGE = (-0.015, 0.015)  # 上游现用值 ±0.86°
COM_RANGE = 0.003                     # 上游 ±3mm(起步值, 课程序上升)
HEAD_BODY_NAMES = ("neck", "neck_pitch", "yaw_roll_motion", "jaw_soft")
MASS_INERTIA_RANGE = (0.95, 1.05)
ARMATURE_RANGE = (0.9, 1.1)


def _make_env(extra_events: dict | None = None, obs_patch=None, num_envs: int = NUM,
              with_push: bool = False) -> ManagerBasedRlEnv:
    cfg = microduck_walk_env_cfg()
    cfg.scene.num_envs = num_envs
    if not with_push:
        cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.episode_length_s = 4.0
    for name, term in (extra_events or {}).items():
        cfg.events[name] = term
    for path, value in (obs_patch or {}).items():
        group, term_name, key = path
        if term_name is None:               # 组级属性, 例如 enable_corruption
            setattr(cfg.observations[group], key, value)
        else:
            term = copy.deepcopy(cfg.observations[group].terms[term_name])
            import dataclasses as _dc
            if key in {f.name for f in _dc.fields(type(term))}:   # 是 term 自己的字段
                setattr(term, key, value)                         # 例如 delay_min_lag
            else:
                term.params[key] = value
            cfg.observations[group].terms[term_name] = term
    return ManagerBasedRlEnv(cfg=cfg, device="cpu")


def _id_of(mj, obj_type, name: str) -> int:
    """编译后的模型里名字带实体前缀(如 `robot/trunk_base`), 所以要前缀/后缀都能匹配。

    (踩过: 直接 mj_name2id 返回 -1, 于是所有读取都落在"最后一个 body"上,
     看起来像"DR 没生效" —— 实际是读错了对象。)
    """
    i = mujoco.mj_name2id(mj, obj_type, name)
    if i >= 0:
        return i
    for cand in (f"robot/{name}",):
        i = mujoco.mj_name2id(mj, obj_type, cand)
        if i >= 0:
            return i
    for k in range(mj.nbody if obj_type == mujoco.mjtObj.mjOBJ_BODY else mj.ngeom):
        n = mujoco.mj_id2name(mj, obj_type, k)
        if n and (n.endswith("/" + name) or n == name):
            return k
    raise KeyError(f"模型里找不到 {name}")


def _geoms_on_bodies(env: ManagerBasedRlEnv, body_names: tuple[str, ...]) -> list[int]:
    mj = env.sim.mj_model
    ids = {_id_of(mj, mujoco.mjtObj.mjOBJ_BODY, b) for b in body_names}
    return [g for g in range(mj.ngeom) if mj.geom_bodyid[g] in ids]


def _body_id(env: ManagerBasedRlEnv, name: str) -> int:
    return _id_of(env.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)


def _default(env: ManagerBasedRlEnv, field: str):
    return torch.as_tensor(getattr(env.sim.mj_model, field)).clone()


# --------------------------------------------------------------------------- ①
def check_foot_friction() -> tuple[str, str]:
    """脚底摩擦: 只看两个脚底 geom, 且各环境不同、两脚相同(shared)。"""
    env = _make_env()
    mj = env.sim.mj_model
    # 注意: 脚踝 body 上挂着 5 个 geom, 只有 *_collision 那两个参与碰撞; 读全部会把视觉 geom
    #       (没被随机化、默认摩擦 1.0)混进来, 于是"值域/是否共享"都算错(第一版就踩了)
    feet = [_id_of(mj, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision"),
            _id_of(mj, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")]
    print(f"  脚底碰撞 geom ids = {feet}"
          f" ({[mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, g) for g in feet]})")
    print(f"  (脚踝上共 {len(_geoms_on_bodies(env, ('ankle_left', 'ankle_right')))} 个 geom,"
          " 其中只有这两个参与碰撞)")
    env.close()

    # 真正打开这个事件, 再看两个碰撞 geom 的采样值
    ev = EventTermCfg(func=dr.geom_friction, mode="startup", params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(
            "left_foot_collision", "right_foot_collision")),  # 模型里的真实名字
        "operation": "abs", "ranges": FOOT_FRICTION_RANGE, "shared_random": True})
    env = _make_env({"foot_friction": ev})
    f = env.sim.model.geom_friction[:, feet, 0]            # [num_envs, 2]
    d0 = _default(env, "geom_friction")[feet, 0]
    lo, hi = float(f.min()), float(f.max())
    spread = float(f.max(dim=0).values.min() and (f.max(dim=0).values - f.min(dim=0).values).mean())
    shared = bool(torch.allclose(f[:, 0], f[:, 1]))
    print(f"  默认摩擦 {float(d0[0]):.3f}; 采样值域 [{lo:.3f}, {hi:.3f}] (目标 {FOOT_FRICTION_RANGE})")
    print(f"  环境间跨度(每只脚) {spread:.3f};  两只脚同值(shared_random=True): {shared}")
    if not shared:
        d = (f[:, 0] - f[:, 1]).abs()
        print(f"  ⚠️ shared_random=True 并没有让两只脚共享同一个值: 两脚差 "
              f"均值 {float(d.mean()):.3f}, 最大 {float(d.max()):.3f} "
              "→ 实际会随机出'左脚比右脚滑'这种不对称, 与上游注释的预期不符")
    env.close()

    # 陷阱: 上游模板里 geom_names 是空元组("Set per-robot"), 不填会怎样?
    ev2 = EventTermCfg(func=dr.geom_friction, mode="startup", params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),
        "operation": "abs", "ranges": FOOT_FRICTION_RANGE, "shared_random": True})
    try:
        env2 = _make_env({"foot_friction": ev2})
        f2 = env2.sim.model.geom_friction[:, :, 0]
        d0all = _default(env2, "geom_friction")[None, :, 0]
        changed = int((f2 - d0all).abs().max(dim=0).values.gt(1e-6).sum())
        print(f"  [陷阱] geom_names 留空时, 被改动的 geom 个数 = {changed} / {f2.shape[1]}"
              "  → 留空 = 什么都不随机(静默 no-op), 必须按模型填名字")
        env2.close()
    except Exception as exc:
        print(f"  [陷阱] geom_names 留空直接报错: {type(exc).__name__}: {exc}")

    ok = (lo >= FOOT_FRICTION_RANGE[0] - 1e-6 and hi <= FOOT_FRICTION_RANGE[1] + 1e-6
          and spread > 1e-3)
    verdict = "生效" if ok else "no-op/异常"
    return verdict, (f"两脚 {lo:.2f}~{hi:.2f}, 环境间跨度 {spread:.2f}, 两脚共享={shared}"
                     + ("" if shared else " ⚠️ shared_random 没生效"))


# --------------------------------------------------------------------------- ②
def check_encoder_bias() -> tuple[str, str]:
    """编码器偏置: ① 字段真的被采样了吗 ② actor 观测真的读到了吗(biased 开关)。"""
    ev = EventTermCfg(func=dr.encoder_bias, mode="startup",
                      params={"asset_cfg": SceneEntityCfg("robot"),
                              "bias_range": ENCODER_BIAS_RANGE})

    # 不设 biased(默认) -> 期望观测与"真实关节角"完全一致 = 偏置被忽略
    env_a = _make_env({"encoder_bias": ev})
    bias = env_a.scene["robot"].data.encoder_bias
    nz = int((bias.abs() > 1e-9).sum())
    print(f"  encoder_bias 形状 {tuple(bias.shape)}, 非零元素 {nz}/{bias.numel()}, "
          f"范围 [{float(bias.min()):+.4f}, {float(bias.max()):+.4f}] (目标 ±0.015)")
    robot = env_a.scene["robot"]
    obs_a = env_a.observation_manager.compute()["actor"]
    true_rel = robot.data.joint_pos - robot.data.default_joint_pos
    diff_a = (obs_a[:, :14] - true_rel).abs().max()
    env_a.close()

    env_b = _make_env({"encoder_bias": ev},
                      obs_patch={("actor", "joint_pos", "biased"): True,
                                 ("critic", "joint_pos", "biased"): False})
    robot_b = env_b.scene["robot"]
    obs_b = env_b.observation_manager.compute()["actor"]
    true_rel_b = robot_b.data.joint_pos - robot_b.data.default_joint_pos
    diff_b = (obs_b[:, :14] - true_rel_b).abs().max()
    matches_bias = (obs_b[:, :14] - true_rel_b - robot_b.data.encoder_bias).abs().max()
    env_b.close()

    print(f"  biased 不设 : |actor观测 - 真实关节角| 最大 {float(diff_a):.3e}  "
          f"→ {'偏置被忽略(no-op)' if float(diff_a) < 1e-9 else '偏置生效'}")
    print(f"  biased=True: |actor观测 - 真实关节角| 最大 {float(diff_b):.3e}; "
          f"与 encoder_bias 的差 {float(matches_bias):.3e}")
    verdict = "生效" if float(matches_bias) < 1e-6 and float(diff_a) < 1e-9 else "需检查"
    return verdict, ("字段采样 OK; 必须 biased=True 才会进 actor 观测"
                     if verdict == "生效" else "见上面两行")


# --------------------------------------------------------------------------- ③
def check_com() -> tuple[str, str]:
    """质心偏移: 躯干 + 头链 5 个 body。重点验证"跨 reset 不累积"(上游记录过的坑)。"""
    body_names = ("trunk_base",) + HEAD_BODY_NAMES
    ev = EventTermCfg(func=dr.body_ipos, mode="reset", params={
        "asset_cfg": SceneEntityCfg("robot", body_names=body_names),
        "operation": "add", "ranges": (-COM_RANGE, COM_RANGE)})
    env = _make_env({"randomize_com": ev})
    ids = [_body_id(env, b) for b in body_names]
    default = _default(env, "body_ipos")
    ipos = env.sim.model.body_ipos
    d = (ipos[:, ids, :] - default[None, ids, :])
    print(f"  第一次采样后 |偏移| 最大 {float(d.abs().max()):.4f} m (目标 ≤ {COM_RANGE})")
    # 连续 reset, 看偏移会不会越滚越大(accumulating footgun)
    worst = 0.0
    for i in range(5):
        env.reset()
        d = (env.sim.model.body_ipos[:, ids, :] - default[None, ids, :])
        worst = max(worst, float(d.abs().max()))
    per_env_std = float((env.sim.model.body_ipos[:, ids, :] - default[None, ids, :])
                        .std(dim=0).mean())
    print(f"  连续 5 次 reset 后 |偏移| 最大 {worst:.4f} m "
          f"→ {'不累积(正确)' if worst <= COM_RANGE * 1.5 else '**在累积!**'}")
    print(f"  环境间标准差 {per_env_std:.4f} m (0 = 所有环境同一个值 = 没按环境展开)")
    env.close()
    verdict = "生效" if worst <= COM_RANGE * 1.5 and per_env_std > 1e-4 else "no-op/累积"
    return verdict, f"偏移 ≤{worst:.4f}m, 不累积, 环境间 σ={per_env_std:.4f}"


# --------------------------------------------------------------------------- ④
def check_mass_inertia() -> tuple[str, str]:
    """质量+惯量: 走 pseudo_inertia(直接写 body_mass 在 1.3.0 不按环境展开)。"""
    import math
    lo, hi = MASS_INERTIA_RANGE
    ev = EventTermCfg(func=dr.pseudo_inertia, mode="startup", params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        "alpha_range": (math.log(lo) / 2.0, math.log(hi) / 2.0)})
    env = _make_env({"randomize_mass_inertia": ev})
    bid = _body_id(env, "trunk_base")
    mass = env.sim.model.body_mass[:, bid]
    inertia = env.sim.model.body_inertia[:, bid, :]
    m0 = float(_default(env, "body_mass")[bid])
    i0 = float(_default(env, "body_inertia")[bid].mean())
    print(f"  躯干质量 {m0 * 1000:.1f} g → [{float(mass.min()) * 1000:.1f}, "
          f"{float(mass.max()) * 1000:.1f}] g  (跨度 {float(mass.max() - mass.min()) / m0 * 100:.1f}%)")
    print(f"  躯干惯量均值 {i0:.3e} → [{float(inertia.mean(dim=1).min()):.3e}, "
          f"{float(inertia.mean(dim=1).max()):.3e}]  (跨度 "
          f"{float((inertia.mean(dim=1).max() - inertia.mean(dim=1).min()) / i0) * 100:.1f}%)")
    env.close()
    spread = float(mass.max() - mass.min()) / m0
    verdict = "生效" if spread > 0.01 else "no-op"
    return verdict, f"质量跨度 {spread * 100:.1f}%"


# --------------------------------------------------------------------------- ⑤
def check_armature() -> tuple[str, str]:
    """转子惯量(armature) DR。

    ⚠️ 实测判决是 **no-op**(见模块 docstring ⑤): BAM 每步都用自己配置里的 armature
    重写 `dof_armature`, reset/startup 时写进去的随机值被覆盖 —— 倍数恒为 1.000,
    跑一步之后仍 1.000。上游注释说"BAM 下这条真生效"是在他们打过补丁的 bam 上;
    我们这版不生效, 要生效得改 BAM 侧, 不是加个 cfg 能解决的。
    """
    ev = EventTermCfg(func=dr.joint_armature, mode="reset", params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
        "operation": "scale", "ranges": ARMATURE_RANGE})
    env = _make_env({"randomize_armature": ev})
    a = env.sim.model.dof_armature
    d0 = _default(env, "dof_armature")
    # 只看被驱动的关节: 浮动基座那 6 个 dof 的 armature 本来就是 0(除 0 会变 inf/0)
    jids = [_id_of(env.sim.mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in env.scene["robot"].joint_names]
    dofs = [int(env.sim.mj_model.jnt_dofadr[j]) for j in jids]
    rel = (a[:, dofs] / d0[None, dofs].clamp(min=1e-12))
    print(f"  统计 {len(dofs)} 个被驱动 dof")
    print(f"  reset 后相对默认的倍数: min {float(rel.min()):.3f}, max {float(rel.max()):.3f} "
          f"(目标 {ARMATURE_RANGE}); 环境间 σ {float(a.std(dim=0).mean()):.5f}")
    # BAM 每步都会写 dof_armature —— 如果它把自己的值写回来, DR 就被覆盖了(实测就是这个)
    env.step(torch.zeros(env.num_envs, 14))
    a2 = env.sim.model.dof_armature[:, dofs]
    rel2 = (a2 / d0[None, dofs].clamp(min=1e-12))
    print(f"  跑一步之后:              min {float(rel2.min()):.3f}, max {float(rel2.max()):.3f}"
          f"  → {'被 BAM 覆盖(DR 失效)' if float(rel2.max() - rel2.min()) < 1e-6 else '仍是随机值'}")
    env.close()
    ok = (float(rel.min()) >= ARMATURE_RANGE[0] - 1e-3 and float(rel.max()) <= ARMATURE_RANGE[1] + 1e-3
          and float(rel.max() - rel.min()) > 1e-3 and float(rel2.max() - rel2.min()) > 1e-6)
    return ("生效" if ok else "no-op/异常"), f"倍数 {float(rel.min()):.2f}~{float(rel.max()):.2f}"


# --------------------------------------------------------------------------- ⑥
def check_obs_noise() -> tuple[str, str]:
    """观测噪声: actor 组开噪声、critic 组不开 —— 两者相减就是噪声本身。

    (第一版用"站立时槽位的时间标准差", 把机器人自己的晃动也算进去了, 于是对照组
     就有 0.31 的标准差, 判不出噪声。actor-critic 相减才是干净的测法。)
    """
    def diff_std(patch) -> float:
        env = _make_env(obs_patch=patch)
        diffs = []
        for _ in range(40):
            env.step(torch.zeros(env.num_envs, 14))
            o = env.observation_manager.compute()
            diffs.append((o["actor"][:, 14:28] - o["critic"][:, 14:28]).clone())
        d = torch.stack(diffs)
        env.close()
        return float(d.std(dim=0).mean())

    off = diff_std({("actor", None, "enable_corruption"): True})
    on = diff_std({("actor", None, "enable_corruption"): True,
                   ("actor", "joint_vel", "noise"): Unoise(n_min=-0.25, n_max=0.25)})
    theory = 0.25 / 3 ** 0.5
    print(f"  actor(joint_vel 槽) - critic(同一槽) 的标准差: 不开噪声 {off:.5f} → 开噪声 ±0.25 {on:.5f}")
    print(f"  均匀分布 ±0.25 的理论标准差 = 0.25/√3 = {theory:.5f}")
    verdict = "生效" if off < 1e-6 and abs(on - theory) < 0.03 else "no-op/异常"
    return verdict, f"σ {off:.4f} → {on:.4f} (理论 {theory:.4f})"


# --------------------------------------------------------------------------- ⑦
def check_obs_delay() -> tuple[str, str]:
    """观测延迟: joint_vel 设成恒定 1 步(20ms)。

    口径: `observation_manager.compute()` 在 step 之后读, 所以
      delay=1 时它应该等于 **step 之前** 的 joint_vel(我记的 prev_true),
      delay=0 时应该等于 **step 之后** 的 joint_vel。
    (第一版把两栏的标签写反了, 于是把"生效"读成了"没看出延迟"。)
    """
    def score(patch) -> tuple[float, float]:
        env = _make_env(obs_patch=patch)
        prev_true = env.scene["robot"].data.joint_vel.clone()
        same_step, prev_step = [], []
        for _ in range(40):
            env.step(torch.zeros(env.num_envs, 14))
            obs = env.observation_manager.compute()["actor"][:, 14:28]
            now = env.scene["robot"].data.joint_vel.clone()
            same_step.append(float((obs - now).abs().mean()))
            prev_step.append(float((obs - prev_true).abs().mean()))
            prev_true = now
        env.close()
        return sum(same_step) / len(same_step), sum(prev_step) / len(prev_step)

    s0, p0 = score({})
    s1, p1 = score({("actor", "joint_vel", "delay_min_lag"): 1,
                    ("actor", "joint_vel", "delay_max_lag"): 1})
    print(f"  延迟 0: |观测-本步状态| {s0:.5f}   |观测-上一步状态| {p0:.5f}")
    print(f"  延迟 1: |观测-本步状态| {s1:.5f}   |观测-上一步状态| {p1:.5f}")
    ok = s0 < p0 and p1 < s1
    verdict = "生效" if ok else "no-op/需检查"
    return verdict, (f"延迟1时观测=上一步({p1:.5f} < {s1:.5f}); 延迟0时=本步({s0:.5f} < {p0:.5f})"
                     if ok else "两栏关系不对, 需检查")


SECTIONS = [
    ("① 脚底摩擦 geom_friction", check_foot_friction),
    ("② 编码器偏置 encoder_bias", check_encoder_bias),
    ("③ 质心偏移 body_ipos(躯干+头)", check_com),
    ("④ 质量+惯量 pseudo_inertia", check_mass_inertia),
    ("⑤ 转子惯量 joint_armature", check_armature),
    ("⑥ 观测噪声 Unoise", check_obs_noise),
    ("⑦ 观测延迟 delay_lag", check_obs_delay),
]


def main() -> None:
    results = []
    for title, fn in SECTIONS:
        print(f"\n=== {title} ===")
        try:
            verdict, detail = fn()
        except Exception as exc:
            verdict, detail = "报错", f"{type(exc).__name__}: {exc}"
            import traceback
            traceback.print_exc()
        results.append((title, verdict, detail))
    print("\n" + "=" * 78)
    print(f"{'项':<34}{'判决':<10}证据")
    print("-" * 78)
    for title, verdict, detail in results:
        print(f"{title:<34}{verdict:<10}{detail}")
    print("=" * 78)
    print("判决口径: 生效 = 字段确实按环境采样且被消费; no-op = 配置能跑但什么都没变。")


if __name__ == "__main__":
    main()
