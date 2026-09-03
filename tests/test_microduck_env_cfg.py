import mujoco

from mjlab_microduck.tasks import get_microduck_entity_cfg, _STAND_JOINT_POS

JOINT = mujoco.mjtObj.mjOBJ_JOINT

def _model():
    return get_microduck_entity_cfg().spec_fn().compile()

def test_kinematic_counts():
    model = _model()
    assert model.nq == 21
    assert model.nv == 20
    assert model.nu == 14

def test_all_servo_joints_are_actuated():
    """14 个执行器恰好覆盖 14 个舵机关节（正则没误伤 free 根）。"""
    m = _model()
    actuated = {
        mujoco.mj_id2name(m, JOINT, int(m.actuator_trnid[i, 0]))
        for i in range(m.nu)
    }
    assert actuated == set(_STAND_JOINT_POS.keys())

def test_stand_pose_within_joint_limits():
    """STAND 站姿的每个关节角都在 XML 限位内（单位弧度）。"""
    m = _model()
    for name, val in _STAND_JOINT_POS.items():
        jid = mujoco.mj_name2id(m, JOINT, name)
        lo, hi = m.jnt_range[jid]
        assert lo - 1e-6 <= val <= hi + 1e-6, f"{name}={val} 超出 [{lo}, {hi}]"

def test_xml_file_resolves():
    """spec_fn 能编译说明 XML 路径 + mesh 目录都对。"""
    assert _model() is not None