from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg

_REPO = Path(__file__).resolve().parents[1]
_DUCK_DIR = _REPO / "assets" / "microduck"
_DUCK_XML = _DUCK_DIR / "robot_walk.xml"

_STAND_JOINT_POS = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": -0.0873,
    "left_hip_pitch": -0.4579,
    "left_knee": -0.0049,
    "left_ankle": 0.453,
    "neck_pitch": 0.3491,
    "head_pitch": 0.3491,
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0873,
    "right_hip_pitch": 0.4579,
    "right_knee": 0.0049,
    "right_ankle": -0.453,
}


def get_microduck_entity_cfg() -> EntityCfg:
    """
    把robot_walk.xml 接成mjlab认识的实体。
    """
    return EntityCfg(
        # 模型来源：读取 XML。传函数(lambda)而非直接结果，mjlab 每个 env 各建一份
        spec_fn=lambda: mujoco.MjSpec.from_file(str(_DUCK_XML)),
        # 关键：让 XML 里已有的 14 个位置执行器生效。
        # 正则是"关节名筛选"：排除 free 根关节(它没有执行器)，其余 14 个全部包进来
        articulation=EntityArticulationInfoCfg(
            actuators=(
                XmlActuatorCfg(target_names_expr=(r"(?!.*freejoint).*",)),
            ),
        ),
        # 初始位形 = STAND 站姿：根位置(0,0,0.12) + 关节角
        # 关节速度默认全 0，不用写
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.12),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=_STAND_JOINT_POS,
        ),
    )
