#!/usr/bin/env python3
"""打开 MuJoCo 查看器，看 MicroDuck 鸭子模型。

为什么需要这个文件？
    - robot_walk.xml 是"纯机器人"(没地面/没姿势)；scene_walk.xml = 机器人 + 地面 + 4 个姿势快照。
    - 本脚本 = 一条命令把"鸭子以某姿势"展示在可交互的 3D 窗口里。
    - 它是你 scripts/ 工具箱的第 1 个工具：之后看训练回放、看环境效果都会沿用这种"一文件一命令"的套路。

用法（在仓库根目录运行）:
    uv run mjpython scripts/view_microduck.py          # 默认 STAND(站立)
    uv run mjpython scripts/view_microduck.py SIT      # 姿势: INIT / STAND / SIT / FOLD

常见问题:
    1) macOS 必须用 mjpython 启动(不能用普通 python)，否则报
       "launch_passive requires ... run under mjpython" —— mujoco 官方限制。
    2) 窗口一闪而过 ≠ 出错。真实原因(macOS)：launch_passive 不阻塞，
       脚本跑完 Python 退出，窗口跟着没。本脚本已用"等待窗口关闭"解决(见第 6 步)。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 第 1 步：决定渲染后端。必须在 import mujoco 之前完成！
# ---------------------------------------------------------------------------
# MuJoCo 在不同平台用不同 OpenGL 后端:
#   macOS → cgl  (系统自带的 OpenGL)
#   Linux → egl  (无头/服务器渲染常用) 或 glfw(有桌面时)
# 后端名字是 mujoco 在"第一次渲染"时读取的环境变量，所以要先设、再 import。
if sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")
elif sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402  ← import 顺序是刻意的(见上面注释)
from mujoco import viewer  # noqa: E402

# ---------------------------------------------------------------------------
# 第 2 步：定位文件路径
# ---------------------------------------------------------------------------
# 本脚本在 仓库根/scripts/ 下，所以"上一级"就是仓库根。
# 用 Path(__file__) 而不是写死路径 → 仓库挪位置也不用改代码。
_REPO = Path(__file__).resolve().parent.parent
_SCENE_XML = _REPO / "assets" / "microduck" / "scene_walk.xml"


def main() -> None:
    # ---- 第 3 步：读命令行参数(可选)，默认 STAND ----
    keyframe = sys.argv[1] if len(sys.argv) > 1 else "STAND"
    print(f"[view_microduck] 姿势 = {keyframe}")

    # ---- 第 4 步：把 XML 编译成 MuJoCo 模型(model)和数据区(data) ----
    # model = 机器人的"静态定义"(几何/关节/执行器/材质)
    # data  = 机器人的"当前状态"(每个 qpos/qvel 的实时值)
    model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
    data = mujoco.MjData(model)

    # ---- 第 5 步：把机器人摆成指定姿势 ----
    # keyframe = XML 里保存的"姿势快照"(21 个 qpos + 可选 ctrl)
    # mj_name2id: 按名字查 id(查不到返回 -1)；mj_resetDataKeyframe: 重置成该姿势
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    if key_id < 0:
        raise SystemExit(f"没有这个姿势: {keyframe} (可用: INIT/STAND/SIT/FOLD)")
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    # ---- 第 6 步：打开窗口并"保持进程存活"，直到用户关窗口 ----
    # 关键坑(macOS)：launch_passive 并不阻塞到窗口关闭！
    #   看 mujoco/viewer.py 源码：macOS 下它把查看器放到 UI 线程去跑，
    #   然后立刻返回(它等的是"加载完成"信号)。如果脚本到此结束，Python
    #   退出 → 窗口跟着没 → 就是之前看到的"一闪而过"。
    # 修法：拿住返回的 Handle，主线程轮询"窗口是否还活着"，活着就等。
    print("[view_microduck] 窗口已打开。拖拽=转视角，滚轮=缩放；关窗口(Esc)即退出")
    handle = viewer.launch_passive(model, data)
    # Handle 内部用弱引用(weakref)指向查看器；窗口关闭、查看器被销毁后，
    # handle._sim() 会变成 None → 轮询到 None 就结束。
    # 这比 time.sleep(固定秒数) 干净：用户随时关，脚本就随时退。
    try:
        while handle._sim() is not None:  # noqa: SLF001 —— 官方没暴露 is_running
            time.sleep(0.2)
    except ReferenceError:
        pass  # 查看器已被回收，等价于窗口已关闭
    print("[view_microduck] 窗口已关闭，结束。")


if __name__ == "__main__":
    main()
