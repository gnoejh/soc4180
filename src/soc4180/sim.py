"""Simulation control utilities.

The distinction these helpers make explicit: a MuJoCo robot with ``ctrl = 0`` is
**not** an uncontrolled robot. The G1's actuators are position servos, so
``ctrl = 0`` commands every joint to angle zero and the servos hold that pose.
To see the robot without a controller you must disable actuation entirely.
"""

from __future__ import annotations

import contextlib

import mujoco
import numpy as np

__all__ = ["actuation_disabled", "hold", "keyframe_data", "keyframe_names",
           "launch_viewer"]

_ACTUATION = int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)


@contextlib.contextmanager
def actuation_disabled(model: mujoco.MjModel):
    """Temporarily make every actuator limp — a rag-doll robot.

    This is what "no controller" actually means. Without it, ``ctrl = 0`` is
    still a command, and the robot will hold a pose.
    """
    previous = int(model.opt.disableflags)
    model.opt.disableflags = previous | _ACTUATION
    try:
        yield model
    finally:
        model.opt.disableflags = previous


def keyframe_names(model: mujoco.MjModel) -> list[str]:
    """Named poses stored in the model (the G1 ships one, ``stand``)."""
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
        for i in range(model.nkey)
    ]


def keyframe_data(model: mujoco.MjModel, key: int | str = 0) -> mujoco.MjData:
    """Fresh ``MjData`` initialised to a named keyframe pose."""
    if isinstance(key, str):
        names = keyframe_names(model)
        if key not in names:
            raise ValueError(f"No keyframe '{key}'. Available: {names}")
        key = names.index(key)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, key)
    return data


def hold(model: mujoco.MjModel, key: int | str = 0):
    """A controller that holds a keyframe pose.

    Returns a ``ctrl_fn`` for :func:`soc4180.render_rollout`. This is the
    simplest possible Layer 4: a fixed setpoint handed to the joint servos.
    """
    if isinstance(key, str):
        key = keyframe_names(model).index(key)
    target = np.array(model.key_ctrl[key], copy=True)

    def ctrl_fn(model, data):
        data.ctrl[:] = target

    return ctrl_fn


def launch_viewer(model=None, data=None, *, passive: bool = False):
    """Open MuJoCo's interactive 3-D viewer. **Desktop only.**

    Orbit with the left mouse, pan with the right, zoom with the wheel; double
    click a body to select it, then ctrl-drag to push the robot around. Far more
    useful than a rendered video when you are trying to understand why a
    controller misbehaves.

    ``passive=True`` returns a handle instead of blocking, so you can step the
    simulation yourself and watch it live.

    There is no equivalent in Colab: a notebook has no window to draw into, which
    is why every lab renders video instead.
    """
    from ._gl import is_colab

    if is_colab():
        raise RuntimeError(
            "The interactive viewer needs a desktop window and cannot run in "
            "Colab. Render video with soc4180.render_rollout instead, or run "
            "this locally."
        )

    import mujoco.viewer

    if model is None:
        from .models import load_g1

        model = load_g1()
    if data is None:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

    if passive:
        return mujoco.viewer.launch_passive(model, data)
    return mujoco.viewer.launch(model, data)
