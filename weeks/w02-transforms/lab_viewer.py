"""Week 2 lab, on your laptop: pose the leg, predict the foot, then look.

    uv run weeks/w02-transforms/lab_viewer.py

A window opens with the G1 frozen in the first pose of POSES. Physics is never
stepped: this is kinematics, so the robot holds poses it could never balance in.

    SPACE or right arrow   next pose
    left arrow             previous pose
    ENTER                  print the current angles and foot position
    Control sliders        (right panel, or press F3) drag any joint directly
    left-drag / right-drag / wheel   orbit / pan / zoom

The Control sliders normally set actuator *targets*, which only move the robot
through physics. Nothing here is simulated, so this script wires each slider
straight onto its joint angle instead: drag left_knee_joint and the leg bends, and the
red sphere follows your forward kinematics live.

Two spheres are drawn at the left foot. The green one is where MuJoCo puts the
foot site (`site_xpos`). The red one is where *your* forward kinematics puts it,
from `predict_foot` below. Until you write it there is no red sphere.

What to change, in order, and show the instructor:

1. Add three poses of your own to POSES. Predict on paper which way the foot
   moves before you press SPACE.
2. Paste your `paper_fk` from the notebook into `predict_foot`. The red sphere
   should sit inside the green one for every pose whose roll and yaw are zero,
   and the printed error should be about 2e-06.
3. Turn roll or yaw on. Watch the red sphere leave the foot. Say why.
4. Replace `paper_fk` with your full `my_fk` chain. The error drops to 1e-16 and
   the red sphere stays inside the green one for every pose.
5. Find a pose that puts the foot 10 cm forward of the stand pose and still flat
   on the floor. There is more than one answer; explain which joints you used.
"""

from __future__ import annotations

import time

import mujoco
import numpy as np

import soc4180
from soc4180 import kinematics as kin

# name, then [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll] in radians
POSES = [
    ("stand: all six zero",             [0.00, 0.00, 0.00, 0.00,  0.00, 0.00]),
    ("the crouch from the deck",        [-0.35, 0.00, 0.00, 0.70, -0.35, 0.00]),
    ("the validated pose, roll+yaw on", [-0.30, 0.10, 0.05, 0.70, -0.35, 0.02]),
    # add your own below
]

SIDE = "left"


def predict_foot(model: mujoco.MjModel, angles) -> np.ndarray | None:
    """Your forward kinematics. Return the foot site's world position, or None.

    Start with the planar `paper_fk(th1, th2, th3)` from the notebook: it needs
    H (the hip anchor at stand), L1 = 0.3366, L2 = 0.3000 and foot_drop = 0.0176,
    all of which the notebook measures from the model. It returns (x, z); with
    roll and yaw at zero the y never moves, so use the foot site's y at stand
    (0.1185, the first line the script prints). Then replace it with your
    `my_fk` over the body tree, which handles roll and yaw too.
    """
    return None


# --- nothing below needs editing ------------------------------------------

def apply_pose(model, data, angles):
    data.qpos[:] = model.key_qpos[0]                      # back to `stand`
    data.qpos[kin.leg_qpos_indices(model, SIDE)] = angles
    mujoco.mj_forward(model, data)                        # place, do not simulate
    return data.site_xpos[kin.foot_site_id(model, SIDE)].copy()


def report(model, name, angles, truth):
    print(f"\n[{name}]  angles = {np.round(angles, 3).tolist()}")
    print(f"  mujoco site_xpos : {truth.round(6)}")
    mine = predict_foot(model, angles)
    if mine is None:
        print("  predict_foot     : (not written yet)")
        return None
    mine = np.asarray(mine, dtype=float).reshape(3)
    print(f"  predict_foot     : {mine.round(6)}")
    print(f"  max error        : {np.abs(mine - truth).max():.1e}")
    return mine


def sphere(geom, pos, rgba, radius):
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0, 0], dtype=float),
                        np.asarray(pos, dtype=float),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=float))


def main() -> int:
    if soc4180.is_colab():
        print("This lab needs a desktop window; run it on your laptop.")
        return 1

    model = soc4180.load_g1()
    data = soc4180.keyframe_data(model, "stand")

    # every G1 actuator drives exactly one hinge; this is that hinge's qpos slot
    slider_qpos = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    leg = kin.leg_qpos_indices(model, SIDE)
    site = kin.foot_site_id(model, SIDE)

    state = {"i": 0, "pose": True, "print": False}

    def on_key(keycode):
        if keycode in (32, 262):            # SPACE, right arrow
            state["i"] = (state["i"] + 1) % len(POSES)
            state["pose"] = True
        elif keycode == 263:                # left arrow
            state["i"] = (state["i"] - 1) % len(POSES)
            state["pose"] = True
        elif keycode in (257, 335):         # ENTER, keypad ENTER
            state["print"] = True

    def markers(viewer, truth, mine):
        with viewer.lock():
            scn = viewer.user_scn
            sphere(scn.geoms[0], truth, (0.1, 0.9, 0.2, 0.6), 0.03)
            scn.ngeom = 1
            if mine is not None:
                sphere(scn.geoms[1], mine, (0.9, 0.1, 0.1, 0.9), 0.02)
                scn.ngeom = 2

    print(__doc__.split("\n\n")[1])
    last = None
    with soc4180.launch_viewer(model, data, passive=True, key_callback=on_key) as viewer:
        while viewer.is_running():
            if state["pose"]:
                state["pose"] = False
                name, angles = POSES[state["i"]]
                truth = apply_pose(model, data, angles)
                data.ctrl[:] = data.qpos[slider_qpos]     # sliders show the pose
                markers(viewer, truth, report(model, name, angles, truth))
                last = data.qpos[leg].copy()
            else:
                data.qpos[slider_qpos] = data.ctrl         # sliders ARE the angles
                mujoco.mj_forward(model, data)
                angles = data.qpos[leg].copy()
                if state["print"] or not np.array_equal(angles, last):
                    truth = data.site_xpos[site].copy()
                    mine = predict_foot(model, angles)
                    if state["print"]:
                        state["print"] = False
                        report(model, "sliders", angles, truth)
                    markers(viewer, truth, None if mine is None
                            else np.asarray(mine, dtype=float).reshape(3))
                    last = angles
            viewer.sync()
            time.sleep(0.02)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
