"""Week 2 lab, on your laptop: pose the leg, predict the foot, then look.

    uv run weeks/02-transforms/lab_viewer.py

A window opens with the G1 frozen in the first pose of POSES. Physics is never
stepped: this is kinematics, so the robot holds poses it could never balance in.

    SPACE or right arrow   next pose
    left arrow             previous pose
    F                      which forward kinematics draws the red sphere:
                           the full chain (default) <-> the planar paper model
    ENTER                  print the current angles, MuJoCo's foot, yours, the error
    Control sliders        (right panel, or press F3) drag any joint directly
    left-drag / right-drag / wheel   orbit / pan / zoom
    Keys go to the MuJoCo window, not the terminal -- click the window first.
    If no key does anything in the window, press the same key in this terminal
    instead -- single keys, no enter, arrows included. `q` stops it.

The Control sliders normally set actuator *targets*, which only move the robot
through physics. Nothing here is simulated, so this script wires each slider
straight onto its joint angle instead: drag left_knee_joint and the leg bends,
and the red sphere follows your forward kinematics live.

Two spheres are drawn at the left foot. The GREEN one is where MuJoCo puts the
foot site (`site_xpos`). The RED one is where *your* forward kinematics puts
it. Both versions of it are complete in section 1: `chain_fk` composes the six
bodies exactly as the slides derive it and agrees with MuJoCo to 1e-16;
`paper_fk` is the planar three-angle formula and agrees to 2e-6 while roll and
yaw are zero. `fk.py` in this folder prints the chain body by body.

Experiments, in order, and what to show the instructor:

1. Add three poses of your own to POSES. Predict on paper which way the foot
   moves before you press SPACE. Explain one of them from the printed chain
   (`uv run weeks/02-transforms/fk.py --angles ...`).
2. ENTER on the crouch: chain error 1.8e-16, the red sphere inside the green.
   Press F: the paper model, error 2.2e-06, still inside. Say what the 2e-6 is
   (a constant offset in the model at the zero pose, not rounding).
3. Go to the third pose, roll and yaw on. F to the paper model: the red sphere
   leaves the foot by 1.2 mm. Say why a planar model cannot know.
4. Break the chain on purpose: in `chain_fk` swap the two lines that add the
   body offset and apply the body rotation, or drop `(a - Rj @ a)`. The first
   moves the red sphere; the second changes nothing on the G1 (every joint
   sits at its body origin, so a = 0) -- say why that term is still right.
5. Find a pose that puts the foot 10 cm forward of the stand pose and still
   flat on the floor. There is more than one answer; explain which joints you
   used. (Pitch angles that sum to zero keep the foot level.)
"""

from __future__ import annotations

import functools
import os
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


# --- 1. forward kinematics, two ways -------------------------------------------------

def quat2mat(q):
    """MuJoCo's (w, x, y, z) quaternion -> the 3 x 3 matrix that acts on vectors."""
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(q, float))
    return out.reshape(3, 3)


def axis_angle(axis, angle):
    """Rotation by `angle` about `axis`, as a matrix (Rodrigues, via MuJoCo)."""
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), float(angle))
    return quat2mat(q)


def chain_fk(model, qpos, side=SIDE):
    """The full chain: pelvis -> six leg bodies -> foot site. Exact (1e-16).

    Each body carries a fixed offset from its parent (`body_pos`, `body_quat`:
    the MJCF's own numbers) followed by a rotation about its joint axis by the
    joint's angle. `pos`/`rot` are the current frame in world coordinates;
    every offset is rotated by `rot` before it is added -- that ordering IS
    forward kinematics, and swapping it is the classic mistake.
    """
    pos = np.asarray(qpos[:3], float).copy()          # the floating pelvis: position ...
    rot = quat2mat(qpos[3:7])                          # ... and orientation
    for bid in kin.leg_chain(model, side):
        pos = pos + rot @ model.body_pos[bid]          # step to the next body's origin
        rot = rot @ quat2mat(model.body_quat[bid])     # apply any built-in tilt
        jnt = model.body_jntadr[bid]
        if jnt >= 0:
            a = model.jnt_pos[jnt]                     # the axis anchor in the body (0 on the G1)
            Rj = axis_angle(model.jnt_axis[jnt], qpos[model.jnt_qposadr[jnt]])
            pos = pos + rot @ (a - Rj @ a)             # rotate about the anchor, not the origin
            rot = rot @ Rj                             # carry the turn down the chain
    s = kin.foot_site_id(model, side)
    return pos + rot @ model.site_pos[s]               # out to the site inside the last body


@functools.lru_cache(maxsize=None)
def _planar_constants(model_id, side):
    """H, L1, L2, foot_drop and the foot's y, measured from the model at `stand`."""
    model = _planar_constants.models[model_id]
    d = soc4180.keyframe_data(model, "stand"); mujoco.mj_forward(model, d)
    body = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_{n}_link")
    H, K, A = (d.xpos[body(n)].copy() for n in ("hip_pitch", "knee", "ankle_pitch"))
    site = d.site_xpos[kin.foot_site_id(model, side)]
    L1, L2 = np.hypot(*(K - H)[[0, 2]]), np.hypot(*(A - K)[[0, 2]])   # in-plane lengths
    return H, L1, L2, float(np.linalg.norm(site - A)), float(site[1])


_planar_constants.models = {}


def paper_fk(model, qpos, side=SIDE):
    """The planar model: hip pitch, knee, ankle pitch, and nothing but sin and cos.

    Roll and yaw are assumed zero, so y is the foot's y at `stand`. Thigh L1 is
    the IN-PLANE length (0.3366 m; the hip -> knee vector splays 5.4 cm
    sideways), shin L2 = 0.3000 m, and the site hangs 0.0176 m below the ankle.
    Agrees with MuJoCo to 2e-6 m when roll and yaw are zero; the residual is a
    constant offset in the model (at the zero pose MuJoCo's foot is at
    x = -2.3e-6, not exactly under the hip), not rounding.
    """
    _planar_constants.models[id(model)] = model
    H, L1, L2, drop, y = _planar_constants(id(model), side)
    q = qpos[kin.leg_qpos_indices(model, side)]
    th1, th2, th3 = q[0], q[3], q[4]                   # hip pitch, knee, ankle pitch
    kx, kz = H[0] - L1 * np.sin(th1), H[2] - L1 * np.cos(th1)
    ax, az = kx - L2 * np.sin(th1 + th2), kz - L2 * np.cos(th1 + th2)
    return np.array([ax - drop * np.sin(th1 + th2 + th3), y, az - drop * np.cos(th1 + th2 + th3)])


METHODS = [("chain", chain_fk), ("paper", paper_fk)]


# --- 2. helpers: place a pose, report, draw --------------------------------------------

def apply_pose(model, data, angles):
    data.qpos[:] = model.key_qpos[0]                      # back to `stand`
    data.qpos[kin.leg_qpos_indices(model, SIDE)] = angles
    mujoco.mj_forward(model, data)                        # place, do not simulate
    return data.site_xpos[kin.foot_site_id(model, SIDE)].copy()


def report(model, data, name, angles, truth, method):
    mname, fk = METHODS[method]
    mine = np.asarray(fk(model, data.qpos), float).reshape(3)
    print(f"\n[{name}]  angles = {np.round(angles, 3).tolist()}   fk = {mname}")
    print(f"  mujoco site_xpos : {truth.round(6)}")
    print(f"  {mname + '_fk':17}: {mine.round(6)}")
    print(f"  max error        : {np.abs(mine - truth).max():.1e}")
    return mine


def sphere(geom, pos, rgba, radius):
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0, 0], dtype=float),
                        np.asarray(pos, dtype=float),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=float))


# --- 3. the loop ----------------------------------------------------------------------------

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

    state = {"i": 0, "pose": True, "print": False, "method": 0}

    def on_key(keycode):
        if keycode in (32, 262):            # SPACE, right arrow
            state["i"] = (state["i"] + 1) % len(POSES)
            state["pose"] = True
        elif keycode == 263:                # left arrow
            state["i"] = (state["i"] - 1) % len(POSES)
            state["pose"] = True
        elif keycode == 70:                 # 'F'
            state["method"] = (state["method"] + 1) % len(METHODS)
            print(f"\n[red sphere: {METHODS[state['method']][0]}_fk]")
            state["print"] = True
        elif keycode in (257, 335):         # ENTER, keypad ENTER
            state["print"] = True

    def markers(viewer, truth, mine):
        with viewer.lock():
            scn = viewer.user_scn
            sphere(scn.geoms[0], truth, (0.1, 0.9, 0.2, 0.6), 0.03)
            sphere(scn.geoms[1], mine, (0.9, 0.1, 0.1, 0.9), 0.02)
            scn.ngeom = 2

    print("\n\n".join(__doc__.split("\n\n")[2:4]))      # the window + key help
    last = None
    # Second route for every command: type it in this terminal. The viewer's own
    # keys go through GLFW, which an IME or a stray keyboard focus can swallow.
    soc4180.terminal_keys(on_key)
    print("  [terminal] keys dead in the window? press them here instead "
          "-- single keys, no enter, arrows included; 'q' stops.", flush=True)
    # SOC4180_AUTOCLOSE=<seconds> closes the window by itself: the smoke test.
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    with soc4180.launch_viewer(model, data, passive=True, key_callback=on_key) as viewer:
        while viewer.is_running() and time.time() < deadline:
            if state["pose"]:
                state["pose"] = state["print"] = False
                name, angles = POSES[state["i"]]
                truth = apply_pose(model, data, angles)
                data.ctrl[:] = data.qpos[slider_qpos]     # sliders show the pose
                markers(viewer, truth, report(model, data, name, angles, truth, state["method"]))
                last = data.qpos[leg].copy()
            else:
                data.qpos[slider_qpos] = data.ctrl         # sliders ARE the angles
                mujoco.mj_forward(model, data)
                angles = data.qpos[leg].copy()
                if state["print"] or not np.array_equal(angles, last):
                    truth = data.site_xpos[site].copy()
                    if state["print"]:
                        state["print"] = False
                        mine = report(model, data, "sliders", angles, truth, state["method"])
                    else:
                        mine = np.asarray(METHODS[state["method"]][1](model, data.qpos), float)
                    markers(viewer, truth, mine)
                    last = angles
            viewer.sync()
            time.sleep(0.02)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
