"""Forward kinematics by hand, one body at a time, checked against MuJoCo, then shown.

    uv run weeks/02-transforms/fk.py
    uv run weeks/02-transforms/fk.py --angles -0.35 0 0 0.70 -0.35 0
    uv run weeks/02-transforms/fk.py --angles -0.30 0.10 0.05 0.70 -0.35 0.02 --side right
    uv run weeks/02-transforms/fk.py --angles 0.3 0 0 0.2 -0.1 0 --no-viewer

The six angles are hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll
in radians. The script places the leg (mj_forward, no physics), then walks the
chain from the pelvis to the foot site by hand -- printing, for every body,
the offset it adds and where that leaves us in the world -- and compares the
result with MuJoCo's own `site_xpos`. It also evaluates the planar paper model
(sines and cosines, roll and yaw assumed zero) so the two errors sit side by
side. Then it opens the simulator with the pose and two spheres at the foot:
GREEN where MuJoCo puts the site, RED where the hand-made chain puts it.

    --angles a b c d e f    the six leg angles (default: the deck's crouch)
    --side left|right       which leg
    --no-viewer             print only
    --hold 6                seconds to keep the window open

Five numbered steps; read them with the week 2 slides open.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mujoco
import numpy as np

import soc4180
from soc4180 import kinematics as kin

JOINT_NAMES = ("hip pitch", "hip roll", "hip yaw", "knee", "ankle pitch", "ankle roll")


# ---------------------------------------------------------------------------
# The two conversions a chain needs: quaternion -> matrix, axis+angle -> matrix.
# MuJoCo stores orientations as (w, x, y, z) quaternions; a rotation matrix is
# the only form that acts on a vector, so we convert before we multiply.
# ---------------------------------------------------------------------------

def quat2mat(q):
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(q, float))
    return out.reshape(3, 3)


def axis_angle(axis, angle):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), float(angle))
    return quat2mat(q)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--angles", nargs=6, type=float, default=[-0.35, 0, 0, 0.70, -0.35, 0])
    ap.add_argument("--side", choices=("left", "right"), default="left")
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--hold", type=float, default=6.0)
    args = ap.parse_args(argv)
    angles = np.asarray(args.angles, float)

    # -- 1. place the leg: qpos in, world positions out --------------------------------
    # `stand` is the keyframe; the six leg angles overwrite their slice of qpos.
    # mj_forward IS MuJoCo's forward kinematics: it fills every `x` quantity
    # (xpos, xmat, site_xpos ...) from qpos, without stepping physics.
    model = soc4180.load_g1()
    data = soc4180.keyframe_data(model, "stand")
    qidx = kin.leg_qpos_indices(model, args.side)
    data.qpos[qidx] = angles
    mujoco.mj_forward(model, data)
    site = kin.foot_site_id(model, args.side)
    truth = data.site_xpos[site].copy()
    print(f"{args.side} leg, angles {angles.tolist()}   ({', '.join(JOINT_NAMES)})")
    print(f"MuJoCo's answer, site_xpos: {truth.round(6)}\n")

    # -- 2. the chain by hand: pelvis -> six bodies -> foot site ----------------------------
    # Each body carries a fixed offset from its parent (body_pos, body_quat: the
    # MJCF's numbers, no `x` prefix) and a rotation about its joint axis by the
    # joint's angle. Compose them in order. `pos`/`rot` are the current frame in
    # world coordinates; every offset is rotated by `rot` before it is added.
    pos = data.qpos[:3].copy()               # the pelvis: 3 numbers of position ...
    rot = quat2mat(data.qpos[3:7])           # ... and a quaternion for its orientation
    print(f"{'body':22} {'offset from parent (m)':>26} {'joint':12} {'angle':>7}   world position after")
    print(f"{'pelvis':22} {'':>26} {'':12} {'':>7}   {pos.round(4)}")
    for bid in kin.leg_chain(model, args.side):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        pos = pos + rot @ model.body_pos[bid]            # step to where this body sits
        rot = rot @ quat2mat(model.body_quat[bid])       # and apply any built-in tilt
        jnt = model.body_jntadr[bid]
        jname, ang = "", 0.0
        if jnt >= 0:                                      # then turn about the joint
            ang = float(data.qpos[model.jnt_qposadr[jnt]])
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt).replace(f"{args.side}_", "")
            a = model.jnt_pos[jnt]                        # axis anchor in the body (0 on the G1)
            Rj = axis_angle(model.jnt_axis[jnt], ang)
            pos = pos + rot @ (a - Rj @ a)                # rotate about the anchor, not the origin
            rot = rot @ Rj                                # carry the turn down the chain
        print(f"{name:22} {str(np.round(model.body_pos[bid], 4).tolist()):>26} {jname:12} {ang:7.3f}   {pos.round(4)}")
    mine = pos + rot @ model.site_pos[site]              # the site's own offset in the last body
    print(f"{'foot site':22} {str(np.round(model.site_pos[site], 4).tolist()):>26} {'':12} {'':>7}   {mine.round(4)}")
    err_chain = np.abs(mine - truth).max()
    print(f"\nchain by hand : {mine.round(6)}   max error vs MuJoCo {err_chain:.1e} m")

    # -- 3. the paper model: three angles, sines and cosines, roll and yaw ignored ----------
    # In-plane thigh L1 and shin L2, measured from the model at `stand`; the
    # hip -> knee vector splays 5.4 cm sideways, so the in-plane length is the
    # one the planar formula wants. The 2e-6 m residual at the zero pose is a
    # constant offset in the model, not noise.
    d0 = soc4180.keyframe_data(model, "stand"); mujoco.mj_forward(model, d0)
    body = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{args.side}_{n}_link")
    H, K, A = (d0.xpos[body(n)] for n in ("hip_pitch", "knee", "ankle_pitch"))
    L1, L2 = np.hypot(*(K - H)[[0, 2]]), np.hypot(*(A - K)[[0, 2]])
    drop = np.linalg.norm(d0.site_xpos[site] - A)
    th1, th2, th3 = angles[0], angles[3], angles[4]
    kx, kz = H[0] - L1 * np.sin(th1), H[2] - L1 * np.cos(th1)
    ax_, az = kx - L2 * np.sin(th1 + th2), kz - L2 * np.cos(th1 + th2)
    paper = np.array([ax_ - drop * np.sin(th1 + th2 + th3), truth[1], az - drop * np.cos(th1 + th2 + th3)])
    err_paper = np.abs(paper[[0, 2]] - truth[[0, 2]]).max()
    print(f"paper (x, z)  : {paper.round(6)}   max error vs MuJoCo {err_paper:.1e} m"
          + ("   (roll/yaw are not zero: the planar model cannot know)" if angles[1] or angles[2] or angles[5] else ""))

    # -- 4. the verdict --------------------------------------------------------------------
    print(f"\nL1 = {L1:.4f} m (in-plane thigh), L2 = {L2:.4f} m, ankle -> site {drop:.4f} m")
    print("chain error ~1e-16 means the composition is exactly MuJoCo's; paper error ~2e-6 with roll and yaw "
          "at zero is the planar model's constant offset; more than that and the plane assumption is broken.")

    if args.no_viewer or soc4180.is_colab():
        return 0

    # -- 5. show it: the pose, green at MuJoCo's site, red at the chain's answer --------------
    deadline = min(time.time() + args.hold, time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12))

    def sphere(geom, p, rgba, r):
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0], float),
                            np.asarray(p, float), np.eye(3).flatten(), np.asarray(rgba, float))

    with soc4180.launch_viewer(model, data, passive=True) as viewer:
        with viewer.lock():
            sphere(viewer.user_scn.geoms[0], truth, (0.1, 0.9, 0.2, 0.6), 0.03)
            sphere(viewer.user_scn.geoms[1], mine, (0.9, 0.1, 0.1, 0.9), 0.02)
            viewer.user_scn.ngeom = 2
        while viewer.is_running() and time.time() < deadline:
            viewer.sync(); time.sleep(0.05)
    return 0


if __name__ == "__main__":
    sys.exit(main())
