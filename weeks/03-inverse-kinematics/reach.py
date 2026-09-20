"""Put the foot on a target: the whole IK pipeline, top to bottom, in one script.

    uv run weeks/03-inverse-kinematics/reach.py --target 0.10 0 0.05
    uv run weeks/03-inverse-kinematics/reach.py --leg right --target 0 -0.1 0 --target 0.1 -0.1 0.1
    uv run weeks/03-inverse-kinematics/reach.py --target 0 0 0.01 --seed straight --solver inverse --trace
    uv run weeks/03-inverse-kinematics/reach.py --target 0.3 0 0 --trace --jacobian --no-viewer

Each --target is an offset from where the foot starts (metres, world frame:
x forward, y the robot's left, z up). The script solves for the joint angles,
prints what it did, and then opens the simulator and moves the leg there so you
can see it. Several --target flags are solved one after another, each from the
previous answer, and the foot is animated from one to the next.

    --leg left|right        which leg (default left)
    --seed crouch|straight  starting pose: the week 4 crouch, or every leg joint at zero
    --solver dls|inverse    damped least squares (default) or the plain inverse
    --lam 0.01              damping for dls
    --iters 30              solver iterations per target
    --position-only         3 rows: place the foot, let it tilt
    --trace                 print every iteration, not just the summary
    --jacobian              print the Jacobian and its singular values at the start
    --no-viewer             solve and print only (what Colab and the autograder see)
    --hold 4                seconds to keep the window open after the last target

lab_ik.py is the same mathematics made interactive. This file is the version
you read: six numbered steps, each doing one thing, each calling MuJoCo where
MuJoCo is needed and numpy where it is not. Read it with the slides open.
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
# The two solvers. Each answers  J dq = err  for one small joint step.
# ---------------------------------------------------------------------------

def dls_step(J, err, lam):
    """Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 err  (slide: 'Damped least squares')."""
    m = J.shape[0]
    return J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(m), err)


def inverse_step(J, err, lam):
    """The plain inverse: dq = J^-1 err  (slide: 'The obvious answer'). lambda unused."""
    return np.linalg.lstsq(J, err, rcond=None)[0]


SOLVERS = {"dls": dls_step, "inverse": inverse_step}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", nargs=3, type=float, action="append", metavar=("DX", "DY", "DZ"),
                    required=True, help="offset from the foot's start, metres; repeatable")
    ap.add_argument("--leg", choices=("left", "right"), default="left")
    ap.add_argument("--seed", choices=("crouch", "straight"), default="crouch")
    ap.add_argument("--solver", choices=tuple(SOLVERS), default="dls")
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--position-only", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--jacobian", action="store_true")
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--hold", type=float, default=4.0)
    args = ap.parse_args(argv)
    solver = SOLVERS[args.solver]

    # -- 1. the robot, and the pose we start from ---------------------------------
    # load_g1() compiles the Unitree G1's MJCF into an MjModel (the robot's
    # constants: bodies, joints, limits). MjData is the state: joint angles in
    # `qpos`, and every world position MuJoCo derives from them.
    model = soc4180.load_g1()
    data = soc4180.WalkingController(model).initial_data()      # the week 4 crouch
    if args.seed == "straight":
        for side in ("left", "right"):
            data.qpos[kin.leg_qpos_indices(model, side)] = 0.0   # the `stand` singularity
    mujoco.mj_forward(model, data)     # forward kinematics: qpos -> every xpos, site_xpos, xmat

    # -- 2. which numbers belong to this leg --------------------------------------
    # The six leg joints are a contiguous slice of qpos (angles) and of qvel
    # (velocities). The two slices differ by one, because the floating pelvis
    # takes 7 numbers of qpos (position + quaternion) and 6 of qvel. The
    # Jacobian is a velocity object, so its columns follow the qvel indices.
    site = kin.foot_site_id(model, args.leg)              # the foot site we are placing
    qidx = kin.leg_qpos_indices(model, args.leg)          # angles      e.g. 7..12
    didx = kin.leg_dof_indices(model, args.leg)           # dof columns e.g. 6..11
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{args.leg}_{j}")
              for j in kin.LEG_JOINTS]
    lo, hi = model.jnt_range[joints, 0], model.jnt_range[joints, 1]   # joint limits, rad
    flat = data.site_xmat[site].reshape(3, 3).copy()      # the level foot: orientation target
    jac_p, jac_r = np.zeros((3, model.nv)), np.zeros((3, model.nv))   # mj_jacSite writes here

    hip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{args.leg}_hip_pitch_link")
    knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{args.leg}_knee_link")
    ankle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{args.leg}_ankle_pitch_link")
    hip_pos = data.xpos[hip].copy()
    # the farthest the foot SITE can get from the hip: thigh + shin + the
    # site's drop below the ankle joint (0.3409 + 0.3000 + 0.0176 = 0.6585 m)
    reach = (np.linalg.norm(data.xpos[knee] - data.xpos[hip])
             + np.linalg.norm(data.xpos[ankle] - data.xpos[knee])
             + np.linalg.norm(data.site_xpos[site] - data.xpos[ankle]))

    foot0 = data.site_xpos[site].copy()
    print(f"{args.leg} leg, seed {args.seed}, solver {args.solver}"
          f"{'' if args.solver != 'dls' else f' (lambda {args.lam:g})'}, "
          f"{'3 rows (position only)' if args.position_only else '6 rows (position + level foot)'}")
    print(f"foot starts at {foot0.round(4)}   reach {reach:.4f} m from the hip at {hip_pos.round(3)}")
    print(f"angles {np.round(data.qpos[qidx], 3).tolist()}   ({', '.join(JOINT_NAMES)})")

    def jacobian():
        """The 6 x 6 (or 3 x 6) site Jacobian: foot motion per unit joint motion."""
        mujoco.mj_jacSite(model, data, jac_p, jac_r, site)   # fills jac_p, jac_r: 3 x nv each
        J = np.vstack([jac_p, jac_r])[:, didx]               # keep the leg's six columns
        return J[:3] if args.position_only else J

    def error(target):
        """The foot motion that would put the foot on the target (and level it)."""
        e = kin.pose_error(data, site, target, flat)          # [dx dy dz, droll dpitch dyaw]
        return e[:3] if args.position_only else e

    if args.jacobian:
        J = jacobian()
        print("\nJacobian at the seed (rows: foot x y z" + ("" if args.position_only else " roll pitch yaw")
              + "; columns: " + ", ".join(JOINT_NAMES) + ")")
        print(np.array2string(J, precision=3, suppress_small=True))
        s = np.linalg.svd(J, compute_uv=False)
        print(f"singular values {np.round(s, 5)}   condition number {s[0] / max(s[-1], 1e-300):.1e}")

    # -- 3. solve each target, one small step at a time -----------------------------
    poses = [data.qpos.copy()]              # every solved pose, for the animation in step 6
    for k, off in enumerate(args.target, 1):
        target = foot0 + np.asarray(off, float)
        dist = np.linalg.norm(target - hip_pos)
        print(f"\n--- target {k}: offset {np.round(off, 3).tolist()} -> world {target.round(4)}"
              f"   hip->target {dist:.4f} m of reach {reach:.4f} m"
              f"{'   <- OUTSIDE the reach: expect a residual' if dist > reach else ''}")
        if args.trace:
            print(f"  {'it':>3}  {'residual m':>11}  {'|dq| rad':>9}   angles (rad)")
        t0 = time.perf_counter()
        for it in range(1, args.iters + 1):
            err = error(target)                     # 3a. how far the foot is from the target
            J = jacobian()                          # 3b. how the foot moves per joint, here
            dq = solver(J, err, args.lam)           # 3c. the joint step the solver proposes
            q = data.qpos[qidx] + dq
            data.qpos[qidx] = np.clip(q, lo, hi)    # 3d. apply it, inside the joint limits
            mujoco.mj_forward(model, data)          # 3e. recompute the world: the foot moved
            residual = np.linalg.norm(error(target)[:3])
            if args.trace:
                print(f"  {it:3d}  {residual:11.2e}  {np.linalg.norm(dq):9.2e}   "
                      f"{np.round(data.qpos[qidx], 3).tolist()}")
            if residual < 1e-6 and np.linalg.norm(dq) < 1e-6:
                break
        ms = (time.perf_counter() - t0) * 1e3

        # -- 4. report --------------------------------------------------------------
        foot = data.site_xpos[site]
        R = data.site_xmat[site].reshape(3, 3)
        tilt = np.degrees(np.arccos(np.clip((R.T @ flat)[2, 2], -1, 1)))
        print(f"  after {it} iterations ({ms:.1f} ms): foot {foot.round(4)}   residual {residual:.2e} m"
              f"   foot tilt {tilt:.1f} deg")
        print(f"  angles {np.round(data.qpos[qidx], 3).tolist()}")
        at_limit = [n for n, q_, l, h in zip(JOINT_NAMES, data.qpos[qidx], lo, hi)
                    if q_ <= l + 1e-9 or q_ >= h - 1e-9]
        if at_limit:
            print(f"  on a joint limit: {', '.join(at_limit)}")
        poses.append(data.qpos.copy())

    # -- 5. no window? we are done ----------------------------------------------------
    if args.no_viewer or soc4180.is_colab():
        return 0

    # -- 6. show it: animate from pose to pose in the simulator ------------------------
    # The passive viewer draws `data` whenever we call sync(). Physics is never
    # stepped: we write qpos and call mj_forward, so the robot is placed, not
    # simulated, and holds poses it could never balance in.
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    print("\nopening the simulator: the leg moves through the solved poses, then holds "
          f"for {args.hold:g} s (close the window to stop early)")
    with soc4180.launch_viewer(model, data, passive=True) as viewer:
        data.qpos[:] = poses[0]; mujoco.mj_forward(model, data); viewer.sync()
        time.sleep(0.8)
        for a, b in zip(poses, poses[1:]):
            for s in np.linspace(0, 1, 50):                       # one second per target
                if not viewer.is_running() or time.time() > deadline:
                    return 0
                data.qpos[:] = (1 - s) * a + s * b                # straight line in joint space
                mujoco.mj_forward(model, data)
                viewer.sync(); time.sleep(0.02)
            time.sleep(0.5)
        hold_until = min(time.time() + args.hold, deadline)
        while viewer.is_running() and time.time() < hold_until:
            viewer.sync(); time.sleep(0.05)
    return 0


if __name__ == "__main__":
    sys.exit(main())
