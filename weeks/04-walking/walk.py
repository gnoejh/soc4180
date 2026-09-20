"""Walk the G1 with the analytic controller: plan, LIPM, IK, physics -- printed, then replayed.

    uv run weeks/04-walking/walk.py
    uv run weeks/04-walking/walk.py --step-time 0.45
    uv run weeks/04-walking/walk.py --double-support 0 --no-viewer
    uv run weeks/04-walking/walk.py --gravity 1.62
    uv run weeks/04-walking/walk.py --friction 0.2 --steps 6 --step-length 0.20

Every gait parameter is a flag (defaults are the tuned gait). The script
prints the footstep plan and the pendulum's numbers, checks the LIPM closed
form written by hand against the package's, then runs the walk headless with
one printed row per step -- where the robot is, how high the pelvis is, how
far the measured ZMP swung sideways, how well IK tracked -- and a summary.
Then it opens the simulator and replays the walk with the footstep plan (grey
boxes), the pelvis (red) and the measured ZMP (yellow) drawn on the floor.

    --steps 12  --step-length 0.14  --step-time 0.65  --step-height 0.10
    --double-support 0.20  --com-height 0.60  --pelvis-height 0.72
    --gravity 9.81 (1.62: the Moon, the controller is not told)
    --friction 1.0 (0.2: ice)   --no-viewer   --speed 1

Six numbered steps; read them with the week 4 slides open.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import mujoco
import numpy as np

import soc4180
from soc4180 import GaitParams, WalkingController


def predict_com(x0, v0, zmp, t, omega):
    """The linear inverted pendulum, solved by hand.

    x'' = omega^2 (x - zmp) with omega = sqrt(g / z_c). Released at x0 with
    velocity v0 and the ZMP held at `zmp`, the centre of mass follows a
    cosh/sinh pair: it falls away from the ZMP exponentially, which is why a
    walker must keep moving the ZMP (its feet) underneath it.
    """
    return zmp + (x0 - zmp) * math.cosh(omega * t) + (v0 / omega) * math.sinh(omega * t)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--step-length", type=float, default=0.14)
    ap.add_argument("--step-time", type=float, default=0.65)
    ap.add_argument("--step-height", type=float, default=0.10)
    ap.add_argument("--double-support", type=float, default=0.20)
    ap.add_argument("--com-height", type=float, default=0.60)
    ap.add_argument("--pelvis-height", type=float, default=0.72)
    ap.add_argument("--gravity", type=float, default=9.81)
    ap.add_argument("--friction", type=float, default=1.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(argv)

    # -- 1. the robot, the world, and the gait ----------------------------------------
    model = soc4180.load_g1()
    model.opt.gravity[:] = [0, 0, -args.gravity]           # the controller is NOT told
    model.geom_friction[0, 0] = args.friction              # geom 0 is the floor
    params = GaitParams(n_steps=args.steps, step_length=args.step_length,
                        step_time=args.step_time, step_height=args.step_height,
                        double_support=args.double_support, com_height=args.com_height,
                        pelvis_height=args.pelvis_height)
    # WalkingController holds the three pieces of week 4: a footstep plan, a
    # LIPM trajectory for the pelvis that starts each step where the last one
    # ended, and week 3's IK turning foot + pelvis targets into 29 servo angles.
    controller = WalkingController(model, params)
    data = controller.initial_data()                        # the crouch, feet planted
    w = controller.lipm.omega
    print(f"gait: {params.n_steps} steps of {params.step_length} m every {params.step_time} s, "
          f"double support {params.double_support:.0%}, pelvis {params.pelvis_height} m")
    print(f"pendulum: z_c = {params.com_height} m, omega = sqrt(g/z_c) = {w:.2f} rad/s, "
          f"1/omega = {1 / w:.3f} s (the step is {params.step_time / (1 / w):.1f} time constants long)")
    print("footstep plan (support foot, and where the swing foot lands):")
    for k, (support, _, _, land) in enumerate(controller.plan):
        print(f"   step {k + 1:2d}: stand on {support:5s}, put the other foot at x={land[0]:+.3f} y={land[1]:+.3f}")

    # -- 2. the LIPM closed form, by hand, against the package ---------------------------
    x0, vx, y0, vy, zx, zy = controller._segments[0]       # the first step's boundary values
    dev = max(abs(predict_com(x0, vx, zx, tau, w) - controller.lipm.evolve(x0, vx, zx, tau)[0])
              for tau in np.linspace(0, params.step_time, 25))
    print(f"\nLIPM by hand vs soc4180.LIPM over the first step: max difference {dev:.1e} m")

    # -- 3. the walk: controller -> ctrl -> mj_step, one row per step ---------------------
    dt = model.opt.timestep
    frames, zmps = [], []
    zmp_y_all, step_zmp, fell_at = [], [], None
    settle = params.settle_time
    print(f"\n{'step':>4} {'t (s)':>6} {'x (m)':>7} {'pelvis z':>9} {'ZMP y range (m)':>18} {'IK err (m)':>11}")
    k_prev = -1
    n = 0
    while data.time < controller.total_time - 1e-9:
        data.ctrl[:] = controller.control(data.time)       # plan -> LIPM -> IK -> 29 targets
        mujoco.mj_step(model, data)
        n += 1
        zx_, zy_ = controller.zmp(data)                     # centre of pressure under the feet
        if n % 10 == 0:                                     # 50 Hz: a pose and a ZMP for the replay
            frames.append(data.qpos.copy()); zmps.append((zx_, zy_))
        if data.time > settle and not math.isnan(zy_):
            step_zmp.append(zy_); zmp_y_all.append(zy_)
        k = int((data.time - settle) / params.step_time) if data.time > settle else -1
        if k != k_prev and k_prev >= 0 and step_zmp:
            print(f"{k_prev + 1:4d} {data.time:6.2f} {data.qpos[0]:+7.3f} {data.qpos[2]:9.3f} "
                  f"[{min(step_zmp):+.3f}, {max(step_zmp):+.3f}] "
                  f"{float(controller.last_ik_error):11.1e}")
            step_zmp = []
        k_prev = k
        if data.qpos[2] < 0.5 and fell_at is None:
            fell_at = data.time
            print(f"     FELL at t = {fell_at:.2f} s after {data.qpos[0]:+.3f} m")
        if fell_at is not None and data.time > fell_at + 1.0:
            break

    # -- 4. summary --------------------------------------------------------------------------
    print(f"\nwalked {data.qpos[0]:+.3f} m in {data.time:.2f} s; pelvis ends at {data.qpos[2]:.3f} m; "
          f"{'FELL' if fell_at else 'upright'}")
    if zmp_y_all:
        print(f"measured ZMP y swung over [{min(zmp_y_all):+.3f}, {max(zmp_y_all):+.3f}] m "
              f"against stance feet at +-{params.stance_width} m: "
              f"{'OUTSIDE the support polygon -- the model, not the criterion, is what fails' if max(abs(v) for v in zmp_y_all) > params.stance_width + 0.06 else 'inside'}")

    # -- 5. no window? done -------------------------------------------------------------------
    if args.no_viewer or soc4180.is_colab():
        return 0

    # -- 6. replay with the plan, the pelvis and the ZMP drawn on the floor -----------------------
    def sphere(geom, pos, rgba, r):
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0], float),
                            np.asarray(pos, float), np.eye(3).flatten(), np.asarray(rgba, float))

    def box(geom, pos, half, rgba):
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX, np.asarray(half, float),
                            np.asarray(pos, float), np.eye(3).flatten(), np.asarray(rgba, float))

    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    print(f"\nreplaying {len(frames)} poses at {args.speed:g}x (close the window to stop)")
    with soc4180.launch_viewer(model, data, passive=True) as viewer:
        while viewer.is_running() and time.time() < deadline:
            for q, (zx_, zy_) in zip(frames, zmps):
                if not viewer.is_running() or time.time() > deadline:
                    break
                data.qpos[:] = q
                mujoco.mj_forward(model, data)
                with viewer.lock():
                    scn = viewer.user_scn; m = 0
                    for _, _, _, land in controller.plan:
                        box(scn.geoms[m], [*land, 0.002], [0.085, 0.03, 0.002], (0.5, 0.5, 0.5, 0.35)); m += 1
                    sphere(scn.geoms[m], [q[0], q[1], 0.01], (0.9, 0.1, 0.1, 0.9), 0.02); m += 1
                    if not math.isnan(zx_):
                        sphere(scn.geoms[m], [zx_, zy_, 0.01], (1.0, 0.85, 0.1, 0.9), 0.02); m += 1
                    scn.ngeom = m
                viewer.sync()
                time.sleep(0.02 / args.speed)
            time.sleep(0.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
