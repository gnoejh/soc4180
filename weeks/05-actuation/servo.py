"""One servo, measured: a step response you can read, then the walk under new gains.

    uv run weeks/05-actuation/servo.py                          # left knee, step 0.5 rad, gravity off
    uv run weeks/05-actuation/servo.py --kp-scale 4
    uv run weeks/05-actuation/servo.py --kv-scale 0.25
    uv run weeks/05-actuation/servo.py --joint left_hip_pitch --step 0.3 --gravity 9.81
    uv run weeks/05-actuation/servo.py --walk --limit 50
    uv run weeks/05-actuation/servo.py --walk --kp-scale 0.5 --no-viewer

MuJoCo's position actuator is a PD law written into the model file:

    tau = kp (ctrl - q) - kv qdot

Step-response mode (the default) holds `stand`, commands one joint to jump
by --step, and records what happens: the law evaluated by hand against
`actuator_force`, the damping ratio from the mass matrix, the time to reach
1 % of the target, the overshoot, the peak torque. Then the simulator replays
the motion. --walk runs the week 4 gait instead, with the gains scaled and an
optional torque limit, printing the pelvis sag and the loudest joint every
second.

    --joint left_knee   --step 0.5     --kp-scale 1   --kv-scale 1
    --limit 50 (N m, every actuator)   --gravity 0 (step mode) / 9.81 (walk mode)
    --seconds 1 (step) / 7 (walk)      --walk   --no-viewer   --speed 1

Six numbered steps; read them with the week 5 slides open.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mujoco
import numpy as np

import soc4180
from soc4180.walking import GaitParams, WalkingController


def torque_from_pd(kp, kv, ctrl, q, qdot):
    """The position servo, for every actuator at once: tau = kp (ctrl - q) - kv qdot."""
    return kp * (ctrl - q) - kv * qdot


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--joint", default="left_knee")
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--kp-scale", type=float, default=1.0)
    ap.add_argument("--kv-scale", type=float, default=1.0)
    ap.add_argument("--limit", type=float)
    ap.add_argument("--gravity", type=float)
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--walk", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(argv)
    gravity = args.gravity if args.gravity is not None else (9.81 if args.walk else 0.0)
    seconds = args.seconds if args.seconds is not None else (7.0 if args.walk else 1.0)

    # -- 1. the robot, with the servos changed ---------------------------------------------
    # A <position> actuator stores kp in gainprm[0] and (-kp, -kv) in biasprm[1:3].
    # scale_gains rewrites those for all 29; set_torque_limit writes forcerange.
    model = soc4180.load_g1()
    model.opt.gravity[:] = [0, 0, -gravity]
    kp, kv = soc4180.scale_gains(model, args.kp_scale, args.kv_scale)
    soc4180.set_torque_limit(model, args.limit)
    data = soc4180.keyframe_data(model, "stand")
    qadr = model.jnt_qposadr[model.actuator_trnid[:, 0]]      # each actuator's joint angle ...
    dadr = model.jnt_dofadr[model.actuator_trnid[:, 0]]       # ... and velocity
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    a = names.index(f"{args.joint}_joint") if f"{args.joint}_joint" in names else names.index(args.joint)
    print(f"servos: kp x{args.kp_scale:g}, kv x{args.kv_scale:g}, limit {args.limit}, gravity {gravity:g}")
    print(f"{names[a]}: kp = {kp[a]:.0f}, kv = {kv[a]:.2f}")

    # -- 2. the damping ratio, from the mass matrix -------------------------------------------
    # A servo on one joint is a mass-spring-damper: M q'' + kv q' + kp q = kp ctrl.
    # zeta = kv / (2 sqrt(kp M)) with M the joint's diagonal inertia at this pose.
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv)); mujoco.mj_fullM(model, data, M)   # dense mass matrix (3.12: takes data)
    m_ii = M[dadr[a], dadr[a]]
    zeta = kv[a] / (2 * np.sqrt(kp[a] * m_ii))
    print(f"inertia seen by the joint M_ii = {m_ii:.4f} kg m^2  ->  zeta = kv / 2 sqrt(kp M) = {zeta:.2f} "
          f"({'critically damped' if abs(zeta - 1) < 0.1 else 'under-damped: expect overshoot' if zeta < 1 else 'over-damped: sluggish'})")

    dt = model.opt.timestep
    frames, torque_gap = [], 0.0
    if not args.walk:
        # -- 3a. the step: hold `stand`, then jump one target ---------------------------------
        data.ctrl[:] = model.key_ctrl[0]
        target = data.ctrl.copy(); target[a] += args.step
        t_step, q0 = 0.2, data.qpos[qadr[a]]
        rows, first_1pct, peak_tau, q_max = [], None, 0.0, q0
        print(f"\n{'t (s)':>6} {'angle (rad)':>12} {'error (rad)':>12} {'torque (N m)':>13}")
        for k in range(int(seconds / dt) + 1):
            if data.time >= t_step:
                data.ctrl[:] = target
            mine = torque_from_pd(kp, kv, data.ctrl, data.qpos[qadr], data.qvel[dadr])
            mujoco.mj_step(model, data)
            # the law by hand vs the engine's own actuator_force (clipped only by a limit)
            torque_gap = max(torque_gap, float(np.abs(mine - data.actuator_force).max()) if args.limit is None else 0.0)
            q, tau = data.qpos[qadr[a]], data.actuator_force[a]
            if k % 10 == 0:
                frames.append(data.qpos.copy())
            if data.time >= t_step:
                err = target[a] - q
                if first_1pct is None and abs(err) < 0.01 * abs(args.step):
                    first_1pct = data.time - t_step
                peak_tau = max(peak_tau, abs(tau)); q_max = max(q_max, q) if args.step > 0 else min(q_max, q)
            if k % int(0.05 / dt) == 0:
                print(f"{data.time:6.3f} {q:12.4f} {target[a] - q:12.4f} {tau:13.2f}")
        # -- 4a. what the numbers say --------------------------------------------------------
        overshoot = (q_max - target[a]) / args.step * 100
        print(f"\nreached 1 % of the step after {first_1pct if first_1pct is not None else float('nan'):.3f} s; "
              f"overshoot {max(overshoot, 0):.1f} %; peak torque {peak_tau:.1f} N m; "
              f"final error {target[a] - data.qpos[qadr[a]]:+.4f} rad")
        print(f"the law by hand vs actuator_force: max gap {torque_gap:.1e} N m"
              + ("" if args.limit is None else " (not compared: a limit clips the engine's torque)"))
    else:
        # -- 3b. the walk, under the new gains --------------------------------------------------
        controller = WalkingController(model, GaitParams(n_steps=8))
        data = controller.initial_data()
        peak = np.zeros(model.nu); sag = []; fell = None
        print(f"\n{'t (s)':>6} {'x (m)':>7} {'pelvis z':>9} {'sag (mm)':>9} {'loudest joint':>24} {'N m':>7}")
        for k in range(int(seconds / dt)):
            if data.time >= controller.total_time - 1e-3:
                break
            data.ctrl[:] = controller.control(data.time)
            mujoco.mj_step(model, data)
            peak = np.maximum(peak, np.abs(data.actuator_force))
            if data.time > controller.params.settle_time:
                sag.append(controller.targets_at(data.time)[0][2] - data.qpos[2])   # commanded - actual pelvis z
            if k % 10 == 0:
                frames.append(data.qpos.copy())
            if data.qpos[2] < 0.5 and fell is None:
                fell = data.time; print(f"   FELL at t = {fell:.2f} s after {data.qpos[0]:+.2f} m")
            if k % int(1.0 / dt) == 0 and k > 0:
                loud = int(np.argmax(peak))
                print(f"{data.time:6.2f} {data.qpos[0]:+7.3f} {data.qpos[2]:9.3f} "
                      f"{(np.mean(sag) * 1000 if sag else float('nan')):9.1f} {names[loud]:>24} {peak[loud]:7.1f}")
                peak[:] = 0
        # -- 4b. summary ----------------------------------------------------------------------------
        print(f"\nwalked {data.qpos[0]:+.3f} m; mean sag {np.mean(sag) * 1000 if sag else float('nan'):+.1f} mm; "
              f"{'FELL' if fell else 'upright'}")

    # -- 5. no window? done -------------------------------------------------------------------------
    if args.no_viewer or soc4180.is_colab():
        return 0

    # -- 6. replay -------------------------------------------------------------------------------------
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    print(f"\nreplaying {len(frames)} poses at {args.speed:g}x (close the window to stop)")
    with soc4180.launch_viewer(model, data, passive=True) as viewer:
        while viewer.is_running() and time.time() < deadline:
            for q in frames:
                if not viewer.is_running() or time.time() > deadline:
                    break
                data.qpos[:] = q; mujoco.mj_forward(model, data); viewer.sync()
                time.sleep(0.02 / args.speed)
            time.sleep(0.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
