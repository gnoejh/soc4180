"""Run the robot stack one layer at a time, print what physics reports, then watch it.

    uv run weeks/00-robot-stack/stack.py --layer hold
    uv run weeks/00-robot-stack/stack.py --layer limp --seconds 3
    uv run weeks/00-robot-stack/stack.py --layer walk --seconds 9
    uv run weeks/00-robot-stack/stack.py --layer wave --gravity 1.62 --no-viewer

The five layers of week 0, as switches on one simulation:

    --layer limp   layers 1+2 only: physics and the model, every servo dead
    --layer zero   layer 4 in its trap form: ctrl = 0, every joint commanded to angle zero
    --layer hold   layer 4, simplest: the servos hold the `stand` keyframe
    --layer walk   layer 4 in full: the week 4 walker (LIPM + IK)
    --layer wave   layer 4, yours: hold `stand` and wave the left arm (see `wave`)
    --gravity 1.62 change layer 1 only (the Moon); everything else untouched
    --seconds 3    how long to simulate       --report 0.5   seconds between printed rows
    --no-viewer    simulate and print only    --speed 1      replay speed in the viewer

The script simulates first, headless and as fast as the machine allows,
printing one row per `--report` seconds: torso height, number of contacts,
the sum of the vertical contact forces against the robot's weight, and the
physics rate. Then it opens the simulator and replays the recorded motion in
real time. Six numbered steps; read them with the week 0 slides open.
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


# ---------------------------------------------------------------------------
# Layer 4, five ways. Each is a function (model, data, t) -> None that writes
# `data.ctrl`, the 29 servo targets in radians, before every physics step.
# ---------------------------------------------------------------------------

def zero(model, data, t):
    """ctrl = 0 is a COMMAND: every joint to angle zero. The robot stands, straight-legged."""
    data.ctrl[:] = 0.0


def hold(model, data, t):
    """The keyframe's own targets: the simplest controller there is."""
    data.ctrl[:] = model.key_ctrl[0]


def wave(model, data, t):
    """Hold `stand`, and swing the left shoulder pitch (actuator 15) as a sinusoid.

    Every other servo keeps its `stand` target, which is why the rest of the
    robot keeps standing while one arm moves: layer 4 holds everything you do
    not command otherwise. Change the amplitude, the frequency, the actuator.
    """
    ctrl = model.key_ctrl[0].copy()
    ctrl[15] = -1.2 + 0.6 * math.sin(2.0 * t)
    data.ctrl[:] = ctrl


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--layer", choices=("limp", "zero", "hold", "walk", "wave"), default="hold")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--gravity", type=float, default=9.81, help="m/s^2 downward; 1.62 is the Moon")
    ap.add_argument("--report", type=float, default=0.5)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(argv)

    # -- 1. layers 1 and 2: the world and the robot ----------------------------------
    # load_g1() compiles the Unitree G1's MJCF into an MjModel: bodies, joints,
    # geoms, actuators, and the physics options (timestep 0.002 s, gravity).
    # keyframe_data() makes an MjData -- the state -- set to the `stand` pose.
    model = soc4180.load_g1()
    data = soc4180.keyframe_data(model, "stand")
    model.opt.gravity[:] = [0, 0, -args.gravity]                 # layer 1, changed here only
    weight = float(model.body_mass.sum() * args.gravity)          # what the floor must push back
    print(f"model: {model.nbody} bodies, {model.njnt} joints, nq={model.nq} nv={model.nv} "
          f"nu={model.nu} servos, timestep {model.opt.timestep} s, gravity {args.gravity:g} m/s^2")
    print(f"mass {model.body_mass.sum():.1f} kg, weight {weight:.1f} N")

    # -- 2. layer 4: pick the controller ----------------------------------------------
    controllers = {"zero": zero, "hold": hold, "wave": wave}
    if args.layer == "walk":
        walker = soc4180.WalkingController(model)               # LIPM plan + IK, week 4
        data.qpos[:] = walker.initial_data().qpos               # it starts from its crouch
        mujoco.mj_forward(model, data)
        controllers["walk"] = lambda m, d, t: d.ctrl.__setitem__(
            slice(None), walker.control(min(t, walker.total_time - 1e-3)))
    if args.layer == "limp":
        # no controller at all: the actuation flag is off, so `ctrl` is ignored
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        controllers["limp"] = lambda m, d, t: None
    control = controllers[args.layer]
    print(f"layer 4: {args.layer}\n")

    # -- 3. the loop: controller, then one physics step, 500 times a second -----------
    # mj_step advances the state by one timestep: it reads ctrl, computes the
    # servo torques, contacts and constraint forces, and integrates. Everything
    # the printout reports is in `data` afterwards.
    steps = int(round(args.seconds / model.opt.timestep))
    every = max(1, int(round(args.report / model.opt.timestep)))
    frame_every = max(1, int(round(0.02 / model.opt.timestep)))   # record 50 poses per second
    frames, wrench = [], np.zeros(6)
    print(f"{'t (s)':>6} {'torso z (m)':>12} {'contacts':>9} {'sum Fz (N)':>11} {'weight (N)':>11}")
    t0 = time.perf_counter()
    for k in range(steps + 1):
        if k % frame_every == 0:
            frames.append(data.qpos.copy())
        if k % every == 0:
            # mj_contactForce gives each contact's force in the contact frame;
            # its first component is the normal force, which is vertical on
            # a flat floor. Summing them is the floor pushing back (layer 1).
            total = 0.0
            for i in range(data.ncon):
                mujoco.mj_contactForce(model, data, i, wrench)
                total += wrench[0]
            print(f"{data.time:6.2f} {data.qpos[2]:12.3f} {data.ncon:9d} {total:11.1f} {weight:11.1f}")
        if k == steps:
            break
        control(model, data, data.time)          # layer 4 writes ctrl ...
        mujoco.mj_step(model, data)              # ... and layers 1+2 move the world
    elapsed = time.perf_counter() - t0

    # -- 4. summary -------------------------------------------------------------------
    warn = int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
    print(f"\n{steps} physics steps in {elapsed:.2f} s = {steps / elapsed:,.0f} steps/s "
          f"({steps / elapsed / 500:.1f}x real time)")
    print(f"torso z {data.qpos[2]:.3f} m   walked {data.qpos[0]:+.3f} m forward"
          f"   integrator warnings {warn}")

    # -- 5. no window? done -------------------------------------------------------------
    model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    if args.no_viewer or soc4180.is_colab():
        return 0

    # -- 6. replay: the recorded poses, in real time, in the simulator -------------------
    # The passive viewer draws `data` on every sync(). We write the recorded
    # qpos and call mj_forward so the geometry matches; physics is not re-run.
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    print(f"\nreplaying {len(frames)} poses at {args.speed:g}x (close the window to stop)")
    with soc4180.launch_viewer(model, data, passive=True) as viewer:
        while viewer.is_running() and time.time() < deadline:
            for q in frames:
                if not viewer.is_running() or time.time() > deadline:
                    break
                data.qpos[:] = q
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.02 / args.speed)
            time.sleep(0.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
