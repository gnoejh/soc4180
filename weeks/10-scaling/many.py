"""Robot-steps per second: against robots per scene, and against processes -- measured, then shown.

    uv run weeks/10-scaling/many.py
    uv run weeks/10-scaling/many.py --robots 1 2 4 8 16 --seconds 3 --no-viewer
    uv run weeks/10-scaling/many.py --procs 1 2 4 8 16
    uv run weeks/10-scaling/many.py --robots 6 --randomise

Two ways to get more robot-steps out of one laptop, both measured here:

  1. MORE ROBOTS IN ONE SCENE: one MjModel with n G1s attached through MjSpec,
     one MjData, one mj_step. The physics simply has more bodies in it. This
     is the shape of what a GPU simulator does with thousands of environments.
  2. MORE PROCESSES: one robot each, on separate cores (multiprocessing). This
     is what SubprocVecEnv does, and it stops at the core count.

The script builds each scene, walks it with the week 4 controller tiled
across the robots, prints robot-steps/s and the cost per robot, compares it
with `predict_rate` (a model you correct), runs the process benchmark, and
turns the best rate into hours for 150 million steps -- the Berkeley
Humanoid budget. Then it opens the simulator on the largest scene, walking.

    --robots 1 2 4 8     robots per scene to try     --seconds 2   per measurement
    --procs 1 2 4 8      processes to try (one robot each)
    --randomise          different friction and mass per robot (week 11 in one flag)
    --hold 8             seconds to keep the window open      --no-viewer

Six numbered steps; read them with the week 10 slides open.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time

import mujoco
import numpy as np

import soc4180

SPACING = 0.7
COST_EXPONENT = 1.0     # cost of one step ~ n ** COST_EXPONENT; 1.0 = linear in bodies. Correct it.


def predict_rate(n, single):
    """Robot-steps/s for one process stepping n robots, from the single-robot rate.

    If one step of an n-robot scene costs n ** p times a single robot's step,
    the scene runs at single / n ** p steps/s and delivers n times as many
    robot-steps each: single * n ** (1 - p). p = 1 means flat throughput; the
    measured column tells you the real p on your machine.
    """
    return single * n ** (1 - COST_EXPONENT)


def build(n, randomise, seed=0):
    """One scene with n G1s side by side: MjSpec attaches copies, compile() makes one model."""
    robot = str(soc4180.robot_path("unitree_g1", "g1"))            # MjSpec.from_file wants a str
    base = mujoco.MjSpec.from_string("""
<mujoco>
  <option integrator="implicitfast"/>
  <asset>
    <texture type="2d" name="grid" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="grid"/>
  </worldbody>
</mujoco>""")
    for k in range(n):
        child = mujoco.MjSpec.from_file(robot)
        frame = base.worldbody.add_frame()
        frame.attach_body(child.worldbody.first_body(), f"r{k}_", "")   # names get the prefix r{k}_
    model = base.compile()
    if randomise:
        rng = np.random.default_rng(seed)
        for k in range(n):
            scale, mu = rng.uniform(0.8, 1.2), rng.uniform(0.3, 1.0)
            for b in range(model.nbody):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                if name.startswith(f"r{k}_"):
                    model.body_mass[b] *= scale; model.body_inertia[b] *= scale
            for g in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
                if name.startswith(f"r{k}_") and name.endswith("ankle_roll_link"):
                    model.geom_friction[g, 0] = mu
    return model


def run_scene(model, one, n, seconds, walk):
    """Step n robots for `seconds` of wall clock, holding the crouch or walking; return steps/s.

    Holding: ctrl is fixed, so the time is physics alone. Walking: one
    WalkingController call (a 500 Hz IK solve) per step, shared by every robot
    through np.tile -- the controller costs about ten single-robot physics
    steps, so sharing it is where "free" throughput comes from at small n.
    """
    data = mujoco.MjData(model)
    controller = soc4180.WalkingController(one, soc4180.GaitParams(n_steps=12))
    fresh = controller.initial_data()
    for k in range(n):                                       # a free-joint offset goes in qpos, not the frame
        data.qpos[k * one.nq:(k + 1) * one.nq] = fresh.qpos
        data.qpos[k * one.nq + 1] += (k - (n - 1) / 2) * SPACING
    mujoco.mj_forward(model, data)
    data.ctrl[:] = np.tile(controller.control(0.0), n)
    t0, steps = time.perf_counter(), 0
    while time.perf_counter() - t0 < seconds:
        if walk:
            ctrl = controller.control(min(data.time, controller.total_time - 1e-3))
            data.ctrl[:] = np.tile(ctrl, n)                  # the same 29 targets for every robot
        mujoco.mj_step(model, data)
        steps += 1
    return steps / seconds


def _bench_one(seconds):
    """Step one G1 alone for `seconds`; return physics steps done. Runs inside a worker process."""
    m = soc4180.load_g1(); d = soc4180.keyframe_data(m, "stand")
    t0, k = time.perf_counter(), 0
    while time.perf_counter() - t0 < seconds:
        mujoco.mj_step(m, d); k += 1
    return k


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--robots", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--randomise", action="store_true")
    ap.add_argument("--hold", type=float, default=8.0)
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(argv)

    # -- 1. one robot alone: the unit of everything below ----------------------------------------
    one = soc4180.load_g1()
    single = _bench_one(1.0)
    print(f"{os.cpu_count()} CPU cores. One G1 alone, standing: {single:,.0f} physics steps/s "
          f"({single / 500:.0f}x real time); 150M steps at that rate = {150e6 / single / 3600:.1f} h")

    # -- 2. more robots per scene ------------------------------------------------------------------
    # Two measurements per scene: holding the crouch (physics only, what
    # predict_rate models) and walking (physics plus ONE controller call per
    # step, shared by every robot).
    print(f"\nrobots per scene, one process, {args.seconds:g} s each:")
    print(f"{'n':>3} {'nq':>5} {'nbody':>6} | {'physics only':>13} {'robot-steps/s':>14} {'predicted':>10} {'us/robot':>9} | "
          f"{'walking':>9} {'robot-steps/s':>14} {'x real':>7}")
    largest, rates, walks = None, {}, {}
    for n in args.robots:
        model = build(n, args.randomise)
        hold = run_scene(model, one, n, args.seconds, walk=False)
        walk = run_scene(model, one, n, args.seconds, walk=True)
        rates[n], walks[n] = hold * n, walk
        print(f"{n:3d} {model.nq:5d} {model.nbody:6d} | {hold:13,.0f} {hold * n:14,.0f} {predict_rate(n, single):10,.0f} "
              f"{1e6 / (hold * n):9.1f} | {walk:9,.0f} {walk * n:14,.0f} {walk / 500:7.2f}")
        largest = (model, n)
    n_max = max(rates)
    p_fit = 1 - np.log(rates[n_max] / single) / np.log(n_max) if n_max > 1 else float("nan")
    print(f"physics only: predict_rate assumes a step of n robots costs n ** {COST_EXPONENT} single steps; "
          f"the measured curve fits p = {p_fit:.2f} at n = {n_max} (contacts and the mass matrix grow faster "
          f"than the body count).")
    if 1 in rates:
        ctrl_ms = 1000 / walks[1] - 1000 / rates[1]
        print(f"walking: the controller's IK call costs about {ctrl_ms:.2f} ms per step against "
              f"{1000 / rates[1]:.3f} ms of physics for one robot ({ctrl_ms * rates[1] / 1000:.0f}x), and it is "
              f"shared by every robot in the scene, so walking robot-steps/s climbs with n until physics dominates.")

    # -- 3. more processes ---------------------------------------------------------------------------------
    print(f"\nprocesses, one robot each, standing, {args.seconds:g} s each (spawned, so the start-up is paid too):")
    best = single
    for p in args.procs:
        with mp.get_context("spawn").Pool(p) as pool:
            t0 = time.perf_counter()
            counts = pool.map(_bench_one, [args.seconds] * p)
            wall = time.perf_counter() - t0
        total = sum(counts) / args.seconds
        best = max(best, total)
        print(f"   {p:2d} processes: {total:9,.0f} robot-steps/s   ({total / single:4.1f}x one process; wall {wall:.1f} s)")

    # -- 4. the budget -------------------------------------------------------------------------------------
    print(f"\nbest rate here {best:,.0f} robot-steps/s: 150M steps = {150e6 / best / 3600:.1f} hours. "
          f"Berkeley Humanoid trains 150M steps at 8192 envs on one GPU in a few hours -- a different kind of processor.")

    if args.no_viewer or soc4180.is_colab() or largest is None:
        return 0

    # -- 5. and 6. show the largest scene, walking live ------------------------------------------------------
    model, n = largest
    data = mujoco.MjData(model)
    controller = soc4180.WalkingController(one, soc4180.GaitParams(n_steps=12))
    fresh = controller.initial_data()
    for k in range(n):
        data.qpos[k * one.nq:(k + 1) * one.nq] = fresh.qpos
        data.qpos[k * one.nq + 1] += (k - (n - 1) / 2) * SPACING
    mujoco.mj_forward(model, data)
    deadline = min(time.time() + args.hold, time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12))
    print(f"\nshowing {n} robots walking for {args.hold:g} s (close the window to stop)")
    with soc4180.launch_viewer(model, data, passive=True) as viewer:
        while viewer.is_running() and time.time() < deadline:
            wall = time.time()
            data.ctrl[:] = np.tile(controller.control(min(data.time, controller.total_time - 1e-3)), n)
            mujoco.mj_step(model, data)
            viewer.sync()
            lag = model.opt.timestep - (time.time() - wall)
            if lag > 0:
                time.sleep(lag)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
