"""Plan at run time: MPPI on the G1, with MuJoCo as the model -- printed, then replayed.

    uv run weeks/11-planning/plan.py                          # stand, shoved with 150 N
    uv run weeks/11-planning/plan.py --push 300
    uv run weeks/11-planning/plan.py --controller hold --push 70
    uv run weeks/11-planning/plan.py --task crane --seconds 6
    uv run weeks/11-planning/plan.py --samples 4 --push 200
    uv run weeks/11-planning/plan.py --model-mass 10 --push 200     # the planner's model is wrong
    uv run weeks/11-planning/plan.py --no-viewer

At every decision (25 per second) the planner copies the robot's state into
the simulator, tries --samples futures of --horizon seconds with the leg
servos' targets perturbed by --sigma radians, scores every future with the
task's cost, and executes the first step of the cost-weighted average. Then it
does it all again. Nothing is trained; nothing is remembered but the plan.

    --task stand|crane|squat    --controller mppi|hold
    --push 150      sideways force on the torso, newtons, for 0.2 s at t = 1 s
    --samples 64    --horizon 0.4   --knot 0.04   --sigma 0.15   --temperature 0.05
    --chains left_leg right_leg     which servos the planner may move (bodies.CHAINS names)
    --model-mass 0  kilograms added to the PLANNER's torso only: the world is unchanged
    --seconds 4     --seed 0        --no-viewer   --speed 1

Six numbered steps; read them with the week 11 slides open.
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
from soc4180 import planning


# ---------------------------------------------------------------------------
# The tasks. A cost takes the K sampled futures (soc4180.planning.Rollouts)
# and returns one number per future: lower is better. Each is a sum of
# squared errors over every step of the horizon. Nothing here says HOW to do
# the task -- that is the planner's job.
# ---------------------------------------------------------------------------

def stand(r):
    """Tall, upright, centre of mass over the middle of the feet."""
    return planning.standing_cost(r)


def crane(r):
    """Week 3's unsolved problem: stand on the right foot with the left foot 13 cm up."""
    rf, lf = r.foot("right"), r.foot("left")
    c = (10.0 * (r.pelvis_height - 0.72) ** 2 + 5.0 * (1.0 - r.upright)
         + 20.0 * np.sum((r.com[..., :2] - rf[..., :2]) ** 2, -1)           # balance over the STANCE foot
         + 200.0 * np.maximum(0.0, 0.13 - lf[..., 2]) ** 2                   # lift the other one
         + 0.01 * np.sum(r.qvel[..., :6] ** 2, -1) + 100.0 * (r.pelvis_height < 0.5))
    return c.sum(1)


def squat(r, t0=0.0):
    """Down to 0.55 m between t = 1 and 3 s, back to 0.72 after."""
    t = t0 + np.arange(r.qpos.shape[1]) * PLAN_DT
    target = np.where((t > 1.0) & (t < 3.0), 0.55, 0.72)
    mid = 0.5 * (r.foot("left") + r.foot("right"))
    c = (20.0 * (r.pelvis_height - target) ** 2 + 5.0 * (1.0 - r.upright)
         + 20.0 * np.sum((r.com[..., :2] - mid[..., :2]) ** 2, -1)
         + 0.01 * np.sum(r.qvel[..., :6] ** 2, -1) + 100.0 * (r.pelvis_height < 0.4))
    return c.sum(1)


TASKS = {"stand": stand, "crane": crane, "squat": squat}
PLAN_DT = 0.002


def tilt_deg(data):
    x, y = data.qpos[4], data.qpos[5]
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


def main(argv=None) -> int:
    global PLAN_DT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", choices=tuple(TASKS), default="stand")
    ap.add_argument("--controller", choices=("mppi", "hold"), default="mppi")
    ap.add_argument("--push", type=float, default=None)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--horizon", type=float, default=0.4)
    ap.add_argument("--knot", type=float, default=0.04)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--chains", nargs="+", default=["left_leg", "right_leg"])
    ap.add_argument("--model-mass", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(argv)
    push = args.push if args.push is not None else (150.0 if args.task == "stand" else 0.0)
    seconds = args.seconds or {"stand": 4.0, "crane": 6.0, "squat": 5.0}[args.task]

    # -- 1. two models: the world, and the planner's copy of it ---------------------------
    # planning_model() is the G1 plus six position sensors (feet, hands, head,
    # centre of mass) so that every rollout records them. The planner gets its
    # OWN model, so --model-mass can make it wrong without touching the world.
    world = planning.planning_model()
    model = planning.planning_model()
    torso = mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    model.body_mass[torso] += args.model_mass
    PLAN_DT = model.opt.timestep
    q0, ctrl0 = planning.crouch(world)
    data = mujoco.MjData(world)
    data.qpos[:] = q0; data.qpos[2] = 0.74
    mujoco.mj_forward(world, data)
    print(f"G1 + {len(planning.SENSORS)} sensors: nsensordata {world.nsensordata}; "
          f"torso {world.body_mass[torso]:.2f} kg in the world, {model.body_mass[torso]:.2f} kg in the planner")

    # -- 2. the planner ------------------------------------------------------------------------
    acts = planning.actuators_for(world, args.chains)
    mppi = planning.MPPI(model, ctrl0, acts, samples=args.samples, horizon=args.horizon, knot=args.knot,
                         sigma=args.sigma, temperature=args.temperature, seed=args.seed)
    mppi.cost = TASKS[args.task]
    steps_per_plan = args.samples * mppi.H * mppi.substeps
    print(f"MPPI over {len(acts)} servos ({' '.join(args.chains)}): {args.samples} samples x "
          f"{mppi.H} knots of {mppi.knot * 1000:.0f} ms = {args.horizon:.2f} s ahead; "
          f"{steps_per_plan:,} physics steps per decision, {len(mppi._datas)} threads")
    print(f"task {args.task}, controller {args.controller}"
          + (f", push {push:g} N sideways at t = 1 s for 0.2 s" if push else ""))

    # -- 3. the loop: plan, apply for one knot, repeat --------------------------------------------
    feet = [mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_SITE, f"{s}_foot") for s in ("left", "right")]
    start = [data.site_xpos[f].copy() for f in feet]
    frames, plan_s, n_plans, zmin, tmax, lifted = [], 0.0, 0, 9.0, 0.0, 0.0
    ctrl = ctrl0.copy()
    print(f"\n{'t (s)':>6} {'pelvis':>7} {'tilt':>6} {'L foot z':>9} {'feet moved':>11} {'best cost':>10} {'ms/plan':>8}")
    next_row = 0.0
    wall0 = time.time()
    while data.time < seconds - 1e-9:
        if args.controller == "mppi":
            t0 = time.time()
            if args.task == "squat":
                mppi.cost = lambda r, t=data.time: squat(r, t)
            ctrl = mppi.plan(data)                                       # K futures, one decision
            plan_s += time.time() - t0; n_plans += 1
        for _ in range(mppi.substeps):
            data.xfrc_applied[torso, 1] = push if 1.0 <= data.time < 1.2 else 0.0
            data.ctrl[:] = ctrl
            mujoco.mj_step(world, data)
            if int(round(data.time / world.opt.timestep)) % 10 == 0:
                frames.append(data.qpos.copy())
        zmin = min(zmin, data.qpos[2]); tmax = max(tmax, tilt_deg(data))
        if data.site_xpos[feet[0]][2] > 0.083:
            lifted += mppi.knot
        if data.time >= next_row - 1e-9:
            moved = sum(np.linalg.norm(data.site_xpos[f][:2] - s[:2]) for f, s in zip(feet, start))
            best = mppi.last_costs.min() if mppi.last_costs is not None and args.controller == "mppi" else float("nan")
            ms = 1000 * plan_s / max(n_plans, 1)
            print(f"{data.time:6.2f} {data.qpos[2]:7.3f} {tilt_deg(data):5.1f}° {data.site_xpos[feet[0]][2]:9.3f} "
                  f"{moved:10.2f}m {best:10.2f} {ms:8.1f}")
            next_row += 0.5
    wall = time.time() - wall0

    # -- 4. the outcome ---------------------------------------------------------------------------------
    fell = zmin < 0.5 or tilt_deg(data) > 15.0          # on the floor, or on the way there
    moved = sum(np.linalg.norm(data.site_xpos[f][:2] - s[:2]) for f, s in zip(feet, start))
    print(f"\n{'FELL' if fell else 'STOOD'}: lowest pelvis {zmin:.3f} m, worst tilt {tmax:.1f}°, "
          f"feet moved {moved:.2f} m in total, left foot above 5 cm for {lifted:.2f} s")
    if args.controller == "mppi":
        print(f"planning: {n_plans} decisions, {1000 * plan_s / n_plans:.0f} ms each "
              f"({steps_per_plan * n_plans / plan_s:,.0f} physics steps/s); "
              f"{seconds / wall:.2f}x real time")

    # -- 5. what it means -----------------------------------------------------------------------------
    print("\nholding the crouch falls at 70 N; the ankle strategy (week 7) at 80 N. Whatever MPPI did beyond "
          "that, the cost did not ask for -- it asked for balance, and search found a way.")

    if args.no_viewer or soc4180.is_colab():
        return 0

    # -- 6. replay --------------------------------------------------------------------------------------
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    print(f"\nreplaying {len(frames)} frames at {args.speed:g}x (close the window to stop)")
    with soc4180.launch_viewer(world, data, passive=True) as viewer:
        while viewer.is_running() and time.time() < deadline:
            for q in frames:
                if not viewer.is_running() or time.time() > deadline:
                    break
                data.qpos[:] = q; mujoco.mj_forward(world, data); viewer.sync()
                time.sleep(10 * world.opt.timestep / args.speed)
            time.sleep(0.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
