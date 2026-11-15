"""Week 10: survive a shove by planning -- then copy the planner into a reflex.

    uv run weeks/10-planning-and-imitation/lab.py               # plan, copy, test, replay
    uv run weeks/10-planning-and-imitation/lab.py --no-viewer
    uv run weeks/10-planning-and-imitation/lab.py --push 300    # a harder shove

THE TASK, in two parts, both on the crouched G1 shoved sideways on the torso
(0.2 s at t = 1 s). Holding the crouch falls at 70 N.

PART 1 -- PLAN. MPPI (model-predictive control by sampling) decides 25 times a
second: copy the robot's state into MuJoCo, try SAMPLES futures of HORIZON
seconds with the leg servo targets jittered, score each future with `cost()`,
and execute the cost-weighted average of their first step. You never say HOW
to balance -- you say what is BAD, in `cost()`, and the search finds a way.
You change `cost()`.

PART 2 -- COPY. Planning is slow (each decision rolls out thousands of physics
steps). So record the planner's decisions while it recovers from a few
shoves, and fit a LINEAR reflex, action = W x + b, by least squares, from the
INPUTS you choose. Then test the reflex alone, with no planner. You change
`INPUTS`.

THE RULE you will test (measured in the deck three ways): a copy can only be
as good as the expert's action is a FUNCTION of what the copy sees. If the
expert's action depends on something the copy does not see (its plan, its
random samples, the time), no amount of data fixes it.

Measured with this file (~4 min; your laptop differs in ms, not in who falls):

    part 1, 150 N:  hold FALLS; MPPI stands -- by stepping: its feet move 3.04 m
                    in total, because nothing in cost() asks it to stop.
                    81-98 ms per decision on 16 threads (about 0.4x real time).
    part 2 (copy, 400 decisions recorded on 4 shoves), who stands:

        EXPERT     INPUTS                         fit error  60 N  75 N 100 N 150 N
        planner    tilt spin legs speeds (29)     0.188 rad  FELL  FELL  FELL  FELL
        planner    tilt (2)                       0.201 rad  stood stood FELL  FELL
        ankle      tilt spin legs speeds (29)     0.003 rad  FELL  FELL  FELL  FELL
        ankle      tilt (2)                       0.011 rad  stood stood FELL  FELL
        (the experts themselves: planner stands at all four; ankle at 60 and 75 N;
         holding the crouch at 60 N only)

    The planner's spread is 0.202 rad and the fit leaves 0.188-0.201 of it:
    most of what MPPI does is its random samples, which no input explains.
    The ankle rule is linear in tilt (but for its clip), yet the 29-input copy, with a
    tiny fit error, falls: inputs that barely moved in the recordings get
    weights nobody checked, and the copy's own mistakes take it where they
    matter. More inputs made the copy worse.

Try (predict each one first):
    1. delete the "still" term from cost(). Does it still stand? Does it stop?
    2. add a term that keeps the feet where they started. Does the drift stop?
    3. EXPERT = "ankle", then INPUTS = ["tilt"]: reproduce the table above.
    4. add "time" to INPUTS. The planner never looks at the clock -- does the copy improve?
    5. --push 300: does the planner survive? does the copy?
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mujoco
import numpy as np

import soc4180
from soc4180 import planning
from soc4180.estimation import gravity_body

# -----------------------------------------------------------------------------
# 1. THE COST -- what is bad about a future. Lower is better.
#    r holds SAMPLES futures, each HORIZON seconds of physics steps; every
#    quantity below is an array (samples, steps). Return one number per future.
# -----------------------------------------------------------------------------
def cost(r):
    mid = 0.5 * (r.foot("left") + r.foot("right"))                     # between the feet
    c = (10.0 * (r.pelvis_height - 0.72) ** 2                           # stay tall
         + 5.0 * (1.0 - r.upright)                                      # stay upright
         + 20.0 * np.sum((r.com[..., :2] - mid[..., :2]) ** 2, -1)      # mass over the feet
         + 0.01 * np.sum(r.qvel[..., :6] ** 2, -1)                      # "still": pelvis speed
         + 100.0 * (r.pelvis_height < 0.5))                             # fallen
    return c.sum(axis=1)


SAMPLES, HORIZON, SIGMA, TEMPERATURE = 64, 0.4, 0.15, 0.05

# -----------------------------------------------------------------------------
# 2. THE COPY'S INPUTS -- what the reflex is allowed to see. Pick from:
#    "tilt"   gravity in the pelvis frame, x and y (2)
#    "spin"   pelvis angular velocity (3)
#    "legs"   the 12 leg angles minus the crouch (12)
#    "speeds" the 12 leg angular velocities (12)
#    "time"   seconds since the start (1) -- the planner never looks at it
# -----------------------------------------------------------------------------
INPUTS = ["tilt", "spin", "legs", "speeds"]
EXPERT = "planner"      # whom to copy: "planner" (MPPI, part 1) or "ankle" (week 8's rule)
TRAIN_PUSHES = (-150.0, -100.0, 100.0, 150.0)   # the shoves the planner is recorded on


# -----------------------------------------------------------------------------
# 3. The machinery: the world, a run, the inputs, the fit.
# -----------------------------------------------------------------------------
class World:
    def __init__(self):
        self.model = planning.planning_model()
        self.data = mujoco.MjData(self.model)
        self.torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.q0, self.ctrl0 = planning.crouch(self.model)
        self.acts = planning.actuators_for(self.model)
        self.legs = self.model.actuator_trnid[self.acts, 0]
        self.qadr = self.model.jnt_qposadr[self.legs]
        self.vadr = self.model.jnt_dofadr[self.legs]

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.q0; self.data.qpos[2] = 0.74
        mujoco.mj_forward(self.model, self.data)

    def inputs(self):
        d = self.data
        parts = {
            "tilt": gravity_body(d)[:2],
            "spin": d.qvel[3:6],
            "legs": d.qpos[self.qadr] - self.q0[self.qadr],
            "speeds": d.qvel[self.vadr],
            "time": [d.time],
        }
        return np.concatenate([np.asarray(parts[k], float) for k in INPUTS])

    def tilt_deg(self):
        x, y = self.data.qpos[4], self.data.qpos[5]
        return float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))


def run(world, push, decide, seconds=4.0, *, record=False):
    """One shove. decide(world) -> full ctrl, called every knot. Returns (stood, info)."""
    world.reset()
    d, frames, X, Y, zmin = world.data, [], [], [], 9.0
    feet = [mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_foot") for s in ("left", "right")]
    start = [d.site_xpos[f][:2].copy() for f in feet]
    substeps = int(round(0.04 / world.model.opt.timestep))
    while d.time < seconds - 1e-9:
        x = world.inputs()
        ctrl = decide(world)
        X.append(x); Y.append(ctrl[world.acts] - world.ctrl0[world.acts])
        for _ in range(substeps):
            d.xfrc_applied[world.torso, 1] = push if 1.0 <= d.time < 1.2 else 0.0
            d.ctrl[:] = ctrl
            mujoco.mj_step(world.model, d)
            if record and int(round(d.time / world.model.opt.timestep)) % 10 == 0:
                frames.append(d.qpos.copy())
        zmin = min(zmin, d.qpos[2])
    stood = zmin > 0.5 and world.tilt_deg() < 15.0
    moved = sum(float(np.linalg.norm(d.site_xpos[f][:2] - s)) for f, s in zip(feet, start))
    return stood, dict(moved=moved, X=np.array(X), Y=np.array(Y), frames=frames)


def fit(X, Y, ridge=1e-3):
    """Least squares with a little ridge: W, b minimising |X W + b - Y|^2 + ridge |W|^2."""
    mu, sd = X.mean(0), np.maximum(X.std(0), 0.01)      # a floor: an input that never moved must not explode
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    A = Z.T @ Z + ridge * np.eye(Z.shape[1]); A[-1, -1] -= ridge
    Wb = np.linalg.solve(A, Z.T @ Y)
    return lambda x: np.hstack([(x - mu) / sd, 1.0]) @ Wb


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Week 10: plan, then copy the planner.")
    ap.add_argument("--push", type=float, default=150.0, help="the test shove, newtons, sideways")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(argv)

    world = World()
    hold = lambda w: w.ctrl0
    mppi = planning.MPPI(world.model, world.ctrl0, world.acts, samples=SAMPLES, horizon=HORIZON,
                         sigma=SIGMA, temperature=TEMPERATURE, seed=0)
    mppi.cost = cost

    def plan(w):
        return mppi.plan(w.data)

    def ankle(w):                                     # week 8: ankle pitch = 2 x forward tilt
        ctrl = w.ctrl0.copy()
        ctrl[w.acts[[4, 10]]] += 0.3 * np.clip(2.0 * gravity_body(w.data)[0], -1, 1)
        return ctrl

    expert = {"planner": plan, "ankle": ankle}[EXPERT]

    # Part 1: plan
    print(f"PART 1 -- plan. MPPI: {SAMPLES} futures x {HORIZON} s, 25 decisions per second\n")
    print(f"{'controller':>12} {'push':>7} {'result':>7} {'feet moved':>11} {'ms/decision':>12}")
    stood, info = run(world, args.push, hold)
    print(f"{'hold':>12} {args.push:6g}N {'stood' if stood else 'FELL':>7} {info['moved']:10.2f}m {'-':>12}")
    mppi.reset(); t0 = time.time()
    stood, info = run(world, args.push, plan, record=True)
    ms = 1000 * (time.time() - t0) / len(info["X"])
    print(f"{'MPPI':>12} {args.push:6g}N {'stood' if stood else 'FELL':>7} {info['moved']:10.2f}m {ms:11.0f}")
    planned = info

    # Part 2: copy
    print(f"\nPART 2 -- copy. Recording the expert ({EXPERT}) on {len(TRAIN_PUSHES)} shoves {TRAIN_PUSHES} N ...")
    X, Y = [], []
    for p in TRAIN_PUSHES:
        mppi.reset()
        _, rec = run(world, p, expert)
        X.append(rec["X"]); Y.append(rec["Y"])
    X, Y = np.vstack(X), np.vstack(Y)
    reflex = fit(X, Y)
    err = np.sqrt(np.mean((np.array([reflex(x) for x in X]) - Y) ** 2))
    scale = np.sqrt(np.mean(Y ** 2))
    print(f"{len(X)} decisions; inputs {INPUTS} ({X.shape[1]} numbers) -> 12 servo targets; "
          f"fit error {err:.4f} rad against the expert's own spread {scale:.4f} rad")

    def copy(w):
        ctrl = w.ctrl0.copy()
        ctrl[w.acts] += np.clip(reflex(w.inputs()), -0.5, 0.5)
        return ctrl

    print(f"\n{'push':>7} {'hold':>7} {'expert':>7} {'copy':>7}")
    for p in (60.0, 75.0, 100.0, args.push):
        cells = []
        for c in (hold, expert, copy):
            mppi.reset()
            cells.append(run(world, p, c)[0])
        print(f"{p:6g}N " + " ".join(f"{'stood' if s else 'FELL':>7}" for s in cells))

    if args.no_viewer or soc4180.is_colab():
        return 0
    print(f"\nreplaying MPPI under {args.push:g} N; close the window to stop")
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    d = world.data
    with soc4180.launch_viewer(world.model, d, passive=True) as viewer:
        while viewer.is_running() and time.time() < deadline:
            for q in planned["frames"]:
                if not viewer.is_running() or time.time() > deadline:
                    break
                d.qpos[:] = q; mujoco.mj_forward(world.model, d); viewer.sync()
                time.sleep(10 * world.model.opt.timestep)
            time.sleep(0.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
