"""Week 11 lab game, on your laptop: six problems, one planner, a score out of 6.

    uv run weeks/11-planning/lab_plan.py

YOU EDIT ONLY `costs.py`, next to this file. Every problem is solved by the
same planner -- MPPI, with MuJoCo as its model (soc4180.planning, the
lecture's code). What you write is WHAT the robot should want: the weights of
a cost, a few numbers, and the planner's settings. How to achieve it is the
planner's job, and it finds out 25 times a second by trying futures.

    1 ... 6        play problem 1 ... 6 live (slow motion: the planner is thinking)
    SPACE          play the current problem again
    G              GRADE: all six, one after another; the score stays on screen
    ENTER          print the live gauge in this terminal
    left-drag / right-drag / wheel   orbit / pan / zoom
    Keys go to the MuJoCo window -- click it first. If no key works there,
    press the same keys in this terminal (single keys, no enter). `q` quits.

    uv run weeks/11-planning/lab_plan.py --grade --no-viewer      # grade headless
    uv run weeks/11-planning/lab_plan.py --grade --problem 2 5    # just these

costs.py is re-read every time a problem starts: save it and press the
number again, no restart. The terms a cost can weigh are listed in section 1
below, each one line of numpy on the sampled futures.

Things measured with this file's own code before it was written (--grade):

- Holding the crouch with no planner at all falls at a 70 N shove (plan.py
  --controller hold). With STAND = height 10, upright 5, over_feet 20,
  still 0.01, fall 100 the planner survives 150 N and 300 N; so does that
  cost with any ONE of height, upright or over_feet set to 0.
- At 300 N it survives by STEPPING, about 6 m of foot travel in two seconds.
  Nothing in the cost mentions feet or steps.
- A cost of ONLY "fall": 100 falls over with no push at all: the planner
  looks 0.4 s ahead, and a fall that starts now is not below 0.5 m yet. A
  cost of only "still" stands when nobody pushes and falls at 150 N. All
  weights 0 falls in 1.4 s: every future is equally good, and the average
  of random plans is a random walk.
- At 300 N, 16 samples with sigma 0.15 fall; sigma 0.2, 0.25, 0.3 and 0.35
  all stand. Fewer samples need wider ones. (2 samples fall at 150 N;
  4 survive it.)
- CRANE with lift_left 20 or 50 never holds the foot up for 2 s; 200 does --
  by HOPPING on the right foot, 2 m of travel; 2000 ends tilted 18 degrees.
  A term that keeps the right foot planted (plant_right) stops the hopping
  at exactly one weight we found (1000) and falls or hops at 500 and 2000,
  so it is not graded. The planner does what the cost says, not what you
  meant.
- SQUAT with height 20 never moves (lowest 0.745 m): every sampled future
  that starts down also tilts, and costs more than standing still. With
  height 100 it reaches 0.503 m and stands back up. With height 100 and the
  balance terms at 0 it falls.

Everything in this file is complete and explained. You are not meant to edit
it: it is the referee, and the grading screen shows a 4-letter code computed
from it so the instructor can see it is unchanged.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np

import soc4180
from soc4180 import planning
from soc4180.game import AnswersError, Attempt, Problem, arrow, grade_headless, need, number, run_viewer, sphere

HERE = Path(__file__).resolve().parent


# --- 1. the cost terms --------------------------------------------------------------------------------
#
# Each term takes the K sampled futures `r` (soc4180.planning.Rollouts; every
# accessor is K x T) and returns K x T: how bad each step of each future is.
# A cost is sum over terms of weight x term, summed over the T steps. Lower
# is better. The planner prefers the futures with the lowest cost.

def t_height(r, p):        # pelvis away from its target height, squared (target: p["height_target"])
    return (r.pelvis_height - p.get("height_target", 0.72)) ** 2

def t_upright(r, p):       # 1 - cos(tilt): 0 upright, 1 lying down
    return 1.0 - r.upright

def t_over_feet(r, p):     # centre of mass away from the midpoint of the feet, horizontal, squared
    mid = 0.5 * (r.foot("left") + r.foot("right"))
    return np.sum((r.com[..., :2] - mid[..., :2]) ** 2, axis=-1)

def t_over_right(r, p):    # centre of mass away from the RIGHT foot: stand on it
    return np.sum((r.com[..., :2] - r.foot("right")[..., :2]) ** 2, axis=-1)

def t_lift_left(r, p):     # left foot below the lift height (p["lift"]), squared; 0 once it is above
    return np.maximum(0.0, p.get("lift", 0.10) - r.foot("left")[..., 2]) ** 2

def t_plant_right(r, p):  # right foot off the floor, squared: keep the stance foot planted (no hopping)
    return (r.foot("right")[..., 2] - 0.033) ** 2

def t_still(r, p):         # pelvis velocity, linear and angular, squared
    return np.sum(r.qvel[..., :6] ** 2, axis=-1)

def t_fall(r, p):          # 1 on every step the pelvis is below 0.5 m
    return (r.pelvis_height < 0.5).astype(float)


TERMS = {"height": t_height, "upright": t_upright, "over_feet": t_over_feet, "over_right": t_over_right,
         "lift_left": t_lift_left, "plant_right": t_plant_right, "still": t_still, "fall": t_fall}
SETTINGS = {"height_target", "lift"}          # numbers a cost dict may carry that are not weights


def make_cost(weights: dict, where: str):
    """A cost function from a dict of term weights (and settings), checked for typos."""
    for k, v in weights.items():
        if k not in TERMS and k not in SETTINGS:
            raise AnswersError(f"{where}: no term called \"{k}\" -- terms are {', '.join(TERMS)}")
        number(weights, k, where)
    w = {k: float(v) for k, v in weights.items() if k in TERMS}
    p = {k: float(v) for k, v in weights.items() if k in SETTINGS}

    def cost(r):
        return sum(wk * TERMS[k](r, p) for k, wk in w.items()).sum(axis=1)
    return cost


def make_planner(model, ctrl0, settings: dict, where: str, max_samples=256):
    s = int(number(settings, "samples", where, 1, max_samples))
    return planning.MPPI(model, ctrl0, planning.actuators_for(model, ("left_leg", "right_leg")),
                         samples=s, horizon=number(settings, "horizon", where, 0.04, 1.0),
                         sigma=number(settings, "sigma", where, 0.0, 1.0),
                         temperature=number(settings, "temperature", where, 1e-4, 100.0), seed=0)


# --- 2. one attempt: the world, the planner, the loop ------------------------------------------------

class World:
    """Built once: the G1 with the planning sensors, its crouch, and ids the gauges read."""

    def __init__(self):
        self.model = planning.planning_model()
        self.q0, self.ctrl0 = planning.crouch(self.model)
        self.torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.feet = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_foot") for s in ("left", "right")]
        self.data = mujoco.MjData(self.model)
        self.reset(self.data)

    def reset(self, data):
        mujoco.mj_resetData(self.model, data)
        data.qpos[:] = self.q0
        data.qpos[2] = 0.74
        mujoco.mj_forward(self.model, data)


WORLD = None


def world() -> World:
    global WORLD
    if WORLD is None:
        WORLD = World()
    return WORLD


class PlanAttempt(Attempt):
    """Stand in the crouch, maybe get shoved, plan every knot, measure everything."""

    def __init__(self, cost, planner_settings, where, *, seconds, push=0.0, push_at=1.0, schedule=None,
                 max_samples=256):
        super().__init__()
        w = world()
        self.w, self.model, self.data = w, w.model, mujoco.MjData(w.model)
        w.reset(self.data)
        self.mppi = make_planner(w.model, w.ctrl0, planner_settings, where, max_samples)
        self.mppi.cost = cost
        self.schedule = schedule            # t -> cost, for costs that change with time (the squat)
        self.dt = self.mppi.knot
        self.seconds, self.push, self.push_at = seconds, push, push_at
        self.start = [self.data.site_xpos[f][:2].copy() for f in w.feet]
        self.zmin, self.tilt_max, self.lift_run, self.lift_best, self.zlow = 9.0, 0.0, 0.0, 0.0, 9.0

    @property
    def tilt(self):
        x, y = self.data.qpos[4], self.data.qpos[5]
        return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))

    @property
    def moved(self):
        return sum(float(np.linalg.norm(self.data.site_xpos[f][:2] - s)) for f, s in zip(self.w.feet, self.start))

    def step(self):
        d = self.data
        if d.time >= self.seconds - 1e-9 or self.zmin < 0.45:
            return False
        if self.schedule is not None:
            self.mppi.cost = self.schedule(d.time)
        ctrl = self.mppi.plan(d)                                     # sample, roll out, weigh, average
        for _ in range(self.mppi.substeps):
            on = self.push and self.push_at <= d.time < self.push_at + 0.2
            d.xfrc_applied[self.w.torso, 1] = self.push if on else 0.0
            d.ctrl[:] = ctrl
            mujoco.mj_step(self.model, d)
        self.zmin = min(self.zmin, d.qpos[2]); self.zlow = min(self.zlow, d.qpos[2])
        self.tilt_max = max(self.tilt_max, self.tilt)
        up = d.site_xpos[self.w.feet[0]][2] > 0.033 + 0.05
        self.lift_run = self.lift_run + self.dt if up else 0.0
        self.lift_best = max(self.lift_best, self.lift_run)
        self.gauge = (f"t {d.time:4.2f} s   pelvis {d.qpos[2]:.3f} (lowest {self.zlow:.3f})   tilt {self.tilt:4.1f}°   "
                      f"feet moved {self.moved:.2f} m   left foot up {self.lift_run:.1f} s")
        return True

    def standing(self):
        if self.zmin < 0.5:
            return False, f"FELL (pelvis reached {self.zmin:.2f} m)"
        if self.tilt > 15.0:
            return False, f"ended tilted {self.tilt:.0f}° (must be under 15)"
        return True, "stood"

    def draw(self, scn):
        d = self.data
        if self.push and self.push_at <= d.time < self.push_at + 0.25:
            arrow(scn, d.xpos[self.w.torso] - [0, 0.5, 0], [0, 0.4, 0], (1, 0.3, 0.1, 0.9))
        for f in self.w.feet:
            sphere(scn, d.site_xpos[f], (0.2, 0.8, 1.0, 0.7), 0.03)


# --- 3. the six problems ------------------------------------------------------------------------------
#
# Each builds its attempt from the answers and says what counts as a pass.

def p_still(a):
    cost = make_cost(need(a, "STAND"), "STAND")
    att = PlanAttempt(cost, need(a, "PLANNER"), "PLANNER", seconds=3.0)
    att.verdict = lambda: ((False, f"feet moved {att.moved:.2f} m (must stay under 0.05)")
                           if att.standing()[0] and att.moved >= 0.05 else att.standing())
    return att


def p_shove(a, force):
    def make(a):
        cost = make_cost(need(a, "STAND"), "STAND")
        att = PlanAttempt(cost, need(a, "PLANNER"), "PLANNER", seconds=3.0, push=force)
        att.verdict = att.standing
        return att
    return make


def p_cheap(a):
    cost = make_cost(need(a, "STAND"), "STAND")
    att = PlanAttempt(cost, need(a, "CHEAP_PLANNER"), "CHEAP_PLANNER", seconds=3.0, push=300.0, max_samples=16)
    att.verdict = att.standing
    return att


def p_crane(a):
    weights = dict(need(a, "CRANE"))
    cost = make_cost(weights, "CRANE")
    att = PlanAttempt(cost, need(a, "PLANNER"), "PLANNER", seconds=5.0)

    def verdict():
        ok, why = att.standing()
        if not ok:
            return ok, why
        if att.lift_best < 2.0:
            return False, f"left foot up for {att.lift_best:.1f} s at most (need 2.0 in a row)"
        hop = float(np.linalg.norm(att.data.site_xpos[att.w.feet[1]][:2] - att.start[1]))
        return True, f"foot up for {att.lift_best:.1f} s; the right foot travelled {hop:.2f} m"
    att.verdict = verdict
    return att


def p_squat(a):
    down = dict(need(a, "SQUAT"))
    make_cost(down, "SQUAT")
    up = {**down, "height_target": 0.72}

    def schedule(t):            # down between 1 and 3 s, back up after (the cost looks 0.4 s ahead of now)
        return make_cost(down if 1.0 <= t < 3.0 else up, "SQUAT")
    att = PlanAttempt(schedule(0.0), need(a, "PLANNER"), "PLANNER", seconds=5.0, schedule=schedule)

    def verdict():
        ok, why = att.standing()
        if not ok:
            return ok, why
        if att.zlow > 0.58:
            return False, f"lowest pelvis {att.zlow:.3f} m (need below 0.58)"
        if att.data.qpos[2] < 0.68:
            return False, f"did not stand back up: pelvis {att.data.qpos[2]:.2f} m at the end (need 0.68)"
        return True, f"down to {att.zlow:.3f} m and back up"
    att.verdict = verdict
    return att


PROBLEMS = {
    1: Problem("STAND", "No push. Stand for 3 s without moving the feet more than 5 cm.  (STAND, PLANNER)", p_still),
    2: Problem("SHOVE", "A 150 N shove from the side at t = 1 s. Do not fall.  (STAND, PLANNER)", p_shove(None, 150.0)),
    3: Problem("BIG SHOVE", "A 300 N shove. Holding the crouch falls at 70.  (STAND, PLANNER)", p_shove(None, 300.0)),
    4: Problem("CHEAP", "300 N again, with at most 16 samples (a quarter of the thinking).  (STAND, CHEAP_PLANNER)", p_cheap),
    5: Problem("CRANE", "Left foot up (5 cm clear) for 2 s in a row; do not fall. Watch HOW.  (CRANE, PLANNER)", p_crane),
    6: Problem("SQUAT", "Pelvis below 0.58 m between 1 and 3 s, back above 0.68 by 5 s.  (SQUAT, PLANNER)", p_squat),
}


# --- 4. main ------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--costs", type=Path, default=HERE / "costs.py", help="the answers file (default: costs.py here)")
    ap.add_argument("--grade", action="store_true", help="grade headless and print the table")
    ap.add_argument("--problem", type=int, nargs="*", help="with --grade: only these problems")
    ap.add_argument("--no-viewer", action="store_true", help="never open a window")
    args = ap.parse_args(argv)
    if args.grade or args.no_viewer:
        grade_headless(PROBLEMS, args.costs, __file__, args.problem)
        return 0
    if soc4180.is_colab():
        print("This lab needs a desktop window; run it on your laptop.")
        return 1
    print(__doc__.split("\n\n")[1])
    w = world()

    def camera(viewer):
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.5, 150.0, -15.0
        viewer.cam.lookat[:] = (0.0, 0.3, 0.7)
    return run_viewer(PROBLEMS, args.costs, __file__, idle_model=w.model, idle_data=w.data, camera=camera,
                      title="week 11: planning")


if __name__ == "__main__":
    raise SystemExit(main())
