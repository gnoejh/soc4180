"""Week 10 lab, on your laptop: one model, many robots, and where the time goes.

    uv run weeks/10-scaling/lab_many.py

A window opens with N_ROBOTS copies of the G1 in one scene, all walking the
week 4 gait from one controller. There is ONE model and ONE `mj_step`; the
physics simply has more bodies in it. That is the cheapest possible version
of what a GPU does with 8192 environments, and it is enough to measure the
thing that matters: robot-steps per second.

    + / -                  one more / one fewer robot (rebuilds the scene)
    F                      randomise each robot's foot friction and mass (toggle, rebuilds)
    P                      benchmark real parallelism: 1, 2, 4, 8 processes, one
                           robot each, two seconds -- printed as robot-steps/s
    R                      restart the walk
    ENTER                  robot-steps/s measured so far, against your prediction,
                           and how far behind real time the window is
    double-click a body, then ctrl-drag    push one robot
    Keys go to the MuJoCo window, not the terminal -- click the window first.
    If no key does anything in the window, press the same key in this terminal
    instead -- single keys, no enter. `q` stops it.

`predict_rate` below is your model of the throughput of one process with n
robots in it, given the single-robot rate the script measures at start-up.
It returns None until written; ENTER then prints measured against predicted.

What to change, in order, and show the instructor:

1. Press ENTER with the default N. Write down robot-steps/s and "x real time".
   Press + three times, ENTER each time. Does the total throughput of one
   process grow with N, stay flat, or shrink? Write `predict_rate` to match
   what you think, then check.
2. Press P. Read the four numbers. At how many processes does it stop
   scaling, and how many cores does this laptop have (`os.cpu_count()`)?
3. Take your best rate from step 2. How many hours is 150 million steps?
   How many laptops would you need to do it in one lecture? (The deck's
   answer is a different kind of processor, not more laptops.)
4. Press F. The robots now differ slightly in friction and mass and drift
   apart within a few steps. This is week 11 in one key: say what a policy
   trained on all of them at once would have to learn that one robot cannot
   teach it.
5. Bring N up until the window drops below 0.25x real time. Say what an
   "environment" costs, in milliseconds, on your machine.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time

import mujoco
import numpy as np

import soc4180

N_ROBOTS = 4
SPACING = 0.7          # metres between robots, sideways
SEED = 0


def predict_rate(n: int, single: float) -> float | None:
    """Predicted robot-steps per second for one process stepping n robots at once.

    `single` is the measured rate for one robot alone. Return None until written.
    """
    return None


# --- nothing below needs editing ------------------------------------------

def build(n, randomise):
    """One scene with n G1s side by side, and the index arrays to drive them."""
    robot = str(soc4180.robot_path("unitree_g1", "g1"))
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
        frame.attach_body(child.worldbody.first_body(), f"r{k}_", "")
    model = base.compile()
    one = soc4180.load_g1()
    rng = np.random.default_rng(SEED)
    if randomise:
        for k in range(n):
            scale = rng.uniform(0.8, 1.2)
            mu = rng.uniform(0.3, 1.0)
            for b in range(model.nbody):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                if name.startswith(f"r{k}_"):
                    model.body_mass[b] *= scale
                    model.body_inertia[b] *= scale
            for g in range(model.ngeom):
                b = model.geom_bodyid[g]
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                if name.startswith(f"r{k}_") and name.endswith("ankle_roll_link"):
                    model.geom_friction[g, 0] = mu
    return model, one


class Scene:
    def __init__(self, n, randomise):
        self.n = n
        self.model, self.one = build(n, randomise)
        self.data = mujoco.MjData(self.model)
        self.controller = soc4180.WalkingController(self.one, soc4180.GaitParams(n_steps=12))
        self.nq, self.nu = self.one.nq, self.one.nu
        self.act_q = self.one.jnt_qposadr[self.one.actuator_trnid[:, 0]]
        self.reset()

    def reset(self):
        d, fresh = self.data, self.controller.initial_data()
        mujoco.mj_resetData(self.model, d)
        for k in range(self.n):
            d.qpos[k * self.nq:(k + 1) * self.nq] = fresh.qpos
            d.qpos[k * self.nq + 1] += (k - (self.n - 1) / 2) * SPACING     # sideways offset
        mujoco.mj_forward(self.model, d)
        self.controller = soc4180.WalkingController(self.one, soc4180.GaitParams(n_steps=12))
        self.steps, self.t0 = 0, time.time()

    def step(self):
        c, d = self.controller, self.data
        ctrl = c.control(min(d.time, c.total_time - 1e-3))
        d.ctrl[:] = np.tile(ctrl, self.n)
        mujoco.mj_step(self.model, d)
        self.steps += 1

    def rate(self):
        return self.steps * self.n / max(time.time() - self.t0, 1e-6)


def _bench_one(seconds):
    """Step one G1 alone for `seconds`; return physics steps done. (Runs in a worker.)"""
    m = soc4180.load_g1(); d = soc4180.keyframe_data(m, "stand")
    t0, k = time.time(), 0
    while time.time() - t0 < seconds:
        mujoco.mj_step(m, d); k += 1
    return k


def single_rate(seconds=1.0):
    return _bench_one(seconds) / seconds


def parallel_benchmark(seconds=2.0):
    print(f"\n  P: {os.cpu_count()} CPU cores reported. One robot per process:")
    for n in (1, 2, 4, 8):
        with mp.get_context("spawn").Pool(n) as pool:
            t0 = time.time()
            counts = pool.map(_bench_one, [seconds] * n)
            wall = time.time() - t0
        print(f"    {n} processes: {sum(counts) / seconds:8.0f} robot-steps/s   (wall {wall:.1f} s incl. start-up)")


def main() -> int:
    if soc4180.is_colab():
        print("This lab needs a desktop window; run it on your laptop.")
        return 1

    single = single_rate()
    print(f"one robot alone: {single:,.0f} physics steps/s on this machine")
    state = {"n": N_ROBOTS, "random": False, "rebuild": False, "reset": False,
             "print": False, "bench": False}

    def on_key(keycode):
        if keycode in (43, 61):                              # '+' or '='
            state["n"] = min(state["n"] + 1, 32); state["rebuild"] = True
        elif keycode == 45:                                  # '-'
            state["n"] = max(state["n"] - 1, 1); state["rebuild"] = True
        elif keycode == 70:                                  # 'F'
            state["random"] = not state["random"]; state["rebuild"] = True
        elif keycode == 80:                                  # 'P'
            state["bench"] = True
        elif keycode == 82:
            state["reset"] = True
        elif keycode in (257, 335):
            state["print"] = True

    def report(scene):
        measured = scene.rate()
        pred = predict_rate(scene.n, single)
        sim_per_wall = scene.steps * scene.model.opt.timestep / max(time.time() - scene.t0, 1e-6)
        print(f"  {scene.n} robots in one process: {measured:8,.0f} robot-steps/s measured"
              + ("   (predict_rate not written)" if pred is None else f"   vs {pred:8,.0f} predicted")
              + f"   window at {sim_per_wall:.2f}x real time")
        print(f"  150M steps at this rate: {150e6 / measured / 3600:6.1f} hours"
              f"   nq = {scene.model.nq}, nbody = {scene.model.nbody}, contacts now {scene.data.ncon}")

    print("\n\n".join(__doc__.split("\n\n")[2:4]))
    soc4180.terminal_keys(on_key)
    print("  [terminal] keys dead in the window? press them here instead; 'q' stops.",
          flush=True)
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    while time.time() < deadline:
        scene = Scene(state["n"], state["random"])
        print(f"\n[{scene.n} robots{', randomised' if state['random'] else ''}]  "
              f"nq = {scene.model.nq}, nu = {scene.model.nu}")
        state["rebuild"] = False
        closed = True
        with soc4180.launch_viewer(scene.model, scene.data, passive=True, key_callback=on_key) as viewer:
            while viewer.is_running() and time.time() < deadline:
                wall = time.time()
                if state["rebuild"]:
                    closed = False
                    break
                if state["reset"]:
                    state["reset"] = False; scene.reset()
                if state["bench"]:
                    state["bench"] = False; parallel_benchmark(); scene.t0 = time.time(); scene.steps = 0
                scene.step()
                if state["print"]:
                    state["print"] = False; report(scene)
                viewer.sync()
                lag = scene.model.opt.timestep - (time.time() - wall)
                if lag > 0:
                    time.sleep(lag)
            if closed:
                return 0
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
