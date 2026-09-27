"""Copy an expert: record demonstrations, train a student by regression, and measure whether it can do the job -- printed, then replayed.

    uv run weeks/13-imitation/imitate.py                                   # the walker, copied from sensors
    uv run weeks/13-imitation/imitate.py --features clock time             # ... from the clock alone
    uv run weeks/13-imitation/imitate.py --expert policy                   # week 12's PPO policy, all 42 numbers
    uv run weeks/13-imitation/imitate.py --expert policy --features gravity gyro joints previous --history 5 --dagger 3
    uv run weeks/13-imitation/imitate.py --expert planner                  # week 11's MPPI (slow: it plans every label)
    uv run weeks/13-imitation/imitate.py --no-viewer

Three experts, one recipe. The expert acts; every step becomes an example
(what the student is shown -> what the expert did). A network is fitted by
squared error -- behaviour cloning. Then the student drives, and we count.
With --dagger N the student drives N more rounds and the expert labels the
states the STUDENT visits (DAgger).

    --expert walker|policy|planner
        walker   the week-4 LIPM walker, at 100 Hz: 10 demonstrations of a 10 s walk
        policy   week 12's PPO policy trained on randomised worlds (checkpoints/ in week 12)
        planner  week 11's MPPI, 64 samples (about 12 s of planning per demonstration)
    --features ...   what the student sees: gravity gyro joints velocities previous clock time
                     (default: walker -> gravity gyro joints velocities clock time;
                               policy, planner -> gravity gyro joints velocities previous)
    --history 1      how many past steps the student sees at once
    --demos N        demonstrations (default 10 walker, 40 policy, 8 planner)
    --dagger 0       DAgger rounds after behaviour cloning
    --push 0         walker only: shove the student sideways with this many N at t = 3 s
    --epochs 30      --seed 0   --no-viewer   --speed 1

Six numbered steps; read them with the week 13 slides open.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

import soc4180

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE.parent / "12-robustness" / "checkpoints" / "push_random.zip"


class PlannerExpert:
    """Week 11's MPPI with an SB3-style predict(): plans from the env's current state, returns env actions."""

    def __init__(self, env, samples=64):
        from soc4180 import planning

        self.env = env
        self.model = planning.planning_model()                 # same robot + sensors; same state layout
        q0, ctrl0 = planning.crouch(self.model)
        self.acts = planning.actuators_for(self.model)
        self.mppi = planning.MPPI(self.model, ctrl0, self.acts, samples=samples)
        self.ctrl0 = ctrl0

    def predict(self, obs, deterministic=True):
        if self.env.data.time < 1e-9:
            self.mppi.reset()                                    # a new episode: forget the old plan
        ctrl = self.mppi.plan(self.env.data)
        residual = (ctrl[self.acts] - self.ctrl0[self.acts]) / self.env.action_scale
        return np.clip(residual, -1.0, 1.0).astype(np.float32), None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--expert", choices=("walker", "policy", "planner"), default="walker")
    ap.add_argument("--features", nargs="+", default=None)
    ap.add_argument("--history", type=int, default=1)
    ap.add_argument("--demos", type=int, default=None)
    ap.add_argument("--dagger", type=int, default=0)
    ap.add_argument("--push", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(argv)

    try:
        import torch
        from stable_baselines3 import PPO
        from soc4180 import imitation as im
        from soc4180.envs import G1PushEnv
    except ImportError:
        print("This script needs torch and stable-baselines3: run `uv sync --extra rl`, then try again.")
        return 1
    torch.set_num_threads(4)
    walker = args.expert == "walker"
    features = args.features or (["gravity", "gyro", "joints", "velocities", "clock", "time"] if walker
                                 else ["gravity", "gyro", "joints", "velocities", "previous"])
    frames, env = [], None

    # -- 1. the expert ---------------------------------------------------------------------------------
    if walker:
        from soc4180.walking import GaitParams
        env = im._walker_env()
        params = GaitParams(n_steps=40)
        print("expert: the week-4 walker (LIPM + IK, open loop: its action depends only on the time), 100 Hz")
    else:
        env = G1PushEnv(control_hz=25.0, action_scale=1.0) if args.expert == "planner" else G1PushEnv()
        if args.expert == "planner":
            expert = PlannerExpert(env)
            print("expert: week 11's MPPI, 64 samples x 0.4 s, 25 decisions/s (every label is a plan)")
        else:
            expert = PPO.load(CHECKPOINT, device="cpu")
            print(f"expert: week 12's PPO policy ({CHECKPOINT.name}), 50 Hz")
    print(f"student sees: {' + '.join(features)}, history {args.history}")

    # -- 2. demonstrations: the expert drives, every step is an example --------------------------------------
    n = args.demos or (10 if walker else (8 if args.expert == "planner" else 40))
    t0 = time.time()
    if walker:
        X, Y = im.walker_demos(n, features, seed=args.seed, env=env, params=params)
    else:
        X, Y = im.policy_demos(env, expert, n, features, history=args.history, seed=args.seed)
    print(f"\n{n} demonstrations: {len(X):,} examples, {X.shape[1]} inputs -> {Y.shape[1]} actions ({time.time() - t0:.0f} s)")

    # -- 3. behaviour cloning (and DAgger) ------------------------------------------------------------------
    epochs = args.epochs or (100 if walker else 30)
    t0 = time.time()
    if walker or args.dagger == 0:
        student = im.Student.fit(X, Y, epochs=epochs)
        print(f"behaviour cloning, {epochs} epochs: training error (MSE) {student.train_mse:.2e} ({time.time() - t0:.0f} s)")
    else:
        def log(it, size, s):
            print(f"   DAgger round {it}: {size:,} examples, training MSE {s.train_mse:.2e}, "
                  f"survives 100 N {im.shove_test(env, s, features, history=args.history)}/20", flush=True)
        base = im.Student.fit(X, Y, epochs=epochs)
        print(f"behaviour cloning: MSE {base.train_mse:.2e}, survives 100 N "
              f"{im.shove_test(env, base, features, history=args.history)}/20")
        student = im.dagger(env, expert, X, Y, features, history=args.history, iterations=args.dagger,
                            epochs=epochs, log=log)

    # -- 4. the test: can the student do the job? ---------------------------------------------------------------
    print()
    if walker:
        results, frames = im.walk(student, features, env=env, params=params, record=True, push=args.push)
        if args.push:
            print(f"(a {args.push:g} N sideways shove at t = 3 s)")
        for (steps, x), s in zip(results, (100, 101, 102)):
            print(f"student, start {s}: {steps:4d} of 1000 decisions, {x:+.2f} m" + ("  FELL" if steps < 1000 else ""))
        print("(the walker itself: 1000 decisions and about 1.1 m from these starts)")
    else:
        pushes = (60, 80, 100, 120)
        print(f"{'episodes of 20 survived, sideways shove':42s}" + "".join(f"{p:>6} N" for p in pushes))
        if args.expert != "planner":
            print(f"{'  expert':42s}" + "".join(f"{im.shove_test(env, expert, push=p):8d}" for p in pushes))
        print(f"{'  student':42s}" + "".join(f"{im.shove_test(env, student, features, history=args.history, push=p):8d}"
                                             for p in pushes))
        print(f"{'  hold the crouch':42s}" + "".join(f"{im.shove_test(env, _Hold(), push=p):8d}" for p in pushes))
        im.shove_test(env, student, features, history=args.history, push=80, episodes=1, frames=frames)

    # -- 5. what it means -------------------------------------------------------------------------------------------
    print("\na low training error says the student copies the expert on the expert's own states. Whether it can"
          " do the job depends on the states it reaches itself, and on whether the expert's action was a"
          " function of what the student can see.")

    if args.no_viewer or soc4180.is_colab():
        return 0

    # -- 6. replay --------------------------------------------------------------------------------------------------
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    print(f"\nreplaying the student's first test episode at {args.speed:g}x (close the window to stop)")
    with soc4180.launch_viewer(env.model, env.data, passive=True) as viewer:
        while viewer.is_running() and time.time() < deadline:
            for q in frames:
                if not viewer.is_running() or time.time() > deadline:
                    break
                env.data.qpos[:] = q; mujoco.mj_forward(env.model, env.data); viewer.sync()
                time.sleep(env.control_dt / args.speed)
            time.sleep(0.5)
    return 0


class _Hold:
    """Zero action, with an SB3-style predict(): holding the crouch, the baseline."""

    def predict(self, obs, deterministic=True):
        return np.zeros(12, np.float32), None


if __name__ == "__main__":
    sys.exit(main())
