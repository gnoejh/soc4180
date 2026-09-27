"""Week 13 lab game, on your laptop: four students to train, a score out of 4.

    uv run weeks/13-imitation/lab_imitate.py          # torch + SB3: uv sync --extra rl

YOU EDIT ONLY `students.py`, next to this file. Each problem names an expert
-- the week-4 walker, or week 12's PPO policy -- and a job. You decide what
the student is SHOWN (which sensors, how much history), how many
demonstrations it learns from, and how many rounds of DAgger follow. The
referee records the demonstrations, trains the student (a 256 x 256 network,
squared error), and then lets the STUDENT drive while it counts.

    1 ... 4        train and test problem 1 ... 4 (training takes 10-60 s; the
                   window freezes while it does, then the test plays at 5x)
    SPACE          the current problem again
    G              GRADE: all four, one after another; the score stays on screen
    ENTER          print the live gauge in this terminal
    Keys go to the MuJoCo window -- click it first. If no key works there,
    press the same keys in this terminal (single keys, no enter). `q` quits.

    uv run weeks/13-imitation/lab_imitate.py --grade --no-viewer
    uv run weeks/13-imitation/lab_imitate.py --grade --problem 3

What a student may be shown (soc4180.imitation.FEATURES):

    "gravity"     3   gravity in the torso frame (IMU)
    "gyro"        3   angular velocity (IMU)
    "joints"     12   leg angles minus the crouch
    "velocities" 12   leg joint velocities
    "previous"   12   the action taken one step ago
    "clock"       3   sin and cos of the gait phase, and "walking yet?"
    "time"        1   seconds since the start, / 10

Things measured with this file's own code before it was written (--grade),
and with imitate.py:

- The walker copied from gravity + gyro + joints + velocities + clock + time
  fits its demonstrations to a mean squared error of 9e-6 and falls after
  2.3-2.6 s from every start. From clock + time alone it fits them WORSE
  (1.8e-5) and walks 1.14-1.15 m, the walker's own distance, every time.
  With "previous" added the fit is better still and it falls sooner.
- week 12's policy, copied from its own 42 numbers, 40 demonstrations:
  training MSE 8e-5, 12 of 20 at 100 N (the policy itself 16; holding the
  crouch 0). At 80 N, 20 of 20, like the policy.
- From 1 or 2 demonstrations behaviour cloning survives 0 of 20 at 80 N;
  the same demonstrations plus two rounds of DAgger survive 20 of 20. (From
  3 or more, cloning alone already manages 80 N.)
- Without "velocities", one step of history: 0 of 20 at 100 N after
  behaviour cloning, 7 after three rounds of DAgger. With five steps of
  history: 3, and 16 after three rounds -- the policy itself survives 16.

Everything in this file is complete and explained. You are not meant to edit
it: it is the referee, and the grading screen shows a 4-letter code computed
from it so the instructor can see it is unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

import soc4180
from soc4180.game import AnswersError, Attempt, Problem, grade_headless, need, number, run_viewer, sphere

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent / "12-robustness" / "checkpoints" / "push_random.zip"


def features_of(answer, where, forbid=()):
    if not isinstance(answer, list) or not answer or not all(isinstance(f, str) for f in answer):
        raise AnswersError(f"{where}: features must be a list of names, e.g. [\"gravity\", \"gyro\"]")
    from soc4180.imitation import FEATURES

    for f in answer:
        if f not in FEATURES:
            raise AnswersError(f"{where}: no feature called \"{f}\" -- features are {', '.join(FEATURES)}")
        if f in forbid:
            raise AnswersError(f"{where}: \"{f}\" is not allowed in this problem")
    return answer


# --- 1. the walker problem: record, fit, walk ------------------------------------------------------

class WalkAttempt(Attempt):
    """Ten demonstrations of the walker, a student fitted to them, then the student walks from three starts."""

    def __init__(self, features):
        super().__init__()
        from soc4180 import imitation as im
        from soc4180.walking import GaitParams

        self.im, self.features = im, features
        self.env = im._walker_env()
        self.model, self.data = self.env.model, self.env.data
        self.params = GaitParams(n_steps=40)
        self.dt = self.env.control_dt
        self.student, self.seeds, self.k, self.results = None, (100, 101, 102), -1, []
        self.gauge = "recording 10 demonstrations and training the student..."

    def _next_episode(self):
        self.k += 1
        if self.k >= len(self.seeds):
            return False
        self.obs, _ = self.env.reset(seed=self.seeds[self.k])
        self.n = 0
        return True

    def step(self):
        if self.student is None:                               # the first step does all the learning
            X, Y = self.im.walker_demos(10, self.features, env=self.env, params=self.params)
            self.student = self.im.Student.fit(X, Y, epochs=100)
            self.notes.append(f"{len(X)} examples, training MSE {self.student.train_mse:.1e}")
            print(f"   trained: {len(X)} examples, MSE {self.student.train_mse:.1e}")
            return self._next_episode()
        a = self.student.act(self.im.observe(self.obs, self.env.data.time, self.features, self.params))
        self.obs, _, term, trunc, _ = self.env.step(a)
        self.n += 1
        self.gauge = (f"start {self.k + 1} of 3: {self.env.data.time:4.1f} s, {self.env.data.qpos[0]:+.2f} m   "
                      f"(training MSE {self.student.train_mse:.1e})   done: "
                      + ", ".join(f"{x:+.2f} m" + (" fell" if s < 1000 else "") for s, x in self.results))
        if term or trunc:
            self.results.append((self.n, float(self.env.data.qpos[0])))
            return self._next_episode()
        return True

    def verdict(self):
        worst = min((x for _, x in self.results), default=0.0)
        if len(self.results) < 3:
            return False, "did not finish"
        if any(s < 1000 for s, _ in self.results):
            return False, f"fell in {sum(s < 1000 for s, _ in self.results)} of 3 walks"
        if worst < 1.0:
            return False, f"stood, but walked only {worst:.2f} m (need 1.0 in every walk)"
        return True, f"walked {worst:.2f} m or more from every start"


# --- 2. the shove problems: demonstrate, clone, DAgger, shove ---------------------------------------------

class ShoveAttempt(Attempt):
    """Demonstrations from week 12's policy, a student, optional DAgger, then 20 shoves of `push` N."""

    def __init__(self, spec, where, *, push, need_ok, forbid=(), max_demos=200):
        super().__init__()
        from stable_baselines3 import PPO

        from soc4180 import imitation as im
        from soc4180.envs import G1PushEnv

        self.features = features_of(spec.get("features"), where, forbid)
        self.history = int(number(spec, "history", where, 1, 20))
        self.demos = int(number(spec, "demos", where, 1, max_demos))
        self.rounds = int(number(spec, "dagger", where, 0, 5))
        self.im, self.push, self.need_ok = im, push, need_ok
        self.env = G1PushEnv()
        self.model, self.data = self.env.model, self.env.data
        self.teacher = PPO.load(TEACHER, device="cpu")
        self.dt = self.env.control_dt / 5                      # the test plays at 5x
        self.student, self.k, self.ok, self.done = None, -1, 0, 0
        self.gauge = (f"recording {self.demos} demonstrations"
                      + (f", then {self.rounds} rounds of DAgger" if self.rounds else "") + "...")

    def _next_episode(self):
        from soc4180.envs import NOMINAL_WORLD

        self.k += 1
        if self.k >= 20:
            return False
        self.env.world = {k: v for k, v in NOMINAL_WORLD.items()}
        obs, _ = self.env.reset(seed=5000 + self.k)
        self.env.push_force = np.array([0.0, self.push])       # sideways, exactly `push` newtons
        self.past = [obs] * self.history
        return True

    def step(self):
        im = self.im
        if self.student is None:
            X, Y = im.policy_demos(self.env, self.teacher, self.demos, self.features, history=self.history)
            if self.rounds:
                self.student = im.dagger(self.env, self.teacher, X, Y, self.features, history=self.history,
                                         iterations=self.rounds)
            else:
                self.student = im.Student.fit(X, Y)
            print(f"   trained: {len(X)} examples from {self.demos} demonstrations"
                  + (f" + {self.rounds} DAgger rounds" if self.rounds else "") + f", MSE {self.student.train_mse:.1e}")
            return self._next_episode()
        view = im._student_view(self.past, self.features, self.history)
        obs, _, term, trunc, _ = self.env.step(self.student.act(view))
        self.past = self.past[1:] + [obs]
        k, t = self.k, self.env.data.time
        more = True
        if term or trunc:
            self.done += 1
            self.ok += int(not term)
            more = self._next_episode()
        self.gauge = (f"shove {self.push:.0f} N, episode {k + 1} of 20, t {t:3.1f} s   "
                      f"survived so far {self.ok} of {self.done}   need {self.need_ok}")
        self.met = self.ok >= self.need_ok
        return more

    def draw(self, scn):
        d, e = self.data, self.env
        if e.push_time <= d.time < e.push_time + 0.3:
            sphere(scn, d.xpos[e.torso] - [0, 0.35, 0], (1, 0.3, 0.1, 0.9), 0.06)

    def verdict(self):
        if self.done < 20:
            return False, "did not finish"
        if self.ok < self.need_ok:
            return False, f"survived {self.ok} of 20 at {self.push:.0f} N (need {self.need_ok})"
        return True, f"survived {self.ok} of 20 at {self.push:.0f} N"


# --- 3. the problems ------------------------------------------------------------------------------

PROBLEMS = {
    1: Problem("TAPE", "Copy the week-4 walker. The student must walk 1.0 m in 10 s from all 3 starts.  (WALKER_FEATURES)",
               lambda a: WalkAttempt(features_of(need(a, "WALKER_FEATURES"), "WALKER_FEATURES"))),
    2: Problem("COPY", "Copy week 12's policy. Survive a 100 N shove in 10 of 20 episodes.  (COPY)",
               lambda a: ShoveAttempt(need(a, "COPY"), "COPY", push=100.0, need_ok=10)),
    3: Problem("FEW", "Copy it again from at most 2 demonstrations: 80 N, 18 of 20.  (FEW)",
               lambda a: ShoveAttempt(need(a, "FEW"), "FEW", push=80.0, need_ok=18, max_demos=2)),
    4: Problem("BLIND", "No joint velocities allowed. Survive 100 N in 14 of 20.  (BLIND)",
               lambda a: ShoveAttempt(need(a, "BLIND"), "BLIND", push=100.0, need_ok=14, forbid=("velocities",))),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--students", type=Path, default=HERE / "students.py", help="the answers file")
    ap.add_argument("--grade", action="store_true")
    ap.add_argument("--problem", type=int, nargs="*")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(argv)
    try:
        import torch
        import stable_baselines3  # noqa: F401
    except ImportError:
        print("This lab needs torch and stable-baselines3: run `uv sync --extra rl`, then try again.")
        return 1
    torch.set_num_threads(4)
    if args.grade or args.no_viewer:
        grade_headless(PROBLEMS, args.students, __file__, args.problem)
        return 0
    if soc4180.is_colab():
        print("This lab needs a desktop window; run it on your laptop.")
        return 1
    print(__doc__.split("\n\n")[1])
    model = soc4180.load_g1()
    data = soc4180.keyframe_data(model, "stand")
    return run_viewer(PROBLEMS, args.students, __file__, idle_model=model, idle_data=data, title="week 13: imitation")


if __name__ == "__main__":
    raise SystemExit(main())
