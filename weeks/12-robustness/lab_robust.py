"""Week 12 lab game, on your laptop: a robustness audit in five questions, a score out of 5.

    uv run weeks/12-robustness/lab_robust.py          # torch + SB3: uv sync --extra rl

YOU EDIT ONLY `audit.py`, next to this file. You are auditing three
controllers for the shove task -- holding the crouch, week 12's PPO policy
trained in the nominal world, and the one trained on randomised worlds -- in
worlds that differ from the one they trained in. For each question you write
down a NUMBER you measured, or a CHOICE you would defend, and the referee
re-measures it in front of you: ten episodes per shove size, each a sideways
shove of exactly that many newtons for 0.2 s at a random moment, played at
3x speed with the shove drawn as an arrow.

A claimed limit L passes if the controller survives at least 7 of 10
episodes at L, and fewer than 7 of 10 at L + 20: honest, and not sold
short by more than one step of 10 N.

    1 ... 5        check question 1 ... 5 live
    SPACE          the current question again
    G              GRADE: all five; the score stays on screen
    ENTER          print the live gauge in this terminal
    Keys go to the MuJoCo window -- click it first. If no key works there,
    press the same keys in this terminal (single keys, no enter). `q` quits.

    uv run weeks/12-robustness/lab_robust.py --grade --no-viewer

HOW TO MEASURE: robust.py is the instrument --
    uv run weeks/12-robustness/robust.py --policy nominal --no-viewer
prints the survival grid for five worlds and shoves of 0-120 N. Its first
columns answer the first questions; for the rest, change its PUSHES and
WORLDS, or write a loop around its `survived` function.

Things measured with robust.py before this file was written (10 episodes each):

- nominal world: hold survives 60 N (10/10) and falls at 70 (0/10); PPO
  nominal 80 (10/10, 2/10 at 90); PPO random 100 (9/10, 4/10 at 110).
- heavy world (+15 kg torso): hold falls with no shove at all; PPO nominal
  80 (8/10); PPO random 120 (10/10, 5/10 at 130).
- 20 worlds drawn from friction 0.2-1.0, torso -5..+15 kg, stiffness x0.6-1.5,
  one 80 N shove each: hold 2 of 20, PPO nominal 14, PPO random 18.

Everything in this file is complete and explained. You are not meant to edit
it: it is the referee, and the grading screen shows a 4-letter code computed
from it so the instructor can see it is unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import soc4180
from soc4180.game import AnswersError, Attempt, Problem, arrow, grade_headless, need, number, run_viewer

HERE = Path(__file__).resolve().parent
NOMINAL = dict(friction=1.0, mass=0.0, kp=1.0)
HEAVY = dict(friction=1.0, mass=15.0, kp=1.0)
_POLICIES = {}


def policy(name):
    """'hold', 'nominal' or 'random' -> a function obs -> action (checkpoints loaded once)."""
    if name not in ("hold", "nominal", "random"):
        raise AnswersError(f"no controller called {name!r}: choose \"hold\", \"nominal\" or \"random\"")
    if name not in _POLICIES:
        if name == "hold":
            _POLICIES[name] = lambda obs: np.zeros(12, np.float32)
        else:
            from stable_baselines3 import PPO

            agent = PPO.load(HERE / "checkpoints" / f"push_{name}.zip", device="cpu")
            _POLICIES[name] = lambda obs, a=agent: a.predict(obs, deterministic=True)[0]
    return _POLICIES[name]


class Trials(Attempt):
    """A list of (world, shove, seed) episodes, played one decision at a time; counts survivors per group."""

    def __init__(self, act, groups, *, verdict):
        """groups: [(label, world, shove N, episodes, seed0)]"""
        super().__init__()
        from soc4180.envs import G1PushEnv

        self.env = G1PushEnv()
        self.model, self.data = self.env.model, self.env.data
        self.dt = self.env.control_dt / 3                      # 3x speed
        self.act, self.groups, self._verdict = act, groups, verdict
        self.queue = [(g, k) for g in range(len(groups)) for k in range(groups[g][3])]
        self.ok = [0] * len(groups)
        self.done = [0] * len(groups)
        self.current = None

    def _start(self):
        if not self.queue:
            return False
        g, k = self.queue.pop(0)
        label, world, shove, _, seed0 = self.groups[g]
        self.env.world = {key: (v, v) for key, v in world.items()}
        self.obs, _ = self.env.reset(seed=seed0 + k)
        self.env.push_force = np.array([0.0, float(shove)])    # sideways, exactly `shove` newtons
        self.current = g
        return True

    def step(self):
        if self.current is None and not self._start():
            return False
        self.obs, _, term, trunc, _ = self.env.step(self.act(self.obs))
        g = self.current
        if term or trunc:
            self.done[g] += 1
            self.ok[g] += int(not term)
            self.current = None
        if len(self.groups) > 4:                               # many one-episode groups: one total
            self.gauge = f"survived {sum(self.ok)} of {sum(self.done)} so far, of {len(self.groups)}"
        else:
            self.gauge = "   ".join(f"{lab}: {self.ok[i]}/{self.done[i]}" for i, (lab, *_rest) in enumerate(self.groups)
                                   if self.done[i] or i == g)
        return True

    def draw(self, scn):
        e, d = self.env, self.data
        if e.push_time <= d.time < e.push_time + 0.3:
            arrow(scn, d.xpos[e.torso] - [0, 0.6, 0], [0, 0.45, 0], (1, 0.3, 0.1, 0.9), radius=0.02)

    def verdict(self):
        return self._verdict(self.ok, self.done)


def limit_check(act, world, claim, where):
    """Is `claim` an honest limit? >= 7/10 at the claim, < 7/10 at claim + 20."""
    if claim % 10 or not 0 <= claim <= 300:
        raise AnswersError(f"{where}: give a multiple of 10 N between 0 and 300, not {claim}")

    def verdict(ok, done):
        if ok[0] < 7:
            return False, f"survived only {ok[0]}/10 at {claim:.0f} N: that is more than it can take"
        if ok[1] >= 7:
            return False, f"survived {ok[1]}/10 at {claim + 20:.0f} N too: you sold it short"
        return True, f"{ok[0]}/10 at {claim:.0f} N, {ok[1]}/10 at {claim + 20:.0f} N"
    return Trials(act, [(f"{claim:.0f} N", world, claim, 10, 0), (f"{claim + 20:.0f} N", world, claim + 20, 10, 0)],
                  verdict=verdict)


def q_limit(name, world, key):
    def make(a):
        claim = number({key: need(a, key)}, key, key, 0, 300)
        return limit_check(policy(name), world, claim, key)
    return make


def q_heavy(a):
    spec = need(a, "HEAVY")
    if not isinstance(spec, dict) or "policy" not in spec:
        raise AnswersError("HEAVY must be a dict like {\"policy\": \"hold\", \"limit\": 50}")
    claim = number(spec, "limit", "HEAVY", 0, 300)
    att = limit_check(policy(spec["policy"]), HEAVY, claim, "HEAVY")
    inner = att._verdict

    def verdict(ok, done):
        good, why = inner(ok, done)
        if good and claim < 100:
            return False, f"{why} -- honest, but another controller takes 100 N or more here"
        return good, why
    att._verdict = verdict
    return att


def q_ship(a):
    spec = need(a, "SHIP")
    if not isinstance(spec, dict) or "policy" not in spec:
        raise AnswersError("SHIP must be a dict like {\"policy\": \"hold\", \"promise\": 50}")
    promise = number(spec, "promise", "SHIP", 0, 300)
    if promise < 80:
        raise AnswersError("SHIP: the customer needs a promise of at least 80 N")
    rng = np.random.default_rng(7)                       # the same 20 unknown robots for everyone
    worlds = [dict(friction=rng.uniform(0.2, 1.0), mass=rng.uniform(-5, 15), kp=rng.uniform(0.6, 1.5))
              for _ in range(20)]
    groups = [(f"robot {i + 1}", w, promise, 1, 100 + i) for i, w in enumerate(worlds)]

    def verdict(ok, done):
        n = sum(ok)
        return (n >= 16, f"{n} of 20 unknown robots survived {promise:.0f} N (need 16)")
    att = Trials(policy(spec["policy"]), groups, verdict=verdict)
    return att


PROBLEMS = {
    1: Problem("HOLD", "The largest shove holding the crouch survives, nominal world.  (HOLD_LIMIT)",
               q_limit("hold", NOMINAL, "HOLD_LIMIT")),
    2: Problem("NOMINAL", "The same for PPO trained in the nominal world.  (NOMINAL_LIMIT)",
               q_limit("nominal", NOMINAL, "NOMINAL_LIMIT")),
    3: Problem("RANDOM", "The same for PPO trained on randomised worlds.  (RANDOM_LIMIT)",
               q_limit("random", NOMINAL, "RANDOM_LIMIT")),
    4: Problem("HEAVY", "Torso +15 kg: the best controller, and its limit (100 N or more).  (HEAVY)", q_heavy),
    5: Problem("SHIP", "20 robots, mass/friction/motors unknown: what do you ship, promising 80 N or more?  (SHIP)",
               q_ship),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--audit", type=Path, default=HERE / "audit.py", help="the answers file")
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
    torch.set_num_threads(1)
    if args.grade or args.no_viewer:
        grade_headless(PROBLEMS, args.audit, __file__, args.problem)
        return 0
    if soc4180.is_colab():
        print("This lab needs a desktop window; run it on your laptop.")
        return 1
    print(__doc__.split("\n\n")[1])
    from soc4180.envs import G1PushEnv
    env = G1PushEnv()
    env.reset(seed=0)
    return run_viewer(PROBLEMS, args.audit, __file__, idle_model=env.model, idle_data=env.data,
                      title="week 12: robustness")


if __name__ == "__main__":
    raise SystemExit(main())
