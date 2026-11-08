"""Week 7b lab game, on your laptop: six problems, one learning rule, a score out of 6.

    uv run --extra rl weeks/07b-tabular/lab_tables.py

YOU EDIT ONLY `tables.py`, next to this file. Every problem is solved by the
same few lines -- a table Q(s, a), nudged toward each sample:

    Q[s, a] += alpha * (target - Q[s, a])
    target   = r + gamma * max_a' Q[s', a']      Q-learning
    target   = r + gamma * Q[s', a']             SARSA (a' = the action it WILL take)

(soc4180.tabular, the lecture's code). You never write the update. You choose
what it is told: how far it looks ahead, which update it uses, what the table
can see, how big the table is. Each problem trains a table live -- watch the
arrows change -- then plays what it learned and grades it.

    1 ... 6        play problem 1 ... 6 (training is live: arrows = the table's best action)
    SPACE          play the current problem again
    G              GRADE: all six, one after another; the score stays on screen
    ENTER          print the live gauge in this terminal
    left-drag / right-drag / wheel   orbit / pan / zoom
    ctrl + right-drag on the pole (problem 5, after training)   push it
    Keys go to the MuJoCo window -- click it first. If no key works there,
    press the same keys in this terminal (single keys, no enter). `q` quits.

    uv run --extra rl weeks/07b-tabular/lab_tables.py --grade --no-viewer      # grade headless
    uv run --extra rl weeks/07b-tabular/lab_tables.py --grade --problem 5 6    # just these

tables.py is re-read every time a problem starts: save it and press the
number again, no restart.

Things measured with this file's own code before it was written (--grade):

- FAR-SIGHTED: gamma 0.99 and 1.0 reach the goal 0.737 with all three
  tables (value iteration's pi* also 0.737). gamma 0.95: 0.607 / 0.640 /
  0.734; 0.9: 0.730 / 0.506 / 0.681; 0.5: about 0.07 -- yet value iteration
  at 0.9 still finds a 0.72 policy. Samples need the routes' values far
  apart, and a low gamma squeezes them together.
- WATCHER: acting at random for 5000 episodes, Q-learning's greedy table
  reaches the goal 0.730 / 0.747 / 0.646; SARSA's 0.308 / 0.607 / 0.128.
- SAFE / SHORT (cliff, epsilon 0.1, 500 episodes): SARSA falls in 11 / 7 /
  14 of the last 200 episodes, Q-learning in 51 / 49 / 47. Q-learning's
  greedy path is 13 steps every time; SARSA's 17 / 17 / 19.
- PENDULUM: angle (6 bins) + angular velocity (6 bins), 36 rows: the best
  checked table (episode 2750) balances 1000 steps from all 10 test starts.
  Angle only (6 rows): 21-35 steps. All four numbers (324 rows): 43-77.
  The angle cut at 0 alone (2 bins) + 6 velocity bins also passes (12 rows).
  With these cut points seed 0 reaches 1000 at episode 2750 and keeps it;
  seeds 1 and 2 reach 1000 and lose it again by episode 3000 -- a coarse
  table forgets, which is why the best checked table is kept.
- G1 BALANCE (J over the second half of 99,840 episodes; hold 12.66, value
  iteration 72.12): 5 bins 19.1, 11 bins 51.1, 21 bins 65.5, 41 bins 66.1,
  81 bins 53.3. Too coarse never settles; too fine is visited too rarely.
- Grading all six takes about 2 minutes; problem 6 alone about a minute.
  Every world here is pure numpy or MuJoCo's deterministic stepping, so a
  grade repeats exactly on one machine.

Everything in this file is complete and explained. You are not meant to edit
it: it is the referee, and the grading screen shows a 4-letter code computed
from it so the instructor can see it is unchanged.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

import soc4180
from soc4180 import tabular
from soc4180.game import AnswersError, Attempt, Problem, arrow, grade_headless, need, number, run_viewer, sphere

HERE = Path(__file__).resolve().parent
VIEW = {"lookat": (0.75, -0.75, 0.0), "distance": 4.0, "azimuth": 90.0, "elevation": -89.0}
BUDGET = 0.03     # seconds of training per viewer step: the window stays live while tables learn


# --- 1. the worlds, drawn as MuJoCo scenes --------------------------------------------------------------
#
# FrozenLake (4 x 4, slippery: the agent moves where it meant only 1 time in 3)
# and CliffWalking (4 x 12) are gymnasium's toy-text worlds: a state is a tile
# number, row * columns + column. MuJoCo only DRAWS them here -- a tile per
# cell and a ball for the agent; the arrows are the table's best action.

TILE = 0.5
COLOURS = {"S": "0.95 0.85 0.3 1", "F": "0.75 0.88 0.97 1", "H": "0.15 0.2 0.3 1", "G": "0.3 0.75 0.4 1",
           "C": "0.75 0.25 0.2 1", ".": "0.85 0.85 0.85 1"}
FROZEN_MOVES = {0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1)}       # left, down, right, up  (dx, dy)
CLIFF_MOVES = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}        # up, right, down, left


def grid_scene(rows):
    """An MJCF with one tile per character of ``rows`` and a mocap ball called 'agent'."""
    geoms = []
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            geoms.append(f'<geom type="box" size="{TILE * 0.47} {TILE * 0.47} 0.01" '
                         f'pos="{c * TILE} {-r * TILE} 0" rgba="{COLOURS[ch]}"/>')
    return f"""<mujoco>
  <visual><headlight ambient="0.6 0.6 0.6" diffuse="0.4 0.4 0.4"/></visual>
  <worldbody>
    {''.join(geoms)}
    <body name="agent" mocap="true" pos="0 0 0.12">
      <geom type="sphere" size="0.11" rgba="0.91 0.45 0.23 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>"""


FROZEN_ROWS = ["SFFF", "FHFH", "FFFH", "HFFG"]
CLIFF_ROWS = ["." * 12, "." * 12, "." * 12, "S" + "C" * 10 + "G"]


class GridAttempt(Attempt):
    """Train one or more tables (one per seed), then play the first one's greedy policy, then grade.

    ``learners`` are soc4180.tabular.Learner objects; ``episodes`` each;
    ``judge(learners) -> (ok, why, gauge)`` grades them when training ends.
    """

    dt = 0.05

    def __init__(self, rows, moves, learners, episodes, judge, terminal_tiles):
        super().__init__()
        self.rows, self.moves, self.learners, self.episodes, self.judge = rows, moves, learners, episodes, judge
        self.terminal = set(terminal_tiles)
        self.cols = len(rows[0])
        self.model = mujoco.MjModel.from_xml_string(grid_scene(rows))
        self.data = mujoco.MjData(self.model)
        self.k, self.result, self.path, self.t = 0, None, [], 0
        VIEW.update(lookat=((self.cols - 1) * TILE / 2, -(len(rows) - 1) * TILE / 2, 0.0),
                    distance=1.25 * self.cols * TILE + 1.5, azimuth=90.0, elevation=-89.0)
        self.place(0)
        self.gauge = "training ..."

    def place(self, s):
        r, c = divmod(int(s), self.cols)
        self.data.mocap_pos[0] = (c * TILE, -r * TILE, 0.12)
        mujoco.mj_forward(self.model, self.data)       # mocap moved: recompute before drawing

    def step(self):
        if self.result is None:                        # 1. training: episodes for up to 30 ms per frame
            L = self.learners[self.k]
            t0 = time.perf_counter()
            while L.episodes < self.episodes and time.perf_counter() - t0 < BUDGET:
                L.episode()
            self.gauge = (f"training table {self.k + 1} of {len(self.learners)}: episode {L.episodes} / {self.episodes}"
                          f"   epsilon {L.epsilon:.2f}   mean return of the last 100 {np.mean(L.returns[-100:]):+.2f}")
            if L.episodes >= self.episodes:
                self.k += 1
                if self.k == len(self.learners):
                    self.result = self.judge(self.learners)
                    self.met = self.result[0]
                    self.gauge = self.result[2]
                    self.path = self.greedy_path(self.learners[0])
            return True
        if self.t < 6 * len(self.path):                # 2. then play the greedy path: one move per 0.3 s
            if self.t % 6 == 0:
                self.place(self.path[self.t // 6])
            self.t += 1
            return True
        return False

    def greedy_path(self, L, limit=40):
        """The first table's greedy policy, played once (on the slippery lake, wherever the ice sends it)."""
        s, _ = L.env.reset(seed=12345)
        path = [s]
        for _ in range(limit):
            s, r, te, tr, _ = L.env.step(int(np.argmax(L.Q[s])))
            path.append(s)
            if te or tr:
                break
        return path

    def verdict(self):
        return self.result[0], self.result[1]

    def draw(self, scn):
        Q = self.learners[min(self.k, len(self.learners) - 1)].Q
        for s in range(Q.shape[0]):
            if s in self.terminal or not Q[s].any():
                continue                               # nothing learned there yet: no arrow
            r, c = divmod(s, self.cols)
            dx, dy = self.moves[int(np.argmax(Q[s]))]
            arrow(scn, (c * TILE - 0.12 * dx, -r * TILE - 0.12 * dy, 0.03), (0.24 * dx, 0.24 * dy, 0),
                  (0.1, 0.1, 0.1, 1), radius=0.03)


def frozen_success(L, n=2000):
    return float(tabular.run_greedy(L.env, L.Q, n, 10**6)[0].mean())


FROZEN_STAR = None


def frozen_star():
    """pi*'s success rate on the lake, from value iteration (week 7's method, using P)."""
    global FROZEN_STAR
    if FROZEN_STAR is None:
        env = gym.make("FrozenLake-v1", is_slippery=True)
        _, Q, _ = tabular.value_iteration(env.unwrapped.P, 16, 4, 0.99)
        FROZEN_STAR = float(tabular.run_greedy(env, Q, 2000, 10**6)[0].mean())
    return FROZEN_STAR


def frozen_learners(gamma, algo, epsilon, eps_decay, eps_min, seeds=(0, 1, 2)):
    return [tabular.Learner(gym.make("FrozenLake-v1", is_slippery=True), 16, 4, algo=algo, alpha=0.05,
                            gamma=gamma, epsilon=epsilon, eps_decay=eps_decay, eps_min=eps_min, seed=s)
            for s in seeds]


def algo_of(d, where):
    a = d.get("algo")
    if a not in ("q", "sarsa"):
        raise AnswersError(f"{where}: \"algo\" must be \"q\" or \"sarsa\", not {a!r}")
    return a


# --- 2. the problems ---------------------------------------------------------------------------------------

def p_far(answers):
    """1. FrozenLake, Q-learning, 5000 episodes; you choose gamma. Graded: mean success of 3 tables >= 0.70."""
    d = need(answers, "FROZEN")
    gamma = number(d, "gamma", "FROZEN", 0.0, 1.0)

    def judge(Ls):
        s = [frozen_success(L) for L in Ls]
        ok = np.mean(s) >= 0.70
        return ok, (f"reaches the goal {np.mean(s):.0%} of the time" + ("" if ok else ", needs 70 %")), \
            f"success {' / '.join(f'{x:.3f}' for x in s)}   (pi* by value iteration: {frozen_star():.3f})"
    return GridAttempt(FROZEN_ROWS, FROZEN_MOVES, frozen_learners(gamma, "q", 1.0, 0.9995, 0.05), 5000, judge,
                       {5, 7, 11, 12, 15})


def p_watcher(answers):
    """2. FrozenLake, acting at RANDOM all along (epsilon 1); you choose the update. Graded: greedy table >= 0.60."""
    algo = algo_of(need(answers, "RANDOM"), "RANDOM")

    def judge(Ls):
        s = [frozen_success(L) for L in Ls]
        ok = np.mean(s) >= 0.60
        return ok, (f"its greedy table reaches the goal {np.mean(s):.0%}" + ("" if ok else ", needs 60 %")), \
            f"greedy success {' / '.join(f'{x:.3f}' for x in s)}   (it acted at random all along)"
    return GridAttempt(FROZEN_ROWS, FROZEN_MOVES, frozen_learners(0.99, algo, 1.0, 1.0, 1.0), 5000, judge,
                       {5, 7, 11, 12, 15})


def cliff_learners(algo, seeds=(0, 1, 2)):
    return [tabular.Learner(gym.make("CliffWalking-v1"), 48, 4, algo=algo, alpha=0.5, gamma=1.0, epsilon=0.1,
                            seed=s, max_steps=1000) for s in seeds]


def p_safe(answers):
    """3. CliffWalking, epsilon 0.1, 500 episodes: fall off the cliff in fewer than 25 of the last 200, every table."""
    algo = algo_of(need(answers, "CLIFF_SAFE"), "CLIFF_SAFE")

    def judge(Ls):
        falls = [int((np.array(L.returns[-200:]) <= -100).sum()) for L in Ls]
        ok = max(falls) < 25
        return ok, (f"fell in at most {max(falls)} of the last 200 episodes" + ("" if ok else " (needs < 25)")), \
            f"episodes with a fall, last 200: {' / '.join(map(str, falls))}   while still exploring (epsilon 0.1)"
    return GridAttempt(CLIFF_ROWS, CLIFF_MOVES, cliff_learners(algo), 500, judge, {47})


def p_short(answers):
    """4. CliffWalking, same training: the greedy path must reach the goal in 13 steps, every table."""
    algo = algo_of(need(answers, "CLIFF_SHORT"), "CLIFF_SHORT")

    def judge(Ls):
        lens = []
        for L in Ls:
            rets, steps, _ = tabular.run_greedy(L.env, L.Q, 1, 0, max_steps=100)
            lens.append(int(steps[0]) if rets[0] > -100 else 999)
        ok = max(lens) <= 13
        shown = " / ".join("never arrives" if n >= 100 else str(n) for n in lens)
        return ok, ("every greedy path is 13 steps" if ok else f"greedy paths {shown} steps (needs 13)"), \
            f"greedy path length: {shown}   (the shortest possible is 13)"
    return GridAttempt(CLIFF_ROWS, CLIFF_MOVES, cliff_learners(algo), 500, judge, {47})


class PendulumAttempt(Attempt):
    """5. Train on InvertedPendulum-v5 (MuJoCo) for 3000 episodes, keep the best table, test it, then play it."""

    def __init__(self, learner, episodes=3000):
        super().__init__()
        self.L, self.kb = learner, tabular.KeepBest(learner, episodes)
        env = learner.env.unwrapped
        self.model, self.data = env.model, env.data
        VIEW.update(lookat=(0.0, 0.0, 0.3), distance=2.6, azimuth=90.0, elevation=-10.0)
        self.dt, self.result, self.t = env.dt, None, 0          # 0.04 s: the env's own decision period
        self.gauge = "training ..."

    def step(self):
        if self.result is None:                                  # training: episodes for up to 30 ms per frame
            t0 = time.perf_counter()
            while not self.kb.finished and time.perf_counter() - t0 < BUDGET:
                self.kb.advance(1)
            self.gauge = (f"training: episode {self.kb.done} / {self.kb.episodes}   last episode "
                          f"{self.L.lengths[-1] if self.L.lengths else 0} steps   best greedy check {self.kb.best:.0f} "
                          f"of 1000 (episode {self.kb.when})   table rows {self.L.binner.n}")
            if self.kb.finished:
                _, lens, _ = tabular.run_greedy(self.L.env, self.kb.best_Q, 10, 999_000, index=self.L.index,
                                                act=self.L.act)
                ok = lens.min() >= 1000
                self.result = (ok, ("the kept table balances 1000 steps from all 10 test starts" if ok else
                                    f"from 10 test starts it balanced {lens.mean():.0f} steps on average (needs 1000 each)"))
                self.met = ok
                self.gauge = (f"kept table (episode {self.kb.when}): test steps {' '.join(map(str, lens))}   "
                              f"rows {self.L.binner.n}   -- now playing it: ctrl + right-drag the pole to push")
                self.obs, _ = self.L.env.reset(seed=4242)
            return True
        if self.t < 300:                                # 10 s of the kept table, live
            self.obs, _, te, tr, _ = self.L.env.step(self.L.act(int(np.argmax(self.kb.best_Q[self.L.index(self.obs)]))))
            if te or tr:
                self.obs, _ = self.L.env.reset(seed=4243 + self.t)
            self.t += 1
            return True
        return False

    def verdict(self):
        return self.result


def p_pendulum(answers):
    d = need(answers, "PENDULUM")
    if not isinstance(d, dict) or not d:
        raise AnswersError("PENDULUM must be a dict: observation name -> list of cut points")
    take, edges = [], []
    for name, cuts in d.items():
        if name not in tabular.PENDULUM_OBS:
            raise AnswersError(f"PENDULUM: {name!r} is not one of {', '.join(tabular.PENDULUM_OBS)}")
        if not isinstance(cuts, (list, tuple)) or not cuts or not all(isinstance(x, (int, float)) for x in cuts):
            raise AnswersError(f"PENDULUM[{name!r}] must be a list of numbers (cut points), got {cuts!r}")
        if list(cuts) != sorted(cuts):
            raise AnswersError(f"PENDULUM[{name!r}]: cut points must increase")
        take.append(tabular.PENDULUM_OBS.index(name))
        edges.append(cuts)
    L = tabular.pendulum_learner(edges, take, seed=0)
    if L.binner.n > 20_000:
        raise AnswersError(f"PENDULUM: {L.binner.n} table rows; keep it under 20,000")
    return PendulumAttempt(L)


BALANCE_SCENE = """<mujoco>
  <visual><headlight ambient="0.5 0.5 0.5"/></visual>
  <worldbody>
    <light pos="0 -1 3"/>
    <geom type="plane" size="1 1 0.01" rgba="0.9 0.9 0.9 1"/>
    <geom type="box" pos="0.035 0 0.01" size="0.085 0.05 0.01" rgba="0.3 0.3 0.3 1"/>
  </worldbody>
</mujoco>"""


class BalanceAttempt(Attempt):
    """6. Q-learning on week 7's G1 balance, 100,000 episodes in batches of 64; J checked 8 times.

    Drawn as week 7 did: blue ball = centre of mass, orange = centre of
    pressure on the foot (grey), grey line = the leg. While it trains, the
    current greedy table catches pushes of -0.2, +0.3, +0.45 m/s in turn.
    """

    ROUNDS, CHECKS, PASS = 1560, 8, 60.0

    def __init__(self, bins):
        super().__init__()
        self.L = tabular.BalanceLearner(bins, seed=0)
        self.model = mujoco.MjModel.from_xml_string(BALANCE_SCENE)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        VIEW.update(lookat=(0.05, 0.0, 0.35), distance=1.6, azimuth=90.0, elevation=-5.0)
        self.dt, self.result, self.rounds = 0.04, None, 0       # two 50 Hz decisions per frame: real time
        self.x, self.v, self.p, self.push_k, self.ticks = 0.0, -0.2, 0.0, 0, 0
        self.gauge = "training ..."

    def step(self):
        m = self.L.mdp
        if self.result is None and self.rounds < self.ROUNDS:   # training: one batch of 64 episodes per frame
            self.L.advance(1)
            self.rounds += 1
            if self.rounds % (self.ROUNDS // self.CHECKS) == 0:
                self.L.check()
            js = [j for _, j in self.L.curve]
            self.gauge = (f"training: {self.L.episodes:,} episodes   J checks: {' '.join(f'{j:.1f}' for j in js)}   "
                          f"(hold 12.66, pi* {m.J_STAR})   table {self.L.bins} x {self.L.bins}")
        if self.result is None and self.rounds >= self.ROUNDS:
            js = [j for _, j in self.L.curve]
            late = float(np.mean(js[-4:]))
            ok = late >= self.PASS
            self.met = ok
            self.result = (ok, f"J over the second half {late:.1f}" + (" (>= 60)" if ok else ", needs 60"))
            self.gauge = (f"J checks: {' '.join(f'{j:.1f}' for j in js)}   second half {late:.1f}   "
                          f"(hold 12.66, value iteration's pi* {m.J_STAR})")
            self.replays = 0
        # the current greedy table, catching a push: two decisions (50 Hz) per frame, 3 s per push
        for _ in range(2):
            self.p = float(self.L.policy(np.array([self.x]), np.array([self.v]))[0])
            x, v = m.step(np.array([self.x]), np.array([self.v]), self.p)
            self.x, self.v = float(x[0]), float(v[0])
            self.ticks += 1
            if m.fallen(np.array([self.x]), np.array([self.v]))[0] or self.ticks >= 150:
                self.push_k, self.ticks = self.push_k + 1, 0
                self.x, self.v = 0.0, (-0.2, 0.3, 0.45)[self.push_k % 3]
                if self.result is not None:
                    self.replays += 1
                    if self.replays >= 3:
                        return False
        return True

    def verdict(self):
        return self.result

    def draw(self, scn):
        h = self.L.mdp.H
        sphere(scn, (self.x, 0, h), (0.23, 0.49, 0.85, 1), 0.04)
        sphere(scn, (self.p, 0, 0.025), (0.91, 0.45, 0.23, 1), 0.015)
        if scn.ngeom < scn.maxgeom:
            mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, 0.006,
                                 np.array([self.p, 0, 0.025]), np.array([self.x, 0, h]))
            scn.geoms[scn.ngeom].rgba[:] = (0.5, 0.5, 0.5, 1)
            scn.ngeom += 1


def p_balance(answers):
    d = need(answers, "BALANCE")
    bins = number(d, "bins", "BALANCE", 2, 201)
    if bins != int(bins):
        raise AnswersError("BALANCE: \"bins\" must be a whole number")
    return BalanceAttempt(int(bins))


PROBLEMS = {
    1: Problem("FAR-SIGHTED", "Slippery lake, Q-learning, 5000 episodes. Reach the goal 70 % of the time.  (FROZEN)", p_far),
    2: Problem("WATCHER", "Act at RANDOM the whole time. Its greedy table must still reach the goal 60 %.  (RANDOM)",
               p_watcher),
    3: Problem("SAFE", "Cliff, exploring (epsilon 0.1): fall in fewer than 25 of the last 200 episodes.  (CLIFF_SAFE)",
               p_safe),
    4: Problem("SHORT", "Cliff, same training: the greedy path must be the shortest, 13 steps.  (CLIFF_SHORT)", p_short),
    5: Problem("PENDULUM", "MuJoCo inverted pendulum, 3000 episodes: balance 1000 steps from 10 test starts.  (PENDULUM)",
               p_pendulum),
    6: Problem("G1 BALANCE", "Week 7's balance from samples, 100,000 episodes: J >= 60 (pi* 72.12).  (BALANCE)",
               p_balance),
}


# --- 3. main ------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tables", type=Path, default=HERE / "tables.py", help="the answers file (default: tables.py here)")
    ap.add_argument("--grade", action="store_true", help="grade headless and print the table")
    ap.add_argument("--problem", type=int, nargs="*", help="with --grade: only these problems")
    ap.add_argument("--no-viewer", action="store_true", help="never open a window")
    args = ap.parse_args(argv)
    if args.grade or args.no_viewer:
        grade_headless(PROBLEMS, args.tables, __file__, args.problem)
        return 0
    if soc4180.is_colab():
        print("This lab needs a desktop window; run it on your laptop.")
        return 1
    print(__doc__.split("\n\n")[1])
    idle = mujoco.MjModel.from_xml_string(grid_scene(FROZEN_ROWS))

    def camera(viewer):
        viewer.cam.lookat[:] = VIEW["lookat"]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = VIEW["distance"], VIEW["azimuth"], VIEW["elevation"]
    return run_viewer(PROBLEMS, args.tables, __file__, idle_model=idle, idle_data=mujoco.MjData(idle), camera=camera,
                      title="week 7b: tables")


if __name__ == "__main__":
    raise SystemExit(main())
