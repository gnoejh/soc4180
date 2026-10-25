"""lab_habit: memory first, habit later -- two learners, and a hand-over, on the G1's brake task.

    uv run weeks/24-memory-habit/lab_habit.py                 # the three brains, 20 seeds each
    uv run weeks/24-memory-habit/lab_habit.py --trials 60     # early learning only
    uv run weeks/24-memory-habit/lab_habit.py --gamma 0.9     # a memory that forgets fast
    uv run weeks/24-memory-habit/lab_habit.py --live --brain chooser --trials 30 --viewer
    uv run weeks/24-memory-habit/lab_habit.py --remeasure     # measure the world again (~2 min)

The task is week 18's: a cue says a push from behind comes SOON (0.2 s) or LATE (0.5 s);
the brain picks brake early, brake late or keep walking; the reward is 1 if the G1 is
still up 3 s after the push. Each trial starts at a random point of the stride (one of 40).

THE WORLD IS DETERMINISTIC: the same cue, move and stride point always give the same
outcome (the body runs in lockstep). So the lab measures it once -- all 2 x 3 x 40 = 240
trials on the robot -- keeps it in traces/world.json, and then runs any number of brains
on it in seconds. --live runs the robot itself, trial by trial, for one brain.

THREE BRAINS, the same senses and the same rewards:
  striatum   week 18's player: a value per (cue, move), Q <- Q + eta (r - Q), a race
  memory     the hippocampus as a controller (episodic control): every (cue, move,
             outcome) bound into one decaying vector; at a decision each move's recalled
             vote is read and the best taken -- greedy, from one experience
  chooser    both, with a hand-over (ubrain's `chooser` profile): memory acts while its
             votes' margin clears its own crosstalk scale, the striatum otherwise; both
             learn from every outcome
    --trials T        trials per run (default 200)     --seeds N   runs per brain (20)
    --gamma G         the memory's decay per outcome (default 0.995)
    --d D             the memory's dimension: its room, against crosstalk (default 256)
    --eta E           the striatum's learning rate (default 0.1, the profile's)
    --bandit S        no robot: ubrain's laptop bandit instead, S situations x 2 actions
                      paying 0.7 / 0.3 (ubrain's example 09 uses 256), to see habit overtake
    --live            run the robot for every trial (one brain: --brain, one seed)
    --brain B         striatum, memory or chooser (for --live; default chooser)
    --viewer          with --live: watch it

Measured: the share of trials the robot stayed up, by how many times the cue had been
seen, for each brain; and how often the chooser handed the decision to memory. Every run
appends one row per brain to results.csv. ubrain is the course's local copy.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import ubrain_local

ubrain_local.load()                 # the vendored wheel onto sys.path, before `import ubrain`

from ubrain import Brain, Channel, Manifest, Wiring, profile  # noqa: E402
from ubrain.bodies import BodyClient, kill_tree  # noqa: E402
from ubrain.observe import observe  # noqa: E402

HERE = Path(__file__).resolve().parent
WORLD_FILE = HERE / "traces" / "world.json"
HZ = 50
VX = 0.5
SETTLE = 100
PHASES = 40
WATCH = 150
FORCE, BRAKE = 250.0, 0.4
CUES = ("soon", "late")
DELAY = (10, 25)
MOVES = ("brake early", "brake late", "keep walking")
WINDOW = ((3, 15), (18, 30), None)
BUCKETS = ((1, 1), (2, 3), (4, 10), (11, 30), (31, 60), (61, 10 ** 9))   # times the cue had been seen
BRAINS = ("striatum", "memory", "chooser")


class Robot:
    """Week 18's trial: reset, walk, cue + push, the move, watch 3 s."""

    def __init__(self, viewer: bool = False):
        self.proc, port = ubrain_local.start_body(viewer=viewer)
        self.body = BodyClient("127.0.0.1", port)
        self.walk = self.body.primitives["walk"]["index"]
        self.viewer, self.t0, self.ticks = viewer, time.perf_counter(), 0

    def step(self, vx: float) -> dict:
        state = self.body.act(self.walk, [vx, 0.0, 0.0])
        self.ticks += 1
        if self.viewer:
            lag = self.t0 + self.ticks / HZ - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
        return state

    def trial(self, cue: int, move: int, phase: int) -> int:
        self.body.reset(0)
        for _ in range(SETTLE + phase):
            self.step(VX)
        self.body._send({"type": "world", "tone": True,
                         "push": {"force": [FORCE, 0.0], "delay": DELAY[cue], "ticks": 5}})
        win = WINDOW[move]
        for t in range(DELAY[cue] + WATCH):
            if not self.step(VX - BRAKE if win and win[0] <= t < win[1] else VX)["upright"]:
                return 0
        return 1

    def close(self):
        try:
            self.body.close()
        finally:
            kill_tree(self.proc)


def measure_world() -> np.ndarray:
    print("measuring the world: 2 cues x 3 moves x 40 stride points on the robot (about 2 minutes) ...", flush=True)
    r = Robot()
    try:
        world = np.array([[[r.trial(c, m, ph) for ph in range(PHASES)] for m in range(3)] for c in range(2)])
    finally:
        r.close()
    WORLD_FILE.parent.mkdir(exist_ok=True)
    WORLD_FILE.write_text(json.dumps({"force": FORCE, "brake": BRAKE, "outcome[cue][move][phase]": world.tolist()}),
                          encoding="utf-8")
    return world


def make_brain(kind: str, gamma: float, eta: float, seed: int, d: int = 256, S: int = 2, A: int = 3) -> Brain:
    body = Manifest("course/g1-brake", "none", (Channel("situation", "event", S), Channel("reward", "reward", 1),
                                               Channel("action", "choice", A)))
    if kind == "striatum":
        w = profile("player")
        w.regions["basal_ganglia.striatum"]["eta"] = eta
    else:
        w = profile("chooser")
        regions = {k: dict(v) for k, v in w.regions.items()}
        regions["hippocampus.episodic_control"]["gamma"] = gamma
        regions["hippocampus.episodic_control"]["d"] = d
        regions["basal_ganglia.striatum"]["eta"] = eta
        if kind == "memory":
            regions["hippocampus.episodic_control"].pop("kappa", None)     # no criterion: memory always acts
        w = Wiring(w.name, w.contract, regions, dict(w.inputs), dict(w.outputs))
    return Brain(w, body, rng=np.random.default_rng(seed))


def run(kind: str, trials: int, seed: int, gamma: float, eta: float, outcome, d: int = 256,
        S: int = 2, A: int = 3) -> tuple[list, list, float | None]:
    """outcome(cue, move, phase, rng) -> 0/1. Returns (per trial (visit, reward)), choices, memory share."""
    rng = np.random.default_rng(seed)
    brain = make_brain(kind, gamma, eta, seed + 1, d, S, A)
    visits, log, moves, r = [0] * S, [], [], None
    for _ in range(trials):
        cue = int(rng.integers(S))
        move = int(brain.tick({"situation": np.eye(S)[cue], "reward": r})["action"])
        r = float(outcome(cue, move, int(rng.integers(PHASES)), rng))
        visits[cue] += 1
        log.append((visits[cue], r))
        moves.append((cue, move))
    brain.tick({"reward": r})
    share = None
    if kind == "chooser":
        h = observe(brain)["control.handover"]
        share = h["on_memory"] / max(h["decisions"], 1)
    return log, moves, share


def by_bucket(logs: list) -> list[float]:
    out = []
    for lo, hi in BUCKETS:
        vals = [r for log in logs for v, r in log if lo <= v <= hi]
        out.append(float(np.mean(vals)) if vals else float("nan"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.split("\n")[7:]))
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--bandit", type=int, default=0)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--brain", choices=BRAINS, default="chooser")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--remeasure", action="store_true")
    ap.add_argument("--name", default="")
    a = ap.parse_args(argv)

    if a.bandit:
        S = a.bandit
        good = np.random.default_rng(99).integers(2, size=S)          # which action pays 0.7, per situation
        pay = lambda c, m, ph, rng: rng.random() < (0.7 if m == good[c] else 0.3)
        print(f"ubrain's laptop bandit: {S} situations x 2 actions paying 0.7 / 0.3 (no robot); "
              "always the best 0.70, matching 0.58, a coin 0.50", end="\n\n")
        curves, shares, logs = {}, {}, {}
        for kind in BRAINS:
            results = [run(kind, a.trials, s, a.gamma, a.eta, pay, a.d, S, 2) for s in range(a.seeds)]
            logs[kind] = [r[0] for r in results]
            curves[kind] = by_bucket(logs[kind])
            sh = [r[2] for r in results if r[2] is not None]
            shares[kind] = float(np.mean(sh)) if sh else None
        table = np.array([[0.7, 0.3]])
        report(a, curves, shares, logs, table)
        return 0
    if a.remeasure or not WORLD_FILE.exists():
        world = measure_world()
    else:
        world = np.array(json.loads(WORLD_FILE.read_text(encoding="utf-8"))["outcome[cue][move][phase]"])
    table = world.mean(axis=2)
    print("the world (share of the 40 stride points at which the robot stays up):")
    print("              " + "".join(f"{m:>14s}" for m in MOVES))
    for c in range(2):
        print(f"  {CUES[c]:10s}  " + "".join(f"{p:14.2f}" for p in table[c]))
    best, matching = table.max(1).mean(), (table ** 2 / table.sum(1, keepdims=True)).sum(1).mean()
    print(f"  always the right move: {best:.3f}; matching: {matching:.3f}; at random: {table.mean():.3f}\n")

    if a.live:
        robot = Robot(a.viewer)
        try:
            log, moves, share = run(a.brain, a.trials, 0, a.gamma, a.eta, lambda c, m, ph, _r: robot.trial(c, m, ph), a.d)
        finally:
            robot.close()
        print(f"{a.brain}, live on the robot, {a.trials} trials:")
        print("  " + "".join("." if r else "X" for _v, r in log) + "   (X fell, . stayed up)")
        curves = {a.brain: by_bucket([log])}
        shares = {a.brain: share}
        logs = {a.brain: [log]}
    else:
        curves, shares, logs = {}, {}, {}
        for kind in BRAINS:
            results = [run(kind, a.trials, s, a.gamma, a.eta, lambda c, m, ph, _r: world[c][m][ph], a.d)
                       for s in range(a.seeds)]
            logs[kind] = [r[0] for r in results]
            curves[kind] = by_bucket(logs[kind])
            sh = [r[2] for r in results if r[2] is not None]
            shares[kind] = float(np.mean(sh)) if sh else None
    report(a, curves, shares, logs, table)
    return 0


def report(a, curves, shares, logs, table) -> None:
    rows = []
    label = lambda lo, hi: f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 8 else f"{lo}+")
    print(f"stayed up, by how many times the cue had been seen ({'1 run' if a.live else f'{a.seeds} seeds'}, "
          f"{a.trials} trials each):")
    print("  times seen  " + "".join(f"{label(lo, hi):>9s}" for lo, hi in BUCKETS) + "      all")
    for kind, cur in curves.items():
        allr = float(np.mean([r for log in logs[kind] for _v, r in log]))
        print(f"  {kind:10s}  " + "".join(f"{x:9.2f}" for x in cur) + f"   {allr:6.3f}")
        rows.append({"brain": kind, "all": round(allr, 3), "curve": " ".join(f"{x:.2f}" for x in cur),
                     "memory_share": "" if shares.get(kind) is None else round(shares[kind], 3)})
    if shares.get("chooser") is not None:
        print(f"\nthe chooser handed {shares['chooser']:.2f} of its decisions to memory")
    best, matching = table.max(1).mean(), (table ** 2 / table.sum(1, keepdims=True)).sum(1).mean()
    print(f"(the first time a cue is seen nobody knows anything: at random {table.mean():.2f}; "
          f"matching {matching:.2f}; always right {best:.2f})")

    new = not (HERE / "results.csv").exists()
    with open(HERE / "results.csv", "a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["when", "name", "world", "live", "trials", "seeds", "gamma", "eta", "d", "brain", "all",
                                           "curve", "memory_share"])
        if new:
            wr.writeheader()
        for r in rows:
            wr.writerow({"when": datetime.now().isoformat(timespec="seconds"), "name": a.name,
                         "world": f"bandit {a.bandit}" if a.bandit else "robot", "live": a.live,
                         "trials": a.trials, "seeds": 1 if a.live else a.seeds, "gamma": a.gamma, "eta": a.eta,
                         "d": a.d, **r})
    print("saved: one row per brain in results.csv")


if __name__ == "__main__":
    raise SystemExit(main())
