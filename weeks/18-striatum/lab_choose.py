"""lab_choose: the G1's striatum learns WHEN to brake, from nothing but staying up.

    uv run weeks/18-striatum/lab_choose.py                    # 120 trials (about a minute)
    uv run weeks/18-striatum/lab_choose.py --trials 60 --eta 0.3
    uv run weeks/18-striatum/lab_choose.py --map              # the world's payoffs, no brain
    uv run weeks/18-striatum/lab_choose.py --viewer           # watch it (real time, 6 s a trial)

The Unitree G1 walks at 0.5 m/s. On every trial a cue announces a push from behind
(250 N for 0.1 s) and says when it will come:
    situation 0  "soon"   the push lands 0.2 s after the cue (10 ticks)
    situation 1  "late"   the push lands 0.5 s after the cue (25 ticks)
and the striatum (ubrain's `player` profile: basal_ganglia.striatum) picks one move:
    action 0  brake early   walk 0.4 m/s slower from 3 to 15 ticks after the cue
    action 1  brake late    the same, from 18 to 30 ticks after the cue
    action 2  keep walking
The only teacher is the outcome, 3 s after the push: reward 1 if the robot is still up,
0 if it fell. No error, no direction, nothing about the push: a scalar.

What the world pays (measured with --map, 40 gait phases each, a trial's phase is drawn
at random from the same 40):
                 brake early   brake late   keep walking
    soon            0.85          0.25          0.55
    late            0.48          0.85          0.55
The right brake is the one timed to the push; the wrong one is worse than doing nothing.

THE STRIATUM keeps a value Q per (situation, action), moves the chosen one toward the
outcome by Q <- Q + eta (r - Q), and releases actions by a race between them, each
faster the higher its value.
    --eta E          learning rate (default 0.2)
    --q0 V           the values it starts from (default 0.5: unknown, neither good nor bad)
    --trials N       trials (default 120; about 0.5 s each headless, 6 s with --viewer)
    --seed S         the trial sequence (situations and gait phases) and the race
    --blind          the striatum is not told the cue: it sees one situation for both
                     (the robot still hears the push coming; the brain cannot tell when)
And the world:
    --force F        the push, newtons (default 250)
    --brake B        how much slower a brake walks, m/s (default 0.4)
    --map            no brain: try every (situation, action) at every 4th gait phase
                     (--phases to change) and print the table the striatum has to learn

Every run prints each trial (situation, move, outcome), the values every 20 trials, and
at the end the values beside the measured payoffs and the choices of the last third.
It appends one row to results.csv beside this file and saves every trial (with the values
before its outcome arrived) in last_run.json. ubrain itself is the course's copy
(see ubrain_local.py); nothing else beyond `uv sync`.
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

from ubrain import Brain, Channel, Manifest, profile  # noqa: E402
from ubrain.bodies import BodyClient, kill_tree  # noqa: E402

HERE = Path(__file__).resolve().parent
HZ = 50
VX = 0.5                    # walking speed, m/s
SETTLE = 100                # ticks of walking after a reset before a trial (2 s)
PHASES = 40                 # a stride is 40 ticks: a trial starts at a random one of them
WATCH = 150                 # ticks after the push before the outcome is read (3 s)
SITUATIONS = ("soon", "late")
DELAY = (10, 25)            # ticks from the cue to the push, per situation
MOVES = ("brake early", "brake late", "keep walking")
WINDOW = ((3, 15), (18, 30), None)    # ticks after the cue during which the move brakes
# measured with --map (40 phases each), for the summary
WORLD = np.array([[0.85, 0.25, 0.55],
                  [0.475, 0.85, 0.55]])


class Trials:
    """The G1 as a bandit: reset, walk, cue, move, push, watch."""

    def __init__(self, body: BodyClient, force: float, brake: float, viewer: bool):
        self.body, self.force, self.brake, self.viewer = body, force, brake, viewer
        self.walk = body.primitives["walk"]["index"]
        self.t0, self.ticks = time.perf_counter(), 0

    def step(self, vx: float) -> dict:
        state = self.body.act(self.walk, [vx, 0.0, 0.0])
        self.ticks += 1
        if self.viewer:
            lag = self.t0 + self.ticks / HZ - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
        return state

    def run(self, situation: int, action: int, phase: int) -> int:
        """One trial; returns 1 if the robot is still up WATCH ticks after the push."""
        self.body.reset(0)
        for _ in range(SETTLE + phase):
            self.step(VX)
        delay = DELAY[situation]
        self.body._send({"type": "world", "tone": True,
                         "push": {"force": [self.force, 0.0], "delay": delay, "ticks": 5}})
        win = WINDOW[action]
        for t in range(delay + WATCH):
            braking = win is not None and win[0] <= t < win[1]
            if not self.step(VX - self.brake if braking else VX)["upright"]:
                return 0
        return 1


def payoff_map(trials: Trials, phases: list[int]) -> np.ndarray:
    table = np.zeros((len(SITUATIONS), len(MOVES)))
    print(f"the world, no brain: every (situation, move) at {len(phases)} gait phases")
    for s in range(len(SITUATIONS)):
        for a in range(len(MOVES)):
            ups = [trials.run(s, a, ph) for ph in phases]
            table[s, a] = np.mean(ups)
            print(f"  {SITUATIONS[s]:5s} {MOVES[a]:13s} up {sum(ups):2d}/{len(ups)}  "
                  f"{''.join('.' if u else 'X' for u in ups)}", flush=True)
    print("\n              " + "".join(f"{m:>14s}" for m in MOVES))
    for s in range(len(SITUATIONS)):
        print(f"  {SITUATIONS[s]:10s}  " + "".join(f"{p:14.2f}" for p in table[s]))
    return table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.split("\n")[6:]))
    ap.add_argument("--eta", type=float, default=0.2)
    ap.add_argument("--q0", type=float, default=0.5)
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", type=float, default=250.0)
    ap.add_argument("--brake", type=float, default=0.4)
    ap.add_argument("--blind", action="store_true", help="the striatum is not told the cue")
    ap.add_argument("--map", action="store_true", help="measure the payoff table, no brain")
    ap.add_argument("--phases", type=int, default=10, help="--map: how many gait phases (of 40)")
    ap.add_argument("--viewer", action="store_true", help="show the robot (real time)")
    ap.add_argument("--name", default="", help="a label for this run in results.csv")
    ap.add_argument("--log", default="last_run.json", help="where to save every trial (beside this file)")
    a = ap.parse_args(argv)

    proc, port = ubrain_local.start_body(viewer=a.viewer)
    try:
        trials = Trials(BodyClient("127.0.0.1", port), a.force, a.brake, a.viewer)
        if a.map:
            phases = [round(i * PHASES / a.phases) for i in range(a.phases)]
            payoff_map(trials, phases)
            trials.body.close()
            return 0

        S, A = len(SITUATIONS), len(MOVES)
        w = profile("player")
        w.regions["basal_ganglia.striatum"]["eta"] = a.eta
        w.regions["basal_ganglia.striatum"]["q0"] = a.q0
        seen = 1 if a.blind else S                  # how many situations the striatum can tell apart
        manifest = Manifest("course/g1-brake", "none", (Channel("situation", "event", seen),
                                                       Channel("reward", "reward", 1),
                                                       Channel("action", "choice", A)))
        brain = Brain(w, manifest, rng=np.random.default_rng(a.seed + 1))
        rng = np.random.default_rng(a.seed)
        print(f"{a.trials} trials, push {a.force:.0f} N from behind, brake {a.brake} m/s; "
              f"striatum eta {a.eta}, q0 {a.q0}{', BLIND to the cue' if a.blind else ''}")
        print(" trial  cue    move           outcome")
        log, values, r = [], [], None
        for t in range(a.trials):
            s = int(rng.integers(S))
            cue = np.eye(seen)[0 if a.blind else s]  # an event: which situation this is
            act = int(brain.tick({"situation": cue, "reward": r})["action"])
            r = trials.run(s, act, int(rng.integers(PHASES)))
            log.append((s, act, r))
            values.append(np.array(brain.regions["basal_ganglia.striatum"].Q).round(4).tolist())
            print(f"  {t + 1:4d}  {SITUATIONS[s]:5s}  {MOVES[act]:13s}  {'up' if r else 'FELL'}", flush=True)
            if (t + 1) % 20 == 0:
                Q = brain.regions["basal_ganglia.striatum"].Q
                print("        values Q: " + "   ".join(
                    f"{'both' if a.blind else SITUATIONS[k]}: " + " ".join(f"{q:.2f}" for q in Q[k])
                    for k in range(seen)), flush=True)
        brain.tick({"reward": r})                    # the last outcome
        trials.body.close()
    finally:
        kill_tree(proc)

    Q = brain.regions["basal_ganglia.striatum"].Q
    third = log[len(log) - len(log) // 3:]
    counts = np.zeros((S, A))
    for s, act, _r in third:
        counts[s, act] += 1
    alloc = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    first = log[:len(log) // 3]
    rate = lambda part: float(np.mean([x[2] for x in part])) if part else float("nan")
    default_world = a.force == 250.0 and a.brake == 0.4      # WORLD was measured for these only
    best = float(WORLD.max(1).mean())
    random = float(WORLD.mean())
    print("\nlearned values Q" + ("                          the world's payoffs (measured with --map)"
                                 if default_world else
                                 f"   (this world's payoffs: run --map --force {a.force:g} --brake {a.brake:g})"))
    for s in range(S):
        q = Q[0 if a.blind else s]
        print(f"  {SITUATIONS[s]:5s} " + " ".join(f"{v:5.2f}" for v in q)
              + ("                   " + " ".join(f"{p:5.2f}" for p in WORLD[s]) if default_world else ""))
    if a.blind and default_world:
        print("  (blind: one row of values for both cues; the world's average is "
              + " ".join(f"{p:.2f}" for p in WORLD.mean(0)) + ")")
    print("\nchoices in the last third               (columns: " + ", ".join(MOVES) + ")")
    for s in range(S):
        print(f"  {SITUATIONS[s]:5s} " + " ".join(f"{x:5.2f}" for x in alloc[s]))
    print(f"\nstayed up: first third {rate(first):.2f}, last third {rate(third):.2f}"
          + (f"  (choosing at random: {random:.2f}; always the right brake: {best:.2f})" if default_world else ""))

    row = {"when": datetime.now().isoformat(timespec="seconds"), "name": a.name, "trials": a.trials,
           "eta": a.eta, "q0": a.q0, "seed": a.seed, "blind": a.blind, "force": a.force, "brake": a.brake,
           "up_first_third": round(rate(first), 3), "up_last_third": round(rate(third), 3),
           "Q": json.dumps(np.round(Q, 3).tolist()), "choices_last_third": json.dumps(np.round(alloc, 2).tolist()),
           "record": "".join("." if x[2] else "X" for x in log)}
    new = not (HERE / "results.csv").exists()
    with open(HERE / "results.csv", "a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(row))
        if new:
            wr.writeheader()
        wr.writerow(row)
    (HERE / a.log).write_text(json.dumps(
        {**row, "trials_log": [{"cue": s, "move": m, "up": u, "Q_before_outcome": q}
                               for (s, m, u), q in zip(log, values)]}, indent=1), encoding="utf-8")
    print(f"saved: one row in results.csv, every trial in {a.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
