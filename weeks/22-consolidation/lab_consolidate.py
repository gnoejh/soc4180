"""lab_consolidate: the G1's episodes of pushes are replayed from the hippocampus into the cortex.

    uv run weeks/22-consolidation/lab_consolidate.py                   # 10 episodes of 8 pushes
    uv run weeks/22-consolidation/lab_consolidate.py --episodes 20     # 160 pushes consolidated
    uv run weeks/22-consolidation/lab_consolidate.py --distinct        # the law's world, no robot
    uv run weeks/22-consolidation/lab_consolidate.py --gamma 0.97      # a cortex that forgets
    uv run weeks/22-consolidation/lab_consolidate.py --viewer --episodes 2

The pushes are week 20's: the G1 walks, is pushed (100 N, 0.1 s) in one of 8 directions
in its own frame, and senses each direction with its pelvis accelerometer.

THE MEMORY SYSTEM (ubrain's `rememberer` profile): the hippocampus writes each percept
into one episode vector (week 20). At the episode's end -- a BOUNDARY -- it retrieves
every slot and REPLAYS what it retrieved into the cortex, then clears itself. The cortex
is a slow store: one d x d matrix written by Hebb's rule, W += z k^T, under a fresh random
key k for each replayed push, read by z = W k. Afterwards the lab asks the cortex for
every push it was given, the n-th consolidated memory.

Two laws, computed before anything is stored:
  the episode     an item of a k-item episode is read at the decode law's rate in d
  the cortex      N pairs are read at the same law in the effective dimension d^2
                  (ubrain.datasheet.weighted_law(1, N, d*d, V, d_self=d)): hundreds of
                  pairs where an episode holds a handful. The law counts N DIFFERENT pairs.
The robot's world has only 8 kinds of push, so its pairs repeat; --distinct runs the same
schedule on a laptop with every item a different one, the world the law assumes.
    --episodes E      episodes (default 10), each ended by a boundary
    --pushes K        pushes per episode (default 8)
    --d D             the dimension of the episode, the codes and the cortex (default 64)
    --gamma G         the cortex's decay per write event (default 1.0: forgets nothing)
    --distinct        no robot: every item different (a vocabulary of 512 symbols)
    --seed S

Measured: the hippocampus (retrieved = perceived, before each boundary), the cortex
(recollected = retrieved: faithful), and end to end (recollected = perceived). Every run
appends one row to results.csv. ubrain is the course's local copy (see ubrain_local.py).
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import ubrain_local

ubrain_local.load()                 # the vendored wheel onto sys.path, before `import ubrain`

from ubrain import Brain, Channel, Manifest, profile  # noqa: E402
from ubrain.bodies import BodyClient, kill_tree  # noqa: E402
from ubrain.codecs import GaussianCodes, Sense  # noqa: E402
from ubrain.datasheet import weighted_law  # noqa: E402
from ubrain.observe import decode_law  # noqa: E402

HERE = Path(__file__).resolve().parent
HZ = 50
VX = 0.5
SETTLE = 100
PUSH_TICKS = 5
GAP = (60, 100)
BASELINE = 40
PELVIS_ACC = slice(9, 12)
SLOTS = 32                      # the episode's positions (more than any episode here)
DISTINCT_V = 512                # the --distinct world's vocabulary
NAMES = ("F", "FL", "L", "BL", "B", "BR", "R", "FR")
ANGLES = [2 * math.pi * i / 8 for i in range(8)]


def perceive(before: np.ndarray, during: np.ndarray) -> int:
    """Week 20's sense: the pelvis accelerometer during the push minus the stride before,
    as an angle in the robot's frame, rounded to the nearest of the 8 directions."""
    a = during[:, PELVIS_ACC].mean(0) - before[:, PELVIS_ACC].mean(0)
    ang = math.atan2(a[1], a[0])
    return int(np.argmin([abs((ang - t + math.pi) % (2 * math.pi) - math.pi) for t in ANGLES]))


def rememberer(d: int, V: int, N: int, gamma: float, seed: int) -> Brain:
    sense = Sense.fit(GaussianCodes(dim=d), np.arange(V), d=d)
    w = profile("rememberer")
    w.regions["hippocampus.episode"]["slots"] = SLOTS
    w.regions["cortex.store"]["gamma"] = gamma
    body = Manifest("course/g1-pushes", "none", (
        Channel("camera", "codec", 1), Channel("recall", "event", SLOTS), Channel("boundary", "event", 1),
        Channel("recollect", "event", N), Channel("memory", "command", d), Channel("remembered", "choice", V),
        Channel("recollection", "command", d), Channel("recollected", "choice", V)))
    return Brain(w, body, codecs={"camera": sense}, rng=np.random.default_rng(seed))


class Pusher:
    """The robot side: walk, push in the robot's frame, perceive."""

    def __init__(self, viewer: bool, rng):
        self.proc, port = ubrain_local.start_body(viewer=viewer)
        self.body = BodyClient("127.0.0.1", port)
        self.walk = self.body.primitives["walk"]["index"]
        self.viewer, self.t0, self.ticks, self.rng = viewer, time.perf_counter(), 0, rng
        self.body.reset(0)
        self.history = [self.step() for _ in range(SETTLE)]

    def step(self) -> np.ndarray:
        self.state = self.body.act(self.walk, [VX, 0.0, 0.0])
        self.ticks += 1
        if self.viewer:
            lag = self.t0 + self.ticks / HZ - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
        return np.asarray(self.state["proprio"])

    def push(self) -> tuple[int, int]:
        """One push; returns (true direction, perceived direction)."""
        self.history += [self.step() for _ in range(int(self.rng.integers(*GAP)))]
        d = int(self.rng.integers(8))
        self.body._send({"type": "world", "field": [100.0 * math.cos(ANGLES[d]), 100.0 * math.sin(ANGLES[d])]})
        during = [self.step() for _ in range(PUSH_TICKS)]
        self.body._send({"type": "world", "field": [0.0, 0.0]})
        seen = perceive(np.array(self.history[-BASELINE:]), np.array(during))
        self.history = (self.history + during)[-200:]
        if not self.state["upright"]:
            self.body.reset(0)
            self.history = [self.step() for _ in range(SETTLE)]
        return d, seen

    def close(self):
        try:
            self.body.close()
        finally:
            kill_tree(self.proc)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.split("\n")[7:]))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--pushes", type=int, default=8)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--distinct", action="store_true", help="no robot: every item a different one")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--name", default="")
    a = ap.parse_args(argv)
    E, K = a.episodes, a.pushes
    N = E * K
    if not 1 <= K <= SLOTS:
        ap.error(f"--pushes must be 1 .. {SLOTS}")
    if a.distinct and N > DISTINCT_V:
        ap.error(f"--distinct has {DISTINCT_V} symbols: at most {DISTINCT_V} pushes in all")
    V = DISTINCT_V if a.distinct else 8
    rng = np.random.default_rng(a.seed)
    brain = rememberer(a.d, V, N, a.gamma, a.seed + 1)
    law_hip = decode_law(K, a.d, V)
    law_ctx = weighted_law(1.0, N, a.d * a.d, V, d_self=a.d)
    print(f"{E} episodes of {K} {'different symbols (no robot)' if a.distinct else 'pushes'}, d {a.d}, "
          f"cortex gamma {a.gamma}; laws: hippocampus {law_hip:.2f}, cortex {law_ctx:.2f} for {N} different pairs")
    pool = rng.permutation(DISTINCT_V)[:N] if a.distinct else None
    robot = None if a.distinct else Pusher(a.viewer, rng)
    perceived, retrieved, true, after = [], [], [], []
    try:
        for e in range(E):
            ep = []
            for k in range(K):
                if a.distinct:
                    d = seen = int(pool[e * K + k])
                else:
                    d, seen = robot.push()
                brain.tick({"camera": seen})
                ep.append(seen)
                true.append(d)
            got = [int(brain.tick({"recall": j})["remembered"]) for j in range(K)]
            perceived += ep
            retrieved += got
            brain.tick({"boundary": True})           # replay into the cortex; the episode is cleared
            after.append(brain.tick({"recall": 0})["remembered"] is None)   # cleared: nothing comes back
            if not a.distinct and e < 3:
                print(f"  episode {e + 1}: perceived " + " ".join(NAMES[x] for x in ep)
                      + "   retrieved " + " ".join(NAMES[x] for x in got))
    finally:
        if robot:
            robot.close()
    recollected = [int(brain.tick({"recollect": n})["recollected"]) for n in range(N)]
    hip = np.mean([r == p for r, p in zip(retrieved, perceived)])
    faith = np.mean([c == r for c, r in zip(recollected, retrieved)])
    e2e = np.mean([c == p for c, p in zip(recollected, perceived)])
    per_ep = [np.mean([c == r for c, r in zip(recollected[i * K:(i + 1) * K], retrieved[i * K:(i + 1) * K])])
              for i in range(E)]
    print(f"\nover {N} pushes:")
    print(f"  hippocampus, retrieved before each boundary   {hip:.2f}   (law {law_hip:.2f})")
    print(f"  cortex, recollected as it was given            {faith:.2f}   (law {law_ctx:.2f}, for {N} DIFFERENT pairs)")
    print(f"  end to end, recollected as perceived           {e2e:.2f}   (the product of the laws {law_hip * law_ctx:.2f})")
    print(f"  the hippocampus after each boundary: empty (recall returns nothing) in {sum(after)} of {E}")
    print("  cortex by episode (oldest first): " + " ".join(f"{x:.2f}" for x in per_ep))
    if not a.distinct:
        print(f"  perception (perceived = pushed): {np.mean([p == t for p, t in zip(perceived, true)]):.2f}")
    row = {"when": datetime.now().isoformat(timespec="seconds"), "name": a.name, "world": "distinct" if a.distinct else "robot",
           "episodes": E, "pushes": K, "d": a.d, "gamma": a.gamma, "seed": a.seed,
           "hippocampus": round(float(hip), 3), "law_hippocampus": round(law_hip, 3), "cortex": round(float(faith), 3),
           "law_cortex": round(law_ctx, 3), "end_to_end": round(float(e2e), 3),
           "by_episode": " ".join(f"{x:.2f}" for x in per_ep)}
    new = not (HERE / "results.csv").exists()
    with open(HERE / "results.csv", "a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(row))
        if new:
            wr.writeheader()
        wr.writerow(row)
    print("saved: one row in results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
