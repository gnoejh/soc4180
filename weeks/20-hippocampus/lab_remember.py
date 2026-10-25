"""lab_remember: the G1 remembers which way it was pushed -- one shot, one vector, a capacity.

    uv run weeks/20-hippocampus/lab_remember.py                    # 3 episodes of 8 pushes
    uv run weeks/20-hippocampus/lab_remember.py --pushes 16        # a longer episode
    uv run weeks/20-hippocampus/lab_remember.py --d 32             # a smaller workspace
    uv run weeks/20-hippocampus/lab_remember.py --certify          # the prediction, no robot
    uv run weeks/20-hippocampus/lab_remember.py --viewer           # watch it (real time)

The Unitree G1 walks at 0.5 m/s. Now and then (every 1.2-2 s) the experimenter pushes
its pelvis for 0.1 s, 100 N, in one of 8 directions in the robot's own frame:
    F  forward   FL forward-left   L  left   BL back-left
    B  back      BR back-right     R  right  FR forward-right
(the direction the robot is pushed). Nothing tells the brain the direction. The robot
SENSES it: its pelvis accelerometer, during the push, minus its reading over the stride
before (perceive() below), gives an angle, rounded to the nearest of the 8. That is a
real sense and it can be wrong.

THE HIPPOCAMPUS (ubrain's `observer` profile, hippocampus.episode) stores each percept as
it arrives: the percept's code (a random vector of D numbers, one per direction) is bound
to its slot's role by circular convolution, scrambled by its position, and ADDED to one
vector. After the episode the lab asks for every slot back: unbind, and take the
direction whose code is nearest. The more pushes share the vector, the more they get in
each other's way (crosstalk): recall fails gradually as the episode fills.
    --pushes K        pushes per episode (default 8)
    --episodes E      episodes (default 3); each starts from a reset and a fresh brain
    --d D             the workspace: numbers per code and in the episode vector (default 64)
    --force F         the push, newtons (default 100: the robot does not fall)
    --directions N    8 (default) or 4 (F, L, B, R): fewer, easier to tell apart
    --seed S          the pushes and the brain's random roles
    --certify         no robot: print the certificate, the predicted recall of a K-push
                      episode at this D, for K = 4 ... 24

Measured, per episode and in total: perception (percept = the true push), memory
(recalled = percept), and both (recalled = the true push), beside the certificate.
Every run appends one row to results.csv beside this file. ubrain is the course's local
copy (see ubrain_local.py).
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
from ubrain.codecs import GaussianCodes, Sense, certify  # noqa: E402

HERE = Path(__file__).resolve().parent
HZ = 50
VX = 0.5
SETTLE = 100                    # ticks of walking after a reset (2 s)
PUSH_TICKS = 5                  # 0.1 s
GAP = (60, 100)                 # ticks between pushes (1.2-2 s)
BASELINE = 40                   # ticks before a push that set the accelerometer's baseline
PELVIS_ACC = slice(9, 12)       # proprio: torso gyro 0-2, torso acc 3-5, pelvis gyro 6-8, pelvis acc 9-11
SLOTS = 32                      # the episode's slots (positions): more than any episode here
NAMES8 = ("F", "FL", "L", "BL", "B", "BR", "R", "FR")


def directions(n: int) -> tuple[list[str], list[float]]:
    names = list(NAMES8) if n == 8 else ["F", "L", "B", "R"]
    return names, [2 * math.pi * i / n for i in range(n)]


def perceive(before: np.ndarray, during: np.ndarray, angles: list[float]) -> int:
    """The robot's sense of a push: the pelvis accelerometer during the push, minus its
    mean over the stride before, as an angle in the robot's frame (x forward, y left),
    rounded to the nearest direction it knows. Change this function to change the sense."""
    a = during[:, PELVIS_ACC].mean(0) - before[:, PELVIS_ACC].mean(0)
    ang = math.atan2(a[1], a[0])
    return int(np.argmin([abs((ang - t + math.pi) % (2 * math.pi) - math.pi) for t in angles]))


def observer(d: int, n_dir: int, seed: int) -> tuple[Brain, Sense]:
    codec = GaussianCodes(dim=d)
    sense = Sense.fit(codec, np.arange(n_dir), d=d)
    w = profile("observer")
    w.regions["hippocampus.episode"]["slots"] = SLOTS
    body = Manifest("course/g1-pushes", "none", (
        Channel("camera", "codec", 1), Channel("recall", "event", SLOTS), Channel("boundary", "event", 1),
        Channel("memory", "command", d), Channel("remembered", "choice", n_dir)))
    return Brain(w, body, codecs={"camera": sense}, rng=np.random.default_rng(seed)), sense


def certificate(d: int, n_dir: int, ks) -> list[float]:
    return list(certify(GaussianCodes(dim=d), np.arange(n_dir), d=d, ks=tuple(ks), n_trials=3000).accuracy)


class Walker:
    def __init__(self, viewer: bool):
        self.proc, port = ubrain_local.start_body(viewer=viewer)
        self.body = BodyClient("127.0.0.1", port)
        self.walk = self.body.primitives["walk"]["index"]
        self.viewer, self.t0, self.ticks = viewer, time.perf_counter(), 0
        self.state = None

    def step(self) -> np.ndarray:
        self.state = self.body.act(self.walk, [VX, 0.0, 0.0])
        self.ticks += 1
        if self.viewer:
            lag = self.t0 + self.ticks / HZ - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
        return np.asarray(self.state["proprio"])

    def close(self):
        try:
            self.body.close()
        finally:
            kill_tree(self.proc)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.split("\n")[7:]))
    ap.add_argument("--pushes", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--force", type=float, default=100.0)
    ap.add_argument("--directions", type=int, choices=(4, 8), default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--certify", action="store_true", help="the prediction only, no robot")
    ap.add_argument("--viewer", action="store_true", help="show the robot (real time)")
    ap.add_argument("--name", default="", help="a label for this run in results.csv")
    a = ap.parse_args(argv)
    if not 1 <= a.pushes <= SLOTS:
        ap.error(f"--pushes must be 1 .. {SLOTS}")
    names, angles = directions(a.directions)

    if a.certify:
        ks = (4, 8, 12, 16, 24)
        print(f"certificate: the predicted chance a push comes back right from a K-push episode,\n"
              f"D = {a.d}, {a.directions} directions (the law, evaluated on these codes; no robot)")
        for k, p in zip(ks, certificate(a.d, a.directions, ks)):
            print(f"  K {k:3d}: {p:.2f}")
        return 0

    law = certificate(a.d, a.directions, [a.pushes])[0]
    rng = np.random.default_rng(a.seed)
    w = Walker(a.viewer)
    totals = {"perceived": 0, "remembered": 0, "both": 0, "n": 0}
    falls = 0
    print(f"{a.episodes} episodes of {a.pushes} pushes ({a.force:.0f} N, {a.directions} directions), D = {a.d}; "
          f"the certificate predicts memory right {law:.2f}")
    try:
        for e in range(a.episodes):
            brain, _sense = observer(a.d, a.directions, a.seed * 1000 + e)
            w.body.reset(0)
            proprio = [w.step() for _ in range(SETTLE)]
            truth, percept = [], []
            for _k in range(a.pushes):
                proprio += [w.step() for _ in range(int(rng.integers(*GAP)))]
                d = int(rng.integers(a.directions))
                fx, fy = a.force * math.cos(angles[d]), a.force * math.sin(angles[d])
                w.body._send({"type": "world", "field": [fx, fy]})     # heading frame: the robot's own
                during = []
                for t in range(PUSH_TICKS):
                    during.append(w.step())
                w.body._send({"type": "world", "field": [0.0, 0.0]})
                seen = perceive(np.array(proprio[-BASELINE:]), np.array(during), angles)
                brain.tick({"camera": seen})                             # one shot: stored now
                truth.append(d)
                percept.append(seen)
                proprio += during
                if not w.state["upright"]:
                    falls += 1
                    w.body.reset(0)
                    proprio += [w.step() for _ in range(SETTLE)]
            recalled = [int(brain.tick({"recall": j})["remembered"]) for j in range(a.pushes)]
            brain.tick({"boundary": 0})                                  # the episode ends
            ok_p = sum(t == p for t, p in zip(truth, percept))
            ok_m = sum(p == r for p, r in zip(percept, recalled))
            ok_b = sum(t == r for t, r in zip(truth, recalled))
            for k, v in (("perceived", ok_p), ("remembered", ok_m), ("both", ok_b)):
                totals[k] += v
            totals["n"] += a.pushes
            print(f"\nepisode {e + 1}")
            print("  pushed     " + " ".join(f"{names[x]:>2s}" for x in truth))
            print("  perceived  " + " ".join(f"{names[x]:>2s}" for x in percept) + f"    {ok_p}/{a.pushes} right")
            print("  recalled   " + " ".join(f"{names[x]:>2s}" for x in recalled)
                  + f"    {ok_m}/{a.pushes} as perceived, {ok_b}/{a.pushes} as pushed")
    finally:
        w.close()

    n = totals["n"]
    print(f"\nover {n} pushes: perception {totals['perceived'] / n:.2f}, memory {totals['remembered'] / n:.2f} "
          f"(certificate {law:.2f}), both {totals['both'] / n:.2f}" + (f"; falls {falls}" if falls else ""))
    row = {"when": datetime.now().isoformat(timespec="seconds"), "name": a.name, "pushes": a.pushes,
           "episodes": a.episodes, "d": a.d, "force": a.force, "directions": a.directions, "seed": a.seed,
           "perception": round(totals["perceived"] / n, 3), "memory": round(totals["remembered"] / n, 3),
           "certificate": round(law, 3), "both": round(totals["both"] / n, 3), "falls": falls}
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
