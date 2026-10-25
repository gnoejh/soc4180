"""lab_hold: the G1 holds its last few pushes in working memory -- newest sharp, oldest gone.

    uv run weeks/21-working-memory/lab_hold.py                      # 5 lists of 16 pushes
    uv run weeks/21-working-memory/lab_hold.py --gamma 0.9          # a slower fade
    uv run weeks/21-working-memory/lab_hold.py --rehearse 16        # a full rehearsal cycle first
    uv run weeks/21-working-memory/lab_hold.py --law                # the prediction, no robot
    uv run weeks/21-working-memory/lab_hold.py --viewer --lists 1   # watch it (real time)

The pushes are week 20's: the G1 walks at 0.5 m/s and is pushed now and then (100 N,
0.1 s) in one of 8 directions in its own frame; its pelvis accelerometer senses each
direction (perceive(), as week 20). A list is 16 pushes in one walk.

WORKING MEMORY (ubrain's `holder` profile, prefrontal.buffer) keeps the list in ONE
vector, by position: push j is bound to position j's role and added in; before every
write the whole vector fades by gamma. So the newest push has weight 1, the one before
gamma, then gamma^2, ... After the list the lab asks for every position: unbind its
role, take the nearest direction. There is no cleanup behind it: old, faint pushes
drown in the crosstalk of the newer ones. Unlike week 20's hippocampus, where every slot
was equally good until the episode filled, recall here depends on AGE.

THE COUNT LAW (ubrain.datasheet.buffer_count) predicts, before anything is stored, how
many of the list come back: push j at weight gamma^(L-1-j) is read right with the decode
law's probability at d against the 8 directions; the expected count is the sum.
    --pushes L        the list length (1-16, default 16: the buffer's positions)
    --lists N         lists (default 5), each from a reset and a fresh brain
    --d D             the buffer's dimension (default 64)
    --gamma G         the fade per write (default 0.75)
    --rehearse R      R idle moments before recall: the buffer re-adds (rehearses) held
                      pushes by its schedule (default 0: none)
    --schedule S      which ones: cycle (all in turn, default), newest, oldest, random
    --law             no robot: the law's recall by age and its count
    --seed S          the pushes and the brain's random roles

Measured: recall by age (as perceived), the count held against the law, and perception.
Every run appends one row to results.csv. ubrain is the course's local copy.
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
from ubrain.datasheet import buffer_count, weighted_law  # noqa: E402

HERE = Path(__file__).resolve().parent
HZ = 50
VX = 0.5
SETTLE = 100
PUSH_TICKS = 5
GAP = (60, 100)
BASELINE = 40
PELVIS_ACC = slice(9, 12)
POSITIONS = 16                  # the holder profile's slots
NAMES = ("F", "FL", "L", "BL", "B", "BR", "R", "FR")
ANGLES = [2 * math.pi * i / 8 for i in range(8)]


def perceive(before: np.ndarray, during: np.ndarray) -> int:
    """Week 20's sense: the pelvis accelerometer during the push minus the stride before,
    as an angle in the robot's frame, rounded to the nearest of the 8 directions."""
    a = during[:, PELVIS_ACC].mean(0) - before[:, PELVIS_ACC].mean(0)
    ang = math.atan2(a[1], a[0])
    return int(np.argmin([abs((ang - t + math.pi) % (2 * math.pi) - math.pi) for t in ANGLES]))


def law_by_position(L: int, gamma: float, d: int) -> list[float]:
    w = gamma ** np.arange(L - 1, -1, -1)
    S = float(np.sum(w * w))
    return [weighted_law(float(x), S, d, 8) for x in w]


def holder(d: int, gamma: float, schedule: str, seed: int) -> Brain:
    sense = Sense.fit(GaussianCodes(dim=d), np.arange(8), d=d)
    w = profile("holder")
    w.regions["prefrontal.buffer"]["gamma_m"] = gamma
    w.regions["prefrontal.buffer"]["schedule"] = schedule
    body = Manifest("course/g1-pushes", "none", (
        Channel("camera", "codec", 1), Channel("recall", "event", POSITIONS), Channel("idle", "event", 1),
        Channel("clear", "event", 1), Channel("memory", "command", d), Channel("remembered", "choice", 8)))
    return Brain(w, body, codecs={"camera": sense}, rng=np.random.default_rng(seed))


class Walker:
    def __init__(self, viewer: bool):
        self.proc, port = ubrain_local.start_body(viewer=viewer)
        self.body = BodyClient("127.0.0.1", port)
        self.walk = self.body.primitives["walk"]["index"]
        self.viewer, self.t0, self.ticks, self.state = viewer, time.perf_counter(), 0, None

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
    ap.add_argument("--pushes", type=int, default=16)
    ap.add_argument("--lists", type=int, default=5)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--gamma", type=float, default=0.75)
    ap.add_argument("--rehearse", type=int, default=0)
    ap.add_argument("--schedule", choices=("cycle", "newest", "oldest", "random"), default="cycle")
    ap.add_argument("--law", action="store_true", help="the prediction only, no robot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--name", default="")
    a = ap.parse_args(argv)
    L = a.pushes
    if not 1 <= L <= POSITIONS:
        ap.error(f"--pushes must be 1 .. {POSITIONS}")
    law = law_by_position(L, a.gamma, a.d)
    count = buffer_count(L, a.gamma, a.d, 8)
    if a.law:
        print(f"the count law: a list of {L} at gamma {a.gamma}, d {a.d}, 8 directions: {count:.2f} held "
              f"(chance alone would give {L / 8:.2f})")
        print("  age (pushes since)   " + " ".join(f"{k:5d}" for k in range(L)))
        print("  recall predicted     " + " ".join(f"{law[L - 1 - k]:5.2f}" for k in range(L)))
        return 0

    rng = np.random.default_rng(a.seed)
    w = Walker(a.viewer)
    hits = np.zeros(L)                  # per position, recalled == perceived
    held, perceived_ok, n = [], 0, 0
    print(f"{a.lists} lists of {L} pushes; buffer d {a.d}, gamma {a.gamma}"
          + (f", {a.rehearse} rehearsals ({a.schedule})" if a.rehearse else "")
          + f"; the count law predicts {count:.2f} held")
    try:
        for li in range(a.lists):
            brain = holder(a.d, a.gamma, a.schedule, a.seed * 1000 + li)
            w.body.reset(0)
            proprio = [w.step() for _ in range(SETTLE)]
            truth, percept = [], []
            for _ in range(L):
                proprio += [w.step() for _ in range(int(rng.integers(*GAP)))]
                d = int(rng.integers(8))
                w.body._send({"type": "world", "field": [100.0 * math.cos(ANGLES[d]), 100.0 * math.sin(ANGLES[d])]})
                during = [w.step() for _ in range(PUSH_TICKS)]
                w.body._send({"type": "world", "field": [0.0, 0.0]})
                seen = perceive(np.array(proprio[-BASELINE:]), np.array(during))
                brain.tick({"camera": seen})                  # written into the buffer: everything else fades
                truth.append(d)
                percept.append(seen)
                proprio += during
                if not w.state["upright"]:
                    w.body.reset(0)
                    proprio += [w.step() for _ in range(SETTLE)]
            for _ in range(a.rehearse):
                brain.tick({"idle": True})                    # a free moment: rehearse by the schedule
            recalled = [int(brain.tick({"recall": j})["remembered"]) for j in range(L)]
            brain.tick({"clear": 0})
            ok = np.array([p == r for p, r in zip(percept, recalled)])
            hits += ok
            held.append(int(ok.sum()))
            perceived_ok += sum(t == p for t, p in zip(truth, percept))
            n += L
            print(f"\nlist {li + 1}  (oldest first; the last is the newest)")
            print("  perceived  " + " ".join(f"{NAMES[x]:>2s}" for x in percept))
            print("  recalled   " + " ".join(f"{NAMES[x]:>2s}" for x in recalled) + f"    {ok.sum()}/{L} held")
    finally:
        w.close()

    print(f"\nrecall by age (pushes since; 0 = the newest), {a.lists} lists:")
    print("  age          " + " ".join(f"{k:5d}" for k in range(L)))
    print("  law          " + " ".join(f"{law[L - 1 - k]:5.2f}" for k in range(L)))
    print("  measured     " + " ".join(f"{hits[L - 1 - k] / a.lists:5.2f}" for k in range(L)))
    print(f"\nheld per list: {np.mean(held):.2f} (law {count:.2f}; chance alone {L / 8:.2f}); "
          f"perception {perceived_ok / n:.2f}")
    row = {"when": datetime.now().isoformat(timespec="seconds"), "name": a.name, "pushes": L, "lists": a.lists,
           "d": a.d, "gamma": a.gamma, "rehearse": a.rehearse, "schedule": a.schedule, "seed": a.seed,
           "held": round(float(np.mean(held)), 3), "law": round(count, 3), "perception": round(perceived_ok / n, 3),
           "by_age": " ".join(f"{hits[L - 1 - k] / a.lists:.2f}" for k in range(L))}
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
