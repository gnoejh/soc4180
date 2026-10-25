"""lab_notice: the G1 notices only what is new -- a salience gate in front of its memory.

    uv run weeks/23-attention/lab_notice.py                     # 20 bursts, random directions
    uv run weeks/23-attention/lab_notice.py --world gusty       # mostly the same two sides
    uv run weeks/23-attention/lab_notice.py --alpha 0.3         # a more forgetful gate
    uv run weeks/23-attention/lab_notice.py --window 2          # a gate with a short memory
    uv run weeks/23-attention/lab_notice.py --viewer --bursts 3

The pushes are week 20's (100 N, 0.1 s, 8 directions in the robot's own frame, sensed by
the pelvis accelerometer). They come in BURSTS of pushes with quiet walking between.

THE ATTENDER (ubrain's `attender` profile) puts a gate in front of memory:
  salience.gate     judges each percept against the last `window` that ARRIVED (8): its
                    novelty is 1 - the largest cosine with any of them. Below a threshold
                    it is FAMILIAR and skipped; above, NOVEL and written into the episode.
                    The threshold is not set by hand: it is derived from alpha, the chance
                    that something fresh is mistaken for familiar.
  modes.controller  a novel push is a demand: executive mode while demands keep coming,
                    default mode when nothing new is pending.
  hippocampus +     the first idle (default-mode) moment ends the episode and replays it
  cortex            into the cortex (week 22).
A push the robot felt recently has novelty exactly 0: it is always skipped.
    --world W         random (each push from any of the 8) or gusty (from the same two
                      sides with probability 0.8, any other side otherwise)
    --bursts N        bursts (default 20)      --burst B   pushes per burst (default 6)
    --quiet Q         idle moments (1 s each) between bursts (default 8; 4 is too little rest)
    --window W        the gate's memory of arrivals (default 8)
    --alpha A         the chance a fresh push is judged familiar (default 0.05)
    --d D             the dimension of the codes (default 64)
    --seed S

Measured: how often a push is judged familiar, split by whether its direction arrived
within the gate's window (a recent repeat) or not, beside the predicted rate; how much of
the stream was written; the modes; what the cortex recalls of what was written. Every
run appends one row to results.csv. ubrain is the course's local copy (see ubrain_local.py).
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

from ubrain import Brain, Channel, Manifest, Wiring, profile  # noqa: E402
from ubrain.bodies import BodyClient, kill_tree  # noqa: E402
from ubrain.codecs import GaussianCodes, Sense  # noqa: E402
from ubrain.regions import familiar_threshold, spherical_tail  # noqa: E402

HERE = Path(__file__).resolve().parent
HZ = 50
VX = 0.5
SETTLE = 100
PUSH_TICKS = 5
GAP = (60, 100)                 # ticks between pushes inside a burst
QUIET_TICKS = 50                # walking ticks per idle moment between bursts (1 s)
BASELINE = 40
PELVIS_ACC = slice(9, 12)
NAMES = ("F", "FL", "L", "BL", "B", "BR", "R", "FR")
ANGLES = [2 * math.pi * i / 8 for i in range(8)]
GUSTY_SIDES = (2, 6)            # L and R: the gusty world's usual two sides
GUSTY_P = 0.8


def perceive(before: np.ndarray, during: np.ndarray) -> int:
    """Week 20's sense: the pelvis accelerometer during the push minus the stride before,
    as an angle in the robot's frame, rounded to the nearest of the 8 directions."""
    a = during[:, PELVIS_ACC].mean(0) - before[:, PELVIS_ACC].mean(0)
    ang = math.atan2(a[1], a[0])
    return int(np.argmin([abs((ang - t + math.pi) % (2 * math.pi) - math.pi) for t in ANGLES]))


def attender(d: int, window: int, alpha: float, seed: int) -> Brain:
    sense = Sense.fit(GaussianCodes(dim=d), np.arange(8), d=d)
    w = profile("attender")
    w.regions["salience.gate"]["window"] = window
    w.regions["salience.gate"]["alpha"] = alpha
    w.regions["hippocampus.episode"]["slots"] = 32
    wiring = Wiring(w.name, w.contract, w.regions, w.inputs, {**w.outputs, "novel": "salience.gate.novel"})
    body = Manifest("course/g1-pushes", "none", (
        Channel("camera", "codec", 1), Channel("recall", "event", 32), Channel("recollect", "event", 4096),
        Channel("memory", "command", d), Channel("remembered", "choice", 8),
        Channel("recollection", "command", d), Channel("recollected", "choice", 8),
        Channel("mode", "choice", 2), Channel("novel", "choice", 2)))
    return Brain(wiring, body, codecs={"camera": sense}, rng=np.random.default_rng(seed))


class Pusher:
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

    def walk_for(self, ticks: int) -> None:
        self.history = (self.history + [self.step() for _ in range(ticks)])[-200:]

    def push(self, d: int) -> int:
        self.walk_for(int(self.rng.integers(*GAP)))
        self.body._send({"type": "world", "field": [100.0 * math.cos(ANGLES[d]), 100.0 * math.sin(ANGLES[d])]})
        during = [self.step() for _ in range(PUSH_TICKS)]
        self.body._send({"type": "world", "field": [0.0, 0.0]})
        seen = perceive(np.array(self.history[-BASELINE:]), np.array(during))
        self.history = (self.history + during)[-200:]
        if not self.state["upright"]:
            self.body.reset(0)
            self.history = [self.step() for _ in range(SETTLE)]
        return seen

    def close(self):
        try:
            self.body.close()
        finally:
            kill_tree(self.proc)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.split("\n")[7:]))
    ap.add_argument("--world", choices=("random", "gusty"), default="random")
    ap.add_argument("--bursts", type=int, default=20)
    ap.add_argument("--burst", type=int, default=6)
    ap.add_argument("--quiet", type=int, default=8)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--name", default="")
    a = ap.parse_args(argv)

    theta = familiar_threshold(a.d, a.window, a.alpha)
    t = spherical_tail(1.0 - theta, a.d)
    rng = np.random.default_rng(a.seed)
    brain = attender(a.d, a.window, a.alpha, a.seed + 1)
    print(f"the gate: window {a.window}, alpha {a.alpha}, d {a.d}: threshold {theta:.3f}; "
          f"a fresh push is judged familiar with probability 1 - (1 - {t:.4f})^n, n = distinct directions in the window")
    robot = Pusher(a.viewer, rng)
    window: list[int] = []
    judged = {"recent": [], "other": []}           # (familiar?, predicted P(familiar))
    written, total, executive, ticks, lines, replays = [], 0, 0, 0, [], []
    episode = brain.regions["hippocampus.episode"]

    def tick(senses):
        """One brain moment; an idle moment with something in the episode replays it."""
        load = episode.n
        out = brain.tick(senses)
        if out["mode"] == 0 and load > 0:
            replays.append(load)
        return out
    try:
        for b in range(a.bursts):
            row_p, row_g, row_m = [], [], []
            for _ in range(a.burst):
                if a.world == "gusty" and rng.random() < GUSTY_P:
                    d = int(rng.choice(GUSTY_SIDES))
                else:
                    d = int(rng.integers(8))
                seen = robot.push(d)
                kind = "recent" if seen in window else "other"
                n = len(set(window))
                p = 1.0 if kind == "recent" else 1.0 - (1.0 - t) ** n
                out = tick({"camera": seen})
                judged[kind].append((not out["novel"], p))
                if out["novel"]:
                    written.append(seen)
                window = (window + [seen])[-a.window:]
                total += 1
                executive += out["mode"] == 1
                ticks += 1
                row_p.append(NAMES[seen])
                row_g.append("+" if out["novel"] else "-")
                row_m.append("E" if out["mode"] == 1 else "d")
            for _ in range(a.quiet):                     # quiet walking: idle moments, no push
                robot.walk_for(QUIET_TICKS)
                out = tick({"camera": None})
                executive += out["mode"] == 1
                ticks += 1
                row_m.append("E" if out["mode"] == 1 else "d")
            if b < 3:
                lines.append((row_p, row_g, row_m))
    finally:
        robot.close()

    for i, (rp, rg, rm) in enumerate(lines):
        print(f"\nburst {i + 1}")
        print("  perceived  " + " ".join(f"{x:>2s}" for x in rp))
        print("  gate       " + " ".join(f"{x:>2s}" for x in rg) + "     + novel (written), - familiar (skipped)")
        print("  mode       " + " ".join(f"{x:>2s}" for x in rm) + "     E executive, d default (the last "
              f"{a.quiet}: quiet moments)")
    store = brain.regions["cortex.store"]
    got = [brain.tick({"recollect": i})["recollected"] for i in range(len(store.keys))]
    # the cortex holds the replayed pushes, in the order they were written; pushes still in the
    # episode when the run ends (the brain never rested after them) were never consolidated
    cortex_ok = float(np.mean([g == w for g, w in zip(got, written)])) if got else float("nan")
    print("\njudged familiar             predicted   measured   (pushes)")
    for kind, label in (("recent", "felt within the window"), ("other", "not felt within the window")):
        if judged[kind]:
            fam = np.mean([f for f, _ in judged[kind]])
            pred = np.mean([p for _, p in judged[kind]])
            print(f"  {label:26s} {pred:.3f}      {fam:.3f}      ({len(judged[kind])})")
    print(f"\nwritten into memory: {len(written)} of {total} pushes ({len(written) / total:.2f})")
    print(f"executive mode: {executive / ticks:.2f} of the brain's moments; {len(replays)} episodes replayed into the "
          f"cortex, {np.mean(replays) if replays else 0:.1f} pushes each on average (largest {max(replays) if replays else 0})")
    print(f"consolidated: {len(got)} of the {len(written)} written pushes reached the cortex "
          f"(the rest were still in the episode); the cortex recalls {cortex_ok:.2f} of those")
    row = {"when": datetime.now().isoformat(timespec="seconds"), "name": a.name, "world": a.world, "bursts": a.bursts,
           "burst": a.burst, "window": a.window, "alpha": a.alpha, "d": a.d, "seed": a.seed,
           "familiar_recent": round(float(np.mean([f for f, _ in judged["recent"]])), 3) if judged["recent"] else "",
           "familiar_other": round(float(np.mean([f for f, _ in judged["other"]])), 3) if judged["other"] else "",
           "predicted_other": round(float(np.mean([p for _, p in judged["other"]])), 3) if judged["other"] else "",
           "written": round(len(written) / total, 3), "executive": round(executive / ticks, 3),
           "replays": len(replays), "mean_replay": round(float(np.mean(replays)), 2) if replays else 0,
           "consolidated": round(len(got) / max(len(written), 1), 3), "cortex": round(cortex_ok, 3)}
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
