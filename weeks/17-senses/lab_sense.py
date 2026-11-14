"""lab_sense: the whole brain takes the G1's body -- its streams, its senses, and what each is.

    uv run weeks/29-senses/lab_sense.py --step brace          # the twin: the same pushes, heard and unheard
    uv run weeks/29-senses/lab_sense.py --step push           # which way was it pushed? (a detector)
    uv run weeks/29-senses/lab_sense.py --step contact        # the feet's forces (privileged)
    uv run --extra vision weeks/29-senses/lab_sense.py --step vision   # the head camera through CLIP
    uv run weeks/29-senses/lab_sense.py --step brace --viewer # watch it

Release 0.46 gives the whole brain the body's classes (week 28 ran the G1 on the walker profile):
  body.proprio  the body's proprioception: 70 numbers (IMU of torso and pelvis, 29 joint angles, 29 velocities)
  cue.tone      a sound it hears: the tone the experimenter plays before a push
  goal          the walk the program wants, (vx, vy, wz)
  command       the walk the brain sends: the goal plus the cerebellum's counter-move
`ubrain.bodies.g1.G1` attaches them and turns a body state into a tick and the command into an act request.

  step brace    TWO brains, one stream of pushes (250 N from the front, every 4 s). Brain A hears the tone
                (cue.tone attached); brain B does not (no cue). The difference is the cue and nothing else.
  step push     the program pushes from one of four sides; push_direction reads the pelvis accelerometer
                (during the push minus the stride before) and names the nearest side. A DETECTOR the
                program owns, not a sense the brain has: the program knows when it pushed.
  step contact  the feet's contact forces, asked for each tick (want_contact=True). PRIVILEGED: a real
                humanoid has no foot force sensors; use them for a reward or an analysis.
  step vision   the G1 walks down the gallery with its head camera; every 10th tick a frame goes to
                sense.vision through CLIP (week 26). The whole brain's salience gate reports what is NEW
                (read("novel")); the brain engages, and chooses, only on what is new or surprising.

Options:
    --step S        brace | push | contact | vision
    --pushes N      pushes in steps brace and push (default 20 / 16)
    --force F       newtons (default 250 in brace, 100 in push: it does not fall)
    --vx V          walking speed (default 0.4)
    --delay K       ticks from the tone to the push, step brace (default 10: 0.2 s)
    --seconds S     how long, steps contact and vision (default 4 / 40)
    --viewer        real time, with the viewer
    --seed N        the push sides in step push

Every run appends rows to results.csv (step, setting, measure, value).
ubrain and the body are the course's local copy (week 29: release 0.46).
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

ubrain_local.load()

import ubrain  # noqa: E402
from ubrain import Brain, GaussianCodes  # noqa: E402
from ubrain.bodies import BodyClient  # noqa: E402
from ubrain.bodies.g1 import G1, privileged_contact, push_direction, upright  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.csv"
HZ = 50
SIDES = ["front", "left", "back", "right"]          # the side the push comes FROM
ANGLES = [0.0, math.pi / 2, math.pi, -math.pi / 2]  # the direction the body is pushed TOWARD (x fwd, y left)


def record(rows):
    new = not RESULTS.exists()
    with RESULTS.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["when", "step", "setting", "measure", "value"])
        stamp = datetime.now().isoformat(timespec="seconds")
        for r in rows:
            w.writerow([stamp, *r])


class Run:
    """The body, its G1 streams, and a lockstep (or real-time) clock."""

    def __init__(self, a, **body_kw):
        self.proc, port = ubrain_local.start_body(viewer=a.viewer, **body_kw)
        self.body = BodyClient("127.0.0.1", port)
        self.viewer, self.t0, self.ticks = a.viewer, time.perf_counter(), 0

    def wait(self):
        self.ticks += 1
        if self.viewer:
            time.sleep(max(0.0, self.t0 + self.ticks / HZ - time.perf_counter()))

    def push(self, fx, fy, tone=False, delay=10):
        self.body._send({"type": "world", "tone": tone, "push": {"force": [fx, fy], "delay": delay, "ticks": 5}})

    def close(self):
        self.body.close()
        self.proc.kill()


def whole_brain(g1: G1, seed=1) -> Brain:
    """A whole brain with the G1's streams; it also needs a sense and an action (here one fixed thing seen)."""
    brain = Brain(variant="small", rng=np.random.default_rng(seed))
    brain.attach("sense", "cue", codec=GaussianCodes(dim=32, seed=3))
    brain.attach("action", n=2)
    g1.attach(brain)
    return brain


# ------------------------------------------------------------------ step brace: the twin
def step_brace(a):
    print(f"\n== step brace: {a.pushes} pushes of {a.force:.0f} N from the front, {a.delay / HZ:.2f} s after a tone ==")
    rows, lines = [], {}
    for heard in (True, False):
        run = Run(a)
        try:
            g1 = G1(run.body.hello, cue=heard)             # brain B gets no cue stream: it cannot hear
            brain = whole_brain(g1)
            state, goal, stayed = run.body.reset(0), (a.vx, 0.0, 0.0), []
            for _ in range(a.pushes):
                fell = False
                for k in range(200):
                    out = brain.tick(g1.streams(state, goal) | {"sense.cue": 0})
                    if k == 160 - a.delay:
                        run.push(a.force, 0.0, tone=True, delay=a.delay)
                    state = run.body.act(*g1.act(out))
                    run.wait()
                    if not upright(state):
                        fell = True
                        state = run.body.reset(0)
                        brain.body_reset()
                stayed.append(not fell)
            name = "A hears the tone " if heard else "B does not hear  "
            falls = sum(not s for s in stayed)
            lines[heard] = f"  {name} " + "".join("." if s else "X" for s in stayed) + f"   falls {falls}"
            rows.append(("brace", f"heard{int(heard)},F{a.force:.0f},D{a.delay}", "falls", falls))
            rows.append(("brace", f"heard{int(heard)},F{a.force:.0f},D{a.delay}", "falls_last_10",
                         sum(not s for s in stayed[-10:])))
        finally:
            run.close()
    print(lines[True]); print(lines[False]); print("  (X fell, . stayed up)")
    record(rows)


# ------------------------------------------------------------------ step push: a detector
def step_push(a):
    print(f"\n== step push: {a.pushes} pushes of {a.force:.0f} N from a random side; push_direction names it ==")
    rng = np.random.default_rng(a.seed)
    run = Run(a)
    try:
        g1 = G1(run.body.hello)
        brain = whole_brain(g1)
        state, goal = run.body.reset(0), (a.vx, 0.0, 0.0)
        hist, right, confusion = [], 0, np.zeros((4, 4), int)
        for _ in range(a.pushes):
            side = int(rng.integers(4))
            gap = int(rng.integers(60, 100))
            before, during = [], []
            for k in range(gap + 5):
                out = brain.tick(g1.streams(state, goal) | {"sense.cue": 0})
                if k == gap - 1:
                    ang = ANGLES[side]
                    run.push(a.force * math.cos(ang), a.force * math.sin(ang), delay=1)
                state = run.body.act(*g1.act(out))
                run.wait()
                p = np.asarray(state["proprio"])
                (during if k >= gap else before).append(p)
                if not upright(state):
                    state = run.body.reset(0)
                    brain.body_reset()
            ang = push_direction(np.array(before[-40:]), np.array(during))
            seen = int(np.argmin([abs((ang - t + math.pi) % (2 * math.pi) - math.pi) for t in ANGLES]))
            confusion[side, seen] += 1
            right += seen == side
        print(f"  named right: {right} of {a.pushes}")
        print("  pushed from \\ named as  " + "  ".join(f"{s:>5s}" for s in SIDES))
        for i, s in enumerate(SIDES):
            print(f"  {s:>22s}  " + "  ".join(f"{confusion[i, j]:5d}" for j in range(4)))
        record([("push", f"F{a.force:.0f}", "right", right), ("push", f"F{a.force:.0f}", "of", a.pushes)])
    finally:
        run.close()


# ------------------------------------------------------------------ step contact: privileged
def step_contact(a):
    print(f"\n== step contact: the feet's forces over {a.seconds:.0f} s of walking (PRIVILEGED) ==")
    run = Run(a)
    try:
        g1 = G1(run.body.hello)
        brain = whole_brain(g1)
        state, goal, fz = run.body.reset(0), (a.vx, 0.0, 0.0), []
        for t in range(int(a.seconds * HZ)):
            out = brain.tick(g1.streams(state, goal) | {"sense.cue": 0})
            state = run.body.act(*g1.act(out), want_contact=True)
            run.wait()
            c = privileged_contact(state)
            if c is not None:
                fz.append((abs(c[0]), abs(c[6])))        # each foot's normal force: the contact frame's x
        fz = np.array(fz)
        on = fz > 20.0                                   # newtons: the foot is bearing weight
        both, left, right_, none = (on.all(1).mean(), (on[:, 0] & ~on[:, 1]).mean(),
                                    (~on[:, 0] & on[:, 1]).mean(), (~on).all(1).mean())
        print(f"  share of ticks: both feet {both:.2f}, left only {left:.2f}, right only {right_:.2f}, "
              f"neither {none:.2f}")
        print(f"  mean normal force on a loaded foot: {fz[on].mean():.0f} N; on both feet together: "
              f"{fz.sum(1).mean():.0f} N (the robot's weight, carried by the feet)")
        record([("contact", f"vx{a.vx}", k, round(float(v), 3)) for k, v in
                (("both", both), ("left_only", left), ("right_only", right_), ("neither", none))])
    finally:
        run.close()


# ------------------------------------------------------------------ step vision: what is new
def step_vision(a):
    print(f"\n== step vision: the head camera through CLIP, every 10th tick, {a.seconds:.0f} s down the gallery ==")
    from ubrain.extras.clip import CLIP
    run = Run(a, head_camera=True, gallery=True)
    try:
        g1 = G1(run.body.hello, camera=CLIP("image"))
        brain = Brain(variant="standard", rng=np.random.default_rng(1))
        brain.attach("action", n=2)
        g1.attach(brain)                                 # adds sense.vision
        state, goal = run.body.reset(0), (0.5, 0.0, 0.0)
        glances, novel = 0, []
        for t in range(int(a.seconds * HZ)):
            look = t % 10 == 0
            ins = g1.streams(state, goal)
            if not look:
                ins.pop("sense.vision", None)
            out = brain.tick(ins)
            if look and "sense.vision" in ins:
                glances += 1
                novel.append(bool(brain.read("novel")))
            state = run.body.act(*g1.act(out), want_image=(t + 1) % 10 == 0, image_size=(224, 224))
            run.wait()
            if not upright(state):
                state = run.body.reset(0)
                brain.body_reset()
        n = sum(novel)
        print(f"  {glances} glances; the salience gate called {n} of them new")
        print("  per glance (N new, . seen before): " + "".join("N" if x else "." for x in novel))
        record([("vision", f"s{a.seconds:.0f}", "glances", glances), ("vision", f"s{a.seconds:.0f}", "novel", n)])
    finally:
        run.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--step", required=True, choices=["brace", "push", "contact", "vision"])
    ap.add_argument("--pushes", type=int, default=None)
    ap.add_argument("--force", type=float, default=None)
    ap.add_argument("--vx", type=float, default=0.4)
    ap.add_argument("--delay", type=int, default=10, help="ticks from the tone to the push (step brace)")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    a.pushes = a.pushes or (20 if a.step == "brace" else 16)
    a.force = a.force or (250.0 if a.step == "brace" else 100.0)
    a.seconds = a.seconds or (4.0 if a.step == "contact" else 40.0)
    print(f"ubrain {ubrain.__version__}")
    {"brace": step_brace, "push": step_push, "contact": step_contact, "vision": step_vision}[a.step](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
