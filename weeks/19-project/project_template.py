"""project_template: a working capstone project to copy -- question, prediction, twin, measurement, check.

    uv run weeks/19-project/project_template.py               # runs as it is and ends with CHECK PASS
    uv run weeks/19-project/project_template.py --viewer      # watch it

Copy this file (e.g. weeks/19-project/team3.py), then change the places marked # CHANGE.

THE QUESTION    (# CHANGE) does a warning tone keep the walking G1 up when it is pushed from the front?
THE PREDICTION  (# CHANGE, written before the run, dated) "the brain that hears the tone falls at least
                5 times fewer in 20 pushes than the twin, because the cerebellum learns the counter-move
                from the tone's trace" -- 2026-11-17.
THE TWIN        the same brain, the same pushes, without the tone (one thing differs).
THE MEASUREMENT falls per brain, appended to results.csv beside this file.
THE CHECK       (# CHANGE, decided before the final run) falls(twin) - falls(brain) >= 5.

Every signal, sorted (put this table in your report):
    body.proprio   sense       the body's proprioception
    cue.tone       sense       the tone it hears
    goal           program's   the walk the program wants
    command        brain's     the walk the brain sends
    upright        privilege   used only to count falls (and reset)
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ubrain import Brain, GaussianCodes
from ubrain.bodies import BodyClient
from ubrain.bodies.g1 import G1, upright

import ubrain_local                                   # the course's G1 body (vendor/ubrain/g1_body)

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- the world: # CHANGE
PUSHES = 20              # pushes per brain
PUSH_N = 250.0           # newtons, from the front ([fx, fy] below)
LEAD = 10                # ticks from the tone to the push (0.2 s at 50 Hz)
TRIAL = 200              # ticks between pushes (4 s)
WALK = (0.4, 0.0, 0.0)   # the goal: vx (m/s), vy (m/s), wz (rad/s)


def make_brain(g1: G1) -> Brain:
    """The whole brain with the G1's streams. # CHANGE the variant, add senses, an action meaning."""
    brain = Brain(variant="small", rng=np.random.default_rng(1))
    brain.attach("sense", "cue", codec=GaussianCodes(dim=32, seed=3))   # a whole brain needs a sense...
    brain.attach("action", n=2)                                         # ... and an action
    g1.attach(brain)                                                    # body.proprio, (cue.tone), goal, command
    return brain


def run_one(body, hears: bool, viewer: bool) -> str:
    """One brain's life on the robot: '.' stayed up, 'X' fell, one mark per push. # CHANGE the measure."""
    g1 = G1(body.hello, cue=hears)                     # the twin differs here, and only here
    brain = make_brain(g1)
    state, marks, t0, tick = body.reset(0), "", time.perf_counter(), 0
    for _ in range(PUSHES):
        fell = False
        for k in range(TRIAL):
            out = brain.tick(g1.streams(state, WALK) | {"sense.cue": 0})
            if k == TRIAL - 50 - LEAD:                 # the experimenter's hand: the tone, then the push
                body._send({"type": "world", "tone": True,
                            "push": {"force": [PUSH_N, 0.0], "delay": LEAD, "ticks": 5}})
            state = body.act(*g1.act(out))
            tick += 1
            if viewer:
                time.sleep(max(0.0, t0 + tick * 0.02 - time.perf_counter()))
            if not upright(state):
                fell = True
                state = body.reset(0)
                brain.body_reset()
        marks += "X" if fell else "."
    return marks


def record(rows: list[tuple]) -> None:
    path = HERE / "results.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["when", "script", "brain", "measure", "value"])
        stamp = datetime.now().isoformat(timespec="seconds")
        for r in rows:
            w.writerow([stamp, Path(__file__).name, *r])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--viewer", action="store_true")
    a = ap.parse_args(argv)
    proc, port = ubrain_local.start_body(viewer=a.viewer)
    try:
        with BodyClient("127.0.0.1", port) as body:
            brain_marks = run_one(body, True, a.viewer)
            twin_marks = run_one(body, False, a.viewer)
    finally:
        proc.kill()
    fb, ft = brain_marks.count("X"), twin_marks.count("X")
    print(f"the brain  {brain_marks}   falls {fb}")
    print(f"the twin   {twin_marks}   falls {ft}      (X fell, . stayed up)")
    record([("brain", "falls", fb), ("twin", "falls", ft)])
    ok = ft - fb >= 5                                  # THE CHECK: # CHANGE, before the final run
    print("CHECK", "PASS" if ok else "FAIL", f"(twin {ft} - brain {fb} >= 5)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
