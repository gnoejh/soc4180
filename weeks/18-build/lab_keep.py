"""lab_keep (ubrain's examples/06_g1_keep.py on the course's copy): train the whole brain in the G1, save it, demonstrate the loaded brain (needs: the body).

    uv run weeks/18-build/lab_keep.py
    uv run weeks/18-build/lab_keep.py --viewer

What it shows  a brain's state is its knowledge, so a demonstration runs the SAVED brain, loaded, not a
               lucky live run.

The training   20 pushes of 250 N from the front, each 0.2 s after a tone: the cerebellum learns to brace.
Saving         brain.save(path): one file; it names the body's streams and the sense's codec.
The twin       10 more pushes, twice on the same body: the LOADED brain and a FRESH twin (never trained).
The check      the loaded brain falls at most once; the fresh twin at least twice (it falls while nothing
               is learned, as in ubrain's examples/04_g1_brace.py: 3 of its first 6 pushes).
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

import ubrain_local

ubrain_local.load()

from ubrain import Brain, GaussianCodes  # noqa: E402
from ubrain.bodies import BodyClient, kill_tree  # noqa: E402

start_body = ubrain_local.start_body
from ubrain.bodies.g1 import G1, upright  # noqa: E402

TRIAL, TONE_AT, DELAY, PUSH_N, VX = 200, 150, 10, 250.0, 0.4
CODEC = dict(dim=32, seed=3)


def fresh(g1: G1) -> Brain:
    brain = Brain(variant="small", rng=np.random.default_rng(1))
    brain.attach("sense", "cue", codec=GaussianCodes(**CODEC))
    brain.attach("action", n=2)
    g1.attach(brain)
    return brain


def pushes(body, g1: G1, brain: Brain, n: int, viewer: bool) -> list[bool]:
    """n pushes after a tone; True where the robot stayed up."""
    state, stayed, t0, tick = body.reset(0), [], time.perf_counter(), 0
    for _ in range(n):
        fell = False
        for k in range(TRIAL):
            out = brain.tick(g1.streams(state, (VX, 0.0, 0.0)) | {"sense.cue": 0})
            if k == TONE_AT:
                body._send({"type": "world", "tone": True,
                            "push": {"force": [PUSH_N, 0.0], "delay": DELAY, "ticks": 5}})
            state = body.act(*g1.act(out))
            tick += 1
            if viewer:
                time.sleep(max(0.0, t0 + tick * 0.02 - time.perf_counter()))
            if not upright(state):
                fell = True
                state = body.reset(0)
                brain.body_reset()
        stayed.append(not fell)
    return stayed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="train, save, demonstrate the loaded brain")
    ap.add_argument("--train", type=int, default=20)
    ap.add_argument("--demo", type=int, default=10)
    ap.add_argument("--viewer", action="store_true")
    a = ap.parse_args(argv)
    proc, port = start_body(viewer=a.viewer)
    try:
        with BodyClient("127.0.0.1", port) as body, tempfile.TemporaryDirectory() as tmp:
            g1 = G1(body.hello)
            brain = fresh(g1)
            trained = pushes(body, g1, brain, a.train, False)
            path = brain.save(Path(tmp) / "braced.ubrain.npz")
            print(f"trained on {a.train} pushes: " + "".join("." if s else "X" for s in trained)
                  + f"; saved {Path(path).stat().st_size / 1e6:.1f} MB")
            loaded = Brain.load(path, codecs={"sense.cue": GaussianCodes(**CODEC)})
            demo_loaded = pushes(body, g1, loaded, a.demo, a.viewer)
            demo_fresh = pushes(body, g1, fresh(g1), a.demo, a.viewer)
    finally:
        kill_tree(proc)
    fl, ff = sum(not s for s in demo_loaded), sum(not s for s in demo_fresh)
    print("loaded brain: " + "".join("." if s else "X" for s in demo_loaded) + f"   falls {fl}")
    print("fresh twin:   " + "".join("." if s else "X" for s in demo_fresh) + f"   falls {ff}")
    ok = fl <= 1 and ff >= 2
    print("CHECK", "PASS" if ok else "FAIL", f"(loaded falls {fl} <= 1; fresh falls {ff} >= 2)")
    import csv
    from datetime import datetime
    from pathlib import Path as _P
    res = _P(__file__).resolve().parent / "results.csv"
    new = not res.exists()
    with res.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["when", "lab", "check", "argv"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), "keep", "PASS" if ok else "FAIL", " ".join(argv or [])])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
