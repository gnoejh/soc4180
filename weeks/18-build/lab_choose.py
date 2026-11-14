"""lab_choose (ubrain's examples/05_g1_choose.py on the course's copy): the whole brain in the G1 learns which way to turn from a picture (needs: the body).

    uv run weeks/18-build/lab_choose.py
    uv run weeks/18-build/lab_choose.py --viewer

The world      each trial (2 s) begins with a picture: GO LEFT or GO RIGHT. The brain chooses to turn left
               or right (the program maps the action to the goal: walk at 0.3 m/s turning at +-0.6 rad/s).
The reward     1 if the robot really turned the way the picture asked, 0 if not. The program measures the
               turn by integrating the torso gyro's z over the trial: a DETECTOR it owns.
The question   does the whole brain learn to turn the way each picture asks?
The twin       the same brain shown a fresh, unrelated picture every trial: it engages, but nothing it
               sees says which way to turn, so it can be right only half the time.
The check      over the last half: the brain that sees is right in more than 80% of trials; the blind
               twin in less than 70%.

One rule every choice project needs (measured while writing this example): the whole brain explores only
while its values are near zero. Once an action has paid, an action never tried is almost never tried (the Go
race picks in proportion to value). A first version of this example rewarded walking through a warned push
(it fells a walking G1 about half the time) and stopping with 0.8: the brain kept whatever it did first, on
every trial. Make a wrong choice pay nothing. (Whether the reward comes on the trial's last tick with its
picture, or with the next trial's picture, made no difference here: both learned in two trials.)

Whatever you change, write your prediction down first.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import ubrain_local

ubrain_local.load()

from ubrain import Brain, GaussianCodes  # noqa: E402
from ubrain.bodies import BodyClient, kill_tree  # noqa: E402

start_body = ubrain_local.start_body
from ubrain.bodies.g1 import G1, upright  # noqa: E402

TURNS = [(0.3, 0.0, 0.6), (0.3, 0.0, -0.6)]   # action 0: turn left, 1: turn right
TRIAL = 100                                   # ticks (2 s at 50 Hz)
GYRO_Z = 2                                    # proprio: the torso gyro's z (rad/s, left positive)


def run(body, trials: int, sees: bool, seed: int, viewer: bool) -> list[tuple[int, int, float]]:
    """One brain's life: (asked, chose, reward) per trial."""
    g1 = G1(body.hello, cue=False)
    brain = Brain(variant="small", rng=np.random.default_rng(seed))
    brain.attach("sense", "picture", codec=GaussianCodes(dim=32, seed=7))
    brain.attach("reward")
    brain.attach("action", n=2)
    g1.attach(brain)
    world = np.random.default_rng(seed + 100)
    state, rows, t0, tick = body.reset(0), [], time.perf_counter(), 0
    for _ in range(trials):
        asked = int(world.integers(2))                # 0 go left, 1 go right
        picture = asked if sees else 100 + len(rows)  # the twin: a fresh, unrelated picture
        out = brain.tick(g1.streams(state, (0.3, 0.0, 0.0)) | {"sense.picture": picture, "reward": None})
        chose = int(out["action"]) if out["action"] is not None else int(world.integers(2))
        goal, turned = TURNS[chose], 0.0
        state = body.act(*g1.act(out))
        for _k in range(1, TRIAL):
            out = brain.tick(g1.streams(state, goal) | {"sense.picture": None, "reward": None})
            state = body.act(*g1.act(out))
            turned += state["proprio"][GYRO_Z] * 0.02          # the detector: the turn, integrated
            tick += 1
            if viewer:
                time.sleep(max(0.0, t0 + tick * 0.02 - time.perf_counter()))
            if not upright(state):
                state = body.reset(0)
                brain.body_reset()
        went = 0 if turned > 0 else 1
        reward = 1.0 if went == asked else 0.0
        out = brain.tick(g1.streams(state, goal) | {"sense.picture": picture, "reward": reward})
        state = body.act(*g1.act(out))                # the reward, with the trial's picture
        rows.append((asked, chose, reward))
    return rows


def right_rate(rows) -> float:
    half = rows[len(rows) // 2:]
    return float(np.mean([r for *_, r in half]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="the whole brain learns which way to turn")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--viewer", action="store_true")
    a = ap.parse_args(argv)
    proc, port = start_body(viewer=a.viewer)
    try:
        with BodyClient("127.0.0.1", port) as body:
            seeing = run(body, a.trials, True, a.seed, a.viewer)
            blind = run(body, a.trials, False, a.seed, a.viewer)
    finally:
        kill_tree(proc)
    for name, rows in (("sees the picture", seeing), ("blind twin      ", blind)):
        print(f"{name}  right in the last half: {right_rate(rows):.2f}")
        print("   trials (. right, X wrong):  " + "".join("." if r else "X" for *_, r in rows))
    s, b = right_rate(seeing), right_rate(blind)
    ok = s > 0.8 and b < 0.7
    print("CHECK", "PASS" if ok else "FAIL", f"(seeing {s:.2f} > 0.80; blind {b:.2f} < 0.70)")
    import csv
    from datetime import datetime
    from pathlib import Path as _P
    res = _P(__file__).resolve().parent / "results.csv"
    new = not res.exists()
    with res.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["when", "lab", "check", "argv"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), "choose", "PASS" if ok else "FAIL", " ".join(argv or [])])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
