"""P16 default solution: learn to dance faster -- the G1 learns WHEN to move.

    uv run weeks/projects/examples/p16_learn_timing.py                 # learn, test, replay
    uv run weeks/projects/examples/p16_learn_timing.py --no-viewer     # headless
    uv run weeks/projects/examples/p16_learn_timing.py --updates 0     # no learning: the baseline
    uv run weeks/projects/examples/p16_learn_timing.py --replay        # replay the saved result

THE TASK. P15's dance stands up to tempo 1.35 and falls from 1.4. Here the dance must
last as long as it does at TARGET_TEMPO (1.5: 12.7 s instead of 19.0 s), but the robot
may split that time between the 18 glides as it likes -- more time for moving the weight
onto one foot, less for waving. It learns the split from one number per try: how much
of the dance it got through before falling.

THE LEARNER is week 8's (weeks/08-learning-to-act/lab.py), unchanged, on 18 numbers:

    theta[i]   one per glide; glide i gets  T * base[i] * exp(theta[i]) / sum(...)
    repeat:
        draw 8 random directions eps_k ~ N(0, 1) (18 numbers each)
        play the dance with theta + sigma*eps_k and with theta - sigma*eps_k
        G = fraction of the dance done before falling, + 1 if it never fell
        theta <- theta + lr * mean_k[ (G+_k - G-_k) * eps_k ] / (2 sigma)

The simulation is deterministic, so the two tries of a pair differ only by theta: the
difference G+ - G- says which way to move. No GPU: each try is a few seconds of MuJoCo
on one CPU core, and the 16 tries of an update run in parallel.

MEASURED with this file (seed 0, 30 updates of 8 pairs, 219 s on a 36-core machine;
the run repeats exactly; p16_learned.json is its result, `--replay` shows it):
    G(theta) 0.32 at update 0 (the default split falls 32 % of the way in), 0.89 by
    update 13, 2.00 (stood) from update 18; at update 30, 14 of the 16 tries stand.
    Tested at other tempos, the default split stands up to 1.35 and falls at 1.4-1.8;
    the learned split stands at 1.0, 1.2, 1.35, 1.4 and 1.5 (and, once, at 1.8 while
    falling at 1.6 and 1.7 -- an isolated point, not a limit).
    Where the time went (seconds at tempo 1.5): the weight shifts onto one foot
    1.00 -> 2.33 and 1.00 -> 1.55, the arms-up sways 0.67 -> 1.25 and 1.04; the waves,
    twists and returns to the middle gave it up (e.g. arms_out 1.00 -> 0.39).
    Failed first, and kept as lessons: lr 1.0 found splits that stood, then jumped into
    one that rushes 15 glides (stuck at 0.93); target 1.6 with lr 0.3 crept to 0.90 in
    25 updates and never stood. Hence the smaller step, |theta| <= 1 and keeping the best.

Try: TARGET_TEMPO 1.6 with more updates; SIGMA 0.1 or 1.0; reward only "stood" (0 or
1) -- with no partial credit, does it still learn?
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p15_dance import DANCE  # noqa: E402  (the dance whose limit P15 measured)

TARGET_TEMPO = float(os.environ.get("P16_TEMPO", 1.5))   # the dance lasts sum(seconds) / this
SIGMA = 0.3               # how far apart the two tries of a pair are
LR = 0.3                  # learning rate (1.0 found a split that stood, then jumped away)
BOUND = 1.0               # |theta| <= BOUND: each glide keeps 0.37x .. 2.7x its share
PAIRS = 8                 # pairs of tries per update
MIN_GLIDE = 0.12          # no glide shorter than this (seconds)
SAVED = HERE / "p16_learned.json"


def durations(theta, tempo=None):
    """theta (one number per glide) -> seconds for each glide, summing to the dance length."""
    tempo = TARGET_TEMPO if tempo is None else tempo      # read now, not when the file loaded
    base = np.array([sec for _, sec in DANCE])
    total = base.sum() / tempo
    w = base * np.exp(np.asarray(theta, float))
    d = total * w / w.sum()
    d = np.maximum(d, MIN_GLIDE)
    return d * total / d.sum()


_ROBOT = None


def robot():
    """One G1 per worker process, reset between tries (building one takes memory and time)."""
    global _ROBOT
    from soc4180.actions import G1
    if _ROBOT is None:
        _ROBOT = G1()
    else:
        _ROBOT.reset()
    return _ROBOT


def reward(theta):
    """Play the dance with these glide times. G = fraction done before a fall, + 1 if it stood."""
    d = durations(theta)
    g1 = robot()
    r = g1.play([(spec, sec) for (spec, _), sec in zip(DANCE, d)])
    return float(min(r.seconds / d.sum(), 1.0) + (0.0 if r.fell else 1.0))


def learn(updates, seed, jobs):
    rng = np.random.default_rng(seed)
    theta = np.zeros(len(DANCE))
    best, best_G = theta.copy(), -1.0          # keep the best theta seen (week 7b's KeepBest)
    history = []
    print(f"{'update':>6} {'G(theta)':>9} {'mean G of tries':>16} {'tries that stood':>17}")
    with ProcessPoolExecutor(jobs) as ex:
        for k in range(updates + 1):
            eps = rng.standard_normal((PAIRS, len(DANCE)))
            tries = [theta] + [theta + SIGMA * e for e in eps] + [theta - SIGMA * e for e in eps]
            G = np.array(list(ex.map(reward, tries)))
            g0, Gp, Gm = G[0], G[1:1 + PAIRS], G[1 + PAIRS:]
            stood = int(np.sum(np.concatenate([Gp, Gm]) >= 1.0))
            print(f"{k:6d} {g0:9.3f} {np.mean(G[1:]):16.3f} {stood:>11d}/{2 * PAIRS}", flush=True)
            history.append(dict(update=k, G=float(g0), mean_tries=float(np.mean(G[1:])), stood=stood))
            if g0 > best_G:
                best, best_G = theta.copy(), g0
            if k == updates:
                break
            theta = theta + LR * np.mean((Gp - Gm)[:, None] * eps, axis=0) / (2 * SIGMA)
            theta = np.clip(theta, -BOUND, BOUND)
    print(f"kept the best: G = {best_G:.3f}")
    return best, history


def test(theta, tempos, jobs):
    """Play the default split and the learned split at several tempos."""
    from soc4180.actions import G1

    def plays(th):
        return [[(spec, sec) for (spec, _), sec in zip(DANCE, durations(th, t))] for t in tempos]

    with ProcessPoolExecutor(jobs) as ex:
        base = list(ex.map(_stands, plays(np.zeros(len(DANCE)))))
        learned = list(ex.map(_stands, plays(theta)))
    print(f"\n{'tempo':>6} {'default split':>14} {'learned split':>14}")
    for t, b, l in zip(tempos, base, learned):
        print(f"{t:6.2f} {'stood' if b else 'FELL':>14} {'stood' if l else 'FELL':>14}")
    return base, learned


def _stands(seq):
    return not robot().play(seq).fell


def main(argv=None) -> int:
    global TARGET_TEMPO, LR
    ap = argparse.ArgumentParser(description="P16: learn the timing of a dance.")
    ap.add_argument("--updates", type=int, default=30)
    ap.add_argument("--tempo", type=float, default=TARGET_TEMPO, help="target tempo")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--save", default=str(SAVED))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 2 * PAIRS + 1))
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--replay", action="store_true", help="skip learning; replay the saved result")
    args = ap.parse_args(argv)
    TARGET_TEMPO, LR = args.tempo, args.lr
    os.environ["P16_TEMPO"] = str(args.tempo)          # the worker processes read it

    if args.replay and pathlib.Path(args.save).exists():
        saved = json.loads(pathlib.Path(args.save).read_text(encoding="utf-8"))
        theta, TARGET_TEMPO = np.array(saved["theta"]), saved["target_tempo"]
    else:
        t0 = time.time()
        theta, history = learn(args.updates, args.seed, args.jobs)
        print(f"learned in {time.time() - t0:.0f} s")
        if args.updates > 0:
            pathlib.Path(args.save).write_text(json.dumps(dict(target_tempo=TARGET_TEMPO, seed=args.seed,
                                             theta=theta.tolist(), history=history), indent=1),
                             encoding="utf-8")
        tempos = [1.0, 1.2, 1.35, 1.4, 1.5, 1.6, 1.7, 1.8]
        test(theta, tempos, args.jobs)
        d0, d = durations(np.zeros(len(DANCE))), durations(theta)
        print("\nwhere the time went (seconds at the target tempo):")
        for (spec, _), a, b in zip(DANCE, d0, d):
            print(f"  {spec:26s} {a:5.2f} -> {b:5.2f}  {'+' * int(max(b - a, 0) * 20)}")

    if args.no_viewer:
        return 0
    import soc4180
    from soc4180.actions import G1
    if soc4180.is_colab():
        return 0
    g1 = G1()
    seq = [(spec, sec) for (spec, _), sec in zip(DANCE, durations(theta))]
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    print(f"\nreplaying the learned timing at tempo {TARGET_TEMPO}; close the window to stop")
    with soc4180.launch_viewer(g1.model, g1.data, passive=True) as viewer:
        start = time.time()

        def tick(g):
            if not viewer.is_running() or time.time() > deadline:
                raise KeyboardInterrupt
            viewer.sync()
            ahead = g.data.time - (time.time() - start)
            if ahead > 0:
                time.sleep(ahead)

        g1.on_tick = tick
        try:
            print(g1.play(seq))
            while viewer.is_running() and time.time() < deadline:
                g1.idle(0.1)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
