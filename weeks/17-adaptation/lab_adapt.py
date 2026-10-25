"""lab_adapt: the G1 walks in a side wind, and its cerebellum learns to walk straight.

    uv run weeks/17-adaptation/lab_adapt.py                          # adapt, then the wind stops
    uv run weeks/17-adaptation/lab_adapt.py --learner off            # no learning: it just turns
    uv run weeks/17-adaptation/lab_adapt.py --learner single         # one learning rate
    uv run weeks/17-adaptation/lab_adapt.py --schedule savings       # the same wind twice
    uv run weeks/17-adaptation/lab_adapt.py --schedule recovery      # wind, counter-wind, clamp
    uv run weeks/17-adaptation/lab_adapt.py --viewer                 # watch it (real time)

The Unitree G1 walks straight ahead at 0.5 m/s under its frozen walking policy. The
experimenter then blows a steady SIDE WIND: a sustained force on the pelvis, sideways
in the robot's own heading frame (protocol 0.4, a "field"). The wind turns the walking
robot. The brain never feels the wind; it feels the TURN, the torso gyroscope's z
(yaw rate), and that is its teaching signal (the climbing fibre).

THE ADAPTER (ubrain's cerebellum.adapt) learns a counter-turn and adds it to the
walking command's yaw rate, every tick:
    single   u <- u + eta * e                   one rate (eta 0.005 per tick)
    dual     x_f <- a_f x_f + b_f e             a fast state: learns fast, forgets fast
             x_s <- a_s x_s + b_s e             a slow state: learns slow, keeps it
             u = x_f + x_s                      (two rates: Smith, Ghazizadeh & Shadmehr 2006)
    off      no learning at all (the error clamp held for the whole run)
  where e is minus the sensed turn and u the counter-turn sent with the walk.

The schedule is a list of blocks, each a number of strides (40 ticks, 0.8 s) with a
wind in newtons (left positive), optionally CLAMPED: the teaching signal is silenced,
the learned counter-turn is still sent and only forgets (the error clamp).
    --schedule basic      5 calm, 30 wind, 30 calm             adaptation, then the after-effect
    --schedule savings    5 calm, 30 wind, 30 calm, 20 wind    is the second time faster?
    --schedule recovery   5 calm, 40 wind, counter-wind until the counter-turn is back
                          to zero (de-adaptation), 30 calm clamped
                                                               does the old learning come back?
    --blocks "5:0,30:15,30:0c"    your own (strides:newtons, c = clamped; strides "u" means
                                  until the counter-turn crosses zero, at most 20)
    --wind F              the wind for the presets (default 15 N)

Measured: every stride, the robot's turn (rad/s, + is to the left) and the learned
counter-turn; per block, how far the robot turned in total (degrees).

Every run appends one row to results.csv beside this file and saves the whole trace
in last_run.json. ubrain itself is the course's copy (see ubrain_local.py); nothing
else beyond `uv sync`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import ubrain_local

ubrain_local.load()                 # the vendored wheel onto sys.path, before `import ubrain`

from ubrain import Brain, Channel, Wiring, act_03, manifest_from_hello_03, profile, senses_from_state_03  # noqa: E402
from ubrain.bodies import BodyClient, kill_tree  # noqa: E402

HERE = Path(__file__).resolve().parent
HZ = 50
STRIDE = 40          # ticks per stride (0.8 s)
GYRO_Z = 2           # the torso gyroscope's z (yaw rate) in the 70 body numbers
SETTLE = 200         # ticks of calm walking before the schedule (4 s), not reported

SCHEDULES = {
    "basic": "5:0,30:W,30:0",
    "savings": "5:0,30:W,30:0,20:W",
    "recovery": "5:0,40:W,u:-W,30:0c",
}


UNTIL_MAX = 20       # an "until the counter-turn crosses zero" block stops here regardless


def parse_blocks(text: str, wind: float) -> list[tuple[int | None, float, bool]]:
    """'5:0,30:W,u:-W,30:0c' -> [(strides, newtons, clamped), ...]; W is the wind,
    strides None ('u') means: until the learned counter-turn crosses zero."""
    blocks = []
    for part in text.split(","):
        n, f = part.strip().split(":")
        clamped = f.endswith("c")
        f = f.rstrip("c")
        f = f.replace("W", repr(wind))
        strides = None if n.strip() == "u" else int(n)
        blocks.append((strides, float(eval(f, {"__builtins__": {}})), clamped))   # "-15.0" etc.
    return blocks


def build_brain(learner: str, manifest) -> Brain:
    """The adapter profile, with the learner chosen ('off' builds the dual learner and
    clamps it for the whole run)."""
    w = profile("adapter")
    regions = {k: dict(v) for k, v in w.regions.items()}
    if learner == "single":
        regions["cerebellum.adapt"] = {"arm": "single"}
    return Brain(Wiring(w.name, w.contract, regions=regions, inputs=dict(w.inputs),
                        outputs=dict(w.outputs)), manifest)


def sparkline(values, lo, hi) -> str:
    bars = " .:-=+*#%@"
    return "".join(bars[min(int((min(max(v, lo), hi) - lo) / (hi - lo) * (len(bars) - 1) + 0.5),
                            len(bars) - 1)] for v in values)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.split("\n")[8:]))
    ap.add_argument("--learner", choices=("dual", "single", "off"), default="dual")
    ap.add_argument("--schedule", choices=tuple(SCHEDULES), default="basic")
    ap.add_argument("--blocks", default="", help="your own schedule, e.g. '5:0,30:15,30:0c'")
    ap.add_argument("--wind", type=float, default=15.0, help="the presets' wind, newtons (left +)")
    ap.add_argument("--vx", type=float, default=0.5, help="walking speed, m/s")
    ap.add_argument("--viewer", action="store_true", help="show the robot (real time)")
    ap.add_argument("--name", default="", help="a label for this run in results.csv")
    a = ap.parse_args(argv)
    spec = a.blocks or SCHEDULES[a.schedule]
    blocks = parse_blocks(spec, a.wind)

    proc, port = ubrain_local.start_body(viewer=a.viewer)
    rows, fell = [], None
    try:
        body = BodyClient("127.0.0.1", port)
        manifest = manifest_from_hello_03(body.hello).with_channels(Channel("clamp", "flag", 1))
        brain = build_brain(a.learner, manifest)
        goal = (a.vx, 0.0, 0.0)
        state = body.reset(0)
        t0 = time.perf_counter()
        tick = 0

        def step(clamped: bool) -> tuple[float, float]:
            """One tick: sense, think, act. Returns (sensed turn, sent counter-turn)."""
            nonlocal state, tick
            clamp = clamped or a.learner == "off"
            out = brain.tick(senses_from_state_03(state) | {"goal": goal, "clamp": clamp})
            state = body.act(*act_03(manifest, out))
            tick += 1
            if a.viewer:
                lag = t0 + tick / HZ - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
            return state["proprio"][GYRO_Z], out["walk"][2]

        for _ in range(SETTLE):
            step(False)
        print(f"learner {a.learner}; schedule {spec} (W = {a.wind:g} N); walking at {a.vx} m/s")
        print(" stride  wind N  turn rad/s   counter-turn   (+ is to the left)")
        for b, (n, force, clamped) in enumerate(blocks):
            body._send({"type": "world", "field": [0.0, force]})    # the experimenter's wind
            sign0 = None
            for s in range(n if n is not None else UNTIL_MAX):
                turns, cmds = zip(*(step(clamped) for _ in range(STRIDE)))
                if not state["upright"]:           # fallen: what follows would measure a robot on the floor
                    fell = (b, s + 1)
                    break
                rows.append({"block": b, "wind": force, "clamped": clamped,
                             "turn": float(np.mean(turns)), "counter": float(np.mean(cmds))})
                r = rows[-1]
                if s < 3 or (n is not None and s == n - 1) or s % 5 == 4 or n is None:
                    print(f"  {len(rows):4d}  {force:+6.1f}{'c' if clamped else ' '}  {r['turn']:+8.4f}"
                          f"     {r['counter']:+8.4f}", flush=True)
                if n is None:                      # de-adapt until the counter-turn changes sign
                    sign0 = sign0 if sign0 is not None else np.sign(cmds[0])
                    if np.sign(cmds[-1]) != sign0:
                        break
            if fell:
                print("\n" f"  FELL in block {fell[0]}, stride {fell[1]}: the wind is stronger than the"
                      f" walking policy can take; the run stops here.")
                break
        body.close()
    finally:
        kill_tree(proc)

    print("\nper block: how far the robot turned (+ left), and its turn at the start and end")
    summary = []
    for b, (_n, force, clamped) in enumerate(blocks):
        rb = [r for r in rows if r["block"] == b]
        n = len(rb)
        if not n:
            continue
        turned = math.degrees(sum(r["turn"] for r in rb) * STRIDE / HZ)
        first10 = math.degrees(sum(r["turn"] for r in rb[:10]) * STRIDE / HZ)
        first = float(np.mean([r["turn"] for r in rb[:3]]))
        last = float(np.mean([r["turn"] for r in rb[-5:]]))
        end_u = float(np.mean([r["counter"] for r in rb[-5:]]))
        summary.append({"block": b, "strides": n, "wind": force, "clamped": clamped,
                        "turned_deg": round(turned, 1), "turned_first10_deg": round(first10, 1),
                        "turn_first3": round(first, 4),
                        "turn_last5": round(last, 4), "counter_last5": round(end_u, 4)})
        print(f"  block {b}: {n:3d} strides, wind {force:+5.1f} N{' clamped' if clamped else '':8s}"
              f"  turned {turned:+7.1f} deg ({first10:+6.1f} in the first 10)   turn first 3 {first:+.4f}, last 5 {last:+.4f}"
              f"   counter-turn at the end {end_u:+.4f}")
    turns = [r["turn"] for r in rows]
    print("\nturn per stride (taller = more to the left; '-' row is the counter-turn):")
    print("  [" + sparkline(turns, -0.15, 0.15) + "]")
    print("  [" + sparkline([r["counter"] for r in rows], -0.15, 0.15) + "]")
    if fell:
        print(f"fell: block {fell[0]}, stride {fell[1]}")

    row = {"when": datetime.now().isoformat(timespec="seconds"), "name": a.name,
           "learner": a.learner, "schedule": spec, "wind": a.wind, "vx": a.vx,
           "blocks": json.dumps(summary), "fell": "" if not fell else f"block {fell[0]} stride {fell[1]}"}
    new = not (HERE / "results.csv").exists()
    with open(HERE / "results.csv", "a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(row))
        if new:
            wr.writeheader()
        wr.writerow(row)
    (HERE / "last_run.json").write_text(json.dumps({**row, "blocks": summary, "strides": rows}, indent=1),
                                        encoding="utf-8")
    print("saved: one row in results.csv, every stride in last_run.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
