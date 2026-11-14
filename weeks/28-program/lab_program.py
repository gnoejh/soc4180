"""lab_program: programming the whole brain -- Brain(), streams, ticks, asking it, keeping it.

    uv run weeks/28-program/lab_program.py                    # all four steps, headless
    uv run weeks/28-program/lab_program.py --step whole       # one step: whole | ask | keep | g1
    uv run weeks/28-program/lab_program.py --step g1 --viewer # the G1, watched in real time

Week 28 is the project's first week. Until now every lab built a CIRCUIT: a profile (walker,
explorer, chooser ...) wired from chosen regions. `Brain()` is the WHOLE brain instead: every
region on the route from the senses to the body, wired once, the same in every copy. A program
does not wire it; it attaches STREAMS by PORT CLASS and ticks it.

  step whole  a world of pictures and actions: picture p asks for action right[p], the right
              one pays 1. The brain sees the picture through a codec (any frozen model; here a
              fixed random one), gets the reward, chooses. The printout sets what it earned
              beside the Go race's law on the values it learned (week 18's law, now inside
              the whole brain).
  step ask    the brain answers questions about itself before and after running:
              describe() (its classes, its observables, its guarantees), read() (an
              observable as the last tick left it), datasheet() (its capacities from the
              laws), health() (what it holds against its laws).
  step keep   save, load, continue: the loaded brain chooses exactly what the original
              would have, tick for tick.
  step g1     the G1. In ubrain 0.45 the whole brain has no body classes yet (they come in
              0.46, week 29), so the body runs through the walker profile (week 16): its
              cerebellum predicts the body and its error falls as it walks.

Options:
    --step S        whole | ask | keep | g1 | all (default all)
    --pictures P    pictures in the world (default 4)        --actions A  actions (default 4)
    --trials T      trials in step whole (default 3000)       --variant V  small | standard | large
    --reverse       halfway through step whole, the right actions are shuffled (a new rule)
    --seconds S     how long the G1 walks (default 10)        --vx V       its speed (default 0.3)
    --viewer        step g1 in real time with the viewer      --seed N     the world's seed

Every run appends rows to results.csv (step, setting, measure, value).
ubrain and the body are the course's local copy (week 28: release 0.45).
"""

from __future__ import annotations

import argparse
import csv
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import ubrain_local

ubrain_local.load()

import ubrain  # noqa: E402
from ubrain import Brain, GaussianCodes, Sense  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.csv"
FLOOR = 1e-3                                    # the race's floor on a value


def record(rows: list[tuple]) -> None:
    new = not RESULTS.exists()
    with RESULTS.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["when", "step", "setting", "measure", "value"])
        stamp = datetime.now().isoformat(timespec="seconds")
        for r in rows:
            w.writerow([stamp, *r])


# ---------------------------------------------------------------- 1. the world and the brain
def make_brain(a, seed: int = 1):
    """The whole brain with three streams: a sense through a codec, a reward, an action. The
    codec is the frozen model itself (here a fixed random one; CLIP in week 27): the brain maps
    its codes onto its own sphere and keeps its own record of what it meets, at any variant."""
    camera = GaussianCodes(dim=64, seed=1)
    brain = Brain(variant=a.variant, rng=np.random.default_rng(seed))
    brain.attach("sense", "vision", codec=camera)   # -> the tick key "sense.vision"
    brain.attach("reward")                          # -> "reward"
    brain.attach("action", n=a.actions)             # -> "action" in what tick returns
    return brain, camera


def race_law(brain, camera, right, pictures) -> float:
    """The expected reward of the Go race on the striatum's values now: per picture, the right
    action's floored value over the sum of all. (It taps the striatum, an IMPLEMENTATION port:
    a program written against the interface reads `read("value")` instead.)"""
    st = brain.regions["basal_ganglia.striatum"]
    share = []
    for p in range(pictures):
        q = np.clip(st.W @ st.phi(brain.codecs["sense.vision"]([p])[0]), FLOOR, None)
        share.append(q[right[p]] / q.sum())
    return float(np.mean(share))


def step_whole(a) -> None:
    print("\n== step whole: a picture asks for an action ==")
    rng = np.random.default_rng(a.seed)
    brain, camera = make_brain(a)
    right = rng.permutation(a.actions)[: a.pictures] if a.pictures <= a.actions else \
        rng.integers(a.actions, size=a.pictures)
    block = a.trials // 6
    r, paid, rows = None, [], []
    print(f"  variant {a.variant!r}; {a.pictures} pictures, {a.actions} actions; chance {1 / a.actions:.2f}")
    print("  block   earned   the race's law on the learned values")
    for t in range(a.trials):
        if a.reverse and t == a.trials // 2:
            right = rng.permutation(right)
            print("  -- the rule changes: the right actions are shuffled --")
        p = int(rng.integers(a.pictures))
        out = brain.tick({"sense.vision": p, "reward": r})
        act = out["action"]
        r = float(act == right[p]) if act is not None else 0.0
        paid.append(r)
        if (t + 1) % block == 0:
            k = (t + 1) // block
            earned = float(np.mean(paid[-block:]))
            law = race_law(brain, camera, right, a.pictures)
            print(f"  {k:5d}   {earned:6.3f}   {law:6.3f}")
            rows.append(("whole", f"{a.variant},P{a.pictures},A{a.actions},rev{int(a.reverse)}",
                         f"block{k}_earned", round(earned, 4)))
            rows.append(("whole", f"{a.variant},P{a.pictures},A{a.actions},rev{int(a.reverse)}",
                         f"block{k}_law", round(law, 4)))
    record(rows)


# ---------------------------------------------------------------- 2. asking the brain
def step_ask(a) -> None:
    print("\n== step ask: what the brain says about itself ==")
    brain, camera = make_brain(a)
    d = brain.describe()
    print(f"  implementation {d['implementation']}; interface {d['interface']}")
    print("  port classes built:", ", ".join(d["classes"]))
    print("  observables read() serves:", ", ".join(d["observables"]))
    print("  streams attached:", d["instance"]["streams"] if d["instance"] else None)
    rng = np.random.default_rng(a.seed)
    right = rng.integers(a.actions, size=a.pictures)
    r = None
    for t in range(300):
        p = int(rng.integers(a.pictures))
        act = brain.tick({"sense.vision": p, "reward": r})["action"]
        r = float(act == right[p]) if act is not None else 0.0
    print("  after 300 ticks, read():")
    for name in d["observables"]:
        v = brain.read(name)
        shown = v if not isinstance(v, np.ndarray) else f"array {v.shape}"
        print(f"    {name:18s} {shown}")
    print("  the datasheet (capacities from the laws):")
    sheet = brain.datasheet()
    for row in sheet[:8]:
        print("    " + ", ".join(f"{k}={row[k]}" for k in list(row)[:4]))
    bad = [h for h in brain.health() if not h.get("ok", True)]
    print(f"  health: {len(brain.health())} checks, {len(bad)} warnings")
    record([("ask", a.variant, "classes", len(d["classes"])),
            ("ask", a.variant, "observables", len(d["observables"])),
            ("ask", a.variant, "datasheet_rows", len(sheet)),
            ("ask", a.variant, "health_warnings", len(bad))])


# ---------------------------------------------------------------- 3. keeping the brain
def step_keep(a) -> None:
    print("\n== step keep: save, load, continue ==")
    brain, camera = make_brain(a)
    rng = np.random.default_rng(a.seed)
    right = rng.integers(a.actions, size=a.pictures)
    r = None
    for t in range(1000):
        p = int(rng.integers(a.pictures))
        act = brain.tick({"sense.vision": p, "reward": r})["action"]
        r = float(act == right[p]) if act is not None else 0.0
    with tempfile.TemporaryDirectory() as tmp:
        path = brain.save(Path(tmp) / "student.ubrain.npz")
        size = Path(path).stat().st_size
        try:                                               # the file names the codec it learned with:
            Brain.load(path)                               # loading without it is refused
        except Exception as e:
            print(f"  loaded without its codec: refused ({type(e).__name__})")
        twin = Brain.load(path, codecs={"sense.vision": camera})   # a new object, from the file
        world = np.random.default_rng(a.seed + 1)
        differ, rs = 0, [r, r]
        for t in range(500):
            p = int(world.integers(a.pictures))
            a1 = brain.tick({"sense.vision": p, "reward": rs[0]})["action"]
            a2 = twin.tick({"sense.vision": p, "reward": rs[1]})["action"]
            differ += a1 != a2
            rs = [float(a1 == right[p]), float(a2 == right[p])]
        print(f"  saved {size / 1e6:.1f} MB; loaded; 500 more ticks: {differ} choices differ")
        print(f"  lineage of the loaded brain: {len(twin.lineage)} saved state(s)")
    record([("keep", a.variant, "file_MB", round(size / 1e6, 2)), ("keep", a.variant, "choices_differ", differ)])


# ---------------------------------------------------------------- 4. the G1
def step_g1(a) -> None:
    print("\n== step g1: the body, through the walker profile ==")
    from ubrain import act_03, manifest_from_hello_03, profile, senses_from_state_03
    from ubrain.bodies import BodyClient
    proc, port = ubrain_local.start_body(viewer=a.viewer)
    try:
        with BodyClient("127.0.0.1", port) as body:
            manifest = manifest_from_hello_03(body.hello)
            brain = Brain(profile("walker"), manifest)
            state = body.reset(0)
            ticks = int(a.seconds * 50)
            falls, shown, rows = 0, [], []
            t0 = time.perf_counter()
            for t in range(ticks):
                senses = senses_from_state_03(state) | {"goal": (a.vx, 0.0, 0.0)}
                state = body.act(*act_03(manifest, brain.tick(senses)))
                if not state["upright"]:
                    falls += 1
                    state = body.reset(0)
                    brain.body_reset()
                fw = brain.forward
                if (t + 1) % 50 == 0 and fw is not None and fw.stride_ratio is not None:
                    shown.append(round(100 * fw.stride_ratio, 1))
                if a.viewer:
                    time.sleep(max(0.0, t0 + (t + 1) * 0.02 - time.perf_counter()))
            print(f"  walked {ticks} ticks at vx {a.vx}; falls {falls}")
            print("  the cerebellum's prediction error, % of predicting no change, each second:")
            print("   ", shown)
            rows = [("g1", f"vx{a.vx}", f"error_pct_s{i + 1}", v) for i, v in enumerate(shown)]
            record(rows + [("g1", f"vx{a.vx}", "falls", falls)])
    finally:
        proc.kill()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--step", default="all", choices=["whole", "ask", "keep", "g1", "all"])
    ap.add_argument("--pictures", type=int, default=4)
    ap.add_argument("--actions", type=int, default=4)
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--variant", default="small")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    print(f"ubrain {ubrain.__version__}")
    steps = ["whole", "ask", "keep", "g1"] if a.step == "all" else [a.step]
    for s in steps:
        {"whole": step_whole, "ask": step_ask, "keep": step_keep, "g1": step_g1}[s](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
