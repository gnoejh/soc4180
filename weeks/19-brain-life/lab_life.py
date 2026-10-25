"""lab_life: a G1 brain as a device -- save what it learned, load it, read its datasheet, watch it.

    uv run weeks/19-brain-life/lab_life.py train                 # learn the brace (12 pushes), save it
    uv run weeks/19-brain-life/lab_life.py test                  # load it: braced from the first push?
    uv run weeks/19-brain-life/lab_life.py test --fresh          # the same pushes, an untrained brain
    uv run weeks/19-brain-life/lab_life.py identical             # loaded == never stopped, byte for byte
    uv run weeks/19-brain-life/lab_life.py refuse                # a brain will not load into another body
    uv run weeks/19-brain-life/lab_life.py sheet --tau 0.97      # the datasheet, before a single tick
    uv run weeks/19-brain-life/lab_life.py watch                 # health and the monitor while it walks

The brain is week 16's `walker` (the cerebellum's forward model and brace) on the G1.
An untrained brain learns all its life, so what it holds IS what it learned. ubrain 0.14
makes that a file and a set of instruments:

  brain.save(path)          every region's matrices and traces, every random generator,
                            the tick count, with a header naming the wiring and the body
  Brain.load(path, body)    a brain that continues EXACTLY where the saved one stopped;
                            it refuses a body (or a codec) it did not learn with
  datasheet(brain)          figures from the regions' laws, for a configuration that has
                            not run a tick: the brace's lead is 1 / (1 - tau) ticks
  health(brain)             what a region holds now against its law (the forward model's
                            error against six tenths, the walking robot's bar)
  observe / monitor         read-only copies of the regions' numbers; monitor_manifest
                            turns them into channels a second brain could take as senses

The steps (each starts its own G1 body; brains are saved in brains/ beside this file):
  train     --pushes N (12) --tau T (0.9) --save FILE (brains/brace.npz)
  test      --load FILE (brains/brace.npz) or --fresh;  --pushes N (6)
  identical --ticks N (500): learn for N ticks with pushes, save, load, then run the
            original and the loaded brain for N more ticks on two identical bodies
  refuse    tries to load a saved brain against a different body, and with another wiring
  sheet     --tau T: the datasheet; --measure N also trains N pushes and reports the
            learned peak (observe) beside the predicted lead
  watch     --seconds S (20): health and monitor every second; the command turns at S/2;
            --load FILE watches a saved brain

The push is week 16's: a tone, then 250 N from behind 10 ticks later (--delay), on a
fixed point of the stride after each reset, every 4 s. Every run appends one row to
results.csv. ubrain is the course's local copy (see ubrain_local.py).
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

import ubrain_local

ubrain_local.load()                 # the vendored wheel onto sys.path, before `import ubrain`

from ubrain import Brain, Manifest, Wiring, act_03, manifest_from_hello_03, profile, senses_from_state_03  # noqa: E402
from ubrain.bodies import BodyClient, kill_tree  # noqa: E402
from ubrain.datasheet import datasheet, datasheet_text  # noqa: E402
from ubrain.observe import health, monitor, monitor_manifest, observe  # noqa: E402
from ubrain.state import StateError, read_header, snapshot_bytes  # noqa: E402

HERE = Path(__file__).resolve().parent
BRAINS = HERE / "brains"
HZ = 50
EVERY = 200          # ticks between pushes (4 s), as week 16
HOLD = 60            # ticks a fallen robot stays down before it is set back up
GOAL = (0.5, 0.0, 0.0)


class Session:
    """One G1 body and its manifest."""

    def __init__(self):
        self.proc, port = ubrain_local.start_body(echo=False)
        self.body = BodyClient("127.0.0.1", port)
        self.manifest = manifest_from_hello_03(self.body.hello)

    def close(self):
        try:
            self.body.close()
        finally:
            kill_tree(self.proc)


def walker(manifest, tau: float) -> Brain:
    w = profile("walker")
    regions = {k: dict(v) for k, v in w.regions.items()}
    regions["cerebellum.brace"]["tau"] = tau
    return Brain(Wiring(w.name, w.contract, regions=regions, inputs=dict(w.inputs), outputs=dict(w.outputs)),
                 manifest)


def pushes(s: Session, brain: Brain, n: int, delay: int, force: float = 250.0) -> str:
    """Week 16's schedule: n tone-then-push trials; returns the record (X fell, . stayed up).
    A fall with no push being watched is counted apart and reported after the record."""
    state = s.body.reset(0)
    since, k, watch, down, rec, walking_falls = 0, 0, None, None, [], 0
    while k < n or watch is not None:
        if k < n and watch is None and down is None and since % EVERY == 100 and state["upright"]:
            s.body._send({"type": "world", "tone": True,
                          "push": {"force": [force, 0.0], "delay": delay, "ticks": 5}})
            watch, k = state["tick"] + delay + 100, k + 1
        state = s.body.act(*act_03(s.manifest, brain.tick(senses_from_state_03(state) | {"goal": GOAL})))
        since += 1
        if down is not None and state["tick"] >= down:
            state = s.body.reset(0)
            brain.body_reset()
            since, down = 0, None
        elif not state["upright"] and down is None:
            if watch is not None:
                rec.append("X")
            else:
                walking_falls += 1
            watch, down = None, state["tick"] + HOLD
        elif watch is not None and state["tick"] >= watch:
            rec.append(".")
            watch = None
    return "".join(rec) + (f"  (+{walking_falls} fall{'s' if walking_falls > 1 else ''} without a push)"
                           if walking_falls else "")


def brace_line(brain: Brain) -> str:
    ob = observe(brain)["cerebellum.brace"]
    lead = next(r["value"] for r in datasheet(brain) if r["region"] == "cerebellum.brace")
    return f"brace: {ob['trials']} trials learned, counter-move peaks at tick {ob['peak_tick']} (datasheet lead {lead})"


def log(row: dict) -> None:
    row = {"when": datetime.now().isoformat(timespec="seconds"), **row}
    new = not (HERE / "results.csv").exists()
    with open(HERE / "results.csv", "a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["when", "step", "name", "details"])
        if new:
            wr.writeheader()
        wr.writerow({"when": row["when"], "step": row["step"], "name": row.get("name", ""),
                     "details": json.dumps({k: v for k, v in row.items() if k not in ("when", "step", "name")})})


# ------------------------------------------------------------------ the steps

def step_train(a) -> None:
    s = Session()
    try:
        brain = walker(s.manifest, a.tau)
        rec = pushes(s, brain, a.pushes, a.delay)
        path = Path(a.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        brain.save(path)
        print(f"trained on {a.pushes} pushes: {rec}   (X fell, . stayed up)")
        print(brace_line(brain))
        h = read_header(path)
        print(f"saved {path.name}: {path.stat().st_size / 1e6:.1f} MB; header: ubrain {h['ubrain']}, wiring "
              f"{h['wiring']['name']}, body {h['manifest']['body']} ({h['manifest']['digest']}), {h['ticks']} ticks")
        log({"step": "train", "name": a.name, "pushes": a.pushes, "tau": a.tau, "record": rec, "file": path.name})
    finally:
        s.close()


def step_test(a) -> None:
    s = Session()
    try:
        if a.fresh:
            brain, what = walker(s.manifest, a.tau), "a fresh, untrained brain"
        else:
            brain, what = Brain.load(a.load, s.manifest), f"the brain loaded from {Path(a.load).name}"
        rec = pushes(s, brain, a.pushes, a.delay)
        print(f"{what}: {a.pushes} pushes: {rec}   (X fell, . stayed up)")
        print(brace_line(brain))
        log({"step": "test", "name": a.name, "fresh": a.fresh, "load": "" if a.fresh else Path(a.load).name,
             "pushes": a.pushes, "record": rec})
    finally:
        s.close()


def step_identical(a) -> None:
    """Learn, save, load; then the original and the loaded brain each drive their own,
    identical body for N ticks on the same schedule. Compare every command and the end state."""
    s1, s2 = Session(), Session()
    try:
        brain = walker(s1.manifest, a.tau)
        st1 = s1.body.reset(0)
        for t in range(a.ticks):                         # learn: a walk with a push every 4 s
            if t % EVERY == 100:
                s1.body._send({"type": "world", "tone": True, "push": {"force": [200.0, 0.0], "delay": 10, "ticks": 5}})
            st1 = s1.body.act(*act_03(s1.manifest, brain.tick(senses_from_state_03(st1) | {"goal": GOAL})))
        BRAINS.mkdir(exist_ok=True)
        path = brain.save(BRAINS / "midwalk.npz")
        # both bodies restart from a reset, so the original brain and its loaded copy
        # face identical worlds from here on
        loaded = Brain.load(path, s2.manifest)
        same_at_save = snapshot_bytes(loaded) == snapshot_bytes(brain)
        st1, st2 = s1.body.reset(0), s2.body.reset(0)
        brain.body_reset()
        loaded.body_reset()
        agree, worst = 0, 0.0
        for t in range(a.ticks):
            if t % EVERY == 100:
                for s in (s1, s2):
                    s.body._send({"type": "world", "tone": True, "push": {"force": [200.0, 0.0], "delay": 10, "ticks": 5}})
            o1 = brain.tick(senses_from_state_03(st1) | {"goal": GOAL})
            o2 = loaded.tick(senses_from_state_03(st2) | {"goal": GOAL})
            d = float(np.max(np.abs(np.asarray(o1["walk"]) - np.asarray(o2["walk"]))))
            worst = max(worst, d)
            agree += d == 0.0
            st1, st2 = s1.body.act(*act_03(s1.manifest, o1)), s2.body.act(*act_03(s2.manifest, o2))
        same_at_end = snapshot_bytes(loaded) == snapshot_bytes(brain)
        print(f"learned for {a.ticks} ticks, saved ({path.stat().st_size / 1e6:.1f} MB), loaded into a second body")
        print(f"  identical state at the save:          {same_at_save}")
        print(f"  walk commands identical, next {a.ticks} ticks: {agree} of {a.ticks}  (largest difference {worst:g})")
        print(f"  identical state at the end:           {same_at_end}")
        log({"step": "identical", "name": a.name, "ticks": a.ticks, "same_at_save": same_at_save,
             "agree": agree, "same_at_end": same_at_end})
    finally:
        s1.close()
        s2.close()


def step_refuse(a) -> None:
    s = Session()
    try:
        path = BRAINS / "refuse.npz"
        BRAINS.mkdir(exist_ok=True)
        walker(s.manifest, 0.9).save(path)
        print(f"a walker brain saved for body {read_header(path)['manifest']['body']} "
              f"({read_header(path)['manifest']['digest']})")
        other = Manifest("course/another-robot", "none", s.manifest.channels)
        for what, man in (("the same channels, another body's name", other),
                          ("the G1 it learned with", s.manifest)):
            try:
                Brain.load(path, man)
                print(f"  load into {what}: loaded")
            except StateError as e:
                print(f"  load into {what}: refused -- StateError: {e}")
        log({"step": "refuse", "name": a.name})
    finally:
        s.close()


def step_sheet(a) -> None:
    s = Session()
    try:
        brain = walker(s.manifest, a.tau)
        print(datasheet_text(brain))
        if a.measure:
            rec = pushes(s, brain, a.measure, a.delay)
            print(f"\nmeasured: {a.measure} pushes, the push {a.delay} ticks after the tone: {rec}")
            print(brace_line(brain))
        log({"step": "sheet", "name": a.name, "tau": a.tau, "measure": a.measure, "delay": a.delay})
    finally:
        s.close()


def step_watch(a) -> None:
    s = Session()
    try:
        brain = Brain.load(a.load, s.manifest) if a.load else walker(s.manifest, 0.9)
        print(f"{'the brain loaded from ' + Path(a.load).name if a.load else 'a fresh brain'}")
        print(f"monitor channels ({len(monitor_manifest(brain).channels)}): "
              + ", ".join(c.name for c in monitor_manifest(brain).channels))
        print("  time  goal wz   forward error (stride_ratio)   health ok   brace trials")
        state = s.body.reset(0)
        n = int(a.seconds * HZ)
        for t in range(1, n + 1):
            goal = GOAL if t < n // 2 else (0.5, 0.0, 0.5)          # turn halfway
            state = s.body.act(*act_03(s.manifest, brain.tick(senses_from_state_03(state) | {"goal": goal})))
            if t % HZ == 0:
                m = monitor(brain)
                h = next((x for x in health(brain) if x["region"] == "cerebellum.forward"), None)
                r = m["cerebellum.forward.stride_ratio"][0]       # nan until a stride has been seen
                print(f"  {t // HZ:3d} s   {goal[2]:+.1f}     {'watching' if h is None or np.isnan(r) else f'{r:.2f}':>10s}"
                      f"                   {'-' if h is None else str(h['ok']):5s}       "
                      f"{int(m['cerebellum.brace.trials'][0])}", flush=True)
        if h is not None:
            print(f"health's bar: {h['note']}")
        log({"step": "watch", "name": a.name, "seconds": a.seconds})
    finally:
        s.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.split("\n")[9:]))
    sub = ap.add_subparsers(dest="step", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tau", type=float, default=0.9, help="the brace's trace decay per tick")
    common.add_argument("--delay", type=int, default=10, help="ticks from the tone to the push")
    common.add_argument("--name", default="", help="a label for this run in results.csv")
    p = sub.add_parser("train", parents=[common])
    p.add_argument("--pushes", type=int, default=12)
    p.add_argument("--save", default=str(BRAINS / "brace.npz"))
    p = sub.add_parser("test", parents=[common])
    p.add_argument("--load", default=str(BRAINS / "brace.npz"))
    p.add_argument("--fresh", action="store_true", help="an untrained brain instead")
    p.add_argument("--pushes", type=int, default=6)
    p = sub.add_parser("identical", parents=[common])
    p.add_argument("--ticks", type=int, default=500)
    sub.add_parser("refuse", parents=[common])
    p = sub.add_parser("sheet", parents=[common])
    p.add_argument("--measure", type=int, default=0, help="also train this many pushes")
    p = sub.add_parser("watch", parents=[common])
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--load", default="", help="watch a saved brain instead of a fresh one")
    a = ap.parse_args(argv)
    {"train": step_train, "test": step_test, "identical": step_identical, "refuse": step_refuse,
     "sheet": step_sheet, "watch": step_watch}[a.step](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
