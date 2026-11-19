"""P15 default solution: choreograph from snapshots -- how fast can the G1 dance?

    uv run weeks/projects/examples/p15_dance.py                  # sweep, results.csv, CHECK, replay
    uv run weeks/projects/examples/p15_dance.py --no-viewer      # headless
    uv run weeks/projects/examples/p15_dance.py --tempos 1.0 1.3 1.4 --replay twin

This is a complete project in the shape of templates/project_template.py, answered
with measurements. Copy it to your team folder and replace the dance, the question
and the twin with your own.

    QUESTION    How fast can this dance be played before the G1 falls -- and does the
                rule "on one foot, move one thing at a time" matter?
    YOURS       DANCE: 18 snapshots, 19.0 s at tempo 1; on one foot it only lifts or
                lowers a knee, the arms were placed before.
    TWIN        the same dance with ONE change: glide 6 is "on_left+right_out", so the
                left arm swings out during glide 7, while the right knee lifts.
    PREDICTION  (written before the first run) yours stands up to about tempo 1.3 and
                falls above; the twin falls at more tempos than yours.
    MEASURED    yours stands at every tempo from 0.5 to 1.35 and falls from 1.4 on.
                The twin falls at every tempo from 0.5 to 1.05, stands at 1.1-1.3 (it
                has less time to drift) and falls from 1.35.
                CHECK PASS: yours stands at 18 of the 25 tempos, the twin at 5.

Why the twin falls when slow: with a knee up, the swinging arm makes the centre of mass
creep (about 2.6 cm in 1 s) toward the lifted foot until it passes the 5 cm sole's edge.
Played faster, the glide ends before the creep reaches the edge.

Each tempo takes 3-10 s of computer time; the sweep runs them in parallel.
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent

QUESTION = ("How fast can this dance be played before the G1 falls, and does "
            "'on one foot, move one thing at a time' matter?")
PREDICTION = ("Yours stands up to about tempo 1.3 and falls above it; the twin falls at "
              "more of the tempos than yours.")
# CHECK, decided before the final run: PASS if yours stands at more tempos than the twin.

DANCE = [
    ("arms_out", 1.0),
    ("arms_up+sway_left", 1.0),
    ("arms_up+sway_right", 1.0),
    ("arms_bent+half_squat", 1.0),
    ("arms_out+twist_left", 1.0),
    ("arms_out+twist_right", 1.0),
    ("on_left+arms_out", 1.5),
    ("right_knee_up+arms_out", 1.0),
    ("on_left+arms_out", 1.0),
    ("arms_out", 1.5),
    ("on_right+arms_out", 1.5),
    ("left_knee_up+arms_out", 1.0),
    ("on_right+arms_out", 1.0),
    ("arms_out", 1.5),
    ("right_wave_a+bow", 1.0),
    ("right_wave_b+bow", 0.5),
    ("right_wave_a", 0.5),
    ("home", 1.0),
]
TWIN = [("on_left+right_out", 1.5) if i == 6 else s for i, s in enumerate(DANCE)]
DANCES = {"yours": DANCE, "twin": TWIN}


def run_once(args):
    """One dance at one tempo -> one results row."""
    which, tempo = args
    from soc4180.actions import G1
    g1 = G1()
    r = g1.play(DANCES[which], tempo=tempo)
    t, glide = 0.0, ""
    for i, (spec, sec) in enumerate(DANCES[which]):
        t += sec / tempo
        if r.fell and not glide and t >= r.seconds - 1e-6:
            glide = f"{i} {spec}"
    return dict(dance=which, tempo=tempo, fell=int(r.fell), seconds=round(r.seconds, 2),
                max_tilt_deg=round(r.max_tilt_deg, 1), tracking=r.info["tracking"],
                fell_in_glide=glide)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P15: how fast can the G1 dance?")
    ap.add_argument("--tempos", type=float, nargs="+",
                    default=[round(0.5 + 0.05 * k, 2) for k in range(25)])
    ap.add_argument("--out", default=str(HERE / "p15_results.csv"))
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--replay", choices=list(DANCES), default="yours")
    ap.add_argument("--replay-tempo", type=float, default=1.35)
    args = ap.parse_args(argv)

    print("QUESTION  ", QUESTION)
    print("PREDICTION", PREDICTION)
    from soc4180.actions import G1
    print("check yours:", G1().check(DANCE) or "no problems found")
    print("check twin: ", G1().check(TWIN) or "no problems found")

    t0 = time.time()
    jobs = [(w, t) for w in DANCES for t in args.tempos]
    with ProcessPoolExecutor(max(1, min(args.jobs, len(jobs)))) as ex:
        rows = list(ex.map(run_once, jobs))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'tempo':>6}  {'yours':>8}  {'twin':>8}      ({time.time() - t0:.0f} s, {args.out})")
    by = {(r["dance"], r["tempo"]): r for r in rows}
    for t in args.tempos:
        print(f"{t:6.2f}  " + "  ".join(f"{'FELL' if by[(d, t)]['fell'] else 'stood':>8}" for d in DANCES))
    stood = {d: sum(1 - by[(d, t)]["fell"] for t in args.tempos) for d in DANCES}
    fastest = {d: max([t for t in args.tempos if not by[(d, t)]["fell"]], default=None) for d in DANCES}
    print(f"\nstood at: yours {stood['yours']} of {len(args.tempos)} tempos (fastest {fastest['yours']}), "
          f"twin {stood['twin']} (fastest {fastest['twin']})")
    print("CHECK", "PASS" if stood["yours"] > stood["twin"] else "FAIL",
          "(yours stands at more tempos than the twin)")

    if args.no_viewer:
        return 0
    import soc4180
    if soc4180.is_colab():
        return 0
    g1 = G1()
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    print(f"\nreplaying {args.replay} at tempo {args.replay_tempo}; close the window to stop")
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
            print(g1.play(DANCES[args.replay], tempo=args.replay_tempo))
            while viewer.is_running() and time.time() < deadline:
                g1.idle(0.1)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
