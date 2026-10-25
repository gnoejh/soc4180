"""lab_explore: the G1 walks past a gallery and remembers what it saw.

    uv run --extra vision weeks/27-explorer/lab_explore.py                  # headless, lockstep, a report
    uv run --extra vision weeks/27-explorer/lab_explore.py --viewer         # watch it walk, real time
    uv run --extra vision weeks/27-explorer/lab_explore.py --steady 1       # no event segmentation
    uv run --extra vision weeks/27-explorer/lab_explore.py --no-snap        # store raw glances

Needs the `vision` extra (week 26: `uv sync --extra vision` once, always `uv run --extra
vision`) and the CLIP weights in the Hugging Face cache.

Everything is from earlier weeks; this week puts them in one robot:
  the body       the G1 walks straight down a gallery: 21 CC0 photographs on stands along
                 its line (vendor/ubrain/g1_body/scenes/gallery_cc0), with a head camera.
  the cerebellum week 16's walker keeps it walking (ubrain's `explorer` profile).
  the eye        every GLANCE ticks the lab takes a 224x224 frame and looks at one half of
                 it, the half with more structure (a FOVEA), and CLIP codes it (week 26).
  recognising    the Sense is fitted on the 21 pictures; with snap it reports the nearest
                 picture's code (perception), not the raw glance.
  segmenting     a thing counts as SEEN when the same picture is recognised on STEADY
                 glances in a row and is not already in the episode; a one-frame flicker
                 is not an event.
  the episode    each seen thing is written once into week 20's hippocampal episode; when
                 the walk is over, a cue recalls each slot.
The report sets what was written, what was recalled, and the gallery's own order side by
side, beside CLIP's certificate for the 21 pictures.
    --seconds S   how long it walks (default 58: the whole gallery at 0.5 m/s)
    --vx V        walking speed, m/s (default 0.5)
    --steer K     hold the heading: command a yaw rate of -K x (gyro-integrated turn); 0 off
    --steady N    glances in a row before a thing counts as seen (default 3)
    --glance G    ticks between frames (default 10: 5 a second at 50 Hz)
    --no-snap     store the raw glance codes instead of the recognised picture's
    --no-fovea    look at the whole frame, not the more structured half
    --d D         the workspace (default 512, CLIP's own)
    --verbose     print every glance

Every run appends a row to results.csv and saves every glance in last_run.json.
ubrain and the body are the course's local copy.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import ubrain_local

ubrain_local.load()                 # the vendored wheel onto sys.path, before `import ubrain`

from ubrain import Brain, Channel, act_03, manifest_from_hello_03, profile, senses_from_state_03  # noqa: E402
from ubrain.bodies import BodyClient, kill_tree  # noqa: E402
from ubrain.codecs import Sense, certify  # noqa: E402

HERE = Path(__file__).resolve().parent
GALLERY = ubrain_local.BODY / "scenes" / "gallery_cc0"
MIN_STD = 20.0           # a half-frame this flat holds no picture
GYRO_Z = 2               # the torso gyroscope's z (yaw rate) in the body numbers (week 17)


def fovea(frame: np.ndarray, use: bool = True) -> tuple[np.ndarray, float, str]:
    """The half of the frame with more structure (the larger grey-level spread)."""
    if not use:
        return frame, float(frame.mean(axis=2).std()), "whole"
    w = frame.shape[1] // 2
    halves = {"left": frame[:, :w], "right": frame[:, w:]}
    spread = {k: float(v.mean(axis=2).std()) for k, v in halves.items()}
    side = max(spread, key=spread.get)
    return halves[side], spread[side], side


def lcs(a: list, b: list) -> int:
    """The longest common subsequence: how many pictures the memory holds in the gallery's order."""
    t = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            t[i + 1][j + 1] = t[i][j] + 1 if x == y else max(t[i][j + 1], t[i + 1][j])
    return t[-1][-1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.split("\n")[7:]))
    ap.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer (real time)")
    ap.add_argument("--seconds", type=float, default=58.0)
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--steer", type=float, default=0.0, help="yaw-rate command per radian turned (0: none)")
    ap.add_argument("--steady", type=int, default=3)
    ap.add_argument("--glance", type=int, default=10)
    ap.add_argument("--no-snap", dest="snap", action="store_false")
    ap.add_argument("--no-fovea", dest="fovea", action="store_false")
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--name", default="", help="a label for the results.csv row")
    a = ap.parse_args(argv)

    from ubrain.extras.clip import CLIP
    pictures = sorted(GALLERY.glob("*.png"))
    names = [p.stem.split("_", 1)[1] for p in pictures]
    clip = CLIP("image")
    # 1. The sense: fitted once on the 21 pictures (its vocabulary); snap = recognise.
    sense = Sense.fit(clip, pictures, d=a.d, snap=a.snap)
    cert = certify(clip, pictures, d=a.d, ks=(4, 8, 12, 16, 21), n_trials=2000)
    print(cert.summary(), flush=True)

    proc, port = ubrain_local.start_body(viewer=a.viewer, head_camera=True, gallery=True)
    glances, stored, t = [], [], 0
    try:
        client = BodyClient("127.0.0.1", port)
        # 2. The brain: the explorer profile (walker + hippocampal episode), CLIP on its camera.
        m = manifest_from_hello_03(client.hello).with_channels(
            Channel("recall", "event", 64), Channel("boundary", "event", 1),
            Channel("memory", "command", a.d), Channel("remembered", "choice", len(pictures)))
        brain = Brain(profile("explorer"), m, codecs={"camera": sense}, rng=np.random.default_rng(0))
        state = client.reset(0)
        run, run_of, fell, heading = 0, None, None, 0.0
        t0 = time.perf_counter()
        for t in range(int(a.seconds * 50)):
            heading += float(state["proprio"][GYRO_Z]) / 50       # the gyro, integrated: how far it has turned
            # the walking goal (vx, vy, yaw rate); --steer turns it back toward the start heading
            senses = senses_from_state_03(state) | {"goal": (a.vx, 0.0, -a.steer * heading)}
            frame = senses.pop("camera", None)
            if frame is not None:
                # 3. The eye: a fovea, CLIP's code, the nearest known picture.
                half, spread, side = fovea(frame, a.fovea)
                code = sense.raw([half])[0]
                fam = sense.vocabulary @ code
                guess = names[int(np.argmax(fam))]
                # 4. Event segmentation: STEADY glances on the same picture, written once.
                run, run_of = (run + 1, run_of) if guess == run_of else (1, guess)
                new = spread >= MIN_STD and run == a.steady and guess not in stored
                if a.verbose:
                    print(f"  t={t:5d} {side:5s} spread {spread:5.1f} familiar {float(fam.max()):+.2f} "
                          f"looks like {guess:10s} run {run} heading {np.degrees(heading):+6.1f} {'STORE' if new else ''}", flush=True)
                if new:
                    senses["camera"] = half
                    stored.append(guess)
                glances.append({"tick": t, "heading_deg": round(float(np.degrees(heading)), 1), "side": side, "spread": round(spread, 1), "familiar": round(float(fam.max()), 3),
                                "guess": guess, "run": run, "stored": bool(new)})
            if not state["upright"]:
                fell = t
                print(f"fell at tick {t}", flush=True)
                break
            action = brain.tick(senses)
            state = client.act(*act_03(m, action), want_image=(t + 1) % a.glance == 0, image_size=(224, 224))
            if a.viewer:
                lag = t0 + (t + 1) / 50 - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
        # 5. The recall: a cue per slot of the one episode.
        ep = brain.regions["hippocampus.episode"]
        recalled = [names[brain.tick({"recall": i})["remembered"]] for i in range(ep.n)]
        dt = time.perf_counter() - t0
        client.close()
    finally:
        kill_tree(proc)

    same = sum(x == y for x, y in zip(stored, recalled))
    order = lcs(recalled, names)
    print(f"\nwalked {t + 1} ticks ({(t + 1) / 50:.0f} s simulated, {dt:.0f} s wall), "
          f"{len(glances)} glances, {len(stored)} written into one episode")
    last = max((g["tick"] for g in glances if g["stored"]), default=0)
    print(f"the last thing seen at {last / 50:.1f} s; the robot had turned {np.degrees(heading):+.0f} degrees "
          "by the end (gyro, + is left)")
    print("written :", " ".join(stored))
    print("recalled:", " ".join(recalled))
    print("standing:", " ".join(names))
    print(f"recall of what was written: {same}/{len(stored)} slots")
    print(f"the walk's memory against the gallery's order: {order}/{len(names)} pictures in order")
    row = {"when": datetime.now().isoformat(timespec="seconds"), "name": a.name, "seconds": a.seconds,
           "vx": a.vx, "steady": a.steady, "glance": a.glance, "snap": a.snap, "fovea": a.fovea, "d": a.d, "steer": a.steer,
           "fell": "" if fell is None else fell, "glances": len(glances), "written": len(stored),
           "recalled_exact": same, "in_order": order, "last_seen_s": last / 50,
           "turned_deg": round(float(np.degrees(heading)), 1)}
    new = not (HERE / "results.csv").exists()
    with open(HERE / "results.csv", "a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(row))
        if new:
            wr.writeheader()
        wr.writerow(row)
    (HERE / "last_run.json").write_text(json.dumps({"args": vars(a), "written": stored, "recalled": recalled,
                                                    "standing": names, "glances": glances}, indent=1),
                                        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
