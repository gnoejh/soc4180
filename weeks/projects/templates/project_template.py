"""Term project template: one question about the G1, answered by measurement.

    uv run weeks/projects/templates/project_template.py                 # run, print, write results.csv, replay
    uv run weeks/projects/templates/project_template.py --no-viewer     # headless (what the checker runs)
    uv run weeks/projects/templates/project_template.py --forces 60 70 80 --seeds 0 1 2
    uv run weeks/projects/templates/project_template.py --replay yours --force 75

Copy this file into your team folder (`weeks/projects/teams/<team>/project.py`)
and change the four marked places: the QUESTION and PREDICTION, `yours`, `twin`,
and `run_once`. Everything else -- the sweep, the results file, the table, the
CHECK line and the replay -- is the method, and stays.

As shipped it answers week 7's question, so you can see the shape working:

    QUESTION   does driving the ankles against the measured tilt (the ankle
               strategy) let the crouched G1 survive a larger sideways shove
               than holding the crouch?
    TWIN       the same robot, the same shove, the same seed; only the
               controller differs (hold the crouch).
    MEASURED   (week 7, env_run.py --push) hold survives 65 N and falls at 70;
               the ankle strategy survives 75 N and falls at 80.

Run it: the table shows hold falling from 70 N and the ankle strategy from
80 N, and the last line reads `CHECK PASS`.

The method (weeks/projects/README.md, section 3):
  1. the PREDICTION is written, with a number, BEFORE the first real run;
  2. a TWIN run changes exactly one thing, so a difference has a cause;
  3. every number in the report comes from results.csv, written by this script;
  4. the CHECK rule is decided before the final run, and a FAIL is reported as FAIL;
  5. falls are chaotic across machines: compare by margins (which force falls),
     never by a float read off a fall.

Needs gymnasium: `uv sync --extra rl` once (it is in every lab laptop's lock).
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
import time

import mujoco
import numpy as np

import soc4180

HERE = pathlib.Path(__file__).resolve().parent

# =============================================================================
# 1. CHANGE: the question and the prediction, written before the first run.
# =============================================================================
QUESTION = ("Does the ankle strategy let the crouched G1 survive a larger sideways "
            "shove (0.2 s on the torso) than holding the crouch?")
PREDICTION = ("The ankle strategy survives at least 10 N more than holding the crouch, "
              "because it moves the centre of pressure toward the side the robot falls.")
# The CHECK rule, decided now: PASS if yours survives a force at least MARGIN N
# larger than the twin's largest survived force, in EVERY seed.
MARGIN = 10.0


# =============================================================================
# 2. CHANGE: your controller, and its twin (the same thing with ONE change).
#    A controller is obs, t, env -> 12 residual actions in [-1, 1] on the crouch.
#    obs[0:3] gravity in the body frame, obs[3:6] gyro, obs[6:18] leg angles
#    minus the crouch, obs[18:30] leg velocities, obs[30:42] the previous action.
# =============================================================================
ANKLE_ROLL, ANKLE_PITCH = 2.0, 2.0       # week 7's measured gains


def yours(obs, t, env):
    """The ankle strategy: both ankles turn against the tilt the torso feels."""
    a = np.zeros(12, np.float32)
    a[5] = a[11] = ANKLE_ROLL * obs[1]   # ankle roll  <- sideways tilt
    a[4] = a[10] = ANKLE_PITCH * obs[0]  # ankle pitch <- forward tilt
    return np.clip(a, -1.0, 1.0)


def twin(obs, t, env):
    """The twin: identical except that the ankles do nothing (hold the crouch)."""
    return np.zeros(12, np.float32)


CONTROLLERS = {"yours": yours, "twin": twin}


# =============================================================================
# 3. CHANGE: one run of the experiment -> one row of numbers.
#    Return a dict; every key becomes a column of results.csv.
# =============================================================================
def run_once(controller, force: float, seed: int, *, seconds: float = 4.0, record: bool = False):
    from soc4180.envs import G1WalkEnv

    env = G1WalkEnv()
    model, data = env.model, env.data
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    obs, _ = env.reset(seed=seed)
    frames, fell, steps = [], False, 0
    while data.time < seconds:
        a = controller(obs, data.time, env)
        data.xfrc_applied[torso, :] = 0.0
        if 1.0 <= data.time < 1.2:                 # the shove: +y, world frame, 0.2 s at t = 1 s
            data.xfrc_applied[torso, 1] = force
        obs, _, terminated, truncated, _ = env.step(a)
        steps += 1
        if record:
            frames.append(data.qpos.copy())
        if terminated:                             # G1WalkEnv: tilt past 44.4 degrees
            fell = True
            break
        if truncated:
            break
    row = {"force_N": force, "seed": seed, "fell": int(fell), "seconds": round(data.time, 2),
           "pelvis_z_m": round(float(data.qpos[2]), 3), "drift_y_m": round(float(data.qpos[1]), 3)}
    return (row, frames, env) if record else row


# =============================================================================
# 4. The method: sweep, write results.csv, print the table, decide the CHECK.
#    Change this part only if your question needs a different sweep.
# =============================================================================
def largest_survived(rows, name, seed):
    ok = [r["force_N"] for r in rows if r["controller"] == name and r["seed"] == seed and not r["fell"]]
    return max(ok) if ok else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--forces", type=float, nargs="+", default=[60, 65, 70, 75, 80])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default=str(HERE / "results.csv"))
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--replay", choices=list(CONTROLLERS), default="yours")
    ap.add_argument("--force", type=float, default=75.0, help="the force of the replayed run")
    args = ap.parse_args(argv)

    try:
        import gymnasium  # noqa: F401
    except ImportError:
        print("This template needs gymnasium: run `uv sync --extra rl` once, then try again.")
        return 1

    print(f"QUESTION   {QUESTION}\nPREDICTION {PREDICTION}\n")
    rows, t0 = [], time.time()
    for name, ctl in CONTROLLERS.items():
        for seed in args.seeds:
            for f in args.forces:
                rows.append({"controller": name, **run_once(ctl, f, seed)})

    out = pathlib.Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} runs in {time.time() - t0:.0f} s -> {out}\n")

    print(f"{'force (N)':>10}  " + "  ".join(f"{n + ' s' + str(s):>12}" for n in CONTROLLERS for s in args.seeds))
    for f in args.forces:
        cells = []
        for n in CONTROLLERS:
            for s in args.seeds:
                r = next(r for r in rows if r["controller"] == n and r["seed"] == s and r["force_N"] == f)
                cells.append(f"{'FELL ' + str(r['seconds']) + ' s' if r['fell'] else 'stood':>12}")
        print(f"{f:>10g}  " + "  ".join(cells))

    verdicts = []
    for s in args.seeds:
        y, tw = largest_survived(rows, "yours", s), largest_survived(rows, "twin", s)
        verdicts.append(y is not None and (tw is None or y >= tw + MARGIN))
        print(f"seed {s}: largest survived  yours {y} N   twin {tw} N")
    print(f"\nCHECK {'PASS' if all(verdicts) else 'FAIL'}  (rule: yours survives >= twin + {MARGIN:g} N in every seed)")

    if args.no_viewer or soc4180.is_colab():
        return 0
    _, frames, env = run_once(CONTROLLERS[args.replay], args.force, args.seeds[0], record=True)
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    print(f"\nreplaying {args.replay} at {args.force:g} N (close the window to stop)")
    with soc4180.launch_viewer(env.model, env.data, passive=True) as viewer:
        while viewer.is_running() and time.time() < deadline:
            for q in frames:
                if not viewer.is_running() or time.time() > deadline:
                    break
                env.data.qpos[:] = q; mujoco.mj_forward(env.model, env.data); viewer.sync()
                time.sleep(env.control_dt)
            time.sleep(0.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
