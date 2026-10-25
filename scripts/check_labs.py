"""Run every lab script the way a student would, briefly, and report what broke.

    uv run scripts/check_labs.py               # everything: pipelines headless, then labs with the viewer
    uv run scripts/check_labs.py --no-viewer   # pipelines only (what a headless machine can run)
    uv run scripts/check_labs.py --only 03 07  # a subset, by week prefix

Each week's pipeline script (`stack.py`, `walk.py`, ...) runs with
`--no-viewer` and the smallest arguments that exercise it; each interactive lab
(`lab_*.py`) runs with `SOC4180_AUTOCLOSE=4`, which closes its window after
four seconds. A script passes if it exits 0 within its time limit. Failures
print the tail of their output. The RL weeks (7 on) need `uv sync --extra rl`. Weeks 11-15's
`lab_*.py` are graded games; their `--grade --problem N` runs one problem headless.
Weeks 16+ run on ubrain from vendor/ubrain/; their scripts are headless by
default and run exactly as written. Weeks 26-27 need CLIP: run them as
`uv run --extra vision scripts/check_labs.py --only 26 27` (a plain `uv run` drops torch).

This is the smoke test behind the "every script passed" claims in CLAUDE.md;
run it after touching the package.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PIPELINES = [
    ("00", "weeks/00-robot-stack/stack.py --layer hold --seconds 1"),
    ("01", "weeks/01-intro/mjcf_run.py --seconds 1"),
    ("02", "weeks/02-transforms/fk.py"),
    ("02b", "weeks/02b-robot-as-code/anatomy.py --nudge waist"),
    ("03", "weeks/03-inverse-kinematics/lab_connected.py --grade --problem 1 4"),
    ("04", "weeks/04-walking/walk.py --steps 2"),
    ("05", "weeks/05-actuation/servo.py --seconds 0.3"),
    ("06", "weeks/06-sensing/imu.py --seconds 1"),
    ("07", "weeks/07-mdp/env_run.py --policy hold"),
    ("08", "weeks/08-ppo/train.py --steps 0 --episodes 1"),
    ("09", "weeks/09-reward/shape.py --steps 2048 --episodes 1"),
    ("10", "weeks/10-scaling/many.py --robots 1 2 --procs 1 --seconds 0.5"),
    ("11", "weeks/11-planning/plan.py --seconds 0.5 --samples 8"),
    ("11", "weeks/11-planning/lab_plan.py --grade --problem 1"),
    ("12", "weeks/12-robustness/robust.py --episodes 1"),
    ("12", "weeks/12-robustness/lab_robust.py --grade --problem 1"),
    ("13", "weeks/13-imitation/imitate.py --expert policy --demos 2"),
    ("13", "weeks/13-imitation/lab_imitate.py --grade --problem 1"),
    ("14", "weeks/14-vision/look.py --train 200 --test 50 --epochs 1"),
    ("14", "weeks/14-vision/lab_look.py --grade --problem 1"),
    ("15", "weeks/15-language/ground.py --train 200 --epochs 1"),
    ("15", "weeks/15-language/lab_agent.py --grade --problem 1"),
]

# run as written: no --no-viewer appended (week 16's scripts use --headless, or no flag)
AS_IS = [
    ("16", "weeks/16-cerebellum/lab_cerebellum.py --seconds 3 --name check_labs"),
    ("16", "weeks/16-cerebellum/lab_cerebellum_walk.py --headless --seconds 3"),
    ("17", "weeks/17-adaptation/lab_adapt.py --blocks 1:0,2:15,1:0 --name check_labs"),
    ("17", "weeks/17-adaptation/lab_adapt_walk.py --headless --seconds 3 --wind-at 1"),
    ("18", "weeks/18-striatum/lab_striatum.py --trials 600 --name check_labs"),
    ("18", "weeks/18-striatum/lab_choose.py --trials 4 --name check_labs"),
    ("18", "weeks/18-striatum/lab_choose.py --map --phases 1"),
    ("19", "weeks/19-brain-life/lab_life.py train --pushes 1 --save weeks/19-brain-life/brains/check.npz"),
    ("19", "weeks/19-brain-life/lab_life.py test --pushes 1 --load weeks/19-brain-life/brains/check.npz"),
    ("19", "weeks/19-brain-life/lab_life.py identical --ticks 60"),
    ("19", "weeks/19-brain-life/lab_life.py sheet"),
    ("20", "weeks/20-hippocampus/lab_remember.py --episodes 1 --pushes 3 --name check_labs"),
    ("20", "weeks/20-hippocampus/lab_remember.py --certify"),
    ("21", "weeks/21-working-memory/lab_hold.py --lists 1 --pushes 4 --name check_labs"),
    ("21", "weeks/21-working-memory/lab_hold.py --law"),
    ("22", "weeks/22-consolidation/lab_consolidate.py --episodes 1 --pushes 3 --name check_labs"),
    ("22", "weeks/22-consolidation/lab_consolidate.py --distinct --episodes 2 --name check_labs"),
    ("23", "weeks/23-attention/lab_notice.py --bursts 1 --burst 3 --quiet 2 --name check_labs"),
    ("24", "weeks/24-memory-habit/lab_habit.py --trials 20 --seeds 2 --name check_labs"),
    ("24", "weeks/24-memory-habit/lab_habit.py --bandit 4 --trials 100 --seeds 1 --name check_labs"),
    ("25", "weeks/25-neuromodulators/lab_patience.py serotonin --tau 4 --trials 200"),
    ("25", "weeks/25-neuromodulators/lab_patience.py noradrenaline --events 1200"),
    ("26", "weeks/26-codecs/lab_codecs.py words"),
    ("26", "weeks/26-codecs/lab_codecs.py capacity --d 128 --episodes 2"),
    ("27", "weeks/27-explorer/lab_explore.py --seconds 12 --name check_labs"),
]

LABS = [
    ("00", "weeks/00-robot-stack/lab_stack.py"),
    ("01", "weeks/01-intro/lab_mjcf.py"),
    ("02", "weeks/02-transforms/lab_viewer.py"),
    ("02b", "weeks/02b-robot-as-code/lab_body.py"),
    ("03", "weeks/03-inverse-kinematics/lab_connected.py"),
    ("04", "weeks/04-walking/lab_walk.py"),
    ("05", "weeks/05-actuation/lab_servo.py"),
    ("06", "weeks/06-sensing/lab_imu.py"),
    ("07", "weeks/07-mdp/lab_env.py"),
    ("08", "weeks/08-ppo/lab_train.py"),
    ("09", "weeks/09-reward/lab_reward.py"),
    ("10", "weeks/10-scaling/lab_many.py"),
    ("11", "weeks/11-planning/lab_plan.py"),
    ("12", "weeks/12-robustness/lab_robust.py"),
    ("13", "weeks/13-imitation/lab_imitate.py"),
    ("14", "weeks/14-vision/lab_look.py"),
    ("15", "weeks/15-language/lab_agent.py"),
]


def run(cmd: str, viewer: bool, limit: float) -> tuple[bool, float, str]:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if viewer:
        env["SOC4180_AUTOCLOSE"] = "4"
    elif cmd not in {c for _, c in AS_IS}:
        cmd += " --no-viewer"
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, *cmd.split()], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=limit, text=True,
                           encoding="utf-8", errors="replace")
        ok, out = p.returncode == 0, p.stdout
    except subprocess.TimeoutExpired as e:
        ok, out = False, (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "") + "\n[timed out]"
    return ok, time.time() - t0, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-viewer", action="store_true", help="skip the interactive labs")
    ap.add_argument("--only", nargs="*", default=None, help="week prefixes to run, e.g. 03 07")
    ap.add_argument("--limit", type=float, default=300.0, help="seconds per script")
    args = ap.parse_args(argv)

    jobs = [(w, c, False) for w, c in PIPELINES + AS_IS]
    if not args.no_viewer:
        jobs += [(w, c, True) for w, c in LABS]
    if args.only:
        jobs = [j for j in jobs if j[0] in args.only]

    failed = []
    print(f"{'week':>4}  {'kind':8}  {'result':6}  {'s':>6}  script")
    for week, cmd, viewer in jobs:
        ok, secs, out = run(cmd, viewer, args.limit)
        print(f"{week:>4}  {'lab' if viewer else 'pipeline':8}  {'ok' if ok else 'FAIL':6}  {secs:6.1f}  {cmd.split()[0]}", flush=True)
        if not ok:
            failed.append((cmd, out))
    for cmd, out in failed:
        print(f"\n--- {cmd}\n" + "\n".join(out.strip().splitlines()[-15:]))
    print(f"\n{len(jobs) - len(failed)} of {len(jobs)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
