"""Run every lab script the way a student would, briefly, and report what broke.

    uv run scripts/check_labs.py               # everything: pipelines headless, then labs with the viewer
    uv run scripts/check_labs.py --no-viewer   # pipelines only (what a headless machine can run)
    uv run scripts/check_labs.py --only 03 07  # a subset, by week prefix

Each week's pipeline script (`stack.py`, `reach.py`, `walk.py`, ...) runs with
`--no-viewer` and the smallest arguments that exercise it; each interactive lab
(`lab_*.py`) runs with `SOC4180_AUTOCLOSE=4`, which closes its window after
four seconds. A script passes if it exits 0 within its time limit. Failures
print the tail of their output. The RL weeks need `uv sync --extra rl`.

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
    ("03", "weeks/03-inverse-kinematics/reach.py --target 0.05 0 0.05"),
    ("03", "weeks/03-inverse-kinematics/lab_connected.py --grade --problem 1 4"),
    ("04", "weeks/04-walking/walk.py --steps 2"),
    ("05", "weeks/05-actuation/servo.py --seconds 0.3"),
    ("06", "weeks/06-sensing/imu.py --seconds 1"),
    ("07", "weeks/07-mdp/env_run.py --policy hold"),
    ("08", "weeks/08-ppo/train.py --steps 0 --episodes 1"),
    ("09", "weeks/09-reward/shape.py --steps 2048 --episodes 1"),
    ("10", "weeks/10-scaling/many.py --robots 1 2 --procs 1 --seconds 0.5"),
]

LABS = [
    ("00", "weeks/00-robot-stack/lab_stack.py"),
    ("01", "weeks/01-intro/lab_mjcf.py"),
    ("02", "weeks/02-transforms/lab_viewer.py"),
    ("02b", "weeks/02b-robot-as-code/lab_body.py"),
    ("03", "weeks/03-inverse-kinematics/lab_ik.py"),
    ("03", "weeks/03-inverse-kinematics/lab_connected.py"),
    ("04", "weeks/04-walking/lab_walk.py"),
    ("05", "weeks/05-actuation/lab_servo.py"),
    ("06", "weeks/06-sensing/lab_imu.py"),
    ("07", "weeks/07-mdp/lab_env.py"),
    ("08", "weeks/08-ppo/lab_train.py"),
    ("09", "weeks/09-reward/lab_reward.py"),
    ("10", "weeks/10-scaling/lab_many.py"),
]


def run(cmd: str, viewer: bool, limit: float) -> tuple[bool, float, str]:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if viewer:
        env["SOC4180_AUTOCLOSE"] = "4"
    else:
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

    jobs = [(w, c, False) for w, c in PIPELINES]
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
