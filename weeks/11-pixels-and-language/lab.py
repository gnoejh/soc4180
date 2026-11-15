"""Week 11: "look at the green ball" -- words, pixels, a turn, and a check.

    uv run weeks/11-pixels-and-language/lab.py                          # the test battery, then a replay
    uv run weeks/11-pixels-and-language/lab.py --say "look at the blue ball"
    uv run weeks/11-pixels-and-language/lab.py --no-viewer

THE TASK. The G1 stands in front of four coloured balls (red, green, blue,
yellow) with a camera in its head. It is told, in words, which one to look
at. It must:

    1. pick the colour out of the words        -- refuse a word it does not know
    2. find that colour in the camera picture  -- `find_ball()`, the function you change
    3. turn its waist until the ball is in the middle of the picture
    4. CHECK that it worked, with the picture   -- refuse if the ball is not there

You change `find_ball()`. As shipped it is the simplest thing that can work:
every pixel close enough to the ball's colour, averaged. The deck shows the
learned versions of steps 1-2 -- a CNN that reads the direction from pixels,
and a network that grounds words in pixels -- and why the check in step 4
matters more than either: a network can be confidently wrong.

From pixel to angle. The camera's picture spans -1..+1 across (u) and down
(v), with a 90 degree field of view, so tan(45 deg) = 1 and

    azimuth = atan(-u)          (u > 0: right of centre -> turn right, negative azimuth)

The turn is a P controller on the waist (week 5): waist target += GAIN * azimuth.

Measured with this file, as shipped: all 8 looks done, every ball within
0.3 deg of the middle (the "true error" column, from the simulator, which
the robot never sees); the hidden green ball and the word "crimson" refused.

Try (predict each one first; what this file measured is in brackets):
    1. THRESHOLD = 0.05 or 0.6: too strict, too loose.
       [0.05: sees nothing, refuses all 10. 0.6: says "done" 9 times, but is
       up to 39 deg off -- and "sees" the hidden green ball, 149 deg away.]
    2. --dim 0.3: the light dims.
       [says "done" 9 times, up to 41 deg off; "sees" the hidden ball. The
       check uses the same sense that failed, so it cannot catch it.]
    3. GAIN = 2.5: the waist overshoots (week 5).
       [4 of 8 refused: the ball swings out of the picture.]
    4. MIN_PIXELS = 1. [no change here: noise does not look like a ball.]
    5. What check would catch 1 and 2? (A second sense; the true-error
       column is the one you are not allowed to use.)
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time

import mujoco
import numpy as np

import soc4180

# -----------------------------------------------------------------------------
# 1. FINDING THE BALL -- the function you change.
#    image: (H, W, 3) uint8 from the head camera. rgb: the ball's colour, 0..1.
#    Return (u, v) in -1..+1 (u right, v down) or None if it is not there.
# -----------------------------------------------------------------------------
THRESHOLD = 0.25      # how close a pixel's colour must be to count (0..1.7)
MIN_PIXELS = 4        # fewer matching pixels than this: "not there"


def find_ball(image, rgb):
    pixels = image.astype(float) / 255.0
    distance = np.linalg.norm(pixels - np.asarray(rgb), axis=-1)       # (H, W)
    rows, cols = np.nonzero(distance < THRESHOLD)
    if len(rows) < MIN_PIXELS:
        return None
    h, w = distance.shape
    u = (cols.mean() + 0.5) / w * 2 - 1
    v = (rows.mean() + 0.5) / h * 2 - 1
    return u, v


# -----------------------------------------------------------------------------
# 2. Words -> a colour. The robot knows four words; anything else is refused.
# -----------------------------------------------------------------------------
def colour_of(words):
    from soc4180.vision import COLOURS
    known = [w for w in re.findall(r"[a-z]+", words.lower()) if w in COLOURS]
    return known[0] if known else None


GAIN = 0.8            # waist turn per radian of azimuth, per 0.1 s
RESOLUTION = 64       # camera picture, pixels square
CENTRED_DEG = 5.0     # the check: the ball within this many degrees of the middle


# -----------------------------------------------------------------------------
# 3. One instruction: parse, look, turn, check. Returns (ok, message, true error).
# -----------------------------------------------------------------------------
def obey(body, renderer, words, dim=1.0):
    from soc4180.vision import CAMERA, COLOURS, ball_direction

    colour = colour_of(words)
    if colour is None:
        return False, "refused: no colour I know in those words", None
    rgb = np.asarray(COLOURS[colour]) * dim
    w = body.act("waist_yaw")
    lo, hi = body.model.actuator_ctrlrange[w]

    def picture():
        renderer.update_scene(body.data, camera=CAMERA)
        return renderer.render()

    for _ in range(20):                                            # 2 s of looking and turning
        uv = find_ball(picture(), rgb)
        if uv is None:
            return False, f"refused: I do not see a {colour} ball", None
        azimuth = math.atan(-uv[0])
        body.data.ctrl[w] = float(np.clip(body.data.ctrl[w] + GAIN * azimuth, lo, hi))
        body.step(0.1)

    uv = find_ball(picture(), rgb)                                 # the check, from the picture
    truth = math.degrees(ball_direction(body.model, body.data, ["red", "green", "blue", "yellow"].index(colour))[0])
    if not body.standing:
        return False, "failed: fell over while turning", truth
    if uv is None or abs(math.degrees(math.atan(-uv[0]))) > CENTRED_DEG:
        return False, f"failed: the {colour} ball is not in the middle", truth
    return True, f"done: {colour} in the middle (waist {math.degrees(body.data.qpos[body.act_q[w]]):+.0f} deg)", truth


def scene(body, seed, hide=None):
    """Balls at random directions (seeded); `hide` puts one behind the robot."""
    rng = np.random.default_rng(seed)
    az = rng.permutation(np.linspace(-0.7, 0.7, 4)) + rng.uniform(-0.1, 0.1, 4)
    balls = {n: (float(a), float(rng.uniform(-0.2, 0.15)), float(rng.uniform(1.4, 2.2)))
             for n, a in zip(("red", "green", "blue", "yellow"), az)}
    if hide:
        balls[hide] = (math.pi, 0.0, 2.0)
    body.reset(balls)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Week 11: look at the ball the words name.")
    ap.add_argument("--say", default=None, help="one instruction instead of the battery")
    ap.add_argument("--dim", type=float, default=1.0, help="scale the lights (1 = normal)")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(argv)
    try:
        from soc4180.agent import Body
        from soc4180.render import _new_renderer
    except ImportError:
        print("This lab needs torch: run `uv sync --extra rl` once, then try again.")
        return 1

    body = Body()
    if args.dim != 1.0:
        body.model.light_diffuse[:] *= args.dim
        body.model.light_ambient[:] *= args.dim
    renderer = _new_renderer(body.model, RESOLUTION, RESOLUTION)

    if args.say:
        tests = [(0, None, args.say)]
    else:
        tests = [(s, None, f"look at the {c} ball") for s in (1, 2) for c in ("red", "green", "blue", "yellow")]
        tests += [(1, "green", "look at the green ball"),          # the ball is not in view
                  (1, None, "look at the crimson ball")]           # a word it does not know
    print(f"{'scene':>5} {'instruction':28s} {'robot says':52s} {'true error':>10}")
    ok_count = 0
    for seed, hide, words in tests:
        scene(body, seed, hide)
        ok, msg, truth = obey(body, renderer, words, args.dim)
        ok_count += ok
        tag = f"{seed}" + (" (no " + hide + ")" if hide else "")
        print(f"{tag:>5} {words:28s} {msg:52s} {('%+.1f deg' % truth) if truth is not None else '-':>10}")
    print(f"\n{ok_count} of {len(tests)} instructions carried out"
          + ("" if args.say else " (the last two should be refused when the robot is right)"))

    if args.no_viewer or soc4180.is_colab():
        return 0
    scene(body, 1)
    obey(body, renderer, args.say or "look at the green ball", args.dim)
    frames = body.frames
    print("\nreplaying the last turn; close the window to stop")
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    d = body.data
    with soc4180.launch_viewer(body.model, d, passive=True) as viewer:
        while viewer.is_running() and time.time() < deadline:
            for q in frames:
                if not viewer.is_running() or time.time() > deadline:
                    break
                d.qpos[:] = q; mujoco.mj_forward(body.model, d); viewer.sync()
                time.sleep(17 * body.model.opt.timestep)
            time.sleep(0.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
