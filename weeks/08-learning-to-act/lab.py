"""Week 8: learn to balance from reward.

    uv run weeks/08-learning-to-act/lab.py                 # learn (~1 min), test, replay in the viewer
    uv run weeks/08-learning-to-act/lab.py --no-viewer     # headless
    uv run weeks/08-learning-to-act/lab.py --updates 0     # no learning: test the starting rule

THE TASK. The crouched G1 is shoved sideways on the torso (60-80 N, 0.2 s,
left or right). Holding the crouch, it falls from 70 N. It can save itself by
turning both ankles (pitch) against the tilt it feels:

    ankle pitch = theta * g_x          g_x = obs[0], the forward tilt in the torso's frame

theta is the one number the robot must LEARN, starting from 0 (hold the
crouch), from nothing but a reward. You change `reward()` and the learner's
settings, and watch what it learns.

THE LEARNING RULE: a policy gradient, exploring in theta itself.

    repeat:
        draw 8 pushes p_i and 8 random directions eps_i ~ N(0, 1)
        try   theta + sigma*eps_i  and  theta - sigma*eps_i  on the same push p_i
        G+_i, G-_i = the returns (sums of reward) of the two tries
        theta <- theta + lr * mean_i[ (G+_i - G-_i) * eps_i ] / (2 sigma)

"Try a bit more and a bit less on the same push; move toward the one that
paid." The pair shares its push, so the difference G+ - G- is caused by theta
alone -- that is why it learns in 20 updates where adding noise to the
actions (REINFORCE as usually written, the deck) mostly knocks the robot over.

MEASURED with this file (seeds 0, 1, 2; 20 updates of 8 pairs, ~50 s each):
    theta 0 -> 1.82 / 2.00 / 1.53   (week 7's hand-written ankle gain is 2)
    the test table: holding the crouch (theta 0) falls at 70 N and 75 N, left
    and right; theta 0.6 to 4 stands at 75 N; theta 8 falls at 75 N again (too
    much gain); every theta falls at 80 N; a negative theta falls at 60 N.
    The roll gain on g_y, tried as well, changed nothing at any of these
    pushes: the pitch strategy is what saves the robot.

Try (one change at a time; write your prediction down first):
    1. ALIVE = 0: is there still a reason to learn?
    2. reward only survival: return 1.0 on every step, 0 on the step it falls.
    3. UPRIGHT = 1000: does the learned theta change?
    4. --pushes 30 50: pushes nobody falls from. Is there anything to learn?
    5. SIGMA = 0.05 or 3: tries too close together, or too far apart.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mujoco
import numpy as np

import soc4180

# -----------------------------------------------------------------------------
# 1. THE REWARD -- the function you change.
#    Called every decision (50 per second). `fell` is True on the step the
#    robot tilts past 44 degrees, which ends the episode.
# -----------------------------------------------------------------------------
ALIVE = 1.0          # per decision, for still being up
UPRIGHT = 10.0       # penalty on tilt squared (g_x^2 + g_y^2 = sin^2 of the tilt)
EFFORT = 0.01        # penalty on the ankle command squared


def reward(obs, action, fell):
    gx, gy = obs[0], obs[1]
    return ALIVE - UPRIGHT * (gx * gx + gy * gy) - EFFORT * float(np.sum(action * action))


# -----------------------------------------------------------------------------
# 2. The learner's settings. Change one at a time.
# -----------------------------------------------------------------------------
SIGMA = 0.5          # how far apart the two tries are, in gain units
LR = 0.01            # learning rate
PAIRS = 8            # pairs of tries per update
SECONDS = 4.0        # episode length; the push comes at t = 1 s and lasts 0.2 s


# -----------------------------------------------------------------------------
# 3. One episode: the G1 in its crouch, one sideways push, the rule acting.
# -----------------------------------------------------------------------------
def episode(env, theta, push, *, record=False):
    """push in newtons, + = toward the robot's left. Returns (G, fell, frames)."""
    model, data = env.model, env.data
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    obs, _ = env.reset(seed=0)
    G, fell, frames = 0.0, False, []
    while data.time < SECONDS:
        action = np.zeros(12, np.float32)
        action[4] = action[10] = theta * obs[0]          # both ankle pitches against the tilt
        action = np.clip(action, -1.0, 1.0)

        data.xfrc_applied[torso, :] = 0.0
        if 1.0 <= data.time < 1.2:
            data.xfrc_applied[torso, 1] = push           # world frame, +y
        obs, _, terminated, truncated, _ = env.step(action)
        fell = bool(terminated)
        G += reward(obs, action, fell)
        if record:
            frames.append(data.qpos.copy())
        if terminated or truncated:
            break
    return G, fell, frames


# -----------------------------------------------------------------------------
# 4. Learning: pairs of tries on shared pushes, one step per update.
# -----------------------------------------------------------------------------
def learn(env, updates, lo, hi, seed):
    rng = np.random.default_rng(seed)
    theta = 0.0                                          # start: hold the crouch
    print(f"{'update':>6} {'theta':>7} {'mean return':>12} {'falls':>7}")
    for k in range(updates):
        pushes = rng.choice([-1.0, 1.0], PAIRS) * rng.uniform(lo, hi, PAIRS)
        eps = rng.standard_normal(PAIRS)
        plus = [episode(env, theta + SIGMA * e, p) for e, p in zip(eps, pushes)]
        minus = [episode(env, theta - SIGMA * e, p) for e, p in zip(eps, pushes)]
        Gp = np.array([r[0] for r in plus])
        Gm = np.array([r[0] for r in minus])
        falls = sum(r[1] for r in plus + minus)
        print(f"{k:6d} {theta:7.2f} {np.mean([Gp, Gm]):12.1f} {falls:3d}/{2 * PAIRS}")
        theta += LR * float(np.mean((Gp - Gm) * eps)) / (2 * SIGMA)
    return theta


# -----------------------------------------------------------------------------
# 5. The test: fixed pushes, three rules side by side.
# -----------------------------------------------------------------------------
def test(env, rules, forces):
    print(f"\n{'push':>12}  " + "  ".join(f"{name:>13}" for name in rules))
    for f in forces:
        for label, push in ((f"{f:g} N left", f), (f"{f:g} N right", -f)):
            cells = ["FELL" if episode(env, th, push)[1] else "stood" for th in rules.values()]
            print(f"{label:>12}  " + "  ".join(f"{c:>13}" for c in cells))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Week 8: learn the ankle gain from reward.")
    ap.add_argument("--updates", type=int, default=20)
    ap.add_argument("--pushes", type=float, nargs=2, default=[60.0, 80.0], metavar=("LO", "HI"),
                    help="training pushes: uniform in [LO, HI] newtons, left or right")
    ap.add_argument("--test", type=float, nargs="+", default=[70.0, 75.0, 80.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(argv)
    try:
        from soc4180.envs import G1WalkEnv
    except ImportError:
        print("This lab needs gymnasium: run `uv sync --extra rl` once, then try again.")
        return 1

    env = G1WalkEnv()
    t0 = time.time()
    theta = learn(env, args.updates, *args.pushes, args.seed)
    print(f"learned theta = {theta:.2f} in {time.time() - t0:.0f} s")
    test(env, {"hold (0)": 0.0, "hand (2)": 2.0, f"learned ({theta:.2f})": theta}, args.test)

    if args.no_viewer or soc4180.is_colab():
        return 0
    push = args.test[0]
    _, fell, frames = episode(env, theta, push, record=True)
    print(f"\nreplaying the learned rule under {push:g} N to the left "
          f"({'fell' if fell else 'stood'}); close the window to stop")
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
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
