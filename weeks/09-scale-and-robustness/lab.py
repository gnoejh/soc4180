"""Week 9: is a learned reflex robust to a world it did not train in?

    uv run weeks/09-scale-and-robustness/lab.py                # the test grid (~3 min), then a replay
    uv run weeks/09-scale-and-robustness/lab.py --no-viewer
    uv run weeks/09-scale-and-robustness/lab.py --episodes 3   # quicker, noisier

THE TASK. Two push-recovery policies were trained with PPO (3 M steps each,
~16 min on a 36-core machine; shipped in checkpoints/):

    PPO nominal   trained in ONE world: the normal G1
    PPO random    trained in RANDOMISED worlds (domain randomisation): every
                  episode drew friction x0.3-1.2, torso mass -5..+10 kg and
                  servo stiffness x0.7-1.3

Both see the same 42 numbers as week 8's environment and output 12 leg
residuals at 50 Hz. You do not train anything this week. You TEST: edit the
WORLDS list -- the worlds the robot meets -- and find out where each policy
breaks. Week 8's one-number ankle rule and plain holding are in the table
too, so you can see what 3 M steps of PPO bought over one number.

The test: a sideways shove of EXACTLY N newtons for 0.2 s at a random moment;
count the episodes survived. Exact sizes, not random ones: a mixed bag of
pushes averages a policy's strengths and weaknesses away.

Measured with this file (10 episodes each, sideways; ~3.5 min): episodes survived

    world                 policy          0 N  60 N  80 N 100 N 120 N
    nominal               hold             10    10     0     0     0
                          ankle (wk 8)     10    10     0     0     0
                          PPO nominal      10    10    10     0     0
                          PPO random       10    10    10     9     1
    heavy (+15 kg)        hold              0     0     0     0     0
                          ankle (wk 8)     10    10     6     0     0
                          PPO nominal      10    10     8     0     0
                          PPO random       10    10    10    10    10
    ice (friction 0.2)    hold             10    10     0     0     0
                          ankle (wk 8)     10    10    10     0     0
                          PPO nominal      10    10     9     6     0
                          PPO random       10    10    10    10     5
    weak servos (kp 0.6)  hold             10     0     0     0     0
                          ankle (wk 8)     10    10     0     0     0
                          PPO nominal      10    10     3     0     0
                          PPO random       10    10    10     0     0

Holding the crouch falls with NO push once the torso is 15 kg heavier; the
policy trained in randomised worlds is the best row in every world here.

Try (predict each one first):
    1. add a world OUTSIDE the training range, e.g. mass=+25 or friction=0.1.
       Does PPO random still win there?
    2. kp=0.5 (weak servos): which policy degrades most gracefully?
    3. --any-direction: pushes from every side. Which policy was trained for that?
    4. Which row would you send to a real robot whose mass you know only to +-5 kg?
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

import soc4180

HERE = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# 1. THE TEST WORLDS -- the list you change.
#    friction: scale on every surface's friction; mass: kg added to the torso;
#    kp: scale on every servo's stiffness (damping follows, same damping ratio).
# -----------------------------------------------------------------------------
WORLDS = {
    "nominal": dict(friction=1.0, mass=0.0, kp=1.0),
    "heavy (+15 kg)": dict(friction=1.0, mass=15.0, kp=1.0),
    "ice (friction 0.2)": dict(friction=0.2, mass=0.0, kp=1.0),
    "weak servos (kp 0.6)": dict(friction=1.0, mass=0.0, kp=0.6),
}
PUSHES = (0, 60, 80, 100, 120)        # newtons, sideways, 0.2 s


# -----------------------------------------------------------------------------
# 2. The policies under test: obs (42 numbers) -> 12 leg residuals in [-1, 1].
# -----------------------------------------------------------------------------
def make_policies():
    from stable_baselines3 import PPO

    nominal = PPO.load(HERE / "checkpoints" / "push_nominal.zip", device="cpu")
    randomised = PPO.load(HERE / "checkpoints" / "push_random.zip", device="cpu")

    def hold(obs):
        return np.zeros(12, np.float32)

    def ankle(obs):                                   # week 8: one learned number
        a = np.zeros(12, np.float32)
        a[4] = a[10] = np.clip(2.0 * obs[0], -1, 1)
        return a

    return {
        "hold": hold,
        "ankle (wk 8)": ankle,
        "PPO nominal": lambda obs: nominal.predict(obs, deterministic=True)[0],
        "PPO random": lambda obs: randomised.predict(obs, deterministic=True)[0],
    }


# -----------------------------------------------------------------------------
# 3. One count: episodes survived in one world, under one exact push.
# -----------------------------------------------------------------------------
def survived(env, policy, world, push, episodes, *, sideways=True, record=False):
    env.world = {k: (v, v) for k, v in world.items()}      # this world, every episode
    ok, frames = 0, []
    for k in range(episodes):
        env.push_max = max(push, 1.0)
        obs, _ = env.reset(seed=k)                         # the moment of the push differs by seed
        if sideways:
            env.push_force = np.array([0.0, float(push)])
        else:                                              # random direction, exactly `push` N
            env.push_force *= push / max(np.linalg.norm(env.push_force), 1e-9)
        while True:
            obs, _, fell, done, _ = env.step(policy(obs))
            if record:
                frames.append(env.data.qpos.copy())
            if fell or done:
                break
        ok += int(not fell)
    return ok, frames


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Week 9: test push-recovery policies across worlds.")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--any-direction", action="store_true", help="pushes from every side")
    ap.add_argument("--replay", default="PPO random", help="which policy to replay")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(argv)
    try:
        from soc4180.envs import G1PushEnv
        policies = make_policies()
    except ImportError:
        print("This lab needs gymnasium and stable-baselines3: run `uv sync --extra rl` once.")
        return 1

    env = G1PushEnv()
    print(f"episodes survived out of {args.episodes}; a shove of exactly N newtons, "
          f"{'any direction' if args.any_direction else 'sideways'}, 0.2 s\n")
    print(f"{'world':22s} {'policy':14s} " + " ".join(f"{p:>5} N" for p in PUSHES))
    t0 = time.time()
    for wname, world in WORLDS.items():
        for pname, policy in policies.items():
            counts = [survived(env, policy, world, p, args.episodes, sideways=not args.any_direction)[0]
                      for p in PUSHES]
            print(f"{wname:22s} {pname:14s} " + " ".join(f"{c:7d}" for c in counts), flush=True)
        print()
    print(f"({time.time() - t0:.0f} s)")

    if args.no_viewer or soc4180.is_colab():
        return 0
    world = WORLDS[next(iter(WORLDS))]
    _, frames = survived(env, policies[args.replay], world, 100, 1, record=True)
    print(f"\nreplaying '{args.replay}' in '{next(iter(WORLDS))}' with a 100 N shove; close the window to stop")
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
