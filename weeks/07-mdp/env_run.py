"""One episode of G1WalkEnv under a policy you can read, printed term by term, then replayed.

    uv run weeks/07-mdp/env_run.py --policy hold
    uv run weeks/07-mdp/env_run.py --policy random --no-viewer
    uv run weeks/07-mdp/env_run.py --policy walker
    uv run weeks/07-mdp/env_run.py --policy walker --action-scale 1.0 --hz 100
    uv run weeks/07-mdp/env_run.py --policy lean --push 75
    uv run weeks/07-mdp/env_run.py --policy hold --push 70 --no-viewer

The environment is the course's own `G1WalkEnv`: 42 observations (gravity in
the body frame, gyro, 12 leg angles minus the crouch, 12 leg velocities, the
previous action), 12 residual actions in [-1, 1] scaled to radians, five
weighted reward terms, and an episode that TERMINATES when the robot falls
or is TRUNCATED when time runs out. A policy is any function obs -> action.
Five are here, each a few lines. The script runs one episode, prints a row
per second with the reward earned in that second and its split by term, the
episode line, and then replays it in the simulator.

    --policy hold|random|walker|lean|step   --seed 0
    --hz 50            decisions per second (physics stays at 500)
    --action-scale 0.3 radians per unit of action
    --push 75          push the torso sideways with this force for 0.2 s at t = 1 s
    --ankle-gain 2 2   the lean policy's ankle gains (roll, pitch); --lean-gain for the hips
    --weights alive=0 tracking=1.5   override reward weights
    --no-viewer   --speed 1

Six numbered steps; read them with the week 7 slides open.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import mujoco
import numpy as np

import soc4180

# ---------------------------------------------------------------------------
# Policies. obs[0:3] gravity in the body frame, obs[3:6] gyro, obs[6:18] the
# 12 leg angles minus the crouch (left leg then right, each hip pitch, hip
# roll, hip yaw, knee, ankle pitch, ankle roll), obs[18:30] their velocities,
# obs[30:42] the previous action. Actions are residuals on the crouch, in the
# same 12-joint order, in [-1, 1].
# ---------------------------------------------------------------------------

def hold(obs, t, env):
    """Zero residual: the servos hold the crouch. The baseline every reward is judged against."""
    return np.zeros(12, np.float32)


def random_policy(obs, t, env):
    """Uniform noise in [-1, 1]: what an untrained stochastic policy looks like."""
    return env.action_space.sample()


def step_in_place(obs, t, env):
    """Open-loop stepping: hip pitch sinusoids of opposite sign on the two legs."""
    a = np.zeros(12, np.float32)
    a[0], a[6] = 0.5 * math.sin(4 * t), -0.5 * math.sin(4 * t)
    return a


def lean(obs, t, env):
    """Close the loop: drive the ankles (and optionally the hips) against the tilt.

    obs[0:3] is gravity in the torso frame: (0, 0, -1) when upright. A
    sideways tilt shows up in obs[1], a forward tilt in obs[0]. The gains and
    signs were chosen by measuring which survive a push (`--push N`), not by
    reasoning: ankle roll and pitch at +2 survive a 75 N push where holding the
    crouch falls at 70 N (the ankle strategy); hip roll alone, either sign,
    changes nothing; ankle pitch at -2 falls with no push at all. Everything
    falls at 80 N -- a stance can lean, only a step can recover more.
    """
    a = np.zeros(12, np.float32)
    a[1] = a[7] = LEAN_ROLL * obs[1]          # both hip rolls
    a[0] = a[6] = LEAN_PITCH * obs[0]         # both hip pitches
    a[5] = a[11] = ANKLE_ROLL * obs[1]        # both ankle rolls  (the ankle strategy)
    a[4] = a[10] = ANKLE_PITCH * obs[0]       # both ankle pitches
    return a


LEAN_ROLL, LEAN_PITCH = 0.0, 0.0        # hip gains: measured to do nothing for a push
ANKLE_ROLL, ANKLE_PITCH = 2.0, 2.0      # ankle gains: survive 75 N where hold falls at 70
POLICIES = {"hold": hold, "random": random_policy, "walker": None, "lean": lean, "step": step_in_place}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--policy", choices=tuple(POLICIES), default="hold")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--action-scale", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--push", type=float, default=0.0, help="sideways force on the torso, N, for 0.2 s at t = 1 s")
    ap.add_argument("--weights", nargs="*", default=[], metavar="TERM=W")
    ap.add_argument("--lean-gain", type=float, nargs=2, metavar=("ROLL", "PITCH"))
    ap.add_argument("--ankle-gain", type=float, nargs=2, metavar=("ROLL", "PITCH"))
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(argv)
    global LEAN_ROLL, LEAN_PITCH, ANKLE_ROLL, ANKLE_PITCH
    if args.lean_gain:
        LEAN_ROLL, LEAN_PITCH = args.lean_gain
    if args.ankle_gain:
        ANKLE_ROLL, ANKLE_PITCH = args.ankle_gain

    try:
        from soc4180.envs import DEFAULT_REWARD, G1WalkEnv, walker_actions
        from soc4180.walking import GaitParams, WalkingController
    except ImportError:
        print("This script needs gymnasium: run `uv sync --extra rl` once, then try again.")
        return 1

    # -- 1. the environment: the MDP written down -------------------------------------------
    weights = {k: float(v) for k, v in (w.split("=") for w in args.weights)}
    env = G1WalkEnv(action_scale=args.action_scale, control_hz=args.hz, reward_weights=weights or None)
    model, data = env.model, env.data
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    print(f"observation {env.observation_space.shape[0]}, action {env.action_space.shape[0]} in [-1, 1] "
          f"x {args.action_scale} rad; {args.hz:.0f} decisions/s over {1 / model.opt.timestep:.0f} Hz physics "
          f"({int(round(1 / model.opt.timestep / args.hz))} physics steps per decision); episode {env.max_steps} decisions")
    print("reward = " + " + ".join(f"{w:+.2f} x {k}" for k, w in env.reward_weights.items()))

    # -- 2. the policy ------------------------------------------------------------------------
    walker = WalkingController(model, GaitParams(n_steps=12))
    policy = POLICIES[args.policy] or (lambda obs, t, env: walker_actions(env, walker, data.time))
    print(f"policy: {args.policy}" + (f"  (lean gains roll {LEAN_ROLL}, pitch {LEAN_PITCH})" if args.policy == "lean" else "")
          + (f"  push {args.push:g} N sideways at t = 1 s for 0.2 s" if args.push else ""))

    # -- 3. one episode: reset, then decide / step / score until done -------------------------------
    obs, _ = env.reset(seed=args.seed)
    ret, steps, done_why = 0.0, 0, None
    totals = {k: 0.0 for k in env.reward_weights}
    second, frames = {k: 0.0 for k in env.reward_weights}, []
    print(f"\n{'t (s)':>6} {'x (m)':>7} {'reward/s':>9}   " + "  ".join(f"{k:>9}" for k in env.reward_weights))
    while True:
        a = policy(obs, data.time, env)
        if args.push and 1.0 <= data.time < 1.2:
            data.xfrc_applied[torso, 1] = args.push           # world-frame force on the torso, +y
        else:
            data.xfrc_applied[torso, :] = 0.0
        obs, r, terminated, truncated, info = env.step(a)      # one decision = several physics steps
        ret += r; steps += 1
        for k in totals:
            totals[k] += env.reward_weights[k] * info.get(k, 0.0)
            second[k] += env.reward_weights[k] * info.get(k, 0.0)
        frames.append(data.qpos.copy())
        if steps % int(args.hz) == 0:
            print(f"{data.time:6.2f} {data.qpos[0]:+7.3f} {sum(second.values()):9.2f}   "
                  + "  ".join(f"{second[k]:9.2f}" for k in second))
            second = {k: 0.0 for k in second}
        if terminated or truncated:
            done_why = "TERMINATED (fell)" if terminated else "TRUNCATED (time ran out)"
            break

    # -- 4. the episode line, and the return split by term ---------------------------------------
    print(f"\nepisode: return {ret:8.2f}  {steps:3d} decisions  {data.time:.2f} s  travelled {data.qpos[0]:+.3f} m  {done_why}")
    print("return by term: " + ", ".join(f"{k} {v:+.1f}" for k, v in totals.items()))
    print("(standing still is worth about 776 over a full episode, walking at the target 1250; "
          "TERMINATED and TRUNCATED are different things -- the deck says why)")

    if args.no_viewer or soc4180.is_colab():
        return 0

    # -- 6. replay -------------------------------------------------------------------------------------
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    print(f"\nreplaying {len(frames)} decisions at {args.speed:g}x (close the window to stop)")
    with soc4180.launch_viewer(model, data, passive=True) as viewer:
        while viewer.is_running() and time.time() < deadline:
            for q in frames:
                if not viewer.is_running() or time.time() > deadline:
                    break
                data.qpos[:] = q; mujoco.mj_forward(model, data); viewer.sync()
                time.sleep(env.control_dt / args.speed)
            time.sleep(0.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
