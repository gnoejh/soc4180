"""Week 7 lab, on your laptop: live inside the environment you specified.

    uv run weeks/07-mdp/lab_env.py

A window opens with the G1 in `G1WalkEnv`, stepping at the policy rate (50 Hz
by default: one decision, ten physics steps) in real time. A policy is any
function `obs -> action`; the list POLICIES holds four, and the last one is
yours.

    1 2 3 4                pick a policy and reset      R   reset
    SPACE                  pause / resume
    ENTER                  print the observation, the action, and every reward
                           term for the current step
    double-click a body, then ctrl-drag    push the robot
    Keys go to the MuJoCo window, not the terminal -- click the window first.
    If no key does anything in the window, press the same key in this terminal
    instead -- single keys, no enter. `q` stops it.

Drawn on the robot:

    white arrow     obs[0:3], gravity in the body frame -- what the policy
                    actually knows about which way is up
    bar above head  this step's reward: green when the tracking term is being
                    earned, orange when the robot is just staying alive
    yellow spheres  a foot the env counts as airborne (the air_time term)

Every episode ends with a line in the terminal: return, steps, and whether it
TERMINATED (fell) or was TRUNCATED (ran out of time). Those are different
things, and the deck says why.

The observation, in order (42 numbers):
    obs[0:3]    gravity in the body frame        obs[3:6]    gyroscope
    obs[6:18]   12 leg angles minus the nominal crouch, left leg then right,
                each [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]
    obs[18:30]  the same 12 joint velocities     obs[30:42]  your previous action
The action is 12 residuals in [-1, 1] in that same joint order, scaled by
ACTION_SCALE radians.

Everything here is complete and explained: `my_policy` in section 1 is the
first closed loop of the course (the ankle strategy), sections 2-3 are the
drawing and the loop, with the environment's calls named. `env_run.py` in
this folder runs one episode of any policy headless with the reward split by
term, can push the robot with a measured force, and then replays.

Experiments, in order, and what to show the instructor (numbers measured
with env_run.py):

1. Press 1, 2, 3 and read the three episode lines: hold 774 over 500
   decisions; random 32 and a fall after 38; the week 4 walker -- which walked
   a metre three weeks ago -- 357 and a fall at 4.3 s. Say why. Then set
   ACTION_SCALE = 1.0 and CONTROL_HZ = 100 at the top and press 3 again:
   1805, a metre walked. Say what the two numbers changed.
2. Change `my_policy` to a constant: bend both knees a little more (a[3] =
   a[9] = 0.3). Press 4. Read the reward terms on ENTER: which went up, which
   went down?
3. Make `my_policy` periodic in `t` on hip pitch, opposite signs on the two
   legs (a[0], a[6] = +-0.5 sin 4t). Watch it terminate at 1.5 s. Read the
   episode line and the last ENTER.
4. Restore `my_policy`: the ankles driven against obs[0:3]. Push the robot
   with ctrl-drag under policy 1 and under policy 4. Measured with
   `env_run.py --push`: hold survives 65 N for 0.2 s and falls at 70; the
   ankle strategy survives 75 and falls at 80; hip roll instead of ankle roll,
   either sign, changes nothing. What did policy 4 know that policy 1 did
   not, and why can no stance survive 80 N (what would)?
5. Change REWARD_WEIGHTS (try alive = 0, or tracking = 0) and re-run 1 and 3.
   Nothing about the *behaviour* changes -- only the numbers. Say why, in one
   sentence, and what that means for next week.
"""

from __future__ import annotations

import os
import time

import mujoco
import numpy as np

import soc4180

ACTION_SCALE = 0.3      # radians per unit of action
CONTROL_HZ = 50.0       # decisions per second
REWARD_WEIGHTS = {}     # e.g. {"alive": 0.0, "tracking": 1.5}; see soc4180.envs.DEFAULT_REWARD


# --- 1. the policy ----------------------------------------------------------------

ANKLE_GAIN = 2.0        # residual per unit of body-frame gravity component


def my_policy(obs: np.ndarray, t: float) -> np.ndarray:
    """A policy: 42 observations (and the time) in, 12 residual actions out.

    As shipped, the first closed loop of the course -- the ankle strategy.
    obs[0:3] is gravity in the torso frame, (0, 0, -1) when upright; obs[1]
    grows with a sideways tilt and obs[0] with a forward one. Both ankles
    are turned against them, so the feet push the body back. Measured with
    env_run.py --push: this survives a 75 N shove (0.2 s) where holding the
    crouch falls at 70 N. Doing the same with hip roll (a[1], a[7]) changes
    nothing, and ankle pitch with the sign flipped falls with no push at all.

    Other starting points, for the experiments:
        a[3] = a[9] = 0.3                                          # both knees, a little more
        a[0], a[6] = 0.5 * np.sin(4 * t), -0.5 * np.sin(4 * t)     # step in place
    """
    a = np.zeros(12, dtype=np.float32)
    a[5] = a[11] = ANKLE_GAIN * obs[1]        # ankle roll, both legs, against the sideways tilt
    a[4] = a[10] = ANKLE_GAIN * obs[0]        # ankle pitch, both legs, against the forward tilt
    return a


# --- 2. drawing --------------------------------------------------------------------

def sphere(geom, pos, rgba, radius):
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0, 0], dtype=float),
                        np.asarray(pos, dtype=float),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=float))


def box(geom, pos, half, rgba):
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX,
                        np.asarray(half, dtype=float), np.asarray(pos, dtype=float),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=float))


def arrow(geom, origin, direction, rgba, length=0.35, radius=0.006):
    quat = np.zeros(4); mat = np.zeros(9)
    mujoco.mju_quatZ2Vec(quat, np.asarray(direction, float))
    mujoco.mju_quat2Mat(mat, quat)
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW,
                        np.array([radius, radius, length], dtype=float),
                        np.asarray(origin, dtype=float), mat, np.asarray(rgba, dtype=float))


# --- 3. the loop ---------------------------------------------------------------------

def main() -> int:
    if soc4180.is_colab():
        print("This lab needs a desktop window; run it on your laptop.")
        return 1
    try:
        from soc4180.envs import G1WalkEnv, walker_actions
        from soc4180.walking import GaitParams, WalkingController
    except ImportError:
        print("This lab needs gymnasium: run `uv sync --extra rl` once, then try again.")
        return 1

    # G1WalkEnv is the MDP written down: observation, action, reward terms and
    # weights, termination (fell) and truncation (time). One env.step is one
    # decision = several physics steps (500 / CONTROL_HZ).
    env = G1WalkEnv(action_scale=ACTION_SCALE, control_hz=CONTROL_HZ,
                    reward_weights=REWARD_WEIGHTS or None)
    model, data = env.model, env.data
    walker = WalkingController(model, GaitParams(n_steps=12))
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu_in_torso")
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_foot") for s in ("left", "right")]

    def hold(obs, t):
        return np.zeros(12, dtype=np.float32)

    def random_policy(obs, t):
        return env.action_space.sample()

    def week4_walker(obs, t):
        return walker_actions(env, walker, data.time)

    def mine(obs, t):
        return np.asarray(my_policy(obs, t), np.float32)

    POLICIES = [("hold the crouch (zero action)", hold),
                ("uniform random", random_policy),
                ("the week 4 walker, as actions", week4_walker),
                ("mine: my_policy", mine)]

    state = {"i": 0, "paused": False, "print": False, "reset": True,
             "obs": None, "ret": 0.0, "steps": 0, "done_at": None, "info": {}, "action": None}
    print(f"observation {env.observation_space.shape[0]}, action {env.action_space.shape[0]}, "
          f"{CONTROL_HZ:.0f} Hz over {1 / model.opt.timestep:.0f} Hz physics, "
          f"episode {env.max_steps} decisions, action scale {ACTION_SCALE} rad")

    def reset():
        state["obs"], _ = env.reset(seed=0)
        walker.__init__(model, GaitParams(n_steps=12))
        state["ret"], state["steps"], state["done_at"], state["info"] = 0.0, 0, None, {}
        print(f"\n[{POLICIES[state['i']][0]}]")

    def on_key(keycode):
        if 49 <= keycode <= 52:
            state["i"] = keycode - 49; state["reset"] = True
        elif keycode == 82:
            state["reset"] = True
        elif keycode == 32:
            state["paused"] = not state["paused"]
        elif keycode in (257, 335):
            state["print"] = True

    def report():
        obs, info, a = state["obs"], state["info"], state["action"]
        print(f"  step {state['steps']}  return so far {state['ret']:.2f}")
        print(f"  obs gravity {np.round(obs[:3], 3)}  gyro {np.round(obs[3:6], 3)}")
        print(f"  obs leg angles - nominal {np.round(obs[6:18], 2)}")
        if a is not None:
            print(f"  action {np.round(a, 2)}")
        for k, v in info.items():
            print(f"    {k:18s} {v:+.4f}")

    def draw(viewer):
        with viewer.lock():
            scn = viewer.user_scn; n = 0
            g = state["obs"][:3]
            R = data.site_xmat[torso].reshape(3, 3)
            arrow(scn.geoms[n], data.site_xpos[torso], R @ g, (1, 1, 1, 0.9)); n += 1
            r = state["info"].get("tracking", 0.0)
            reward = sum(env.reward_weights[k] * state["info"].get(k, 0.0) for k in env.reward_weights)
            h = max(min(reward, 2.5), 0.0) * 0.1
            col = (0.2, 0.8, 0.2, 0.8) if r > 0.5 else (0.95, 0.6, 0.1, 0.8)
            box(scn.geoms[n], [data.qpos[0], data.qpos[1], 1.05 + h / 2], [0.03, 0.03, h / 2 + 1e-3], col); n += 1
            for sid in feet:
                if data.site_xpos[sid][2] > 0.033 + 0.02:
                    sphere(scn.geoms[n], data.site_xpos[sid], (1, 0.9, 0.1, 0.8), 0.04); n += 1
            scn.ngeom = n

    print("\n\n".join(__doc__.split("\n\n")[2:4]))
    soc4180.terminal_keys(on_key)
    print("  [terminal] keys dead in the window? press them here instead; 'q' stops.",
          flush=True)
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    reset(); state["reset"] = False
    with soc4180.launch_viewer(model, data, passive=True, key_callback=on_key) as viewer:
        while viewer.is_running() and time.time() < deadline:
            wall = time.time()
            if state["reset"]:
                state["reset"] = False; reset()
            done = state["done_at"] is not None
            if done and time.time() - state["done_at"] > 1.5:
                reset(); done = False
            if not state["paused"] and not done:
                policy = POLICIES[state["i"]][1]
                a = policy(state["obs"], data.time)
                state["action"] = a
                obs, r, term, trunc, info = env.step(a)      # decide -> act -> observe, score
                state["obs"], state["info"] = obs, info
                state["ret"] += r; state["steps"] += 1
                if term or trunc:
                    state["done_at"] = time.time()
                    why = "TERMINATED (fell)" if term else "TRUNCATED (time ran out)"
                    print(f"  episode over: return {state['ret']:8.2f}  {state['steps']:3d} steps  "
                          f"travelled {data.qpos[0]:+.3f} m  {why}")
            if state["print"]:
                state["print"] = False; report()
            draw(viewer)
            viewer.sync()
            lag = env.control_dt - (time.time() - wall)
            if lag > 0:
                time.sleep(lag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
