"""Week 9 lab, on your laptop: change the reward, retrain, and watch what it buys.

    uv run weeks/09-reward/lab_reward.py      # torch + SB3: uv sync --extra rl

A window opens with the G1 in `G1WalkEnv`, holding the crouch under an
untrained policy. VARIANTS at the top of the file is a list of reward
weightings; pick one with a number key and press T to train it in the
background (40k steps, roughly two minutes on a laptop CPU, progress in the
terminal). When it finishes, the trained policy takes over the window.

    1 2 3 ...              select a reward variant (its trained policy if trained)
    T                      train the selected variant in the background
    R                      reset the episode        SPACE   pause
    ENTER                  every reward term for this step, including yours
    double-click a body, then ctrl-drag    push the robot
    Keys go to the MuJoCo window, not the terminal -- click the window first.
    If no key does anything in the window, press the same key in this terminal
    instead -- single keys, no enter. `q` stops it.

Drawn on the robot: a bar above the head for this step's reward (green while
the tracking term is being earned, orange when it is only staying alive) and
YELLOW feet when the env counts a foot as airborne -- the `air_time` term.
Every episode ends with a line: return, steps, distance, feet-up fraction.

`extra_reward` below is your own term, added on top of the variant's weights
through a wrapper. It is empty. Potential-based shaping, F = gamma * Phi(s') -
Phi(s), is the one kind that provably cannot change what is optimal; anything
else can, and the deck says why.

What to change, in order, and show the instructor:

1. Press 1 and watch the untrained robot. Press T. While it trains, say what
   return you expect from the deck's ranking (standing 776, walking 1250).
2. When training ends, watch the trained robot for a full episode. Read the
   episode line. Is it walking? Press ENTER: which term is it living on?
3. Press 4 (+ stand_still and + air_time) and T. Watch the trained result and
   read the distance sign. Then 2 and 3 (each term alone). Say which of the
   four leaves the standing optimum.
4. Write `extra_reward` as a potential-based term with Phi(s) = x position.
   Train variant 1 again. Compare the episode lines: what changed, and what
   did the theorem promise would not?
5. Now make `extra_reward` pay for torso x-velocity with no upright term (set
   the variant's upright to 0). Train. Watch the video before writing a word.
"""

from __future__ import annotations

import os
import threading
import time

import mujoco
import numpy as np

import soc4180

# name, then reward weights to override (see soc4180.envs.DEFAULT_REWARD). Keys 1-9.
VARIANTS = [
    ("ours, unchanged",                     {}),
    ("+ stand_still -1.0",                  {"stand_still": -1.0}),
    ("+ feet air_time +2.0",                {"air_time": 2.0}),
    ("+ both",                              {"stand_still": -1.0, "air_time": 2.0}),
    ("no alive bonus",                      {"alive": 0.0}),
    # add your own below
]
TRAIN_STEPS = 40_000
GAMMA = 0.99
SEED = 0


def extra_reward(env, info: dict, x_before: float, x_after: float) -> float:
    """Your own reward term for one step. Return 0.0 until written.

    `info` carries every built-in term for this step (tracking, upright,
    effort, smooth, alive, stand_still, air_time, forward_velocity).
    `x_before` and `x_after` are the pelvis x position before and after the
    step -- enough for a potential Phi(s) = x, i.e. return
    GAMMA * x_after - x_before.
    """
    return 0.0


# --- nothing below needs editing ------------------------------------------

def box(geom, pos, half, rgba):
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX,
                        np.asarray(half, dtype=float), np.asarray(pos, dtype=float),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=float))


def sphere(geom, pos, rgba, radius):
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0, 0], dtype=float),
                        np.asarray(pos, dtype=float),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=float))


def main() -> int:
    if soc4180.is_colab():
        print("This lab needs a desktop window; run it on your laptop.")
        return 1
    try:
        import gymnasium as gym
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from soc4180.envs import G1WalkEnv
    except ImportError:
        print("This lab needs torch and stable-baselines3: run `uv sync --extra rl`, then try again.")
        return 1
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

    class Shaped(gym.Wrapper):
        """The variant's reward plus the student's extra term."""
        def step(self, action):
            x0 = float(self.unwrapped.data.qpos[0])
            obs, r, term, trunc, info = self.env.step(action)
            extra = float(extra_reward(self.unwrapped, info, x0, float(self.unwrapped.data.qpos[0])))
            info["extra"] = extra
            return obs, r + extra, term, trunc, info

    def make_env(weights):
        return Shaped(G1WalkEnv(reward_weights=weights or None))

    env = make_env(VARIANTS[0][1])
    model, data = env.unwrapped.model, env.unwrapped.data
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_foot") for s in ("left", "right")]

    def fresh_agent(e):
        return PPO("MlpPolicy", e, verbose=0, seed=SEED, device="cpu", gamma=GAMMA,
                   policy_kwargs=dict(log_std_init=-2.0))        # week 8's fix

    agents = {}
    state = {"i": 0, "env": env, "agent": fresh_agent(env), "trained": False,
             "paused": False, "print": False, "reset": True, "training": None,
             "obs": None, "info": {}, "ret": 0.0, "steps": 0, "air": 0.0, "done_at": None}

    class Progress(BaseCallback):
        def __init__(self):
            super().__init__(); self.next = 5000
        def _on_step(self):
            if self.num_timesteps >= self.next:
                print(f"    training... {self.num_timesteps:,} / {TRAIN_STEPS:,}", flush=True)
                self.next += 5000
            return True

    def train(i):
        name, weights = VARIANTS[i]
        agent = fresh_agent(make_env(weights))
        t0 = time.time()
        print(f"\n  T: training [{name}] for {TRAIN_STEPS:,} steps in the background")
        agent.learn(total_timesteps=TRAIN_STEPS, callback=Progress())
        agents[i] = agent
        print(f"  T: done in {time.time() - t0:.0f} s -- trained policy now drives variant {i + 1}")
        if state["i"] == i:
            state["agent"], state["trained"], state["reset"] = agent, True, True
        state["training"] = None

    def select(i):
        state["i"] = i
        state["env"] = make_env(VARIANTS[i][1])
        state["agent"] = agents.get(i) or fresh_agent(state["env"])
        state["trained"] = i in agents
        state["reset"] = True

    def on_key(keycode):
        if 49 <= keycode <= 57 and keycode - 49 < len(VARIANTS):
            select(keycode - 49)
        elif keycode == 84:
            if state["training"] is None:
                th = threading.Thread(target=train, args=(state["i"],), daemon=True)
                state["training"] = th; th.start()
            else:
                print("  T: a training run is already going")
        elif keycode == 82:
            state["reset"] = True
        elif keycode == 32:
            state["paused"] = not state["paused"]
        elif keycode in (257, 335):
            state["print"] = True

    def reset():
        state["obs"], _ = state["env"].reset(seed=SEED)
        state["ret"], state["steps"], state["air"], state["done_at"], state["info"] = 0.0, 0, 0.0, None, {}
        print(f"\n[{VARIANTS[state['i']][0]}] {'TRAINED' if state['trained'] else 'untrained'}")

    def report():
        e = state["env"].unwrapped
        print(f"  step {state['steps']}  return so far {state['ret']:.1f}")
        for k, v in state["info"].items():
            w = e.reward_weights.get(k)
            print(f"    {k:18s} {v:+.4f}" + (f"   x weight {w:+.2f} = {w * v:+.4f}" if w is not None else ""))

    print("\n\n".join(__doc__.split("\n\n")[2:4]))
    soc4180.terminal_keys(on_key)
    print("  [terminal] keys dead in the window? press them here instead; 'q' stops.",
          flush=True)
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    # every variant's env shares this model; the viewer watches whichever data is live
    with soc4180.launch_viewer(model, data, passive=True, key_callback=on_key) as viewer:
        while viewer.is_running() and time.time() < deadline:
            wall = time.time()
            if state["reset"]:
                state["reset"] = False; reset()
            e = state["env"]
            d = e.unwrapped.data
            done = state["done_at"] is not None
            if done and time.time() - state["done_at"] > 1.5:
                reset(); done = False
            if not state["paused"] and not done:
                a, _ = state["agent"].predict(state["obs"], deterministic=True)
                obs, r, term, trunc, info = e.step(a)
                state["obs"], state["info"] = obs, info
                state["ret"] += r; state["steps"] += 1; state["air"] += info["air_time"]
                if term or trunc:
                    state["done_at"] = time.time()
                    print(f"  episode over: return {state['ret']:8.1f}  {state['steps']:3d} steps  "
                          f"travelled {d.qpos[0]:+.3f} m  feet up {state['air'] / state['steps']:.2f}  "
                          f"{'TERMINATED (fell)' if term else 'TRUNCATED'}")
            # mirror the live env's state into the viewer's data (they differ after a variant switch)
            if d is not data:
                data.qpos[:] = d.qpos; data.qvel[:] = d.qvel; mujoco.mj_forward(model, data)
            if state["print"]:
                state["print"] = False; report()
            with viewer.lock():
                scn = viewer.user_scn; n = 0
                info = state["info"]
                reward = sum(e.unwrapped.reward_weights[k] * info.get(k, 0.0) for k in e.unwrapped.reward_weights)
                h = max(min(reward + info.get("extra", 0.0), 2.5), 0.0) * 0.1
                col = (0.2, 0.8, 0.2, 0.8) if info.get("tracking", 0.0) > 0.5 else (0.95, 0.6, 0.1, 0.8)
                box(scn.geoms[n], [data.qpos[0], data.qpos[1], 1.05 + h / 2], [0.03, 0.03, h / 2 + 1e-3], col); n += 1
                for sid in feet:
                    if data.site_xpos[sid][2] > 0.033 + 0.02:
                        sphere(scn.geoms[n], data.site_xpos[sid], (1, 0.9, 0.1, 0.8), 0.04); n += 1
                scn.ngeom = n
            viewer.sync()
            lag = e.unwrapped.control_dt - (time.time() - wall)
            if lag > 0:
                time.sleep(lag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
