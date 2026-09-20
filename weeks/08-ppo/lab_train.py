"""Week 8 lab, on your laptop: train PPO on the G1, then watch what it learned.

    uv run weeks/08-ppo/lab_train.py        # needs torch + SB3: uv sync --extra rl

A window opens with the G1 in `G1WalkEnv`, driven by an UNTRAINED policy from
the first entry of CONFIGS. Press T and PPO trains that configuration in the
background (about a minute for 30k steps on a laptop CPU, progress in the
terminal) while the window keeps playing; when it finishes, the trained policy
takes over. Then the lesson of the week is one key:

    D                      deterministic (the mean action) <-> stochastic (sampled)
    T                      train the selected config in the background
    1 2 3 ...              select a config (its trained policy if you trained it,
                           else a fresh untrained one)
    R                      reset the episode      SPACE   pause
    ENTER                  return so far, steps, the action std
    double-click a body, then ctrl-drag    push the robot
    Keys go to the MuJoCo window, not the terminal -- click the window first.
    If no key does anything in the window, press the same key in this terminal
    instead -- single keys, no enter. `q` stops it.

Every episode prints return, steps, and TERMINATED (fell) or TRUNCATED. The
number to watch is *steps survived* -- deterministic against stochastic.

Pressing D does nothing until you write `choose_action` below: sampling is
the exercise. PPO learns from the sampled actions and is judged on the mean
ones, and those are two different robots.

What to change, in order, and show the instructor:

1. Press ENTER: the untrained policy's action std is 1.0 on a [-1, 1] range.
   Watch the deterministic robot stand for 500 steps.
2. Write `choose_action`: sample from N(mean, std) and clip to [-1, 1]. Press D.
   Count how many steps the stochastic robot survives. This is the data PPO
   will learn from.
3. Press T. Wait for training. Press D a few times: has the trained policy
   learned anything the untrained one could not do? What did it learn *from*?
4. Press 2 (log_std_init = -2.0, std 0.135), then D. Compare the stochastic
   survival with config 1. Press T, wait, and compare the trained policies.
5. Add a config with log_std_init = -4.0. Predict before training: will less
   noise keep helping? Train it and report the return. (It will not walk.
   Say why not -- the deck has the number.)
"""

from __future__ import annotations

import os
import threading
import time

import mujoco
import numpy as np

import soc4180

# name, then PPO settings. Keys 1-9 select. Add your own.
CONFIGS = [
    ("SB3 defaults: log_std_init = 0 (std 1.0)",   dict(log_std_init=0.0, total_timesteps=30_000)),
    ("quiet start: log_std_init = -2 (std 0.135)", dict(log_std_init=-2.0, total_timesteps=30_000)),
    ("longer: -2, 100k steps",                     dict(log_std_init=-2.0, total_timesteps=100_000)),
    # add your own below
]
SEED = 0


def choose_action(mean: np.ndarray, std: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    """A stochastic action: sample around the policy's mean. Return None until written.

    PPO's Gaussian policy has a mean (what `predict(deterministic=True)` gives
    you) and a per-joint standard deviation. Draw one sample,
    `mean + std * rng.standard_normal(mean.shape)`, and clip it to [-1, 1], the
    action space. That sample is what the environment sees during training.
    """
    return None


# --- nothing below needs editing ------------------------------------------

def box(geom, pos, half, rgba):
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX,
                        np.asarray(half, dtype=float), np.asarray(pos, dtype=float),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=float))


def main() -> int:
    if soc4180.is_colab():
        print("This lab needs a desktop window; run it on your laptop.")
        return 1
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from soc4180.envs import G1WalkEnv
    except ImportError:
        print("This lab needs torch and stable-baselines3: run `uv sync --extra rl`, then try again.")
        return 1
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

    env = G1WalkEnv()                       # the one the viewer watches
    model, data = env.model, env.data
    rng = np.random.default_rng(SEED)

    def fresh_agent(cfg):
        return PPO("MlpPolicy", G1WalkEnv(), verbose=0, seed=SEED, device="cpu",
                   policy_kwargs=dict(log_std_init=cfg["log_std_init"]))

    agents = {}                             # config index -> trained PPO
    state = {"i": 0, "agent": fresh_agent(CONFIGS[0][1]), "trained": False,
             "det": True, "paused": False, "print": False, "reset": True,
             "training": None, "obs": None, "ret": 0.0, "steps": 0, "done_at": None}

    class Progress(BaseCallback):
        def __init__(self, total):
            super().__init__(); self.total = total; self.next = 5000
        def _on_step(self):
            if self.num_timesteps >= self.next:
                print(f"    training... {self.num_timesteps:,} / {self.total:,} steps", flush=True)
                self.next += 5000
            return True

    def train(i):
        name, cfg = CONFIGS[i]
        agent = fresh_agent(cfg)
        t0 = time.time()
        print(f"\n  T: training [{name}] for {cfg['total_timesteps']:,} steps in the background")
        agent.learn(total_timesteps=cfg["total_timesteps"], callback=Progress(cfg["total_timesteps"]))
        agents[i] = agent
        print(f"  T: done in {time.time() - t0:.0f} s -- the trained policy is now driving config {i + 1}")
        if state["i"] == i:
            state["agent"], state["trained"], state["reset"] = agent, True, True
        state["training"] = None

    def select(i):
        state["i"] = i
        state["agent"] = agents.get(i) or fresh_agent(CONFIGS[i][1])
        state["trained"] = i in agents
        state["reset"] = True

    def on_key(keycode):
        if 49 <= keycode <= 57 and keycode - 49 < len(CONFIGS):
            select(keycode - 49)
        elif keycode == 84:                                  # 'T'
            if state["training"] is None:
                th = threading.Thread(target=train, args=(state["i"],), daemon=True)
                state["training"] = th; th.start()
            else:
                print("  T: a training run is already going")
        elif keycode == 68:                                  # 'D'
            state["det"] = not state["det"]
            print(f"  D: {'deterministic (mean action)' if state['det'] else 'stochastic (sampled action)'}")
        elif keycode == 82:
            state["reset"] = True
        elif keycode == 32:
            state["paused"] = not state["paused"]
        elif keycode in (257, 335):
            state["print"] = True

    def reset():
        state["obs"], _ = env.reset(seed=SEED)
        state["ret"], state["steps"], state["done_at"] = 0.0, 0, None
        name = CONFIGS[state["i"]][0]
        print(f"\n[{name}] {'TRAINED' if state['trained'] else 'untrained'}, "
              f"{'deterministic' if state['det'] else 'stochastic'}")

    def act(obs):
        agent = state["agent"]
        mean, _ = agent.predict(obs, deterministic=True)
        if state["det"]:
            return mean
        std = np.exp(agent.policy.log_std.detach().cpu().numpy())
        a = choose_action(np.asarray(mean, np.float32), std.astype(np.float32), rng)
        if a is None:
            return mean
        return np.clip(np.asarray(a, np.float32), -1.0, 1.0)

    def report():
        std = np.exp(state["agent"].policy.log_std.detach().cpu().numpy())
        print(f"  step {state['steps']}  return so far {state['ret']:.1f}  action std {std.mean():.3f}"
              f"  (choose_action {'written' if choose_action(np.zeros(12), np.ones(12), rng) is not None else 'NOT written: D does nothing'})")

    print("\n\n".join(__doc__.split("\n\n")[2:4]))
    soc4180.terminal_keys(on_key)
    print("  [terminal] keys dead in the window? press them here instead; 'q' stops.",
          flush=True)
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    with soc4180.launch_viewer(model, data, passive=True, key_callback=on_key) as viewer:
        while viewer.is_running() and time.time() < deadline:
            wall = time.time()
            if state["reset"]:
                state["reset"] = False; reset()
            done = state["done_at"] is not None
            if done and time.time() - state["done_at"] > 1.5:
                reset(); done = False
            if not state["paused"] and not done:
                obs, r, term, trunc, info = env.step(act(state["obs"]))
                state["obs"] = obs; state["ret"] += r; state["steps"] += 1
                if term or trunc:
                    state["done_at"] = time.time()
                    print(f"  episode over: return {state['ret']:8.1f}  {state['steps']:3d} steps  "
                          f"{'TERMINATED (fell)' if term else 'TRUNCATED'}  "
                          f"[{'det' if state['det'] else 'stoch'}, {'trained' if state['trained'] else 'untrained'}]")
            if state["print"]:
                state["print"] = False; report()
            with viewer.lock():
                scn = viewer.user_scn
                h = min(state["steps"], 500) / 500 * 0.3
                col = (0.2, 0.8, 0.2, 0.8) if state["det"] else (0.95, 0.5, 0.1, 0.8)
                box(scn.geoms[0], [data.qpos[0], data.qpos[1], 1.05 + h / 2], [0.03, 0.03, h / 2 + 1e-3], col)
                scn.ngeom = 1
            viewer.sync()
            lag = env.control_dt - (time.time() - wall)
            if lag > 0:
                time.sleep(lag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
