"""Change the reward, train, and measure what it bought -- on the reward it was trained on and on the original.

    uv run weeks/09-reward/shape.py --variant unchanged             # needs torch + SB3: uv sync --extra rl
    uv run weeks/09-reward/shape.py --variant both
    uv run weeks/09-reward/shape.py --variant unchanged --extra potential
    uv run weeks/09-reward/shape.py --variant no_upright --extra velocity --no-viewer

A reward is a list of weighted terms. This script picks a VARIANT of the
course's own G1 reward (extra terms from the MuJoCo Playground reward, or a
term removed), optionally adds a shaping term through a wrapper (potential-
based, which provably cannot change what is optimal, or a plain velocity
bonus, which can), trains PPO for --steps, and then evaluates the result
twice: on the reward it was trained on, and on the ORIGINAL reward. The
second number is the honest one. Then it replays the trained policy.

    --variant unchanged|stand_still|air_time|both|no_alive|no_upright
    --extra none|potential|velocity      --gamma 0.99
    --steps 40000 (about two minutes on a laptop CPU)   --episodes 3   --seed 0
    --no-viewer   --speed 1

Six numbered steps; read them with the week 9 slides open.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mujoco
import numpy as np

import soc4180

VARIANTS = {
    "unchanged": {},
    "stand_still": {"stand_still": -1.0},
    "air_time": {"air_time": 2.0},
    "both": {"stand_still": -1.0, "air_time": 2.0},
    "no_alive": {"alive": 0.0},
    "no_upright": {"upright": 0.0},
}


def potential_term(gamma, x_before, x_after):
    """Potential-based shaping with Phi(s) = x: F = gamma * Phi(s') - Phi(s).

    Ng, Harada and Russell (1999): adding this to any reward leaves the set of
    optimal policies unchanged, because along any trajectory the terms
    telescope into a constant. It changes the *search*, never the *answer*.
    """
    return gamma * x_after - x_before


def velocity_term(gamma, x_before, x_after):
    """A plain bonus for moving forward: NOT potential-based, so it can change the optimum.

    Measured: with no upright term and 40k steps the policy still stands
    (529 on its own reward, 779 on the original) -- a moved optimum is not a
    found one.
    """
    return 5.0 * (x_after - x_before)


EXTRAS = {"none": None, "potential": potential_term, "velocity": velocity_term}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", choices=tuple(VARIANTS), default="unchanged")
    ap.add_argument("--extra", choices=tuple(EXTRAS), default="none")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(argv)

    try:
        import gymnasium as gym
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from soc4180.envs import DEFAULT_REWARD, G1WalkEnv
    except ImportError:
        print("This script needs torch and stable-baselines3: run `uv sync --extra rl`, then try again.")
        return 1
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

    # -- 1. the reward, written down ---------------------------------------------------------
    weights = dict(DEFAULT_REWARD); weights.update(VARIANTS[args.variant])
    extra = EXTRAS[args.extra]
    print(f"variant '{args.variant}': reward = " + " + ".join(f"{w:+.2f} x {k}" for k, w in weights.items() if w)
          + (f"  +  {args.extra} shaping" if extra else ""))

    # -- 2. the environment, wrapped with the extra term ---------------------------------------
    class Shaped(gym.Wrapper):
        """The variant's reward plus the shaping term, computed from the pelvis x before and after."""
        def step(self, action):
            x0 = float(self.unwrapped.data.qpos[0])
            obs, r, term, trunc, info = self.env.step(action)
            f = extra(args.gamma, x0, float(self.unwrapped.data.qpos[0])) if extra else 0.0
            info["extra"] = f
            return obs, r + f, term, trunc, info

    train_env = Shaped(G1WalkEnv(reward_weights=weights))
    base_env = G1WalkEnv()                                   # the ORIGINAL reward, for the honest score
    agent = PPO("MlpPolicy", train_env, verbose=0, seed=args.seed, device="cpu", gamma=args.gamma,
                policy_kwargs=dict(log_std_init=-2.0))       # week 8's fix: a quiet start

    # -- 3. train -------------------------------------------------------------------------------
    class Progress(BaseCallback):
        def __init__(self):
            super().__init__(); self.next = 5000
        def _on_step(self):
            if self.num_timesteps >= self.next:
                buf = self.model.ep_info_buffer
                if buf:
                    print(f"   {self.num_timesteps:7,} steps   last episodes: mean length {np.mean([e['l'] for e in buf]):5.0f}, "
                          f"mean return {np.mean([e['r'] for e in buf]):7.1f}", flush=True)
                self.next += 5000
            return True

    print(f"\ntraining for {args.steps:,} steps:")
    t0 = time.time()
    agent.learn(total_timesteps=args.steps, callback=Progress())
    print(f"   done in {time.time() - t0:.0f} s")

    # -- 4. evaluate: on the training reward, and on the original -----------------------------------
    def evaluate(env, record=False):
        rets, lens, dists, airs, frames = [], [], [], [], []
        for k in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + k)
            ret, n, air = 0.0, 0, 0.0
            while True:
                a, _ = agent.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(a)
                ret += r; n += 1; air += info["air_time"]
                if record and k == 0:
                    frames.append(env.unwrapped.data.qpos.copy())
                if term or trunc:
                    break
            rets.append(ret); lens.append(n); dists.append(float(env.unwrapped.data.qpos[0])); airs.append(air / n)
        return np.mean(rets), np.mean(lens), np.mean(dists), np.mean(airs), frames

    print(f"\nevaluation, {args.episodes} episodes, deterministic:")
    print(f"{'scored on':22} {'return':>8} {'steps':>6} {'dist m':>8} {'feet up':>8}")
    r1, l1, d1, a1, _ = evaluate(train_env)
    print(f"{'the training reward':22} {r1:8.1f} {l1:6.0f} {d1:+8.3f} {a1:8.2f}")
    r0, l0, d0, a0, frames = evaluate(base_env, record=True)
    print(f"{'the ORIGINAL reward':22} {r0:8.1f} {l0:6.0f} {d0:+8.3f} {a0:8.2f}")

    # -- 5. what it means --------------------------------------------------------------------------
    print("\nranking under the original reward: walking at the target ~1250, standing ~776, a fall ~50.\n"
          "A variant that still stands for 500 steps changed nothing; one that leaves the standing\n"
          "optimum shows up as fewer steps, a distance, or feet in the air. Shaping picks which optimum\n"
          "the search lands in; potential-based shaping cannot move the optimum, only the search.")

    if args.no_viewer or soc4180.is_colab():
        return 0

    # -- 6. replay the trained policy ---------------------------------------------------------------
    model, data = base_env.model, base_env.data
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    print(f"\nreplaying {len(frames)} decisions (close the window to stop)")
    with soc4180.launch_viewer(model, data, passive=True) as viewer:
        while viewer.is_running() and time.time() < deadline:
            for q in frames:
                if not viewer.is_running() or time.time() > deadline:
                    break
                data.qpos[:] = q; mujoco.mj_forward(model, data); viewer.sync()
                time.sleep(base_env.control_dt / args.speed)
            time.sleep(0.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
