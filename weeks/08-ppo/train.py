"""Train PPO on the G1 for a few minutes, and measure the two robots it makes.

    uv run weeks/08-ppo/train.py                              # needs torch + SB3: uv sync --extra rl
    uv run weeks/08-ppo/train.py --log-std -2
    uv run weeks/08-ppo/train.py --log-std -2 --steps 100000
    uv run weeks/08-ppo/train.py --steps 0 --no-viewer        # no training: just the untrained pair

PPO's policy is a Gaussian: a network gives the MEAN action, and a learned
per-joint standard deviation says how far to sample around it. Training
collects experience with the SAMPLED action; the robot you show off uses the
MEAN. Those are two different robots, and this script measures both, before
and after training: episodes survived, return, distance. Then it replays the
trained deterministic robot in the simulator.

    --log-std 0      initial log standard deviation (0 -> std 1.0 on a [-1, 1] range; -2 -> 0.135)
    --steps 30000    training timesteps (about a minute on a laptop CPU)
    --episodes 3     evaluation episodes per case      --seed 0
    --no-viewer      --speed 1

Six numbered steps; read them with the week 8 slides open.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mujoco
import numpy as np

import soc4180


def sample_action(mean, std, rng):
    """What the environment sees during training: mean + std * N(0, 1), clipped to the action space."""
    return np.clip(mean + std * rng.standard_normal(mean.shape), -1.0, 1.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log-std", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(argv)

    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from soc4180.envs import G1WalkEnv
    except ImportError:
        print("This script needs torch and stable-baselines3: run `uv sync --extra rl`, then try again.")
        return 1
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

    # -- 1. the environment and the agent ------------------------------------------------
    # PPO("MlpPolicy") builds two small networks on the 42-number observation:
    # the policy (mean action, 12 numbers, plus a log_std vector) and the value
    # function. seed fixes the initial weights and the sampling.
    env = G1WalkEnv()
    model, data = env.model, env.data
    agent = PPO("MlpPolicy", env, verbose=0, seed=args.seed, device="cpu",
                policy_kwargs=dict(log_std_init=args.log_std))
    n_params = sum(p.numel() for p in agent.policy.parameters())
    std0 = float(np.exp(agent.policy.log_std.detach().cpu().numpy()).mean())
    print(f"G1WalkEnv: obs {env.observation_space.shape[0]}, action {env.action_space.shape[0]} in [-1, 1], "
          f"episode {env.max_steps} decisions at {1 / env.control_dt:.0f} Hz")
    print(f"PPO policy + value networks: {n_params:,} parameters; initial action std {std0:.3f} "
          f"(log_std_init {args.log_std})")

    # -- 2. evaluating a policy, both ways -----------------------------------------------------
    def evaluate(agent, deterministic, episodes, record=False):
        """Run episodes; return mean (return, steps, distance) and, if asked, the poses of the first."""
        rng = np.random.default_rng(args.seed)
        std = np.exp(agent.policy.log_std.detach().cpu().numpy())
        rets, lens, dists, frames = [], [], [], []
        for k in range(episodes):
            obs, _ = env.reset(seed=args.seed + k)
            ret, n = 0.0, 0
            while True:
                mean, _ = agent.predict(obs, deterministic=True)       # the network's mean action
                a = mean if deterministic else sample_action(mean, std, rng)
                obs, r, term, trunc, _ = env.step(a)
                ret += r; n += 1
                if record and k == 0:
                    frames.append(data.qpos.copy())
                if term or trunc:
                    break
            rets.append(ret); lens.append(n); dists.append(float(data.qpos[0]))
        return np.mean(rets), np.mean(lens), np.mean(dists), frames

    rows = []
    def measure(label):
        for det in (True, False):
            ret, steps, dist, _ = evaluate(agent, det, args.episodes)
            rows.append((label, "deterministic (mean)" if det else "stochastic (sampled)", ret, steps, dist))
            print(f"   {label:9s} {'deterministic' if det else 'stochastic':13s}  return {ret:8.1f}  "
                  f"survived {steps:5.0f} of {env.max_steps}  travelled {dist:+.3f} m")

    print(f"\nbefore training ({args.episodes} episodes each):")
    measure("untrained")

    # -- 3. training: collect sampled experience, update, repeat ----------------------------------
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

    if args.steps > 0:
        print(f"\ntraining for {args.steps:,} steps (PPO collects 2048 sampled steps, updates, repeats):")
        t0 = time.time()
        agent.learn(total_timesteps=args.steps, callback=Progress())
        print(f"   done in {time.time() - t0:.0f} s; action std now {float(np.exp(agent.policy.log_std.detach().cpu().numpy()).mean()):.3f}")

        # -- 4. the same two measurements, after ------------------------------------------------------
        print(f"\nafter training:")
        measure("trained")

    # -- 5. what it means ------------------------------------------------------------------------------
    print("\nthe untrained deterministic robot holds the crouch (worth ~774 over a full episode);\n"
          "the stochastic one is what PPO learns FROM -- the fraction of its experience that is a fall is\n"
          "the fraction of training spent learning about falling. Walking at the target would be ~1250.")

    if args.no_viewer or soc4180.is_colab():
        return 0

    # -- 6. replay the trained deterministic robot --------------------------------------------------------
    _, _, _, frames = evaluate(agent, True, 1, record=True)
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    print(f"\nreplaying {len(frames)} decisions of the {'trained' if args.steps else 'untrained'} deterministic policy")
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
