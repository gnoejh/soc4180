"""Train a push-recovery policy with PPO, or load one, and measure it across worlds it did and did not train in -- printed, then replayed.

    uv run weeks/12-robustness/robust.py                                  # load the shipped checkpoints, evaluate
    uv run weeks/12-robustness/robust.py --policy random                  # the domain-randomised one
    uv run weeks/12-robustness/robust.py --train 300000 --world nominal   # train your own (minutes)
    uv run weeks/12-robustness/robust.py --train 3000000 --world random --save push_random.zip
    uv run weeks/12-robustness/robust.py --episodes 5 --no-viewer

G1PushEnv: the robot stands in the crouch and, once per 5-second episode, is
shoved on the torso -- a random size up to --push-max newtons, a random
direction, for 0.2 s. PPO learns 12 leg residuals at 50 Hz. The *world* is
friction, torso mass and servo stiffness; `nominal` fixes them, `random`
draws each from a range every episode (domain randomisation).

The evaluation is the point: every policy is scored in a grid of worlds --
nominal, ice, a heavy torso, weak servos, stiff servos -- at four push sizes,
as episodes survived out of --episodes. Holding the crouch (zero action) is
the baseline row.

    --policy nominal|random|hold|none            which shipped checkpoint to evaluate (checkpoints/ beside this file)
    --train 0       PPO timesteps; 0 evaluates a checkpoint instead of training
    --world nominal|random                       what the TRAINING episodes draw from
    --privileged    train a teacher that also observes velocity, height, the push and the world (week 13)
    --envs 16       parallel training environments (SubprocVecEnv; capped by your cores)
    --save FILE     where to write the trained policy (default: checkpoints/push_<world>.zip)
    --push-max 200  --episodes 10  --seed 0  --no-viewer  --speed 1
    --any-direction evaluate with pushes from every side, not just from the right

Six numbered steps; read them with the week 12 slides open.
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
CHECKPOINTS = HERE / "checkpoints"

# The test worlds: name -> the three world parameters (friction scale, added kg, stiffness scale).
WORLDS = {
    "nominal": dict(friction=1.0, mass=0.0, kp=1.0),
    "ice (friction 0.2)": dict(friction=0.2, mass=0.0, kp=1.0),
    "heavy (+15 kg torso)": dict(friction=1.0, mass=15.0, kp=1.0),
    "light (-5 kg torso)": dict(friction=1.0, mass=-5.0, kp=1.0),
    "weak servos (kp x0.6)": dict(friction=1.0, mass=0.0, kp=0.6),
    "stiff servos (kp x1.5)": dict(friction=1.0, mass=0.0, kp=1.5),
}
PUSHES = (0, 60, 80, 100, 120)


def survived(policy, env, world, push, episodes, seed=0, record=False, sideways=True):
    """Episodes survived out of `episodes`: each one a shove of exactly `push` N at a random moment.

    Sideways (+y, the robot's left) by default, as in weeks 7 and 11: a push
    from the front or back is a different, harder problem (see --any-direction).
    The episodes differ in the moment of the push and the small random crouch
    they start from.
    """
    env.world = {k: (v, v) for k, v in world.items()}
    ok, frames = 0, []
    for k in range(episodes):
        env.push_max = max(push, 1.0)
        obs, _ = env.reset(seed=seed + k)
        if sideways:
            env.push_force = np.array([0.0, float(push)])
        else:
            env.push_force *= push / max(np.linalg.norm(env.push_force), 1e-9)   # exactly `push` newtons
        while True:
            a = policy(obs)
            obs, _, term, trunc, _ = env.step(a)
            if record and k == 0:
                frames.append(env.data.qpos.copy())
            if term or trunc:
                break
        ok += int(not term)
    return ok, frames


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--policy", choices=("nominal", "random", "hold", "none"), default="nominal")
    ap.add_argument("--train", type=int, default=0)
    ap.add_argument("--world", choices=("nominal", "random"), default="nominal")
    ap.add_argument("--privileged", action="store_true")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--save", default=None)
    ap.add_argument("--push-max", type=float, default=200.0)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--any-direction", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(argv)

    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
        from soc4180.envs import G1PushEnv, NOMINAL_WORLD, RANDOM_WORLD
    except ImportError:
        print("This script needs torch and stable-baselines3: run `uv sync --extra rl`, then try again.")
        return 1
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) // 2)))

    # -- 1. the environment -------------------------------------------------------------------
    world = RANDOM_WORLD if args.world == "random" else NOMINAL_WORLD
    privileged = args.privileged
    env = G1PushEnv(push_max=args.push_max, world=world, privileged=privileged)
    print(f"G1PushEnv: obs {env.observation_space.shape[0]}{' (privileged)' if privileged else ''}, "
          f"action {env.action_space.shape[0]} x {env.action_scale} rad, {1 / env.control_dt:.0f} Hz, "
          f"{env.max_steps} decisions per episode; pushes up to {args.push_max:g} N for {env.push_seconds} s")
    print("training world: " + ", ".join(f"{k} {lo:g}..{hi:g}" for k, (lo, hi) in world.items()))

    # -- 2. a policy: trained here, or loaded ----------------------------------------------------------
    agent = None
    if args.train > 0:
        def make(i):
            def f():
                e = G1PushEnv(push_max=args.push_max, world=world, privileged=args.privileged)
                e.reset(seed=1000 * args.seed + i)
                return e
            return f
        venv = VecMonitor(SubprocVecEnv([make(i) for i in range(args.envs)]))
        # log_std_init=-2: week 8's lesson -- with std 1.0 the sampled robot falls
        # before the push ever arrives, and PPO learns only about falling.
        agent = PPO("MlpPolicy", venv, n_steps=256, batch_size=1024, n_epochs=5, learning_rate=3e-4,
                    gamma=0.99, seed=args.seed, device="cpu", verbose=0,
                    policy_kwargs=dict(log_std_init=-2.0, net_arch=[128, 128]))

        class Progress(BaseCallback):
            def __init__(self):
                super().__init__(); self.next, self.t0, self.rows = 100_000, time.time(), []

            def _on_step(self):
                if self.num_timesteps >= self.next:
                    buf = self.model.ep_info_buffer
                    if buf:
                        self.rows.append((self.num_timesteps, round(time.time() - self.t0),
                                          round(np.mean([e['l'] for e in buf])), round(np.mean([e['r'] for e in buf]), 1)))
                        print(f"   {self.num_timesteps:9,} steps {time.time() - self.t0:6.0f} s   episode length "
                              f"{np.mean([e['l'] for e in buf]):5.0f} of {env.max_steps}, return "
                              f"{np.mean([e['r'] for e in buf]):6.1f}", flush=True)
                    self.next += 100_000
                return True

        print(f"\n-- training PPO for {args.train:,} steps on {args.envs} processes")
        t0 = time.time()
        progress = Progress()
        agent.learn(total_timesteps=args.train, callback=progress)
        print(f"   {time.time() - t0:.0f} s")
        venv.close()
        out = Path(args.save) if args.save else CHECKPOINTS / f"push_{'teacher' if args.privileged else args.world}.zip"
        out.parent.mkdir(parents=True, exist_ok=True)
        agent.save(out)
        curve = out.with_suffix(".csv")                   # the learning curve, for the slides
        lines = ["steps,seconds,episode_length,episode_return"] + [",".join(map(str, r)) for r in progress.rows]
        curve.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"   saved {out} ({out.stat().st_size / 1024:.0f} kB) and {curve.name}")
    elif args.policy not in ("hold", "none"):
        path = CHECKPOINTS / f"push_{args.policy}.zip"
        if not path.exists():
            print(f"no checkpoint at {path}; train one with --train, or pass --policy hold")
            return 1
        agent = PPO.load(path, device="cpu")
        print(f"\nloaded {path.name}")

    # -- 3. the two policies being compared -----------------------------------------------------------------
    def hold(obs):
        return np.zeros(12, np.float32)

    def learned(obs):
        return agent.predict(obs, deterministic=True)[0]

    label = "teacher" if privileged else (args.policy if not args.train else args.world)
    rows = [("hold the crouch", hold)] + ([("PPO " + label, learned)]
                                          if agent is not None else [])

    # -- 4. the grid: worlds x push sizes ---------------------------------------------------------------------
    print(f"\nepisodes survived out of {args.episodes}, a shove of exactly N newtons for 0.2 s "
          f"{'in a random direction' if args.any_direction else 'from the right'}:")
    print(f"{'world':24s} {'policy':18s} " + " ".join(f"{p:>5} N" for p in PUSHES))
    t0 = time.time()
    for wname, w in WORLDS.items():
        for pname, pol in rows:
            counts = [survived(pol, env, w, p, args.episodes, args.seed, sideways=not args.any_direction)[0]
                      for p in PUSHES]
            print(f"{wname:24s} {pname:18s} " + " ".join(f"{c:7d}" for c in counts), flush=True)
    print(f"({time.time() - t0:.0f} s)")

    # -- 5. what it means --------------------------------------------------------------------------------------
    print("\na policy is robust to what it trained on and to nothing else it was not forced to be robust to."
          " Compare the nominal and random checkpoints row by row: where did randomising help, and what did it cost?")

    if args.no_viewer or soc4180.is_colab():
        return 0

    # -- 6. replay: the last policy, nominal world, a 150 N shove -------------------------------------------------
    _, frames = survived(rows[-1][1], env, WORLDS["nominal"], 80, 1, args.seed, record=True)
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    print(f"\nreplaying '{rows[-1][0]}' with a 150 N shove at t = {env.push_time:.2f} s")
    with soc4180.launch_viewer(env.model, env.data, passive=True) as viewer:
        while viewer.is_running() and time.time() < deadline:
            for q in frames:
                if not viewer.is_running() or time.time() > deadline:
                    break
                env.data.qpos[:] = q; mujoco.mj_forward(env.model, env.data); viewer.sync()
                time.sleep(env.control_dt / args.speed)
            time.sleep(0.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
