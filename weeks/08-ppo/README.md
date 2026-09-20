# 08 — Policy Gradients and PPO

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/08-ppo/lab.ipynb)

**The first week that trains anything.**

| | |
| --- | --- |
| **Runtime** | Local: CPU. **Colab: pick a GPU runtime** (Runtime > Change runtime type > T4). |
| **Wall clock** | **~6 min** — three training runs. Longest lab so far. |
| **Convergence risk** | Low for CartPole and InvertedPendulum; the G1 run is *designed* to fail. |
| **Needs** | `soc4180[rl]` — torch and stable-baselines3 |

## Objectives

1. State the policy gradient theorem and explain why it avoids differentiating
   the simulator.
2. Implement REINFORCE from scratch and read its learning curve.
3. Explain baselines, advantage, and why PPO clips.
4. Diagnose a failed training run as a **specification** failure rather than an
   algorithm failure.

## Measured results

| Run | Result | Time |
| --- | --- | --- |
| REINFORCE, CartPole (from scratch) | 60 → **489** (max 500) | 173 s |
| PPO, InvertedPendulum-v5 | 26 → **1000** (solved) | 99 s |
| PPO, `G1WalkEnv`, 30k steps | 774 → **95** | 69 s |

## The two results that carry the week

**The untrained policy already scores 774** — exactly the hold-the-crouch
baseline. That is the Week 7 residual-action design paying off: a network with
near-zero outputs commands the nominal crouch and simply stands. The policy
starts from competence. It also sets a cruel bar, since the easiest way to score
774 is to do nothing.

**Training made it worse: 774 → 95.** The trained policy lunges at 1.28 m/s
against a 0.5 target and falls after 46 of 500 steps.

The obvious explanation — that the reward rewards diving — is **wrong**, and the
lab checks it rather than asserting it. Evaluating the reward by hand:

| behaviour | per step | over the episode |
| --- | --- | --- |
| walk at the target | 2.500 | **1250** (the best available) |
| stand still | 1.552 | 776 |
| lunge, fall at 46 steps | 1.129 | 52 |

There is no loophole. The reward wants exactly what we want.

**The real cause is exploration noise.** PPO starts with an action standard
deviation of 1.0 on a `[-1, 1]` action space, so:

- untrained **deterministic** policy (what we evaluate): survives **500 of 500** steps
- untrained **stochastic** policy (what we train on): survives **38**

Over 90% of collected experience is a robot falling over. PPO never sees the
competent behaviour its own mean produces. Setting `log_std_init=-2.0`
(std 0.135) raises stochastic survival to ~180 steps, and the same 30k-step run
then scores **776 instead of 95**.

> A policy is a distribution. If the distribution is too wide, your data
> describes a policy you would never deploy.

**But it still is not walking** — 776 is the standing baseline, and walking is
worth 1250. Fixing this bug removed a bug; 30k steps is 0.02% of a real
locomotion run. That is Week 10's problem.

## Lab class: on your laptop

```bash
uv run weeks/08-ppo/lab_train.py          # needs torch + SB3: uv sync --extra rl
uv run weeks/08-ppo/train.py --log-std -2
```

`G1WalkEnv` in the viewer under a PPO policy from `CONFIGS`. `T` trains the
selected configuration in a background thread (about 70 s for 30k steps here)
while the window keeps playing; when it finishes the trained policy takes
over. `D` switches between the deterministic robot (the mean action) and the
stochastic one (a sample), `1`–`3` select a config, `ENTER` prints the return
so far and the action std. Every episode prints return, steps survived, and
terminated or truncated.

**The code is complete and explained**: `choose_action` is PPO's own sampling
(`mean + std · N(0, 1)`, clipped to the action space) with the reason it
matters in its docstring, and the loop names `PPO("MlpPolicy")`, `learn`,
`predict(deterministic=True)` and `policy.log_std`.

| Step | Do | Right looks like (measured with `train.py`) |
| --- | --- | --- |
| 1 | `ENTER` | std 1.0; the deterministic robot stands 500 steps, return 774 |
| 2 | `D`; then break `choose_action` | the stochastic robot survives ~42 steps; returning the mean makes `D` do nothing; std × 3 falls at once |
| 3 | `T`, wait, `D` a few times | trained on std 1.0 the deterministic robot is *worse*: 94, 47 steps — it learned from falls |
| 4 | `2` (log_std_init −2), `D`, `T` | stochastic survival ~208 steps; trained deterministic 777, 500 steps: it learned not to fall, not to walk |
| 5 | a config with log_std_init −4 | a prediction first; it will not walk either — the deck has the number |

### `train.py`: two robots, measured before and after

```bash
uv run weeks/08-ppo/train.py
uv run weeks/08-ppo/train.py --log-std -2
uv run weeks/08-ppo/train.py --log-std -2 --steps 100000
uv run weeks/08-ppo/train.py --steps 0 --no-viewer
```

Builds the agent, evaluates the untrained policy both ways, trains with a
progress line every 5000 steps (mean episode length and return from PPO's own
buffer), evaluates both ways again, and replays the trained deterministic
robot.

| Step | Does | Calls |
| --- | --- | --- |
| 1 | the env and the agent (14,681 parameters, initial std) | `G1WalkEnv`, `PPO("MlpPolicy", …, policy_kwargs=dict(log_std_init=…))` |
| 2 | evaluate: mean action vs `sample_action` | `predict(deterministic=True)`, `policy.log_std` |
| 3 | train, with progress | `learn(callback=…)`, `ep_info_buffer` |
| 4 | evaluate again | |
| 6 | replay | `launch_viewer(passive=True)` |

Measured, 30k steps, 70 s on this machine:

| log_std_init | untrained det. | untrained stoch. | trained det. | trained stoch. |
| --- | --- | --- | --- | --- |
| 0 (std 1.0) | 774, 500 steps | 51, 42 steps | **94, 47 steps** | 37, 35 steps |
| −2 (std 0.135) | 774, 500 steps | 301, 208 steps | **777, 500 steps** | 240, 169 steps |

The first row is the week's lesson: PPO made the robot worse, because over
90 % of what it learned from was a fall. The second row fixes the noise and
learns not to fall — and only that.

## Rebuild

```bash
quarto render weeks/08-ppo/slides.qmd     # ~6 minutes; it trains
```
