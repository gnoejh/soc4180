# 07 — From Control to Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/07-mdp/lab.ipynb)

**The hinge week.** Six weeks of deriving controllers end; specifying problems
begins. No learning code appears — only the problem definition.

| | |
| --- | --- |
| **Runtime** | Local: CPU. **Colab: pick a GPU runtime** (Runtime > Change runtime type > T4). |
| **Wall clock** | ~1 min |
| **Convergence risk** | None. Nothing is trained. |
| **Needs** | `soc4180[env]` — gymnasium. Lighter than `[rl]`: no torch, no SB3. |
| **Feeds** | Weeks 8–12, which all train in this environment |

## Objectives

1. State the MDP formalism and identify each element in the walking task.
2. Explain why the problem is really partially observable, and how that is
   normally papered over.
3. Justify residual actions, a 50 Hz control rate, and the termination rules.
4. Validate an environment against `gymnasium.utils.env_checker`.
5. **Test whether the environment can represent a solution you already have.**

## Measured results

| Policy | Return | Outcome |
| --- | --- | --- |
| Do nothing (hold the crouch) | 774.09 | survived all 500 steps |
| Uniform random (action space seeded 0) | 25.78 | fell in 33 steps (0.66 s) |
| Analytic walker (Week 4) | — | **fell after 4.2 s, 0.24 m** |

The Week 4 walker **fails inside its own environment**, with actions clipped on
**35.7%** of steps. The diagnosis separates two causes:

| action_scale | 50 Hz | 100 Hz | 200 Hz asked (250 Hz actual: decimation round(2.5) = 2) |
| --- | --- | --- | --- |
| 0.3 | fell (36% clipped) | fell | fell |
| 0.6 | fell (18% clipped) | fell | fell |
| 1.0 | fell (0% clipped) | **survived, 1.05 m** | **survived, 1.05 m** |

Both variables matter independently. Only `action_scale = 1.0` **and** ≥100 Hz
reproduces the walk (1.05 m against 0.99 m at 500 Hz).

## The two lessons

**An action space defines the set of reachable behaviours.** A solution outside
it does not exist in your problem, and no learning curve will ever say so — the
run simply plateaus. The defaults stay at 0.3 / 50 Hz because that is what the
locomotion literature trains with, and learned policies do succeed inside that
envelope; they just find *different* gaits from ours. An environment is a
commitment about what kind of solution you expect.

**Return is not comparable across control rates.** The "200 Hz" (really 250 Hz) runs score ~5×
the 50 Hz ones purely because they contain more steps. Report distance, time
upright, or mean velocity alongside it.

## Lab class: on your laptop

```bash
uv run weeks/07-mdp/lab_env.py                     # needs gymnasium: uv sync --extra rl
uv run weeks/07-mdp/env_run.py --policy lean --push 75
```

`G1WalkEnv` in the interactive viewer, stepping at the policy rate in real
time. `POLICIES` holds four functions from observation to action — hold the
crouch, uniform random, the week 4 walker via `walker_actions`, and
`my_policy(obs, t)` — selected with keys `1`–`4`.

```
1..4    pick a policy and reset      R   reset       SPACE   pause
ENTER   the observation (gravity, gyro, joint angles), the action, every reward term
double-click a body, then ctrl-drag: push it
```

Drawn on the robot: a **white arrow** for `obs[0:3]` (gravity in the body
frame, exactly what the policy knows about up), a **bar above the head** for
this step's reward, and **yellow feet** when the env counts them airborne. Each
episode ends with a terminal line: return, steps, distance, and whether it
**terminated** (fell) or was **truncated** (10 s).

**The code is complete and explained.** `my_policy` as shipped is the first
closed loop of the course, the ankle strategy: both ankles driven against
`obs[0:3]`. `ACTION_SCALE`, `CONTROL_HZ` and `REWARD_WEIGHTS` at the top are
the three specification knobs the deck argues about.

| Step | Change | Right looks like (measured with `env_run.py`) |
| --- | --- | --- |
| 1 | `1`, `2`, `3`; then `ACTION_SCALE = 1.0`, `CONTROL_HZ = 100`, `3` again | hold 774; random 32, falls after 38 decisions; the walker 357, falls at 4.3 s; at the new settings 1805 and a metre walked |
| 2 | `my_policy` returns a constant that bends both knees | `ENTER` shows upright up, effort down, tracking unchanged |
| 3 | `my_policy` periodic in `t` on hip pitch, opposite signs per leg | terminates at 1.5 s; the last `ENTER` read |
| 4 | restore `my_policy`; ctrl-drag under `1` and `4` | hold survives 65 N for 0.2 s and falls at 70; the ankle strategy survives 75 and falls at 80; hip roll instead changes nothing |
| 5 | `REWARD_WEIGHTS = {"alive": 0.0}` and re-run `1`, `3` | the numbers change, the behaviour does not — nothing here is learning yet |

### `env_run.py`: one episode, term by term

```bash
uv run weeks/07-mdp/env_run.py --policy hold
uv run weeks/07-mdp/env_run.py --policy walker --action-scale 1.0 --hz 100
uv run weeks/07-mdp/env_run.py --policy lean --push 75
uv run weeks/07-mdp/env_run.py --policy hold --push 70 --no-viewer
uv run weeks/07-mdp/env_run.py --policy random --weights alive=0 --no-viewer
```

Five policies of a few lines each (`hold`, `random`, `walker`, `lean`, `step`),
one episode printed as a row per second with the reward earned in that second
split by term, the episode line, the return split by term, and a replay.
`--push N` applies a sideways force to the torso for 0.2 s at t = 1 s through
`xfrc_applied`, which is how every push number above was measured.

| Step | Does | Calls |
| --- | --- | --- |
| 1 | the environment: spaces, rate, episode length, the reward weights | `G1WalkEnv`, `observation_space`, `action_space`, `reward_weights` |
| 2 | the policy | `walker_actions`, `WalkingController` |
| 3 | reset, then decide / act / score until done | `env.reset(seed)`, `env.step(a)`, `info[term]`, `xfrc_applied` |
| 4 | the episode line and the split | |
| 6 | replay | `launch_viewer(passive=True)` |

Measured: hold 774.1 (tracking 274, upright 250, alive 250); random 31.6;
walker 356.6 and a fall at 4.34 s; walker at scale 1.0 and 100 Hz 1804.6 and
1.05 m; step-in-place 93.3 and a fall at 1.52 s. Pushes: hold survives 65 N,
falls at 70; ankle gains (2, 2) survive 75, fall at 80; gains of 1 or 4 do no
better; every hip-roll variant behaves like hold.

### `mdp.py`: an MDP written down and solved

```bash
uv run weeks/07-mdp/mdp.py                           # numpy only; no extra needed
uv run weeks/07-mdp/mdp.py --policy ankle --push 0.1
uv run weeks/07-mdp/mdp.py --gamma 0.8 --no-viewer
uv run weeks/07-mdp/mdp.py --alive 0 --no-viewer
```

`G1WalkEnv` is a simulator: it returns one sample (s′, r) per (s, a), so
nothing in `lab_env.py` or `env_run.py` *solves* the MDP — the policy there is
designed by hand and the reward only grades it. `mdp.py` closes that gap on a
problem small enough to write down: the G1's forward balance as week 4's
inverted pendulum (G1 numbers: CoM 0.661 m, foot −0.05 … +0.12 m about the
ankle, 50 Hz). The state is (CoM offset, speed) on an 81 × 81 grid, the action
one of 7 centre-of-pressure positions, P a table (bilinear weights over the
grid, every row summing to 1), R the env's alive / upright / effort terms, and
a FALLEN state worth 0. Value iteration iterates the Bellman optimality
equation until it holds, and π* is read off as the argmax — **here R chooses
the policy**.

| Step | Does |
| --- | --- |
| 1 | the dynamics: one decision of the pendulum, solved exactly |
| 2 | S, A, P, R as arrays |
| 3 | value iteration (optimality) and policy evaluation (expectation) |
| 4 | policies on the real pendulum: π* (greedy on V*), two hand rules, hold |
| 5 | print, then replay one push in the viewer (CoM blue, CoP orange, capture point purple) |

Measured (all in the docstring): 1376 sweeps in 3 s, the error shrinking by
exactly γ per sweep. **π* rediscovers week 4's capture point**: the states it
saves and the states whose capture point lies on the foot agree on 100.0 % of
the grid, and it survives pushes of −0.19 … +0.46 m/s against the capture
limits −0.193 … +0.462. The lean-only rule p = 2x (`lab_env.py`'s idea) is an
undamped spring here and survives only ±0.03–0.04 m/s. γ sets the look-ahead:
0.9 still saves 31.8 % of the grid, 0.8 20.6 %, 0.7 0.7 %, 0.5 nothing. With
`--alive 0` (penalties only) π* lets saveable states fall (26.1 % saved).

## Rebuild

The deck shows the code of every cell that computes a number (`#| echo: true`;
revealjs hides code otherwise); figure-drawing code appears only in
`lab.ipynb`. After an edit, check that no slide is taller than 800 px.
Equations are typeset by an embedded MathJax 3 (`assets/mathjax/`), so the
deck shows them offline too; no CDN is needed.

```bash
quarto render weeks/07-mdp/slides.qmd
```
