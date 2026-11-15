# 08 — Learning to Act: one gain, learned from reward

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/08-learning-to-act/lab.ipynb)

**The first week that trains anything.** One task, one file: `lab.py`. The crouched G1 is shoved
sideways; it can save itself by turning both ankles against the tilt it feels,

    ankle pitch = theta * g_x          (g_x = obs[0], the forward tilt in the torso's frame)

and it must **learn the one number `theta`**, starting from 0 (hold the crouch), from nothing but a
reward. Layer 5: learning.

| | |
| --- | --- |
| **Runtime** | Local: CPU. Colab: a CPU runtime is enough (nothing renders video). |
| **Wall clock** | the lab ~1 min to learn; the notebook ~1 min |
| **Needs** | `soc4180[rl]` (`uv sync --extra rl`): gymnasium, for `G1WalkEnv` |

## The deck

**It opens with a map of RL** (five slides of tables): the problem and its vocabulary on the G1;
four questions that sort every method (value- or policy-based, model-based or free, on- or
off-policy, table or function); ten methods sorted by them, each with the week it appears in (★
today's); why the G1 pushes toward policy methods; and the route through weeks 7–11. Weeks 9–11
open with the same route, their own row in bold.

Then: from week 7's MDP to a policy with a parameter and its return J(θ); the policy gradient
(REINFORCE, the score function ∇log π, derived); why action noise knocks a balancing robot over;
exploring in parameter space with **antithetic pairs on a shared push** (a finite-difference /
Gaussian-smoothing gradient) — the rule in `lab.py`; the learning curve, the test table and the
J(θ) landscape; the reward as a design decision; PPO as the same idea scaled to networks. Every
code slide prints a function of `lab.py` with `inspect.getsource`, so the deck cannot drift from
the file, and is followed by a line-by-line slide (`| code | what it does | why |`) for
`episode`, `learn`, `test` and `main`.

## The learning rule

    repeat 20 times:
        draw 8 pushes p_i (60-80 N, left or right) and 8 directions eps_i ~ N(0, 1)
        try theta + sigma*eps_i and theta - sigma*eps_i on the same push p_i
        theta <- theta + lr * mean_i[(G+_i - G-_i) * eps_i] / (2 sigma)

"Try a bit more and a bit less on the same push; move toward the one that paid."

## Measured with `lab.py`

- θ 0 → **1.82 / 2.00 / 1.53** for seeds 0 / 1 / 2 (20 updates of 8 pairs, ~50 s each). Week 7's
  hand-written ankle gain is 2.
- Holding the crouch (θ 0) **falls at 70 N and 75 N**, left and right; θ 0.6 to 4 stands at 75 N;
  θ 8 falls at 75 N again (too much gain); **every θ falls at 80 N**; a negative θ falls at 60 N.
  J(θ) is an inverted U.
- A roll gain on g_y changed nothing at any of these pushes: the pitch strategy is what saves it.
- REINFORCE with noise on the **action** (σ 0.3) made every exploring episode fall — 8 of 8 — and
  diverged. That is why `lab.py` explores in θ.
- PPO on `G1WalkEnv` (30k steps, from earlier weeks): untrained 774.3, default action std 1.0
  **94.7** (exploration noise: the sampled robot falls), `log_std_init = -2` 776 — it learns not to
  fall, not to walk (walking is worth 1250).

## Lab class

```bash
uv sync --extra rl                                   # once
uv run weeks/08-learning-to-act/lab.py               # learn (~1 min), test, replay in the viewer
uv run weeks/08-learning-to-act/lab.py --no-viewer   # headless
uv run weeks/08-learning-to-act/lab.py --updates 0   # no learning: test the starting rule
```

**What students change** (one at a time, prediction first): `reward()` and `ALIVE`, `UPRIGHT`,
`EFFORT`; the learner's `SIGMA`, `LR`, `PAIRS`; the flags `--pushes LO HI`, `--test`, `--seed`,
`--updates`.

**What they should see**: the update table (θ climbs from 0, the falls column empties; seed 0 ends
at 1.82); the test table (hold FELL at 70 and 75 N, hand (2) and learned stood, all three FELL at
80 N); the replay of the learned rule under 70 N, the ankles turning against the tilt.

**Try** (write the prediction down first):

1. `ALIVE = 0`: is there still a reason to learn?
2. Reward only survival: return 1.0 on every step, 0 on the step it falls.
3. `UPRIGHT = 1000`: does the learned θ change?
4. `--pushes 30 50`: pushes nobody falls from. Is there anything to learn?
5. `SIGMA = 0.05` or `3`: tries too close together, or too far apart.

These are not measured for you; the student's tables are the answer.

## Term project

Weeks 12–15 ([`projects/README.md`](https://github.com/gnoejh/soc4180/blob/main/weeks/projects/README.md),
[`catalogue.md`](https://github.com/gnoejh/soc4180/blob/main/weeks/projects/catalogue.md)):
**P9** reward design for stepping, **P13** a table that balances, **P3** balance from the robot's
own senses, **P14** your own proposal. The method: a prediction before the run, a twin with one
change, `results.csv`, a `CHECK` line. **Never grade convergence; grade the diagnosis.**

## Rebuild

```bash
quarto render weeks/08-learning-to-act/slides.qmd
```

The executed cells run ~100 episodes (the landscape, the action-noise check, 10 learning updates,
the test): about a minute. Equations use the embedded MathJax 3 (`assets/mathjax/`), so the deck
works offline. After an edit, check that no slide is taller than 800 px.
