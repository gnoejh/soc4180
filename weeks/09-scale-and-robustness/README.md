# 09 — Scale and robustness

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/09-scale-and-robustness/lab.ipynb)

**What learning costs, and what a learned reflex depends on.** One task, one file: `lab.py`.

| | |
| --- | --- |
| **Layer** | 5, learning; the sim-to-real gap, measured in simulation |
| **Runtime** | Laptop CPU. Colab: a CPU runtime is enough. **No GPU needed** |
| **Needs** | `uv sync --extra rl` (stable-baselines3, torch) |
| **Wall clock** | the deck's notebook ~1 min; `lab.py` ~3.5 min |
| **Trains** | nothing: two PPO checkpoints are shipped in `checkpoints/` |

The deck opens with **where this week sits on the map** of RL methods from week 8, and every
major function of `lab.py` is printed and then explained line by line (`| code | what it does | why |`).

## The lecture

1. **The bill.** PPO learns from its own episodes, so it needs many: week 8's 30 k decisions
   learned not to fall; these checkpoints took 3 M (about 16 min each on 16 processes of a
   36-core machine, read from `checkpoints/push_*.csv`); a tuned humanoid run uses 150 M. The
   arithmetic of a run and Amdahl's law; measured CPU facts: `DummyVecEnv` gives no speed-up
   (1173 / 1240 / 1250 decisions/s for 1 / 4 / 16 envs, a loop on one core), `SubprocVecEnv`
   scales with cores (1649 / 3205 / 5570 for 2 / 4 / 8); one process puts 150 M decisions
   at more than a day. GPU-parallel simulation (MJX, `jax.jit(jax.vmap(step))`) is taught as a
   concept. **GPU training is optional and not needed for anything in this course.**
2. **The reflex.** `G1PushEnv`: stand in the crouch and survive a 0.2 s shove. Two policies,
   identical except the worlds they trained in: **PPO nominal** (one world) and **PPO random**
   (domain randomisation, `RANDOM_WORLD`: friction ×0.3–1.2, torso mass −5…+10 kg, servo
   stiffness ×0.7–1.3). "Robust" means robust *to a distribution*; `J_DR` is the average of
   the return over it; a world outside it is the sim-to-real gap.
3. **Measuring honestly.** Exact shove sizes, sideways, deterministic policy, counts out of 10,
   the same seeds for every policy. 10 of 10 only says p ≥ 0.69 (95 %); 9 of 10 beside 10 of 10
   is not a difference.
4. **A non-result.** A privileged teacher (velocity, height, the shove, the world in its
   observation), same 3 M budget, was *worse*: 3 of 10 at 60 N. Not shipped.

## Lab class on your laptop

```bash
uv sync --extra rl                                          # once
uv run weeks/09-scale-and-robustness/lab.py                 # the table, then a replay in the viewer
uv run weeks/09-scale-and-robustness/lab.py --no-viewer
uv run weeks/09-scale-and-robustness/lab.py --episodes 3    # quicker, noisier
uv run weeks/09-scale-and-robustness/lab.py --any-direction # shoves from every side
```

**You change** `WORLDS` (friction, torso mass, servo kp per world) and `PUSHES` at the top of
`lab.py`. **You see** the table row by row, then the chosen policy (`--replay`, default
`"PPO random"`) taking a 100 N shove in the first world. As shipped (measured with `lab.py`,
10 episodes per cell, sideways, ~3.5 min), episodes survived:

| world | policy | 0 N | 60 N | 80 N | 100 N | 120 N |
| --- | --- | --- | --- | --- | --- | --- |
| nominal | hold | 10 | 10 | 0 | 0 | 0 |
| | ankle (wk 8) | 10 | 10 | 0 | 0 | 0 |
| | PPO nominal | 10 | 10 | 10 | 0 | 0 |
| | PPO random | 10 | 10 | 10 | 9 | 1 |
| heavy (+15 kg) | hold | 0 | 0 | 0 | 0 | 0 |
| | ankle (wk 8) | 10 | 10 | 6 | 0 | 0 |
| | PPO nominal | 10 | 10 | 8 | 0 | 0 |
| | PPO random | 10 | 10 | 10 | 10 | 10 |
| ice (friction 0.2) | hold | 10 | 10 | 0 | 0 | 0 |
| | ankle (wk 8) | 10 | 10 | 10 | 0 | 0 |
| | PPO nominal | 10 | 10 | 9 | 6 | 0 |
| | PPO random | 10 | 10 | 10 | 10 | 5 |
| weak servos (kp 0.6) | hold | 10 | 0 | 0 | 0 | 0 |
| | ankle (wk 8) | 10 | 10 | 0 | 0 | 0 |
| | PPO nominal | 10 | 10 | 3 | 0 | 0 |
| | PPO random | 10 | 10 | 10 | 0 | 0 |

Holding the crouch falls with no shove once the torso is 15 kg heavier. PPO random is the best
row in every world here — and all three changed worlds lie just outside its training box.
Falls are chaotic: counts repeat exactly on one machine but may differ on another, so compare which
shove a policy falls at, never a single count.

**Try** (write the prediction first):

1. A world outside the training range, e.g. `mass=+25` or `friction=0.1`. Does PPO random still win?
2. `kp=0.5`: which policy degrades most gracefully?
3. `--any-direction`: which policy was trained for that?
4. Which row would you send to a real robot whose mass you know only to ±5 kg?

## For your term project

[`weeks/projects/README.md`](../projects/README.md), list in
[`catalogue.md`](../projects/catalogue.md). **P8** (learned push recovery: randomisation that
helps) continues this week directly: the shipped checkpoints are the twin, and retraining is 3 M
decisions — hours on a laptop, so plan the compute. **P2** (the walker on hard ground), **P7**
(a planner with a wrong model) and **P14** (your own) use the same method: a prediction first, a
twin with one change, every number from `results.csv`, a `CHECK` line, exact-size shoves, and
never grade convergence.
