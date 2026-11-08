# 07b — Learning a Table from Experience

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/07b-tabular/lab.ipynb)

**The bridge between week 7 and week 8.** Week 7 solved the G1's balance with
value iteration, because P and R were arrays. `env.step` never gives you P.
This week an agent learns from `(s, a, r, s')` alone, keeping a table Q(s, a)
— first on two gymnasium grid worlds, then on MuJoCo's inverted pendulum, then
on week 7's balance against its exact answer — and finds out why tables stop
working, which is where week 8's policy gradient begins.

| | |
| --- | --- |
| **Layer** | 5 — learning, at its simplest: no network, no gradient, one table |
| **Runtime** | Local: CPU. Colab: CPU is enough (nothing renders video). |
| **Wall clock** | deck ~10 min to render (it trains); lab grading ~2 min |
| **Convergence risk** | Low and measured: every graded result repeats exactly on one machine |
| **Needs** | `soc4180[env]` (gymnasium with MuJoCo). No torch. |
| **Feeds** | Week 8, which drops the table for a function with parameters |

## Objectives

1. Derive the Q-learning update from the Bellman equation for Q, by replacing
   the average over P with a running average of samples.
2. Explain ε-greedy exploration and the difference between off-policy
   (Q-learning) and on-policy (SARSA) targets, with a case where each is right.
3. Put a continuous state into a table by binning, and say what the table must
   see (the Markov requirement) and what seeing more costs (rows multiply).
4. Measure a learned table against an exact answer, and explain the gap.

## Measured results

| World | Result |
| --- | --- |
| FrozenLake, value iteration (knows P), γ 0.99 | 571 sweeps; π\* reaches the goal 0.737 |
| FrozenLake, Q-learning, 5000 episodes | **0.737**, the same action as π\* on every tile that can reach the goal |
| FrozenLake, γ 0.5 | π\* itself 0.465; Q-learning 0.07 |
| FrozenLake, acting at random (ε = 1) | Q-learning's greedy table 0.65–0.75; SARSA's 0.13–0.61 |
| CliffWalking, ε 0.1, 500 episodes | SARSA falls in 7–14 of the last 200 episodes, path 17 steps; Q-learning 47–51, path **13** |
| Inverted pendulum, angle + angular velocity (36 rows), 3000 episodes | kept table balances **1000 / 1000** steps from 10 new starts |
| Inverted pendulum, angle only (6 rows) | 7–35 steps: the angle is not a state |
| Inverted pendulum, all four numbers (324 rows) | 43–77 steps: too many rows for the samples |
| G1 balance from samples, 100,000 episodes, bins 5 / 11 / 21 / 41 / 81 | J 19.1 / 51.1 / **65.5** / **66.1** / 53.3 (hold 12.66; value iteration 72.12) |

## Who needs P, and on- versus off-policy

Two slides put the three update rules side by side:

| | needs P? | uses | on / off-policy | learns |
| --- | --- | --- | --- | --- |
| value iteration (week 7) | yes (model-based) | the whole table of P | neither: it never acts | Q\* |
| Q-learning | no (model-free) | (s, a, r, s′) | **off**: a′ in the target is the greedy argmax, perhaps never taken | Q\* |
| SARSA | no (model-free) | (s, a, r, s′, a′) | **on**: a′ is the ε-greedy action it will take next | Q of the ε-greedy policy |

The test for on/off-policy is whether the a′ inside the target comes from the
same policy that chose a (the behaviour policy, ε-greedy for both). At ε = 0
the two targets coincide.

## The three lessons

**The table can be learned from samples, and it costs samples.** Q-learning
reaches value iteration's exact answer on the lake — in 5000 episodes where P
gave it in 571 sweeps. On the G1's balance, 100,000 episodes leave it 6 points
short of what value iteration found in 3 s.

**What the table is told decides what it learns.** A low γ makes the long safe
route worthless; SARSA's target makes it value the walker it really is
(slipping one move in ten) and keep away from the cliff; Q-learning's target
learns the greedy policy whatever the agent did, so it can learn from random
behaviour — and walks the edge.

**Tables do not scale.** A row learns nothing about its neighbours. The pendulum
seeing two numbers works; seeing four fails in the same budget. The balance has
an inverted U: 11 bins never settle (a row lumps states needing different
actions, so the table forgets), 81 bins are visited too rarely. The walking G1
sees 71 numbers: $10^{71}$ rows. Week 8 replaces the table with a function.

## Lab class: on your laptop

```bash
uv run --extra rl weeks/07b-tabular/lab_tables.py                    # the game: edit tables.py only
uv run --extra rl weeks/07b-tabular/lab_tables.py --grade --no-viewer
```

**A graded game in week 3's shape.** Students edit only `tables.py`; every
problem trains a table **live in the MuJoCo window** (the arrows are the
table's best action and change as it learns), plays what it learned, and grades
it. `G` grades all six; the score and a 4-letter scorer code stay on screen.
As shipped `tables.py` holds the reference answers and scores 6 / 6;
students change an answer and watch what breaks (a `...` blanks one again).

| # | Problem | Students choose | Passes when | Right looks like |
| --- | --- | --- | --- | --- |
| 1 | FAR-SIGHTED | lake: `gamma` | greedy success ≥ 70 % (3 tables) | arrows lead round the holes, the start arrow points into the wall |
| 2 | WATCHER | lake, ε = 1: `"q"` / `"sarsa"` | greedy success ≥ 60 % | the arrows still make sense although it acted at random |
| 3 | SAFE | cliff: `"q"` / `"sarsa"` | < 25 falls in the last 200 episodes | arrows lead up and over, away from the edge |
| 4 | SHORT | cliff: `"q"` / `"sarsa"` | greedy path 13 steps | the ball runs along the cliff edge |
| 5 | PENDULUM | which numbers the table sees, and their cut points | kept table: 1000 steps from 10 starts | the pole stays up; ctrl + right-drag it and it recovers |
| 6 | G1 BALANCE | `bins` | J ≥ 60 over the second half | the CoM catches all three pushes without leaving the foot |

Reference answers are in `instructor/tables_solution.py`, published like
every week's answers. The scorer code for the
unchanged referee is **119A**; any edit to `lab_tables.py` changes it. Wrong answers that fail,
measured: γ 0.5, 0.9 and 0.95; SARSA for problems 2 and 4, Q-learning for 3;
the angle alone or all four numbers for 5; 5, 11 or 81 bins for 6.

## Lab class: `qlearn.py`

```bash
uv run --extra rl weeks/07b-tabular/qlearn.py --task frozen
uv run --extra rl weeks/07b-tabular/qlearn.py --task cliff
uv run --extra rl weeks/07b-tabular/qlearn.py --task pendulum --see angle rate
uv run --extra rl weeks/07b-tabular/qlearn.py --task balance --bins 11 21 41
```

Linear: each task computes and prints, then replays in the viewer — the greedy
agent on the grid, the kept pendulum table live (push it), the G1's CoM catching
three pushes. `--no-viewer` for headless. The numbers above come from it.

## The code

`src/soc4180/tabular.py`: `value_iteration` (for gymnasium's toy-text `P`),
`Learner` (Q-learning / SARSA with ε-greedy; one `episode()` at a time),
`Binner`, `pendulum_learner`, `KeepBest`, `BalanceMDP` and `BalanceLearner`
(week 7's LIPM with the same constants, so J = 72.12 is the exact answer).

## Rebuild

```bash
quarto render weeks/07b-tabular/slides.qmd
```

The deck shows the code of every cell that computes a number (and prints the
package's own source where it trained something), and embeds MathJax 3 so the
equations render offline. After an edit, check that no slide is taller than 800 px.
