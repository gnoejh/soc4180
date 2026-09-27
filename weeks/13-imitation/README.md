# 13 — Imitation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/13-imitation/lab.ipynb)

| | |
| --- | --- |
| **Layer** | 5 — learning without rewards |
| **Runtime** | CPU. Colab: a GPU runtime only for rendering |
| **Wall clock** | a few minutes (six small students) |
| **Needs** | `soc4180[rl]`; week 12's `checkpoints/push_random.zip` |

## Objectives

1. Behaviour cloning: demonstrations → regression.
2. Covariate shift: why a lower training error can mean a worse robot.
3. DAgger: collecting labels where the student goes.
4. The rule that decides whether copying can work: **the expert's action must be a
   function of what the student can see** — deterministic, and closed loop.
5. Distillation and the teacher–student recipe of modern legged robots.

## Measured results (`imitate.py`)

**Expert 1, the week-4 walker** (100 Hz, action scale 1.0; 10 demonstrations, 8,428
examples; 100 epochs; tested from starts 100–102):

| student sees | training MSE | result |
| --- | --- | --- |
| time only | 9.0e-3 | stands, never walks |
| **clock + time** | 1.8e-5 | **1.14–1.15 m every time** (the walker's own distance) |
| sensors + clock + time | 9.0e-6 | falls after 2.3–2.6 s |
| … + previous action | 3e-6 | falls after 1.5–2.8 s |

The clock student is a tape recorder: a 40 N sideways shove at 3 s knocks it over from
every start (`--features clock time --push 40`).

**Expert 2, week 12's PPO policy** (sideways shoves, 20 episodes; the policy itself
20/20/16/0 at 60/80/100/120 N; hold 20/0/0/0):

| student | cloning, 100 N | + 3 DAgger rounds |
| --- | --- | --- |
| all 42 numbers, 40 demos (MSE 8e-5) | 12 | — |
| no joint velocities | 0 | 7 |
| no joint velocities, 5 steps of history | 3 | **16** |
| 1 or 2 demos (80 N) | 0 | 20 (2 rounds) |

**Expert 3, week 11's MPPI** (8 demonstrations, 25 Hz, action scale 1.0): training MSE
1.1e-2, the student survives **0 of 20 even at 60 N** where holding the crouch
survives 20; after 3 DAgger rounds, 2 of 20, with the training error rising each
round (1.1e-2 → 2.0e-2). MPPI's action depends on its internal plan and its random
samples, not on the state alone.

## Lab class: on your laptop

```bash
uv run weeks/13-imitation/lab_imitate.py         # the game: edit students.py
uv run weeks/13-imitation/lab_imitate.py --grade --no-viewer
```

Students edit only `students.py`: each student's features, history, number of
demonstrations and DAgger rounds. The referee trains the student (the window freezes
10–60 s) and plays the test at 5×. As shipped 0 / 4; the reference
(`instructor/students_solution.py`, sealed) 4 / 4 in about 2.5 minutes.

| # | Problem | Pass | Reference |
| --- | --- | --- | --- |
| 1 | TAPE: copy the walker | 1.0 m from all 3 starts | `["clock", "time"]` |
| 2 | COPY: copy the policy | ≥ 10/20 at 100 N | all five features, 40 demos: 12 |
| 3 | FEW: ≤ 2 demonstrations | ≥ 18/20 at 80 N | 2 demos + 2 DAgger rounds: 20 |
| 4 | BLIND: no `velocities` | ≥ 14/20 at 100 N | history 5, 3 rounds: 16 |

### `imitate.py`

```bash
uv run weeks/13-imitation/imitate.py                                   # walker, sensors: falls
uv run weeks/13-imitation/imitate.py --features clock time             # walker, clock: walks
uv run weeks/13-imitation/imitate.py --expert policy
uv run weeks/13-imitation/imitate.py --expert policy --features gravity gyro joints previous --history 5 --dagger 3
uv run weeks/13-imitation/imitate.py --expert planner                  # ~2 min of planning
```

## Package

`soc4180.imitation`: `FEATURES`, `observe`, `clock`, `Student` (fit by MSE),
`walker_demos`, `walk`, `policy_demos`, `dagger`, `shove_test`.

## Rebuild

```bash
quarto render weeks/13-imitation/slides.qmd
```
