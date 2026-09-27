# 11 — Planning at run time

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/11-planning/lab.ipynb)

| | |
| --- | --- |
| **Layer** | 4 — Control & Planning |
| **Runtime** | CPU for the planner; Colab needs a GPU runtime only to render the video |
| **Wall clock** | ~2 min locally (36 cores); several minutes on Colab's two cores |
| **Needs** | `soc4180` only — no torch |

## Objectives

1. Explain model-predictive control: a receding horizon, re-planned every decision.
2. Write MPPI in seven lines: sample, roll out, cost, weight by exp(−J/λ), average, apply, shift.
3. Use MuJoCo itself as the model: `mj_getState` and `mujoco.rollout`.
4. Treat the cost as the specification: dense versus sparse terms, local minima,
   and a planner that does what the cost says rather than what was meant.
5. Measure what planning costs, and what a wrong model does to it.

## Measured results (`plan.py`, seed 0)

| Controller | Sideways shove at t = 1 s for 0.2 s | Notes |
| --- | --- | --- |
| hold the crouch | stands at 65 N, **falls at 70 N** | as week 7 found |
| MPPI, 64 samples × 0.4 s, standing cost | **stands at 150 and 300 N** | by stepping; feet travel ~3–7 m |
| MPPI, 2 samples | falls at 150 N | |
| MPPI, 4 / 16 samples | stands at 150, falls at 300 | |
| MPPI, 16 samples, σ = 0.25 | stands at 300 | fewer samples need wider ones |

One decision is 12,800 physics steps: **74–86 ms** on 16 threads here with the machine
quiet (115–148 ms with other work running), so the planner runs at about
**0.5× real time** when idle and 0.3× when busy. A single `rollout` of 64 × 200 steps took 69 ms (185,000 steps/s).

Model error (`--model-mass`, the planner's torso only; the world's is 7.82 kg):
exact model stands at 150 and 300 N; −10 kg falls at both; +10 and +20 kg stand at
150 and fall at 300; +40 kg falls at both.

Costs (`lab_plan.py --grade`): only `"fall": 100` falls **with no push**; only
`"still"` stands unpushed and falls at 150 N; all weights 0 falls in 1.4 s. The squat
with height weight 20 never moves (lowest pelvis 0.745 m); with 100 it reaches
0.503 m and stands up. CRANE lifts the foot only by **hopping** (2 m of travel);
a `plant_right` term stops that at exactly one weight tried (1000) and fails at 500 and
2000, so hopping is allowed in the graded problem.

At the default temperature λ = 0.05 the weights put effectively **one** sample in
charge (effective sample size 1.0 before and after a 300 N shove): in practice MPPI
here keeps the best of 64 futures. λ = 5 lets 3–5 vote.

## Lab class: on your laptop

```bash
uv run weeks/11-planning/lab_plan.py          # the game: edit costs.py
uv run weeks/11-planning/lab_plan.py --grade --no-viewer
```

**A game modelled on week 3's.** Students edit only `costs.py` — the weights of each
problem's cost (`height`, `upright`, `over_feet`, `over_right`, `lift_left`,
`plant_right`, `still`, `fall`), two settings (`height_target`, `lift`) and the
planner's `samples`, `horizon`, `sigma`, `temperature`. `costs.py` is parsed as data,
never run; `...` is a blank and each problem names its blank lines. As shipped it
scores 0 / 6; the reference (`instructor/costs_solution.py`, sealed) scores 6 / 6.

| # | Problem | Pass |
| --- | --- | --- |
| 1 | STAND | 3 s, no push, feet move < 5 cm |
| 2 | SHOVE | 150 N at t = 1 s; pelvis never below 0.5 m, final tilt < 15° |
| 3 | BIG SHOVE | 300 N |
| 4 | CHEAP | 300 N with at most 16 samples (`CHEAP_PLANNER`) |
| 5 | CRANE | left foot 5 cm clear for 2 s in a row; do not fall |
| 6 | SQUAT | pelvis below 0.58 m between 1 and 3 s, above 0.68 m at 5 s |

Keys: `1`–`6` play (slow motion: the planner thinks), `SPACE` again, `G` grade, `ENTER`
prints the gauge. The terminal takes the same keys. Grading all six takes about a
minute on 36 cores. **Any edit to `lab_plan.py` changes the scorer code.**

### `plan.py`: the planner, printed

```bash
uv run weeks/11-planning/plan.py --push 300
uv run weeks/11-planning/plan.py --controller hold --push 70
uv run weeks/11-planning/plan.py --samples 16 --sigma 0.25 --push 300
uv run weeks/11-planning/plan.py --model-mass -10 --push 150
uv run weeks/11-planning/plan.py --task crane --seconds 6
uv run weeks/11-planning/plan.py --task squat
```

A row every 0.5 s (pelvis, tilt, left-foot height, how far the feet travelled, the
best sampled cost, ms per decision), the outcome (FELL = pelvis below 0.5 m or ending
tilted more than 15°), the planning bill, then a replay.

## Package

`soc4180.planning`: `planning_model()` (the G1 + six `FRAMEPOS`/`SUBTREECOM` sensors
added with `MjSpec.add_sensor`), `Rollouts` (K × T accessors), `MPPI`,
`actuators_for`, `crouch`, `standing_cost`.

## Rebuild

```bash
quarto render weeks/11-planning/slides.qmd
```

The deck runs MPPI four times and renders one video; it is a timing measurement in
places (ms per decision), so render on a quiet machine.
