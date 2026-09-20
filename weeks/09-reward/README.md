# 09 — Reward Shaping

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/09-reward/lab.ipynb)

| | |
| --- | --- |
| **Runtime** | Local: CPU. **Colab: pick a GPU runtime** (Runtime > Change runtime type > T4). |
| **Wall clock** | **~7 min** — four training runs |
| **Convergence risk** | None to the lesson: the interesting result *is* a failure. |
| **Needs** | `soc4180[rl]` |

## Objectives

1. Check a reward by ranking behaviours **before** training anything.
2. Distinguish a wrong destination from a missing route — i.e. local optima.
3. Read a real 24-term locomotion reward and say what each group is for.
4. Recognise reward hacking, and why the reward number cannot reveal it.
5. State the potential-based shaping theorem and why real rewards violate it.

## Measured results

**The reward's ranking is correct** (computed, not asserted): walking at the
target 1250, walking slowly 1023, standing 776, lunging-and-falling 51.

**The planned ablation failed.** All six variants — full reward, and each of five
terms removed — produced the *same* behaviour at 30k steps: survive 500 steps,
travel ~0.01 m, stand still. None of the terms we had changes whether standing is
attractive.

**The real G1 reward** (fetched from MuJoCo Playground source) has 24 terms
against our five. Two matter here: `feet_air_time = +2.0` pays for having a foot
off the ground, and `stand_still = -1.0` penalises idling explicitly. They also
use `alive = 0.0` with `termination = -100.0` — a per-step alive bonus is exactly
what makes standing profitable.

**Adding those two terms:**

| reward | steps | distance | feet lifted |
| --- | --- | --- | --- |
| ours, unchanged | 500 | +0.018 m | 0.00 |
| + `stand_still` −1.0 | 500 | +0.015 m | 0.00 |
| + `air_time` +2.0 | 500 | +0.016 m | 0.00 |
| **+ both** | **224** | **−0.524 m** | **0.08** |

Either alone changes nothing. Together they break the standing optimum — and the
policy falls over travelling *backwards*.

## The lesson

> Shaping decides **which** optimum you land in. It does not buy you the search
> needed to find a good one.

The reward now points the right way and the search is still far too small to
follow it. That is Week 10.

Single seed, single budget. The direction is real; the exact numbers are noise.
Exercise 6 has students re-run with three seeds.

## Lab class: on your laptop

```bash
uv run weeks/09-reward/lab_reward.py      # torch + SB3: uv sync --extra rl
uv run weeks/09-reward/shape.py --variant both
```

`G1WalkEnv` in the viewer. `VARIANTS` lists reward weightings; a number key
selects one and `T` trains it in a background thread (40k steps, about 95 s
here); the trained policy then drives the window. `ENTER` prints every reward
term for the current step, including the extra one. On the robot: the reward
bar and yellow airborne feet. Every episode line: return, steps, distance,
feet-up fraction.

**The code is complete and explained**: `extra_reward` is potential-based
shaping (F = γΦ(s′) − Φ(s), Φ = x) with the theorem in its docstring and a
plain velocity bonus for contrast (`EXTRA` picks); the `Shaped` wrapper shows
how a reward is changed without touching the environment.

| Step | Do | Right looks like (measured with `shape.py`) |
| --- | --- | --- |
| 1 | `1`, `T` | a prediction from the ranking (standing 776, walking 1250) before it finishes |
| 2 | watch the trained robot, `ENTER` | it stands: 778, 500 steps; alive and upright are what it lives on |
| 3 | `4` (+ stand_still and + air_time), `T`; then `2`, `3` | 224 steps, −0.524 m, feet up 0.08: it falls over backwards; each term alone changes nothing |
| 4 | `EXTRA = "potential"`, `1`, `T` | 778 again: the theorem kept its promise |
| 5 | `EXTRA = "velocity"`, a variant with `upright: 0`, `T` | still standing at 40k steps (529 on its reward, 779 on the original): a moved optimum is not a found one |

### `shape.py`: a variant trained, scored twice

```bash
uv run weeks/09-reward/shape.py --variant unchanged
uv run weeks/09-reward/shape.py --variant both
uv run weeks/09-reward/shape.py --variant unchanged --extra potential
uv run weeks/09-reward/shape.py --variant no_upright --extra velocity --no-viewer
```

Prints the reward as a formula, trains PPO on the variant (with the quiet
start from week 8), then evaluates the policy on the reward it was trained on
*and* on the original — the second number is the honest one — and replays it.

| Step | Does | Calls |
| --- | --- | --- |
| 1 | the reward written down | `DEFAULT_REWARD`, the variant's overrides |
| 2 | the env, wrapped with the extra term | `gym.Wrapper`, `G1WalkEnv(reward_weights=…)` |
| 3 | train with progress | `PPO(…, gamma, log_std_init=-2)`, `learn` |
| 4 | evaluate on both rewards | `predict(deterministic=True)`, `info["air_time"]` |
| 6 | replay | `launch_viewer(passive=True)` |

Measured, 40k steps, 95 s each:

| variant | extra | on its reward | on the original | steps | distance | feet up |
| --- | --- | --- | --- | --- | --- | --- |
| unchanged | none | 778.1 | 778.1 | 500 | +0.018 | 0.00 |
| both | none | 260.4 | 324.7 | 224 | −0.524 | 0.08 |
| unchanged | potential | 778.4 | 778.5 | 500 | +0.022 | 0.00 |
| no_upright | velocity | 529.4 | 779.0 | 500 | +0.026 | 0.00 |

The `both` row reproduces the deck's ablation exactly; the potential row is
the theorem; the velocity row is the caution.

## Rebuild

```bash
quarto render weeks/09-reward/slides.qmd     # ~7 minutes; it trains four times
```
