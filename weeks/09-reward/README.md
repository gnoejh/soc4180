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
uv run weeks/09-reward/lab_reward.py       # torch + SB3: uv sync --extra rl
```

`G1WalkEnv` in the viewer under a PPO policy, with `VARIANTS` — the reward
weightings from the lecture, as `reward_weights` overrides — selectable with
keys `1`–`9`. **`T` trains the selected variant in a background thread** (40k
steps, `log_std_init = -2`, roughly two minutes on a laptop CPU) while the
window keeps playing; the trained policy takes over when it finishes.

```
1..9    select a variant (trained if trained)     T   train it (background)
R       reset      SPACE   pause      ENTER   every reward term this step, weighted
```

Drawn on the robot: a reward bar above the head (green while tracking is being
earned, orange when only alive) and **yellow feet** whenever the env counts a
foot airborne — the `air_time` term made visible. Each episode ends with
return, steps, distance and the feet-up fraction, the three columns of the
lecture's table.

`extra_reward(env, info, x_before, x_after)` is the student's own term, added
through a `gym.Wrapper` on top of the variant, and returns 0 as shipped.

| Step | Change | Right looks like |
| --- | --- | --- |
| 1 | `1`, `T` | a return predicted from the hand ranking before training ends |
| 2 | watch a full trained episode | standing; `ENTER` shows alive + upright carrying the score |
| 3 | `4`, `T`; then `2`, `3` | only "+ both" leaves the standing pose, and it goes backwards |
| 4 | potential-based term with $\Phi(s) = x$, `1`, `T` | behaviour unchanged; the telescoping identity quoted |
| 5 | pay for torso $x$-velocity with `upright` at 0, `T` | the student watches the whole episode before describing it |

Training runs are single-seed and short, exactly like the lecture's: the
direction of each result is the lesson, the exact numbers are not.

`SOC4180_AUTOCLOSE=8 uv run weeks/09-reward/lab_reward.py` closes the window
by itself, which is how the script is smoke-tested (training is not triggered).

## Rebuild

```bash
quarto render weeks/09-reward/slides.qmd     # ~7 minutes; it trains four times
```
