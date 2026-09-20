# Week 9 — Reward Shaping

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w09-reward/lab.ipynb)

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

## Rebuild

```bash
quarto render weeks/w09-reward/slides.qmd     # ~7 minutes; it trains four times
```
