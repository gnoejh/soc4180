# 12 — Robustness

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/12-robustness/lab.ipynb)

| | |
| --- | --- |
| **Layer** | 5 — learning; the sim-to-real gap, measured in simulation |
| **Runtime** | CPU. Colab: a GPU runtime only for rendering |
| **Wall clock** | ~3 min (evaluation only — nothing trains in the deck) |
| **Needs** | `soc4180[rl]`; the shipped checkpoints in `checkpoints/` |

## Objectives

1. Train a push-recovery policy with PPO, applying week 8 (`log_std_init = −2`) and
   week 10 (`SubprocVecEnv`).
2. Measure a policy honestly: fixed shoves, fixed worlds, counted survivals.
3. Show the sim-to-real gap without a robot: a policy meets physics it did not train in.
4. Explain domain randomisation, and measure what it bought here.

## The environment

`soc4180.envs.G1PushEnv`: `G1WalkEnv` with target velocity 0; once per 5 s episode a
shove of up to `push_max` (200 N) in a random horizontal direction for 0.2 s at a random
moment in 1–3 s; per episode a world drawn from `world` — friction scale, extra torso
mass, servo stiffness (`kp`, damping scaled by √kp to keep ζ). `NOMINAL_WORLD` fixes
them; `RANDOM_WORLD` is friction 0.3–1.2, mass −5…+10 kg, kp × 0.7–1.3.
`privileged=True` appends nine numbers no robot can measure (a teacher's view).

## The checkpoints

| file | trained on | 3 M steps |
| --- | --- | --- |
| `checkpoints/push_nominal.zip` (+ `.csv` learning curve) | nominal world | 967 s on 36 cores |
| `checkpoints/push_random.zip` (+ `.csv`) | `RANDOM_WORLD` | 975 s |

PPO: `n_steps` 256 × 16 envs, batch 1024, 5 epochs, lr 3e-4, γ 0.99, `net_arch`
[128, 128], `log_std_init` −2, seed 0. Mean episode length climbs for ~1 M steps and
flattens near 170 of 250 — shoves up to 200 N include many no stance survives.
`robust.py --train 3000000 --world random` reproduces one (576 kB).

## Measured results (`robust.py`, 10 episodes per cell, shoves from the right)

Largest shove survived by 10 of 10 unless stated:

| world | hold | PPO nominal | PPO random |
| --- | --- | --- | --- |
| nominal | 60 N | 80 N | **100 N** (9/10) |
| ice (friction 0.2) | 60 | 60 (9/10 at 80) | **100** |
| heavy (+15 kg) | **falls with no shove** | 60 (8/10 at 80) | **120** |
| light (−5 kg) | 60 | 60 (3/10 at 80) | 60 (9/10 at 80) |
| weak servos (kp × 0.6) | falls at 60 | 60 (3/10 at 80) | **80** |
| stiff servos (kp × 1.5) | 80 | 60 (8/10 at 80) | 80 |

Randomising helped in every world, including +15 kg (outside the trained −5…+10), and
cost nothing in the nominal world — here it acted as a regulariser. On 20 worlds drawn
from friction 0.2–1.0, mass −5…+15, kp × 0.6–1.5, one 80 N shove each: hold 2 of 20,
PPO nominal 14, PPO random 18.

A privileged teacher (`--privileged`, same budget) was trained and was **worse** in
the nominal world (3/10 at 60 N) than either policy; it is not shipped, and making one
that helps is week 13's exercise 5.

## Lab class: on your laptop

```bash
uv run weeks/12-robustness/lab_robust.py          # the game: edit audit.py
uv run weeks/12-robustness/lab_robust.py --grade --no-viewer
```

**The robustness audit.** Students measure with `robust.py` and write answers in
`audit.py`; the referee re-measures each one live at 3× speed. A limit L passes if the
controller survives ≥ 7/10 at L and < 7/10 at L + 20. As shipped 0 / 5; the reference
(`instructor/audit_solution.py`, sealed) 5 / 5; claiming 70 N for hold fails (0/10),
shipping the nominal policy fails (14 of 20).

| # | Question | Reference |
| --- | --- | --- |
| 1–3 | `HOLD_LIMIT`, `NOMINAL_LIMIT`, `RANDOM_LIMIT` (nominal world) | 60, 80, 100 |
| 4 | `HEAVY`: best controller at +15 kg and its limit (≥ 100 N) | random, 120 |
| 5 | `SHIP`: one controller for 20 unknown robots, ≥ 16 survive the promise (≥ 80 N) | random, 80 |

### `robust.py`: the instrument

```bash
uv run weeks/12-robustness/robust.py                              # evaluate push_nominal
uv run weeks/12-robustness/robust.py --policy random
uv run weeks/12-robustness/robust.py --policy hold --any-direction
uv run weeks/12-robustness/robust.py --train 300000 --world random --envs 8   # your own, a few minutes
```

## Rebuild

```bash
quarto render weeks/12-robustness/slides.qmd
```
