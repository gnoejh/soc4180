# Week 25 worksheet: reference results (instructor only)

Measured 2026-10-25, ubrain 0.14.0.dev1 (vendor/ubrain). Serotonin runs on the G1's
measured world (`traces/world.json`, week 24's; cue soon, keep walking: up at 22 of 40
stride points, p = 0.55); 2400 trials per arm, the second half measured. Noradrenaline is
ubrain's example 10 (laptop).

## Part A -- serotonin

Law: P(walk) = 0.55 g^8 / (r_S g^2 + 0.55 g^8), g = e^(−1/τ). A1: τ 2: g 0.607, P 0.064;
τ 10: g 0.905, P 0.430. Indifference (P = 0.5) at τ 18.8 for r_S 0.4 (measured at τ 30:
0.528, law 0.530).

| r_S (stop pays) | τ 1.5 | 2 | 3 | 4 | 6 | 10 | 20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4: P(walk on) measured | 0.026 | 0.068 | 0.165 | 0.242 | 0.358 | 0.392 | 0.469 |
| 0.4: reward per moment | 0.098 | 0.093 | 0.087 | 0.081 | 0.075 | 0.072 | 0.069 |
| 0.1: P(walk on) measured | 0.092 | 0.217 | 0.423 | 0.532 | 0.641 | 0.705 | 0.764 |
| 0.1: reward per moment | 0.031 | 0.038 | 0.044 | 0.047 | 0.047 | 0.049 | 0.050 |

Defaults (`serotonin`): τ 2: law 0.064, measured 0.068 (±0.007), 0.093 per moment; τ 10:
law 0.430, measured 0.392 (±0.014), 0.072; adaptive (κ 0.02): law 0.337 at τ* 6.02,
measured 0.303, τ settled at 5.85 (sd 0.35), 0.075.
`--small 0.2`: τ 2 0.118 (law 0.120), 0.051; τ 10 0.530 (0.601), 0.051; adaptive τ 7.08
(predicted 7.28). `--small 0.1`: τ 2 0.217 (0.215), 0.038; τ 10 0.705 (0.751), 0.049;
adaptive τ 8.08 (8.37).

- Q1: a larger τ discounts the late reward less (g^8 nearer 1), so walking on's value rises
  toward its mean and the race picks it more often.
- Q2: per moment stopping pays 0.4 / 4 = 0.100 against walking on's 0.55 / 10 = 0.055: the
  impatient robot is right for a world that counts reward per second.
- Q3: patience moves choice toward the large, late option; it pays where that option is
  better per unit of time (r_S 0.1), costs where it is not (r_S 0.4), and is neutral near
  the tie (r_S 0.2).
- Q4: E[Q/(Q_S + Q)] < E[Q]/(Q_S + E[Q]) for a concave function (Jensen): a noisy value
  is chosen less than its mean would be. At low τ the walk-on value is tiny anyway.
- Q5: τ tracks the time between decisions, which depends on the choices τ produces; it
  settles where the two agree (a fixed point).

## Part B -- noradrenaline

| Arm | Criterion | Error law | Measured |
| --- | --- | --- | --- |
| fixed 0.1, stream 0.1 | 0.815 | 0.017 | 0.019 (±0.002) |
| fixed 0.5, stream 0.5 | 0.790 | 0.026 | 0.024 (±0.003) |
| fixed 0.1, stream changes to 0.5 | 0.815 | 0.049 | 0.051 (±0.006) |
| adaptive η 0.02, after the change | 0.790 | 0.026 | 0.025 (±0.004) |
| adaptive η 0.1 (B3) | 0.995 | 0.500 | 0.492 (±0.013) |

Adaptive level after the change: predicted 0.503, measured 0.501 (sd 0.048); at η 0.1:
predicted 0.000, measured 0.000.

- Q6: new items missed: a criterion tuned for a rare-novelty world is strict, so with half
  the stream new it calls many new items familiar.
- Q7: the level is the running rate of JUDGED surprise; the criterion it sets decides what
  is judged novel; it settles where the judged rate under its own criterion equals itself.
- Q8: a fast level follows its own verdicts: judging less novel lowers the level, a lower
  level raises the criterion (stricter), which judges less novel -- down to 0.
- Q9: noradrenaline, slowly (η 0.02): it recovers the right criterion after a change without
  running away; serotonin if the time between decisions changed.

