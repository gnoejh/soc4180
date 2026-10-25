# Week 18 worksheet: reference results (instructor only)

Measured 2026-10-25 on the instructor's machine, ubrain 0.14.0.dev1 (vendor/ubrain),
body digest 4815eedf56d0. Same-machine reruns are bit-identical (the body runs in
lockstep, the trial sequence and the race are seeded); a fall is chaotic, so another
laptop's trial-by-trial outcomes differ and only the means should be compared.

## Part A (lab_striatum.py, 6000 trials, last third)

| Row | Command | Best action chosen (per situation) | Matching | Reward | Matching's reward | Best |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | default | 0.64 / 0.62 / 0.66 | 0.62 / 0.64 / 0.60 | 0.624 | 0.600 | 0.800 |
| A2 | `--world easy` | 0.93 / 0.88 / 0.89 | 0.82 | 0.820 | 0.755 | 0.900 |
| A3 | `--world close` | 0.39 / 0.40 / 0.47 | 0.40 / 0.40 / 0.40 | 0.530 | 0.513 | 0.600 |
| A4 | `--reverse 3000` | 0.60 / 0.62 / 0.56 (of the new best) | 0.62 / 0.64 / 0.60 | 0.595 | 0.600 | 0.800 |
| A5 | `--eta 0.01` | 0.60 / 0.60 / 0.63; values within 0.05 of p | | 0.609 | 0.600 | 0.800 |

- Q2: an action chosen rarely gets few outcomes, so its value moves rarely and stays near
  where the last few bad outcomes left it: the race picks it even less. With one action
  clearly best, that feedback makes choice more exclusive than matching.
- Q3: yes: after the swap the values end at the new payoffs (e.g. situation 2: 0.95 /
  0.21 / 0.12 against 0.90 / 0.30 / 0.30) and the choices at matching for the new table.

## Part B (lab_choose.py)

The world (no brain). `--map --phases 40` (the deck's table):

|      | brake early | brake late | keep walking |
| --- | --- | --- | --- |
| soon | 0.85 | 0.25 | 0.55 |
| late | 0.475 | 0.85 | 0.55 |

The default `--map` uses 10 phases (every 4th) and prints a coarser table: soon 0.90 /
0.20 / 0.40, late 0.50 / 0.70 / 0.60. Same ranking. At 230 N (`--map --force 230`, 10
phases): soon 1.00 / 0.90 / 1.00, late 0.80 / 1.00 / 1.00. At 270 N (20 phases): soon 0.30
/ 0.00 / 0.20, late 0.15 / 0.40 / 0.15.

| Row | Command | Result |
| --- | --- | --- |
| B2 | default (120 trials, eta 0.2), seeds 0 / 1 / 2 | stayed up, last third 0.65 / 0.80 / 0.75; values e.g. seed 1: soon 0.95 0.16 0.71, late 0.36 0.97 0.49 |
| B3 | `--trials 300`, seeds 0-3 | right brake chosen 0.50 / 0.53 / 0.48 / 0.42 (mean 0.48; matching 0.48); stayed up 0.72 / 0.58 / 0.67 / 0.62 (mean 0.65; matching 0.66) |
| B4 | `--force 230`, seeds 0 / 1 | values near 1 for every move but the rarely tried; choices spread; stayed up 1.00 / 0.95 |

- Q4: a brake that comes before the push has ended is what the trained brace did (vx
  −0.39 exactly while the push lands); a brake 0.3 s off target slows the robot when the
  push is not there and leaves it at full speed when it is.
- Q5: the furthest-off value is usually the rarely chosen one (a move tried a few times
  keeps whatever those few outcomes said). In the deck's trace (eta 0.05, seed 1) soon /
  brake late was chosen 23 times of 144, against 72 and 49.
- Q6: no: at eta 0.2 the race stays near matching for good (a quarter of the choices are
  the wrong brake); it stays up 0.65, not 0.85. The trial outcomes themselves match the map
  (seed 0: soon/early 71 of 87 = 0.82, late/late 57 of 69 = 0.83, soon/late 4 of 23 = 0.17).
- Q7: at 230 N almost everything survives; with no difference in the outcomes there is
  nothing to learn, and nothing to lose by choosing badly.

## Part C (300 trials, last third, mean of seeds 0-3)

| eta | mean abs Q - world | right brake chosen | stayed up |
| --- | --- | --- | --- |
| 0.05 | 0.06 | 0.46 | 0.645 |
| 0.2 | 0.16 | 0.48 | 0.647 |
| 0.5 | 0.27 | 0.72 | 0.745 |
| 0.8 | 0.36 | 0.87 | 0.788 (0.87 / 0.69 / 0.85 / 0.74) |
| 1.0 | 0.44 | 0.57 | 0.680 |
| 0.2, `--blind` | | | 0.60 (0.64 / 0.58 / 0.64 / 0.54); matching predicts 0.59 |

- Q8/Q9: slow learners know the world best and choose near matching. A fast learner's
  value drops most of the way to 0 after one fall and rises near 1 after a success, so the
  values sit far apart and the race is nearly exclusive (win-stay, lose-shift).
- Q10: at eta 1 a value IS the last outcome; every move that worked last time ties at 1 and
  the race splits evenly between them.
- Q11: blind, one value per move for both cues: early 0.66, late 0.55, keep 0.55 on
  average; matching on those earns 0.59. The cue is worth about 0.06 under matching (more
  for a decisive learner). In week 16 the cue was the brace's clock, so without it nothing
  was learned (12 of 12 fell).
- Q12: an error says which way to change; a reward only says whether the last try was good,
  so the learner must try the alternatives, and a race that keeps choosing near matching
  keeps exploring.

