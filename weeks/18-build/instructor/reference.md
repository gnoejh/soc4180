# Week 18 — reference (measured 2026-11-15, ubrain 0.46.0.dev2, the vendored body 4815eedf56d0)

## The three labs, defaults

- **lab_choose.py** (seed 1, 40 trials per brain). The brain that sees: `XX......................................`,
  right in the last half **1.00**. The twin (a fresh, unrelated picture each trial): `XXXXXX.XX..XXXX.XXXX.XXXX..XXXX.XXXX..X.`, **0.35** (7 of 20 at
  chance 0.5).
  `CHECK PASS`.
- **lab_keep.py** (20 training pushes, 10 demo pushes). Training: `..X.XX..............`; the file is **11.2 MB**.
  The loaded brain: `..........`, **0** falls. The fresh twin: `..X.XX....`, **3** falls. `CHECK PASS`.
- **lab_brace.py** (20 pushes per brain, 250 N, 0.4 m/s):

  | lead | 0.9^lead | falls |
  | --- | --- | --- |
  | 5 | 0.590 | 7 |
  | 10 | 0.349 | 3 |
  | 20 | 0.122 | 4 |
  | 40 | 0.015 | 4 |
  | none | — | 14 |

  The prediction from the trace law (falls rise with the lead) failed: the shortest lead is the worst. The cause
  is not known; it is recorded upstream for the research engine. The check that holds: no tone minus the worst
  lead is 14 − 7 = 7, which is ≥ 5. `CHECK PASS`.

## The rule from the first design of lab_choose

The first design was a warning picture before a 250 N push, then stop (0.8) or walk (1.0), with a fall paying 0:
- A walking G1 falls on 6 of 12 such pushes; a standing one on 3 of 12. These are measured, with nothing braced.
- The brain kept its first choice per picture forever: P(stop | warning) 0.00, P(stop | clear) 1.00.
- With outcome rewards (0/1) it walked on everything.

At 300 N standing does not help: 10 of 12 falls standing or walking. At 350 N, 12 of 12 either way.

## Questions F

None has been run. Accept any answer that follows the checklist: a prediction written first, one change, and the
numbers. Question 2 tests the exploration rule directly.
