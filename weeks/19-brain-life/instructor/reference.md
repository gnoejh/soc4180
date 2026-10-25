# Week 19 worksheet: reference results (instructor only)

Measured 2026-10-25, ubrain 0.14.0.dev1 (vendor/ubrain), body digest 4815eedf56d0.
Same-machine reruns are bit-identical; falls are chaotic across machines.

## Part A

| Row | Command | Result |
| --- | --- | --- |
| A1 | `train` | `XXXXX.X..X..`; brace 12 trials, peak tick 9, lead 10.0; `brace.npz` 1.8 MB, 2621 ticks |
| A2 | `test` (loaded) | `.X..X.`; brace 18 trials (12 + 6) |
| A3 | `test --fresh` | `XXXXX.` |
| A4 | `identical` | state equal at the save True; 500 of 500 commands identical (largest difference 0); state equal at the end True; file 1.5 MB |
| A5 | another laptop's file | loads (same body digest); the pushes fall elsewhere (chaotic across machines) |
| A6 | `refuse` | StateError: the state was learned on mujoco/g1 (4815eedf56d0), not course/another-robot (none) |

- Q1: the brace's learned counter-move (observe: 12 trials, peak at tick 9) and its trace
  parameters come back from the file; nothing has to be relearned.
- Q2: the loaded brain falls on pushes 2 and 5, not on the training run's; even a trained
  brace saves only most pushes (week 16), and the loaded brain keeps learning from each.
- Q3: both bodies are deterministic in lockstep, both brains start from the same bytes
  (including every random generator) and see the same senses; without the generators the
  first random draw (the forward model's or a region's) would differ and the two brains
  would part within ticks.
- Q5: the brain learned what these 70 numbers mean on this robot; on another body the
  same channels carry other quantities and every prediction would be wrong silently.

## Part B

| τ | lead | push at 10 ticks (12 pushes) | push at 25 ticks |
| --- | --- | --- | --- |
| 0.5 | 2.0 | XXXXXXXXXXXX (12 fell), peak 12 | XXX.XXX.XX.X (9 fell, +1 walking fall), peak 25 |
| 0.9 | 10.0 | XXXXX.X..X.. (7 fell), peak 9 | X.X.X....... (3 fell), peak 24 |
| 0.97 | 33.3 | X.X......... (2 fell), peak 38 | X.X......... (2 fell), peak 21 |

`watch` (fresh): 2 s 0.89 False; 3 s 0.54 True; 10 s 0.06; the turn (goal wz 0.5 from
10 s): 11 s 0.57; 20 s 0.10. `watch --load brains/brace.npz --seconds 12`: 1 s 1.01
False; 2 s 0.49; 5 s 0.18; the turn from 6 s: 7 s 0.98 False, 8 s 0.61 False, 9 s 0.51.

- Q6: 1/(1−τ) = 2, 10, 33.3.
- Q7: the lead predicts whether the trace still reaches the push (how well and how fast
  it can be learned), not where the learned move peaks: at τ 0.9 the peak sits at the
  push, at 0.97 it drifts late (38) because a long trace spreads credit.
- Q8: 0.9^25 = 0.072: 7 % of the trace is left, enough to learn slowly. The lead is the
  trace's mean reach, not its end.
- Q9: the forward model predicts the body from the command; a new command is a region of
  its input it has not learned yet.
- Q10: the loaded brain's forward model learned straight walking interrupted by pushes and
  falls; it never turned, so a turn is newer to it than to a fresh brain that has just
  learned a calm straight walk (an interpretation: ask the students to test it, e.g. by
  training with turns).
- Q11: slow down or stop (a supervisor in the application); in weeks 16-18 the striatum
  could learn to choose a safer move on that flag, or the adapter could be clamped.

