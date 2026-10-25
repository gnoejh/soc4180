# Week 24 worksheet: reference results (instructor only)

Measured 2026-10-25, ubrain 0.14.0.dev1 (vendor/ubrain), body digest 4815eedf56d0.
`lab_habit.py` on the cached world (`traces/world.json`, identical to week 18's 40-phase
map: soon 0.85 / 0.25 / 0.55, late 0.475 / 0.85 / 0.55; always right 0.85, matching 0.663,
a coin 0.588). 20 seeds of 200 trials unless stated. Buckets: times the cue had been seen.

## Part A

| Row | Command | Brain | 1 | 2-3 | 4-10 | 11-30 | 31-60 | 61+ | all | Handed to memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | default | striatum | 0.65 | 0.64 | 0.55 | 0.62 | 0.65 | 0.66 | 0.642 | |
| A1 | | memory | 0.70 | 0.66 | 0.69 | 0.78 | 0.80 | 0.81 | 0.789 | |
| A1 | | chooser | 0.65 | 0.71 | 0.67 | 0.80 | 0.81 | 0.82 | 0.799 | 0.95 |
| A2 | `--trials 40` | striatum / memory / chooser | | | | | | | 0.599 / 0.724 / 0.730 | 0.87 |
| A3 | `--gamma 0.9` | memory / chooser late | | | | | | 0.83 / 0.83 | 0.807 / 0.804 | 0.94 |
| A3 | `--gamma 0.8` | memory / chooser late | | | | | | 0.79 / 0.78 | 0.766 / 0.771 | 0.89 |
| A4 | `--live --brain memory --trials 20` | memory, on the robot | `...X..X....X.....X..` (4 falls of 20) | | | | | | | |

40 seeds (default otherwise): striatum 0.644, memory 0.781, chooser 0.806 (0.95 to memory).

- Q1: the striatum's values approach the payoffs and its race chooses near matching
  (0.663); memory takes the move with the best recalled outcome, greedily, so it picks the
  right brake almost every time (0.85 is the ceiling).
- Q2: nobody knows anything yet: each move is equally likely, so the robot stays up at the
  world's mean, 0.588 (40 samples per bucket per brain: ±0.08).
- Q3: with 2 cues and 3 moves the store holds 6 pairs; the right brake's votes are
  refreshed every few trials, so a faster fade still leaves them ahead.
- Q4: the falls left are mostly the right brake failing at a bad stride point (0.85, not
  1.00); memory switches to the right move after one or two outcomes.

## Part B

Laptop bandit, 2 actions paying 0.7 / 0.3 (always best 0.70, matching 0.58), late (61+):

| Situations | Striatum | Memory | Chooser | Handed to memory |
| --- | --- | --- | --- | --- |
| 2 (4000 trials, 10 seeds) | 0.59 | 0.70 | 0.70 | 1.00 |
| 16 | 0.58 | 0.68 | 0.67 | 0.84 |
| 64 | 0.57 | 0.61 | 0.61 | 0.60 |
| 256 (20000 trials, 5 seeds) | 0.58 | 0.54 | 0.57 | 0.42 |

256 situations by bucket: striatum 0.50 / 0.51 / 0.52 / 0.55 / 0.57 / 0.58; memory 0.48 /
0.54 / 0.54 / 0.54 / 0.54 / 0.54; chooser 0.49 / 0.52 / 0.54 / 0.55 / 0.57 / 0.57 (memory
ahead on visits 2-10, the striatum from 11-30 on: ubrain's lab 9).

Robot world, small memories: d 16: memory 0.790, chooser 0.757 (0.69 to memory); d 8:
memory 0.793, chooser 0.727 (0.61).

- Q5: every (situation, action, outcome) shares one vector; the crosstalk grows with the
  number of pairs, and a vote is drowned; the striatum keeps a separate value per pair.
- Q6: a small store's votes rarely clear its own crosstalk scale, so the chooser hands to
  the striatum; on this task memory is still the better controller, so each hand-over
  costs reward.
- Q7: they come apart at small d on the robot (memory right but not separable); they agree
  at 256 situations, where memory is neither separable nor right and the chooser follows
  the striatum.
- Q8: measure memory's late reward against the number of situations the robot meets; with
  thousands, habit (or a chooser that hands over) wins late.

