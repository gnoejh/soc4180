# Week 23 worksheet: reference results (instructor only)

Measured 2026-10-25, ubrain 0.14.0.dev1 (vendor/ubrain), body digest 4815eedf56d0.
`lab_notice.py`: 20 bursts of 6 pushes, 8 quiet moments, d 64, window 8, α 0.05 unless
stated. Perception (week 20's sense) about 0.93.

## Part A

| Row | Command | Felt within window: familiar | Not felt lately: familiar (predicted) | Written | Executive | Cortex (of consolidated) |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | default | 1.00 (75) | 0.00 (0.03) | 45 of 120 | 0.62 | 0.98 (45 of 45) |
| A2 | `--world gusty` | 1.00 (91) | 0.00 (0.02) | 29 of 120 | 0.45 | 0.93 (29 of 29) |
| A3 | `--alpha 0.3` | 1.00 (75) | 0.18 (0.19) | 37 of 120 | 0.55 | 0.95 (37 of 37) |
| A4 | `--window 2` | 1.00 | | 0.73 of pushes | 0.71 | 0.77 |
| A4 | `--window 4` | 1.00 | | 0.58 | 0.71 | 0.81 |
| A4 | `--window 16` | 1.00 (101) | 0.00 (0.02) | 0.16 (19) | 0.27 | 0.95 |
| -- | `--world gusty --window 16` | 1.00 (101) | 0.00 | 19 of 120 | 0.30 | 1.00 |

Thresholds: θ = 0.693 at d 64, window 8, α 0.05; 0.786 at α 0.3; 0.757 at window 2; 0.665
at window 16; `familiar_threshold(d, 8, 0.05)` = 0.571 / 0.693 / 0.890 at d 32 / 64 / 512.

- Q1: both: it saves capacity in a repetitive world (a steady wind is not news), but a
  repeat that matters (the same push, now dangerous) is not written either.
- Q2: the prediction averages over random codes; with 8 FIXED codes each pair of
  directions either lies within the threshold or not, every time. At d 64, α 0.05 none
  does; at α 0.3 the threshold is higher and the robot meets the prediction (0.18 vs 0.19).
- Q3: 18 % of fresh directions lost; bought: fewer writes (37 vs 45), a quieter brain.
- Q4: a short window writes repeats as novel (73 % written), crowding the episodes (replays
  of up to 31 with 4 quiet moments) and the cortex with copies; a long window skips a
  direction for as long as it remembers it (16 pushes), so a change back to an old side is
  missed.

## Part B

| Row | Command | Executive | Replays (mean size) | Consolidated | Cortex |
| --- | --- | --- | --- | --- | --- |
| B2 | `--quiet 4` | 0.89 | 9 (2.6) | 23 of 46 | 0.87 |
| A1 | `--quiet 8` | 0.62 | 19 (2.4) | 45 of 45 | 0.98 |
| B3 | `--quiet 12` | 0.49 | 19 (2.4) | 45 of 45 | 0.98 |
| -- | `--window 2 --quiet 4` | 0.99 | 2 (22.0, largest 31) | 44 of 86 | 0.43 |

- Q5: each novel push is a demand served for k = 4 executive moments.
- Q6: with 4 quiet moments the controller is often still executive when the next burst
  starts, so the episode is not ended; the pushes written since the last replay were still
  in the hippocampus when the run ended.
- Q7: 8 moments already give a default-mode moment after every burst (19 replays for 20
  bursts); more rest only lengthens idle time.
- Q8: fewer novel pushes are fewer demands (29 vs 45), so less executive time.

