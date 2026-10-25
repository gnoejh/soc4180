# Week 22 worksheet: reference results (instructor only)

Measured 2026-10-25, ubrain 0.14.0.dev1 (vendor/ubrain), body digest 4815eedf56d0.
`lab_consolidate.py`, d 64, episodes of 8 unless stated; perception 0.93 throughout.

## Part A

| Row | Command | Hippocampus (law) | Cortex (law) | End to end |
| --- | --- | --- | --- | --- |
| A1 | default (10 episodes, 80 pushes) | 0.94 (0.90) | 0.86 (1.00) | 0.81 |
| -- | `--episodes 5` (40) | 0.90 (0.90) | 0.95 (1.00) | 0.85 |
| A2 | `--episodes 20` (160) | 0.94 (0.90) | 0.62 (1.00) | 0.59 |
| A2 | `--episodes 40` (320) | 0.94 (0.90) | 0.41 (0.96) | 0.39 |
| A3 | `--distinct --episodes 5 / 10 / 20 / 40` | 0.45 / 0.46 / 0.45 / 0.44 (0.42) | 1.00 / 1.00 / 0.94 / 0.56 (1.00 / 1.00 / 0.91 / 0.59) | 0.45 / 0.46 / 0.42 / 0.25 |
| A4 | after every boundary | recall returns None in 10 of 10 (20 of 20) episodes | | |

Laptop confirmation (no robot, 3-4 brains per point): 8 items with repeats, N 80 / 160 /
320: cortex 0.82 / 0.62 / 0.46 against law 1.00 / 1.00 / 0.96; 64 items with repeats 0.97
/ 0.87 / 0.56 (law 1.00 / 0.98 / 0.82); 512 distinct 1.00 / 0.91 / 0.57 (1.00 / 0.91 / 0.59).

- Q1: every episode is 8 pushes, so the episode's crosstalk is always that of 8; the
  cortex accumulates every replayed push, and its crosstalk grows with N.
- Q2: a consolidated push needs both stages right, and their errors are independent.
- Q3: the law counts N different pairs. The robot has 8 kinds of push: each direction is
  stored about N/8 times under different keys, and a read under any key picks up all
  copies' crosstalk aligned on the same 8 codes, so the competitors are reinforced.
- Q4: `--distinct` decodes against 512 symbols (law 0.42 at K 8, d 64), the robot against 8.
- Q5: None: the episode is cleared at the boundary; a store that hands on its contents must
  be free for the next episode.

## Part B

| Row | Command | Cortex overall | By episode (oldest → newest) |
| --- | --- | --- | --- |
| B1 | `--episodes 20 --gamma 1.0` | 0.62 | 0.75 0.88 0.62 0.38 0.50 0.75 0.62 0.50 0.88 0.62 0.88 0.50 0.88 0.50 0.38 0.38 0.75 0.50 0.88 0.38 |
| B1 | `--gamma 0.99` | 0.44 | 0.12 0.00 0.25 0.12 0.12 0.00 0.38 0.25 0.38 0.38 0.25 0.25 0.50 0.62 0.50 0.88 0.88 1.00 1.00 1.00 |
| B1 | `--gamma 0.97` | 0.24 | 0.12 0.00 0.12 0.12 0.12 0.00 0.25 0.00 0.12 0.25 0.12 0.00 0.12 0.12 0.00 0.38 0.25 0.62 1.00 1.00 |
| B2 | `--episodes 20 --d 128` | hippocampus 0.99 (0.99), cortex 0.83 (1.00), end to end 0.82 | |
| B2 | `--episodes 40 --d 128` | hippocampus 1.00, cortex 0.67 (1.00), end to end 0.66 | |
| B2 | `--episodes 20 --d 32` | hippocampus 0.78 (0.72), cortex 0.42 (0.81), end to end 0.33 | |
| -- | `--pushes 4 --episodes 40` | hippocampus 1.00 (0.99), cortex 0.61 (1.00), end to end 0.61 | |

- Q6: recency, as in working memory, but per replayed push and on a far slower clock; the
  newest episodes are kept, the oldest go.
- Q7: a world that changes makes old memories wrong; forgetting frees capacity for the
  present (the cost: nothing old is kept for long).
- Q8: no: at d 128 the cortex is at 0.83 where its law says 1.00; the repeats still cost.
  Richer items (more kinds of event, e.g. direction and strength) would bring the robot's
  world closer to the law's.
- Going further: with episodes of 4 the hippocampus is perfect (1.00) and the cortex (0.61
  at 160 pushes) limits end to end entirely.

