# Week 20 worksheet: reference results (instructor only)

Measured 2026-10-25, ubrain 0.14.0.dev1 (vendor/ubrain), body digest 4815eedf56d0.
`lab_remember.py`, 10 episodes unless stated. Same-machine reruns are bit-identical.

## Part A -- the sense

| Row | Command | Perception |
| --- | --- | --- |
| A1 | default (100 N, 8 directions) | 0.90 (8 pushes/ep.), 0.95 (16), 0.91 (24) |
| A2 | `--force 60` | 0.57 |
| A3 | `--force 60 --directions 4` | 1.00 |
| A4 | torso accelerometer instead of the pelvis (pilot, 64 heading-frame pushes, 100 N) | 0.83 against the pelvis's 0.92 |

Pilot, world-frame pushes (the first design): a constant +45° offset (heading drift: each
push turns the robot, and nothing resets its heading); 0.37 right of 8 at 150 N, 0.44 at
200 N. Heading-frame pushes (a 5-tick field): offset 0 to −2°.

- Q1: errors go to neighbouring directions (45° off): the angle is noisy, not reversed.
  Seed 3, episode 3: BR read as B, FR read as F.
- Q2: the walking robot's accelerometer swings by several m/s² at every step (week 6:
  3.4-19.3 m/s²); 60 N on 33 kg is 1.8 m/s², 100 N 3 m/s², close to that noise.
- Q3: with 4 directions a reading must be 45° off to be wrong instead of 22.5°.
- Q4: pushes in world coordinates come from a fixed world direction, but the robot's
  heading drifts (each push turns it) and is not observable (week 6), so the body-frame
  reading rotates with the drift.

## Part B -- the memory

| D | pushes | perception | memory | certificate | both |
| --- | --- | --- | --- | --- | --- |
| 64 | 8 | 0.90 | 0.94 | 0.94 | 0.88 |
| 64 | 16 | 0.95 | 0.79 | 0.76 | 0.76 |
| 64 | 24 | 0.91 | 0.67 | 0.64 | 0.62 |
| 32 | 8 | 0.90 | 0.71 | 0.76 | 0.64 |
| 32 | 16 | 0.95 | 0.53 | 0.56 | 0.51 |
| 32 | 24 | 0.91 | 0.46 | 0.46 | 0.43 |
| 64, 4 dirs | 24 | 1.00 | 0.78 | 0.81 | 0.78 |
| 64, 60 N | 8 | 0.57 | 0.91 | 0.94 | 0.53 |
| 64, 60 N, 4 dirs | 8 | 1.00 | 0.97 | 0.98 | 0.97 |

Certificate (`--certify`), D 64: 1.00 / 0.94 / 0.84 / 0.76 / 0.64 at K 4 / 8 / 12 / 16 /
24; D 32: 0.95 / 0.76 / 0.61 / 0.56 / 0.46. Laptop check (no robot, 200 episodes per
point): recall is the same for the first slot, the last slot and all slots, and matches
the certificate (D 64, K 16: first 0.81, all 0.76, last 0.72, certificate 0.76).

B5: 24 pushes as 3 episodes of 8: memory 1.00 (24 pushes; 0.94 over 80); as one episode
of 24: 0.62 (0.67 over 240).

- Q5: within a few hundredths; the binomial error of 80 pushes at 0.94 is ±0.03, of 240
  at 0.65 ±0.03.
- Q6: at D 32 every episode length loses about a fifth; capacity at 90 % roughly halves
  (D 64: about 9-10 pushes; D 32: about 5).
- Q7: the hippocampus stores the percept, the code the sense delivered; it has no access
  to the push itself, so it cannot know a percept was wrong.
- Q8: the errors are independent: memory fails as often on correctly perceived pushes as
  on misperceived ones.
- Q9: each episode is held in its own vector, so the crosstalk is that of 8 items, not 24;
  what is lost is the order and relation ACROSS episodes (which push came before which,
  across a boundary), and any recall that needs the whole day at once. Consolidation
  (replay into a slow store) is the brain's answer -- a later week.
- Q10: a wrong direction must beat fewer competitors (3 instead of 7).

