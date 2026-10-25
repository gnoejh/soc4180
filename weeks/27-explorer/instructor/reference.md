# Week 27 worksheet: reference results (instructor only)

Measured 2026-10-25, ubrain 0.14.0.dev1 (vendor/ubrain), gallery body digest f787c763e67a,
CLIP openai/clip-vit-base-patch32 (3d74acf9a28c), transformers 5.20.2, CPU. The walk is
deterministic in lockstep: the same command gives the same lists (checked: the subprocess
run and an in-process probe printed identical written lists). Certificate for the 21
pictures at d 512: 1.000 at k 4-16, 0.997 at k 21.

## Part A

| Row | Command | Written | Recalled | In order | Last written |
| --- | --- | --- | --- | --- | --- |
| A1 | default | 12 | 12 of 12 | 11 of 21 | 31.6 s |
| A2 | `--steady 1` | 12 | 12 of 12 | 8 | 25.2 s |
| A3 | `--no-snap` | 12 | 6 of 12 | 8 | 31.6 s |
| A4 | `--no-fovea` | 8 | 8 of 8 | 7 | 33.6 s |

A1 written: `apple banana chair lamp laptop umbrella guitar teapot cup dog cat owl`.
A2 written: `apple laptop banana umbrella chair clock lamp guitar cup teapot dog cat`.
A3 recalled: `apple banana laptop sailboat guitar umbrella cup teapot cup shoe cat laptop`.
A4 written: `apple laptop clock guitar teapot shoe cat owl`.

- Q1: the clock: perception -- its glances (about 12 s) were recognised as the laptop three
  times running, so "laptop" was written there; the clock was recognised only on scattered
  single glances. The shoe: perception/segmentation -- never recognised at all in this walk
  (no glance names it), though A4's whole-frame view does write it.
- Q2: with STEADY 1 a single glance writes; flickers (a glance recognised as a picture
  further on, e.g. the laptop at 3 s, the umbrella at 6 s) are written early and the real
  picture, met later, is then "already in the episode". Memory is unaffected.
- Q3: that stored items ARE vocabulary codes. A raw glance sits at cosine 0.25-0.6 from its
  picture's code (the `familiar` column), so the episode holds noisy codes and the clean-up
  at recall picks wrong neighbours. Snapping splits the errors: perception's (wrong picture
  recognised) and memory's (which the certificate predicts).
- Q4: a frame often holds two panels (one each side) or a panel and the floor; CLIP codes
  the mixture, which matches neither picture well enough to be recognised three times
  running.

## Part B

| Row | Command | Written | Recalled | In order | Last written |
| --- | --- | --- | --- | --- | --- |
| B1 | default | 12 | 12 of 12 | 11 | 31.6 s, heading −5.5° then |
| B2 | `--steer 0.5` | 20 | 20 of 20 | 18 | 46.4 s |
| B3 | `--steer 2` | 20 | 20 of 20 | 17 | 46.4 s |
| B4 | `--steer 0.5 --vx 0.7` | 18 | 18 of 18 | 17 | 35.6 s |
| B4 | `--steer 0.5 --vx 0.3 --seconds 90` | 16 | 16 of 16 | 14 | 74.8 s |

Ground truth (instructor probe: the body run in-process, pelvis qpos, every 5 s): no hold,
x 7.41 / 14.99 / 20.02 / 27.51 m at 15 / 30 / 40 / 55 s, y −0.23 / −0.84 / −1.47 / −2.77 m;
`--steer 0.5`, y +0.08 / +0.04 / −0.10 / −0.49 m. Walking speed 0.505 m/s. The probe
script is not shipped (it reads the simulator directly).

- Q5: the gyro-integrated heading printed at the end (−8° in 58 s; −5.5° when the owl was
  written) and that nothing new is written after 32 s. The hold steers the yaw rate against
  the integrated gyro; it holds the heading, not the position: the sideways offset already
  built up stays, and grows slowly (−0.49 m at 55 s).
- Q6: by the probe, the sideways offset passes the panels' 0.65 m between 25 s (−0.58 m)
  and 30 s (−0.84 m); from then on the right-hand panels are on the robot's left and the
  camera looks past them. (What students see in the viewer was not recorded here.)
- Q7: no: 16 written at 0.3 m/s in 90 s against 20 at 0.5 m/s. Not explained by a
  measurement here; candidates to test: a different gait drifts differently (look at the
  final heading and the last-written time), and more glances per picture give more
  chances for a wrong three-in-a-row.

## Part C

| Row | Command | Written | Recalled | Recall rate | Certificate |
| --- | --- | --- | --- | --- | --- |
| C1 | `--steer 0.5 --d 64` | 15 | 8 of 15 | 0.53 | 0.564 at k 16 |
| C2 | `--steer 0.5 --d 128` | 19 | 14 of 19 | 0.74 | 0.717 at k 21 (0.826 at k 16) |

- Q8: the certificate speaks for memory only: given that clean vocabulary codes are
  written, how many come back. It says nothing about what gets written (perception,
  segmentation, the walk), which is where A1-A4 and B1 lose pictures.
