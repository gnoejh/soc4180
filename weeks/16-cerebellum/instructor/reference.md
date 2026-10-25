# Week 16 worksheet, Part B: reference results (instructor only)

Measured 2026-10-22 on the instructor's machine (Windows, 36 cores), `lab_cerebellum.py`
defaults unless stated: 250 N from behind, delay 10 ticks, tau 0.9, eta-brace 0.3, seed 0.
Same machine reruns are bit-identical; another laptop's X/. strings will differ (falls are
chaotic), so judge only the large contrasts.

| Row | Command | Braced | `--bracing off` | The point |
| --- | --- | --- | --- | --- |
| B1 | `--pushes 12` | `XXXXX.X..X..` 7/12 | `XXXXXXXXXXXX` 12/12 | it learns; the falls stop after ~5 trials |
| B2 | `--tone off` | `XXXXXXXXXXXX` 12/12 | (12/12) | no cue, nothing to learn: the unpaired control |
| B3 | `--delay 5` | `XXX.........` 3/12 | 12/12 | learned faster |
| B3 | `--delay 25` | `X.X.X.......` 3/12 | 12/12 | still learned, although 0.9^25 = 7 % of the trace is left |
| B4 | `--jitter 8` | `.XXXX....X..` 5/12 | — | tolerates an unreliable interval |
| B5 | `--from front` | `..........X.` 1/12 | `............` 0/12 | the robot survives a front push without help: nothing to learn |
| B5 | `--from right` | 12/12 | 12/12 | the brace cannot answer a sideways push (at 200 N it never falls) |
| B6 | `--pushes 10 --tone-only 8 --test 4` | test `...X` 1/4 | — | no extinction |
| B6 | `--pushes 10 --test 4` (control) | test `..XX` 2/4 | — | same as with tone-only trials: the difference is noise |
| B7 | `--tau 0.5` | `XXXXXXXXXXXX` 12/12 | — | 0.5^10 = 0.001: nothing left to credit |
| B7 | `--tau 0.97` | `X.X.........` 2/12 | — | the best run |
| B7 | `--delay 25 --tau 0.97` | `X.X.........` 2/12 | — | |
| B7 | `--delay 25 --tau 0.5` | `XXX.XXX.XX.X` 9/12 | — | mostly fails, as the arithmetic says |
| B8 | `--eta-brace 0.05` | `XXXXXX.X.X..` 8/12 | — | slower |
| B8 | `--eta-brace 0.9` | `X.X.X.X.X..X` 6/12 | — | alternates: overshoot, the delta rule's failure mode (cf. A6) |
| B9 | `--force 220` | `............` 0/12 | — | never falls |
| B9 | `--force 265` | `XXXXXXXXXXXX` 12/12 | 12/12 | too strong for the brace at delay 10 (at delay 25 it learns, 4/12) |

Answers worth having ready:

- B2 / Q5: the brace learns *from* the jolt (the push's effect on proprioception) *to* the
  tone (its trace). Without the tone, nothing precedes the push to attach the counter-move to.
- B5 / Q8: its output is a change of the walking command, `walk = goal + brace`. A push
  from behind can be met by changing vx; a sideways push would need a timed sidestep,
  which it does not find. From the front the walker is already robust.
- B6 / Q9: credit is assigned only when a jolt comes. A tone with no jolt gives no error to
  learn from, so the weights do not move: this brace has no extinction mechanism.
- HZ in `lab_cerebellum_walk.py` is not an experiment: the body always advances 20 ms per
  tick, so it only changes playback speed. Use it as a trick question if at all.

