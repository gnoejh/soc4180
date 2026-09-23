# 03 — Inverse Kinematics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/03-inverse-kinematics/lab.ipynb)

| | |
| --- | --- |
| **Runtime** | Local: CPU. **Colab: pick a GPU runtime** (Runtime > Change runtime type > T4). |
| **Wall clock** | ~1 min |
| **Convergence risk** | None. No learning. |
| **Feeds** | 04 — the walker calls this solver 500 times a second |

## Objectives

1. Solve planar two-link IK in closed form and read off its three lessons:
   reach is an **annulus**, there are **two** solutions, and sensitivity diverges
   as the leg straightens.
2. Explain why closed-form IK does not scale to a 6-DOF foot pose.
3. Derive and use damped least squares, and say what $\lambda$ trades away.
4. **Diagnose a singularity** from the Jacobian's singular values.
5. Justify warm starting and per-iteration joint clamping.

## The two numbers that carry the week

Measured on the G1's left leg:

- **Damping.** Asking the solver to lower the foot 5 cm at the `stand` pose gives
  `|dq| = 55778 rad` with $\lambda = 0$, and `0.04 rad` with $\lambda = 10^{-3}$.
- **Seeding.** The identical request for an 8 cm squat: residual error
  **7.9e-01 from the straight-leg seed** versus **2.2e-06 from the crouch**. Same
  solver, same target, and one of them simply cannot be solved.

The `stand` keyframe has every leg joint at exactly zero, so the Jacobian's
smallest singular value is ~9e-07 and its condition number ~2e+06. This is why
every gait in the course starts crouched.

## Reachability, shown not asserted

The circle demo tracks to **8.4e-05 m**. Stretching it 2 cm taller pushes its
lowest point outside the leg's reach, and the residual jumps to **2.68e-03 m at
279°** — and stays there at 12, 40 or 120 iterations.

That invariance is the diagnostic worth teaching: **a residual that will not
shrink with more iterations is a geometry problem, not a solver problem.**

## Note on the demo

The circle-tracing demo renders with `render_poses` — **kinematics only, physics
never stepped**. IK is a statement about geometry; a pose it solves perfectly may
be one the robot cannot hold. Keeping that separate here is what lets Week 4
combine them deliberately.

## Lab class: on your laptop

```bash
uv run weeks/03-inverse-kinematics/lab_ik.py
```

The G1 frozen in the crouch, a **red** target sphere moved with the arrow keys
(`,` and `.` for sideways), a **green** sphere at the left foot site, and a
large translucent sphere centred on the hip showing the leg's reach, measured
from the model (thigh 0.3409 + shin 0.3000 + ankle-to-site 0.0176 = 0.6585 m).
Physics is never stepped and the pelvis never moves.

```
arrows / , .   move the target          [ ]      lambda ten times smaller / larger
M              next solver: dls -> inverse -> transpose
I              iterations per frame 3 -> 10 -> 1
O              position only (3 rows) <-> position + orientation (6 rows)
S              straight-leg <-> crouch seed      R   target back onto the foot
SPACE          next entry of TARGETS             ENTER   residual, |dq|, condition number, angles
```

**The code is complete and explained, top to bottom.** Nothing is left blank:
the exercise is to run the lecture's mathematics against a real leg, break it,
and explain what happened. The script is in five numbered sections:

| Section | What it holds | What it explains |
| --- | --- | --- |
| 1 | `TARGETS`, `SIDE`, `STEP`, `ITERATION_CHOICES` | the settings students change |
| 2 | `dls_step`, `inverse_step`, `transpose_step` | the three solvers, each a few lines, each commented against the slide it comes from: why $(JJ^{\mathsf T} + \lambda^2 I)$ always inverts, why `solve` not `inv`, why the transpose sits outside, where the transpose method's step length comes from |
| 3 | `leg_lengths`, `sphere` | why lengths are measured between joint anchors, and how the markers are drawn |
| 4 | `class Leg` | `qpos` vs `qvel` indices (the off-by-one trap), what `mj_jacSite` writes and returns, what `pose_error` returns and why a rotation vector, why the clip to joint limits, what `mj_forward` recomputes |
| 5 | `main` | the passive viewer, the key callback shared with the terminal, the per-frame iteration, what ENTER prints |

Eight experiments in the docstring, each with the numbers it should produce
(all measured with the script's own functions before they were written):

| # | Do | See |
| --- | --- | --- |
| 1 | dls, up arrow ×5 | residual 4e-6 after one frame, 1e-10 after two; at 1 it/frame 1.4e-2 then 5e-4 |
| 2 | `M` inverse, `S` straight leg, arrow **up** (1 cm: bend the knee) | \|dq\| = 1.1e4 rad; the clamp is all that stops it; the leg folds onto its limits and never recovers |
| 3 | dls, straight leg, sweep $\lambda$ | 1e-6 folds like the inverse; 1e-3 and 1e-2 take a tiny first step then blow up as the knee leaves zero (60 rad by iteration 10); 1e-1 creeps the knee the *wrong* way to its backward limit; 1 never moves. No $\lambda$ brings the foot up: bend the knee first |
| 4 | `M` transpose, up ×5 | creeps: 8 mm of 50 left after 100 iterations; at the straight leg \|dq\| = 1.5e-5, nothing explodes |
| 5 | right arrow ×30 | hip→target 0.69 m of 0.66 m reach; residual hovers at 5–6 cm, knee and ankle on their limits; `O` and 0.25 m pins at exactly 6.8e-3 |
| 6 | `I` to 1/frame, `]` to $\lambda = 1$ | 3.1 cm of 5 left after 50 frames; 0.3 closes in 50, 0.1 in 10 |
| 7 | `O`, `.` ×15 and up ×5 | foot tilts 16° with 3 rows, ankle roll stays 0; 0.2° with 6 rows, ankle roll −0.26 rad |
| 8 | two new `TARGETS`, one unreachable | the prediction is right before SPACE is pressed |

Then break something: `-dq` from `dls_step` runs the residual to 0.57 m and
sticks on the joint limits; `J.T @ err` without the step length in
`transpose_step` overshoots and diverges (0.050, 0.049, … 0.32, 0.59 in ten
iterations).

`SOC4180_AUTOCLOSE=4 uv run weeks/03-inverse-kinematics/lab_ik.py` closes the
window after four seconds, which is how the script is smoke-tested.

### `reach.py`: the pipeline as a script you read top to bottom

```bash
uv run weeks/03-inverse-kinematics/reach.py --target 0.10 0 0.05
uv run weeks/03-inverse-kinematics/reach.py --leg right --target 0 -0.1 0 --target 0.1 -0.1 0.1
uv run weeks/03-inverse-kinematics/reach.py --target 0 0 0.01 --seed straight --solver inverse --trace
uv run weeks/03-inverse-kinematics/reach.py --target 0.3 0 0 --trace --jacobian --no-viewer
```

Where `lab_ik.py` is interactive, `reach.py` is linear: give it one or more
targets (offsets from the foot's start, metres, x forward / y left / z up), it
solves them one after another, prints what happened, then opens the simulator
and animates the leg through the solved poses. `--trace` prints every
iteration (residual, |dq|, the six angles), `--jacobian` prints $J$ and its
singular values at the seed, `--no-viewer` skips the window (what Colab or an
autograder sees). Six numbered steps, each calling MuJoCo where MuJoCo is
needed and numpy where it is not:

| Step | Code | MuJoCo / package call |
| --- | --- | --- |
| 1 | load the robot, pick the seed pose | `soc4180.load_g1()`, `WalkingController.initial_data()`, `mj_forward` |
| 2 | find the leg's numbers: `qpos` slice, `qvel` slice, foot site, joint limits | `kin.leg_qpos_indices`, `kin.leg_dof_indices`, `kin.foot_site_id`, `model.jnt_range` |
| 3 | per iteration: error → Jacobian → step → clip → forward | `kin.pose_error`, `mj_jacSite`, `dls_step`/`inverse_step`, `np.clip`, `mj_forward` |
| 4 | report: foot, residual, tilt, angles, limits hit | `site_xpos`, `site_xmat` |
| 5 | stop here without a window | |
| 6 | animate pose to pose in the passive viewer | `soc4180.launch_viewer(passive=True)`, `viewer.sync()` |

The first example converges in five iterations (1.6 ms) with a residual of
6e-10 m; the third shows the inverse at the singularity asking for 1.1e4 rad
and folding the leg; the fourth shows a target 0.69 m from the hip against a
0.66 m reach and the residual it leaves. Both scripts are in the same
directory on purpose: read `reach.py` first, then use `lab_ik.py` to play.

### `lab_connected.py`: the lab game, scored out of 10

```bash
uv run weeks/03-inverse-kinematics/lab_connected.py            # play; students edit moves.py only
uv run weeks/03-inverse-kinematics/lab_connected.py --joints   # the 29 joint names, ranges, stand angles
uv run weeks/03-inverse-kinematics/lab_connected.py --grade --no-viewer   # the same referee, headless
```

Ten problems (wave, sway, twist, squat, bow, clap, both hands up, squat-wave,
twist-wave, boss). Students answer each with a **move** in `moves.py`: a list of
`(seconds, pose)` steps, played under physics. A pose names joints in degrees,
or uses **IK**: `"pelvis": [0, 0, -0.2]` lowers the body 20 cm while
`ik_legs` keeps both feet planted. It passes if the goal is reached, the robot
never falls, and it ends standing.

How they solve it by looking, not guessing: `E` freezes the robot and the joint
sliders pose it, `ENTER` prints the pose as a dict to paste, `K` steps through
the current move frozen, and `1`–`0` play a problem with a live gauge
("left hand 1.12 m, need > 1.20"). `moves.py` is re-read on every key press.

**Grading.** Students press `G` whenever they like; all ten play in about 45 s
and the score stays on screen: name, `SCORE n / 10`, `O`/`X` per problem, a
four-letter **scorer code** and the time. The instructor walks the room reading
screens. The scorer code is a hash of `lab_connected.py`; if it differs from the
one on the board, the referee was edited. `moves.py` is parsed as data, never
executed, so it cannot change the rules. Mouse pushes are zeroed while a move
plays.

What it teaches, measured with its own `--grade` before it was written: every
pose here is one IK solves perfectly, and half of them fall over under gravity.
Both arms up at once falls forward; one arm, then the other, reaches 1.36 m. A
clap needs the pelvis 4 cm back as a counterweight. A bow needs the pelvis
pushed back while the hips bend. The boss needs the robot to stand up *before*
lowering its arms. One-foot balance is left out on purpose: 72 open-loop
attempts all fell.

Reference answers (10/10) are in `instructor/moves_solution.py`, which is
gitignored. Check them with
`--grade --moves weeks/03-inverse-kinematics/instructor/moves_solution.py`.

## Rebuild

```bash
quarto render weeks/03-inverse-kinematics/slides.qmd
```
