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
from the model (thigh 0.3409 + shin 0.3000 = 0.6409 m). Physics is never
stepped and the pelvis never moves.

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
| 2 | `M` inverse, `S` straight leg, arrow down | \|dq\| = 1.1e4 rad; the clamp is all that stops it; hip and knee slam to 2.88 rad, foot 0.7 m off |
| 3 | dls, straight leg, `[` to 1e-6 then `]` to 1e-1 | the same fold at 1e-6; at 1e-1 \|dq\| = 1e-4 and the foot never moves: no $\lambda$ brings it down |
| 4 | `M` transpose, up ×5 | creeps: 8 mm of 50 left after 100 iterations; at the straight leg \|dq\| = 1.5e-5, nothing explodes |
| 5 | right arrow ×30 | hip→target 0.69 m of 0.64 m reach; residual hovers at 5–6 cm, knee and ankle on their limits; `O` and 0.25 m pins at exactly 6.8e-3 |
| 6 | `I` to 1/frame, `]` to $\lambda = 1$ | 3.1 cm of 5 left after 50 frames; 0.3 closes in 50, 0.1 in 10 |
| 7 | `O`, `.` ×15 and up ×5 | foot tilts 16° with 3 rows, ankle roll stays 0; 0.2° with 6 rows, ankle roll −0.26 rad |
| 8 | two new `TARGETS`, one unreachable | the prediction is right before SPACE is pressed |

Then break something: `-dq` from `dls_step` runs the residual to 0.57 m and
sticks on the joint limits; `J.T @ err` without the step length in
`transpose_step` overshoots and diverges (0.050, 0.049, … 0.32, 0.59 in ten
iterations).

`SOC4180_AUTOCLOSE=4 uv run weeks/03-inverse-kinematics/lab_ik.py` closes the
window after four seconds, which is how the script is smoke-tested.

## Rebuild

```bash
quarto render weeks/03-inverse-kinematics/slides.qmd
```
