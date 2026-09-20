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
S              straight-leg <-> crouch seed      R   target back onto the foot
SPACE          next entry of TARGETS             ENTER   print target, foot, residual
```

**The solver is the student's.** `dls_step(J, err, lam)` returns `None` as
shipped, and while it does the foot never moves: the script measures the error,
fetches `mj_jacSite`, clamps to the joint limits and iterates three times per
frame, but the damped-least-squares step itself is the exercise.

| Step | Change | Right looks like |
| --- | --- | --- |
| 1 | write `dls_step` | the green sphere chases the red one; residual < 1e-4 within a frame |
| 2 | `[` down to $\lambda = 10^{-6}$, `S` for the straight leg, arrow down | $\|\Delta q\|$ in the thousands and the foot does not come down; `S` again and it drops |
| 3 | push the target out of the translucent sphere | the residual freezes at one value on every ENTER; printed hip distance exceeds reach |
| 4 | `]` up to $\lambda = 1$ | the foot creeps; the student names the trade and finds the largest $\lambda$ that still tracks 5 cm in a second |
| 5 | add two `TARGETS`, one unreachable | the prediction is right before SPACE is pressed |

Step 3 is the one that lands: a residual that does not move with more
iterations is a geometry problem, and the sphere makes the geometry visible.

`SOC4180_AUTOCLOSE=4 uv run weeks/03-inverse-kinematics/lab_ik.py` closes the
window after four seconds, which is how the script is smoke-tested.

## Rebuild

```bash
quarto render weeks/03-inverse-kinematics/slides.qmd
```
