# Week 3 — Inverse Kinematics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w03-inverse-kinematics/lab.ipynb)

| | |
| --- | --- |
| **Runtime** | Local: CPU. **Colab: pick a GPU runtime** (Runtime > Change runtime type > T4). |
| **Wall clock** | ~1 min |
| **Convergence risk** | None. No learning. |
| **Feeds** | Week 4 — the walker calls this solver 500 times a second |

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

## Rebuild

```bash
quarto render weeks/w03-inverse-kinematics/slides.qmd
```
