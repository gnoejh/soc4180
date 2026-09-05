# Week 4 — Making a Humanoid Walk

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w04-walking/lab.ipynb)

**The keystone week.** A 29-DOF humanoid walks using a linear ODE and inverse
kinematics — no learning of any kind, six weeks before RL appears.

| | |
| --- | --- |
| **Runtime** | Local: CPU. **Colab: pick a GPU runtime** (Runtime > Change runtime type > T4). |
| **Wall clock** | ~1 min (the walk is ~9 s of simulation, rendered) |
| **Convergence risk** | None. Nothing is trained. |
| **Depends on** | Week 3 inverse kinematics (reused unchanged) |

## Objectives

1. Explain why the centre of pressure cannot leave the support polygon, and what
   follows when it does.
2. Derive the LIPM from the constant-height assumption and state what $\omega$ means.
3. Solve the LIPM boundary value problem to generate one step, and chain steps
   into a gait.
4. Explain why the lateral sway exists — it buys the time to swing a leg.
5. Measure where the model's assumptions fail on a real 29-DOF robot.

## Result

The G1 walks about **1.0 m in 9 s (~0.12 m/s)** and stays up, with the pelvis
between 0.69 and 0.74 m throughout.

## The honest part

The measured ZMP swings **wider than the feet** — roughly ±0.26 m against stance
feet at ±0.119 m. The LIPM says that is impossible, and the LIPM is right: its
assumptions are what fail (point mass, constant height, massless legs, instant
support exchange, infinitely stiff servos). The robot walks anyway.

That gap is the lesson, not a defect to hide. It is also the direct setup for
Week 13's sim-to-real material.

The controller is **open loop** — it never reads a sensor. It drifts sideways,
cannot take a push, and works on flat ground only. That is precisely the case for
learned policies later in the course.

## Two traps worth knowing

1. **The `stand` keyframe is a singularity.** Every leg joint is exactly zero, so
   the Jacobian has no direction that shortens the leg and IK cannot lower the
   body from it. Gaits start from a crouch instead.
2. **Chain the boundary value problems.** Computing each step's CoM trajectory
   absolutely rather than continuing from the previous step teleports the body
   backwards half a stride at every support exchange. It looks like a balance
   failure and is actually a bookkeeping error.

## Tuning

`GaitParams` holds every knob. The defaults were tuned by sweep; the gait is
genuinely sensitive, and several nearby settings fall over — which is what makes
the exercises worth doing.

## Rebuild

```bash
quarto render weeks/w04-walking/slides.qmd
```
