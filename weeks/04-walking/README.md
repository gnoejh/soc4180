# 04 — Making a Humanoid Walk

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/04-walking/lab.ipynb)

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

## Lab class: on your laptop

```bash
uv run weeks/04-walking/lab_walk.py
```

The walker live, in real time, in the interactive viewer. `GAITS` at the top of
the file is a list of named `GaitParams`; keys `1`–`9` select one and restart.

```
1..9    pick a gait and restart      R   restart       SPACE   pause
G       Moon gravity on/off          F   ice (friction 0.2) on/off
ENTER   distance, pelvis height, ZMP range, fell/upright
double-click a body, then ctrl-drag: push it
```

Drawn on the floor every frame: the **footstep plan** (grey boxes), the
**commanded centre-of-mass path** for the current step (white dots), the
**pelvis** projected down (red) and the **measured ZMP** (yellow). Blue dots are
the student's own LIPM: `predict_com(x0, v0, zmp, t, omega)` returns `None` as
shipped, and the exercise is to write the closed-form cosh/sinh solution so the
blue dots land inside the white ones.

| Step | Change | Right looks like |
| --- | --- | --- |
| 1 | write `predict_com` | blue dots inside white on every step; with the sinh sign flipped they run away backwards |
| 2 | press `2` (no double support) | the student counts the steps before the fall and says which way it went |
| 3 | press `3`, then lower `step_time` in `GAITS` | a failure point, compared against $1/\omega = 0.247$ s printed at start-up |
| 4 | press `G` | a prediction from $\omega = \sqrt{g/z_c}$ made *before* looking |
| 5 | ctrl-drag mid-walk | it never reacts; the student names what a controller would need to know (week 6) |
| 6 | a gait in `GAITS` that goes further | `ENTER` prints a distance beyond the default's ~1.0 m |

The yellow sphere is the point of the class: the theory says it never leaves
the stance foot, and every student watches it swing wide of the feet at each
support exchange.

`SOC4180_AUTOCLOSE=6 uv run weeks/04-walking/lab_walk.py` closes the window by
itself, which is how the script is smoke-tested.

## Rebuild

```bash
quarto render weeks/04-walking/slides.qmd
```
