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
uv run weeks/04-walking/walk.py --step-time 0.45
```

The walker, live and in real time. Keys `1`–`5` pick a gait from the `GAITS`
list; `G` switches to Moon gravity and `F` to ice *without telling the
controller*; ctrl-drag pushes the robot; `ENTER` prints distance, pelvis
height, the ZMP range and the gap between the LIPM written by hand and the
package's. On the floor: the footstep plan as grey boxes, the commanded centre
of mass for the current step as white dots, `predict_com`'s version as blue
dots, the pelvis as a red sphere, the measured ZMP as a yellow one.

**The code is complete and explained.** `predict_com` is the LIPM closed form
with the physics in its docstring; `Walk` is the loop around the package's
controller with the MuJoCo call named at each step.

| Step | Do | Right looks like (measured with `walk.py`) |
| --- | --- | --- |
| 1 | `ENTER`; then flip the sinh sign in `predict_com` | gap 0.0; blue inside white — then the blue dots run backwards |
| 2 | `2` (no double support) | falls at 2.0 s, before the second step completes; which way, and why |
| 3 | `3` (rushed, 0.45 s) | four steps, then a fall at 3.2 s; compare 0.45 s to 1/ω = 0.247 s |
| 4 | `G` | the prediction from ω = √(g/z_c) first; measured: six steps, then a fall at 5.3 s |
| 5 | ctrl-drag while walking | it never reacts: what it would need to know (week 6) |
| 6 | a gait that walks further than 0.99 m in 8.8 s | `ENTER` proves it |

### `walk.py`: plan → LIPM → IK → physics, one row per step

```bash
uv run weeks/04-walking/walk.py
uv run weeks/04-walking/walk.py --double-support 0 --no-viewer
uv run weeks/04-walking/walk.py --gravity 1.62
uv run weeks/04-walking/walk.py --friction 0.2 --steps 6 --step-length 0.20
```

Every gait parameter is a flag. It prints the footstep plan and the pendulum's
numbers (ω = 4.04 rad/s, 1/ω = 0.247 s: a 0.65 s step is 2.6 time constants),
checks the LIPM by hand against the package (0.0), runs the walk headless with
one row per step — position, pelvis height, the ZMP's sideways range, IK
tracking error — and a summary, then replays the walk with the plan, pelvis
and ZMP drawn on the floor.

| Step | Does | Calls |
| --- | --- | --- |
| 1 | robot, world, gait; the plan printed | `opt.gravity`, `geom_friction`, `GaitParams`, `WalkingController`, `controller.plan` |
| 2 | the LIPM by hand vs the package | `controller._segments`, `lipm.evolve` |
| 3 | the loop, one row per step | `controller.control(t)`, `mj_step`, `controller.zmp(data)`, `last_ik_error` |
| 4 | summary: distance, ZMP range vs stance width | |
| 6 | replay with markers | `launch_viewer(passive=True)`, `user_scn` boxes and spheres |

Measured with it: the default gait walks 0.989 m in 8.8 s with the measured
ZMP swinging over [−0.233, +0.105] m against feet at ±0.119 m — outside the
support polygon, which is the model failing, not the criterion; ice (friction
0.2) changes nothing for this gait; the rushed, no-double-support and Moon
gaits fall at 3.2, 2.0 and 5.3 s.

## Rebuild

```bash
quarto render weeks/04-walking/slides.qmd
```
