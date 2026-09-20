# 00 — What a Robot Is: The Five-Layer Stack

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/00-robot-stack/lab.ipynb)

**Day one. Taught before Week 1** — this is the vocabulary lecture the rest of
the semester refers back to.

| | |
| --- | --- |
| **Runtime** | Local: CPU. **Colab: pick a GPU runtime** (Runtime > Change runtime type > T4). |
| **Wall clock** | ~1 min |
| **Convergence risk** | None. No learning. |

> The GPU is not needed for physics — it is needed to *render* the limp/servo videos. MuJoCo uses EGL on Colab, which requires the GPU runtime.

## Objectives

A student who has seen this lecture can:

1. Name the five layers and say what each is responsible for.
2. Explain why layering exists — the three-orders-of-magnitude spread in rates,
   from a 500 Hz physics step to a sub-1 Hz task decision.
3. Map a real robot onto a simulated one, and identify **Layer 1 as the only
   layer that does not transfer** — which is exactly where sim-to-real lives.
4. Explain why `ctrl = 0` is *not* an uncontrolled robot.
5. Ask of any robotics claim: *which layer is this, and what does it assume
   about the layers beneath it?*

## The demonstration

Two videos of the same robot, same physics, same starting pose:

- **actuation disabled** — a rag doll; the torso drops 0.790 m → 0.134 m
- **servos holding `stand`** — upright and stable

That contrast *is* Layer 4. Standing is a property of the loop, not of the robot.

## Departures from the classic five-layer diagram

Three deliberate refinements, each of which heads off a common confusion:

1. **Layer 1 is "Physics / World", not "Physics Engine (simulation only)".**
   A real robot's Layer 1 is reality itself. Framing it this way makes the real
   and simulated stacks line up row for row, so the sim-to-real gap becomes a
   visible asymmetry in a single table rather than a separate topic.
2. **Models are not simulation-only.** Real robots carry models too — IK needs
   link lengths, MPC needs dynamics. The model describes the *robot*; the engine
   simulates the *world*.
3. **Timescales are given as the reason layers exist**, rather than layering
   being presented as mere tidiness.

## Lab class: on your laptop

```bash
uv run weeks/00-robot-stack/lab_stack.py
```

The G1 in the interactive viewer, physics in real time, and each key leaving a
different set of layers switched on:

```
1   layers 1+2 only: every servo dead (rag doll)      0   ctrl = 0: the trap
2   layer 4, simplest: servos hold `stand`            3   layer 4, full: the week 4 walker
4   layer 4, yours: my_controller                     G   Moon gravity, nothing else touched
R   reset     SPACE   pause     ENTER   physics rate, torso height, contacts, force sum vs weight
double-click a body, then ctrl-drag: push it
```

Contact forces are drawn as arrows at the contact points — Layer 1 made
visible. **The code is complete and explained**: `my_controller(model, data, t)`
is a working Layer 4 as shipped (hold `stand`, wave the left arm) that students
change, and the loop around it names the MuJoCo call at each step
(`mj_resetDataKeyframe`, `mj_step`, `mj_contactForce`, the actuation disable
bit, the real-time sleep).

| Step | Do | Right looks like (measured) |
| --- | --- | --- |
| 1 | `1`, `R`, `2` | rag doll ends at torso 0.13 m, held pose stays at 0.792 m; one sentence on which layer differs |
| 2 | `R`, `0` | the robot stands straight-legged at 0.791 m; the student explains why zero is a command |
| 3 | `G`, `R`, `1`; then `3` | a slow-motion fall; forces sum to 54 N (Moon weight); a walker whose timing assumes Earth |
| 4 | `4`, then change `my_controller` | the arm waves and the rest keeps standing; the student names the layer that holds the legs |
| 5 | `ENTER` standing, then during `3` | 327.1 N on 8 contacts; 280–340 N on 4–5 contacts while walking, because the body accelerates |
| 6 | ctrl-drag under `2` and `3` | no reaction either way: there is no sensing layer yet (week 6) |

### `stack.py`: the same stack as a script you read

```bash
uv run weeks/00-robot-stack/stack.py --layer hold
uv run weeks/00-robot-stack/stack.py --layer limp --seconds 3
uv run weeks/00-robot-stack/stack.py --layer walk --seconds 9
uv run weeks/00-robot-stack/stack.py --layer wave --gravity 1.62 --no-viewer
```

The five layers as flags. It simulates headless first, printing a row every
`--report` seconds (torso height, contacts, sum of vertical contact forces
against the weight), then the summary (physics steps per second, distance
walked, integrator warnings), then opens the simulator and replays the
recorded motion in real time. Six numbered steps:

| Step | Does | Calls |
| --- | --- | --- |
| 1 | layers 1+2: load the robot, set gravity, compute the weight | `load_g1`, `keyframe_data`, `model.opt.gravity`, `body_mass` |
| 2 | layer 4: pick the controller (`zero`, `hold`, `wave`, the walker, or none) | `key_ctrl`, `WalkingController`, the actuation disable bit |
| 3 | the loop: controller writes `ctrl`, one physics step, record a pose at 50 Hz | `mj_step`, `mj_contactForce` |
| 4 | summary | `data.warning` |
| 6 | replay the recorded `qpos` in the passive viewer | `launch_viewer(passive=True)`, `mj_forward`, `sync` |

Measured with it: `hold` 0.792 m for ever, 327.1 N on 8 contacts, ~15,000
steps/s here (30× real time); `limp` 0.134 m after 3 s; `zero` 0.791 m; `walk`
1.03 m forward in 9 s; `wave --gravity 1.62` 54 N of contact force.

`SOC4180_AUTOCLOSE=6 uv run weeks/00-robot-stack/lab_stack.py` closes the
window by itself, which is how the script is smoke-tested; the same variable
ends `stack.py`'s replay.

## Files

- `slides.qmd` — single source
- `slides.html` — reveal.js deck, videos embedded (build artifact, gitignored)
- `lab.ipynb` — student notebook, committed for the Colab badge

## Rebuild

```bash
quarto render weeks/00-robot-stack/slides.qmd
```
