# Week 2 — Transforms and Forward Kinematics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w02-transforms/lab.ipynb)

| | |
| --- | --- |
| **Runtime** | Local: CPU. **Colab: pick a GPU runtime** (Runtime > Change runtime type > T4). |
| **Wall clock** | ~1 min |
| **Convergence risk** | None. No learning. |
| **Feeds** | Week 3 (IK), Week 4 (walking) |

## Shape of the session

The deck assumes **no prior robotics and no prior linear algebra beyond matrix
multiplication**. It climbs in four steps, and each one is checked against the
simulator before the next is allowed to build on it:

1. **One link in 2D**, rotated by hand — `tip = (-L sin θ, -L cos θ)`, verified
   with a calculator.
2. **Two links in 2D**, chained — the real G1 thigh and shin, still on paper.
3. **The full 6-joint 3D chain**, composed body by body in twelve lines.
4. **The Jacobian**, introduced as measured millimetres before it is named.

Rotations are shown not to commute with a book on the desk *and* with two 3x3
matrices, because students will not believe the algebra until they have felt it.

## Objectives

1. Say what a frame is, and why a position without one is meaningless.
2. Compose rotations and translations, and say why order matters.
3. Compute a two-link planar leg's foot position **by hand**, and check it.
4. Choose a rotation representation and state its failure mode — including
   MuJoCo's scalar-first `(w,x,y,z)` quaternions.
5. Explain `nq > nv`, and why it means `qpos += qvel * dt` is not a valid step.
6. **Implement forward kinematics from the model tree** and validate it against
   `mj_forward`.
7. Read a workspace plot, and connect its boundary to next week's singularities.

## The two results students must reproduce

**The paper model.** The planar formula agrees with MuJoCo to **2.3e-06 m** —
about the width of a bacterium — across four poses. It is deliberately *not*
exact: at the zero pose MuJoCo puts the foot 2.3 µm behind the hip anchor rather
than exactly beneath it, because the G1's own offsets do not quite cancel. That
gap is the lesson that a flat two-link picture is a **model** of the leg.

**The full chain.** Hand-written FK agrees with `site_xpos` to **~1e-16** —
machine precision, because it is the same computation. Anything larger is a bug,
and the two causes that can actually bite on the G1 are named on the slide:
composition in the wrong order, or scalar-last quaternions.

The third textbook cause — a missing joint anchor — **cannot** produce an error
here: all 30 of the G1's joints have `jnt_pos = 0` (so does every other humanoid
in Menagerie). The slide demonstrates it on a nine-line MJCF instead, where
dropping the term moves the tip 0.215 m.

## Two lengths for the same thigh

Week 2's planar model uses the **in-plane projection, 0.3366 m**. Week 3's law of
cosines uses the **true 3D length, 0.3409 m**. Both are right; the hip-to-knee
vector splays 5.4 cm sideways. Substituting one for the other costs 4 mm.

## Rebuild

```bash
quarto render weeks/w02-transforms/slides.qmd
```
