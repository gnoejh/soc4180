# Week 2 — Transforms and Forward Kinematics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w02-transforms/lab.ipynb)

| | |
| --- | --- |
| **Runtime** | Local: CPU. **Colab: pick a GPU runtime** (Runtime > Change runtime type > T4). |
| **Wall clock** | ~1 min |
| **Convergence risk** | None. No learning. |
| **Feeds** | Week 3 (IK), Week 4 (walking) |

## Objectives

1. Compose rotations and translations, and say why order matters.
2. Choose a rotation representation and state its failure mode — including
   MuJoCo's scalar-first `(w,x,y,z)` quaternions.
3. Explain `nq > nv` from the unit-norm constraint on a quaternion.
4. **Implement forward kinematics from the model tree** and validate it against
   `mj_forward`.
5. Read a workspace plot, and connect its boundary to next week's singularities.

## The result students must reproduce

Hand-written FK agrees with MuJoCo's `site_xpos` to **~1e-16** — machine
precision, because it is the same computation. Anything larger is a bug, and the
three usual causes are named on the slide: a missing joint anchor, composition in
the wrong order, or scalar-last quaternions.

## Rebuild

```bash
quarto render weeks/w02-transforms/slides.qmd
```
