# Week 1 — Robots, Simulation, MuJoCo and MJCF

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w01-intro/lab.ipynb)

| | |
| --- | --- |
| **Runtime** | Local: CPU. **Colab: pick a GPU runtime** (Runtime > Change runtime type > T4). |
| **Wall clock** | ~1 min (first run adds ~5 s to fetch the G1 model). Renders three videos. |
| **Convergence risk** | None. No learning this week. |

> The GPU is not needed for physics — it is needed to *render* the fall video. MuJoCo uses EGL on Colab, which requires the GPU runtime.

## Objectives

By the end of this lab a student can:

1. Explain why robot learning happens in simulation, and name three ways
   simulation misleads.
2. Read a real MJCF file section by section, and identify the kinematic chain in
   its body tree.
3. Explain what `<default>` classes do, and find where an attribute a tag never
   states actually comes from.
4. Distinguish visual from collision geometry, and say why the G1's feet contact
   the ground through spheres rather than meshes.
5. Predict `nq` and `nv` from the joint types in a file, and explain why `nq > nv`
   for a floating-base robot.
6. **Write a working MJCF from scratch**, compile it, and drive it.
7. Load a Menagerie robot, step the physics, and render video.

## Expected output

Three videos: the Unitree G1 collapsing under gravity, and a two-link leg the
student wrote themselves, run twice — once with its servos holding a bent pose,
once with actuation disabled.

**The G1 falling is the correct result** — it motivates every remaining week. The
two leg videos repeat the week-0 lesson (`ctrl = 0` is not an uncontrolled robot)
in a twenty-line model the student owns.

## Files

- `slides.qmd` — the single source. Renders to both outputs.
- `slides.html` — reveal.js lecture deck (video embedded, self-contained).
- `lab.ipynb` — student notebook, opened by the Colab badge above.

## Rebuild

```bash
quarto render weeks/w01-intro/slides.qmd
```
