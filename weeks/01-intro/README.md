# 01 — Robots, Simulation, MuJoCo and MJCF

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/01-intro/lab.ipynb)

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
7. Use the words **roll, pitch and yaw** correctly, and name the axis each one
   turns about.
8. Distinguish `model` (the constant blueprint) from `data` (one robot at one
   instant), and say what `mj_step` advances.
9. Read contact forces back out of a standing robot and check them against `mg`.
10. Explain why a falling robot is deterministic on one machine and irreproducible
    across two — and why convergence is therefore never graded.

## Expected output

Three videos: the Unitree G1 collapsing under gravity, and a two-link leg the
student wrote themselves, run twice — once with its servos holding a bent pose,
once with actuation disabled.

**The G1 falling is the correct result** — it motivates every remaining week. The
two leg videos repeat the week-0 lesson (`ctrl = 0` is not an uncontrolled robot)
in a twenty-line model the student owns.

## Lab class: on your laptop

```bash
uv run weeks/01-intro/lab_mjcf.py
```

The two-link leg from the slides lives as an XML string at the top of the file.
The script compiles it, prints `nq`/`nv`/`nu`/`nbody` and every joint's `qpos`
and `qvel` slot, then opens the viewer with the servos holding `TARGET`.

```
A   actuation on/off        0   ctrl = 0        T   ctrl = TARGET        G   Moon gravity
R   reset to TARGET at rest
ENTER   joint angles, droop from TARGET, contacts, BADQACC warning count
```

Every exercise is an edit to the XML followed by a fresh run — the compiler's
own error messages are part of the lab.

| Step | Change | Right looks like |
| --- | --- | --- |
| 1 | none; `A`, `A`, `ENTER` | a double pendulum, then a held pose; droop of a few hundredths of a radian, explained |
| 2 | `<freejoint/>` on `upper_leg`, then a parent body carrying it | *"more than 6 dofs in body"* read aloud; `nq`, `nv` predicted before the run |
| 3 | `kp` 200 → 100 → 50 → 20 → 5 | droop noted at each; never zero |
| 4 | `timestep="0.02"` | servos off: a different swing and no warning; servos on: `BADQACC` counted and `qpos` reset to zero — the silent failure |
| 5 | a foot body with its own hinge | `nq` up by one; it moves with the knee |
| 6 | the capsules' `rgba` moved into a `<default>` class | the picture unchanged |

`SOC4180_AUTOCLOSE=6 uv run weeks/01-intro/lab_mjcf.py` closes the window by
itself, which is how the script is smoke-tested.

## Files

- `slides.qmd` — the single source. Renders to both outputs.
- `slides.html` — reveal.js lecture deck (video embedded, self-contained).
- `lab.ipynb` — student notebook, opened by the Colab badge above.

## Rebuild

```bash
quarto render weeks/01-intro/slides.qmd
```
