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

**The code is complete and explained**: the XML is annotated tag by tag, the
compile step says what `MjModel` and `MjData` are, and the loop names each
MuJoCo call. Every exercise is an edit to the XML followed by a fresh run — the
compiler's own error messages are part of the lab.

| Step | Change | Right looks like (measured) |
| --- | --- | --- |
| 1 | none; `A`, `A`, `ENTER` | a double pendulum, then a held pose; hip droop −0.080 rad, knee −0.007, explained |
| 2 | `<freejoint/>` on `upper_leg`, then a parent body carrying it | *"more than 6 dofs in body"* read aloud; `nq = 9`, `nv = 8` predicted before the run |
| 3 | `kp` 200 → 100 → 50 → 20 → 5 | hip droop −0.080, −0.151, −0.269, −0.501, −0.858 rad: never zero, and $1/k_p$ while the servo is stiff |
| 4 | `timestep="0.02"` | servos off: a different swing and no warning; servos on: one `BADQACC` and `qpos` reset to zero — the silent failure |
| 5 | a foot body with its own hinge | `nq` up by one; it moves with the knee |
| 6 | the capsules' `rgba` moved into a `<default>` class | the picture unchanged |

### `mjcf_run.py`: compile, print, simulate, replay

```bash
uv run weeks/01-intro/mjcf_run.py
uv run weeks/01-intro/mjcf_run.py --kp 20 --seconds 3
uv run weeks/01-intro/mjcf_run.py --timestep 0.02 --no-viewer
uv run weeks/01-intro/mjcf_run.py --servos off
uv run weeks/01-intro/mjcf_run.py --xml my_robot.xml --target 0.5 0.5 0
```

The same leg (or any MJCF file), with `kp`, the timestep, the servos and
gravity as flags. It compiles, prints the joints' `qpos`/`qvel` slots and the
actuators' gains, holds the target for `--seconds` printing the angles and
their droop every half second with the `BADQACC` count, then replays the
motion in the simulator. Six numbered steps:

| Step | Does | Calls |
| --- | --- | --- |
| 1 | compile the XML; apply `--timestep`, `--gravity`, `--kp` (rescaling `kv` to keep critical damping) | `MjModel.from_xml_string`, `opt.timestep`, `actuator_gainprm`, `actuator_biasprm` |
| 2 | print what the compiler made | `jnt_qposadr`, `jnt_dofadr`, `mj_id2name` |
| 3 | put the leg at the target and command it | `jnt_qposadr[actuator_trnid]`, `ctrl`, the actuation disable bit |
| 4 | the loop, a printed row every `--report` s | `mj_step`, `data.warning` |
| 5 | summary: droop is the servo's steady-state error; `BADQACC` is a reset | |
| 6 | replay | `launch_viewer(passive=True)`, `mj_forward`, `sync` |

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
