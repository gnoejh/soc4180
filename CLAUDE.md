# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Undergraduate course "Robot and AI" (SOC4180): 15 weeks of slides + runnable
labs, taught through MuJoCo simulation. The spine is a **walking humanoid**
(Unitree G1) — classical control first, learned control second.

**Simulation only. No hardware exists.** Never propose a lab that needs a
physical robot.

Weeks 0 and 1 are built and verified end to end. Weeks 2–15 are designed but not
written. **Week 0 is the day-one stack/vocabulary lecture and is taught before
Week 1**; later weeks are expected to name which layer they belong to.

## Commands

```bash
uv sync                                        # environment from uv.lock
uv sync --extra gpu                            # + JAX/MJX/playground (Linux/WSL2)
uv run python -c "import soc4180"              # smoke test
quarto render weeks/w01-intro/slides.qmd       # -> slides.html + lab.ipynb
```

There is no test suite or linter configured. Add the tooling before inventing
commands for it.

## Authoring model: one source, two outputs

Each week is a single `weeks/wNN-*/slides.qmd` that Quarto renders into:

- `slides.html` — self-contained reveal.js deck (**gitignored build artifact**)
- `lab.ipynb` — student notebook (**committed**; Colab loads it from GitHub)

`execute.error: false` in `_quarto.yml` means a broken cell **fails the render** —
verified. Do not set `error: true`; that protection is the point.

Rendered decks are gitignored because reveal.js assets are ~5 MB/week. Only
`lab.ipynb` is committed.

### Verified Quarto behaviours (do not re-litigate)

- `#| eval: false` **does** survive into `lab.ipynb` as a genuine unexecuted code
  cell. No post-render cell injector is needed.
- The Week-1 setup cell uses `try: import soc4180 / except ImportError: %pip install`
  rather than a bare `!pip install`, so the notebook is safe to run locally *and*
  installs on Colab. Keep that shape.
- **Video survives into `slides.html` (embedded base64) but is stripped from
  `lab.ipynb`.** Students still see it when they *run* the notebook, which is the
  actual Colab workflow. This is accepted, not a bug to chase.

## Environment gotchas

### Two execution targets

**JAX CUDA plugins are Linux-only** (`manylinux` wheels only, no `win_amd64`), so
MJX/Brax GPU training cannot run natively on this Windows machine — that work
belongs on Colab or WSL2. PyTorch CUDA *does* work locally and is verified.
Keep `jax`/`brax`/`playground` in the `gpu` optional extra, never in the default
dependency set.

### torch resolves from two sources

PyPI ships CPU-only Windows wheels, so `torch` comes from the `cu130` index on
Windows and from PyPI (already CUDA-enabled) on Linux:

| Platform | torch | Source |
| --- | --- | --- |
| Windows | `2.14.0+cu130` | download.pytorch.org |
| Linux / macOS | `2.14.0` | PyPI |

- **Do not remove the `sys_platform == 'win32'` marker** in `[tool.uv.sources]`.
- **`uv add torchvision`/`torchaudio` bare is wrong on Windows** — they need the
  same marked source, or you get CPU-only builds mismatched against CUDA torch.
- Keep `explicit = true` on the index so only named packages use it.

### Korean Windows locale (cp949)

The system Python defaults to **cp949**, not UTF-8. This bites constantly:

- Always pass `encoding="utf-8"` to `read_text`/`write_text`/`open`. Omitting it
  raises `UnicodeDecodeError` on any file containing an em dash.
- Prefix scripts that print non-ASCII with `PYTHONIOENCODING=utf-8`, or they die
  with `UnicodeEncodeError` on the console.

### ffmpeg

`mediapy` shells out to a **system** ffmpeg, which Colab has and Windows does
not. `render.py` resolves this at import by pointing mediapy at the binary
bundled with `imageio-ffmpeg`. Do not remove that fallback.

### MUJOCO_GL ordering

`MUJOCO_GL` must be set **before** `import mujoco`. `soc4180.render` does this at
import time (`egl` on Colab/headless Linux, default elsewhere). Import
`soc4180` before `mujoco`, and never reorder those statements.

## `ctrl = 0` is not an uncontrolled robot

The G1's actuators are **position servos with gain 500**. Setting `ctrl = 0`
commands every joint to angle zero, and the servos hold that straight-legged
stance — the robot does **not** fall. Verified: torso stays at 0.792 m.

To simulate a robot with no controller you must disable actuation entirely, via
`soc4180.actuation_disabled(model)` (a context manager that restores the flag).
Then the torso drops 0.790 m → 0.134 m.

This bit once already: an early draft of the Week 1 slides asserted the robot
collapsed when it demonstrably did not. **Any slide claiming the robot falls must
disable actuation.** The distinction is now core Week 0 teaching material.

## Robot models

`mujoco-menagerie` is a **pip package** (~28 KB wheel, downloads models lazily and
pins its own upstream commit) — not a git clone. Use `soc4180.load_g1()` /
`load_robot()`. Pinning `mujoco-menagerie` in `pyproject.toml` pins the models.

- G1: `nq=36 nv=35 nu=29`, 500 Hz timestep, one `stand` keyframe. BSD-3-Clause.
  `nq > nv` because the floating base uses a quaternion — that is week-0/1
  teaching material, not a bug.
- Sensors: `nsensor = 4` is **two IMUs**, not four sensors — a gyro +
  accelerometer pair at `imu_in_torso` and another at `imu_in_pelvis`, 3 axes
  each, 12 numbers per step. There are **no cameras, joint-torque sensors, or
  foot contact sensors**; add them to the MJCF if a lab needs them.
- An accelerometer measures *specific force*, so it senses gravity and reads
  ~9.81 m/s² at rest. That is what makes it a tilt sensor, and why policies
  observe the gravity vector in body frame rather than absolute pose. IMUs give
  no position — double integration drifts without bound.
- Entry points: `g1`, `g1_mjx`, `g1_with_hands`, plus matching `scene*` variants.
  Scenes include floor and lighting; bare robot entries do not.
- 11 humanoids available; Cassie is categorised `biped`, not `humanoid`.
- Known upstream gotcha: `assets()` raises for `robotis_op3` (duplicate mesh
  basenames). Use `load()`/`path()` instead.

## Pedagogical constraints that drive the code

- **A robot must walk by week ~4 using analytic control (LIPM/ZMP), long before
  any RL.** If the first walking robot depended on a policy converging, a failed
  training run would leave the course with no walking robot at all.
- **Every RL lab must ship a pre-trained checkpoint fallback** (`checkpoints.py`),
  so a lab never dead-ends on a non-converging run. No suitable public checkpoints
  exist — they must be trained and hosted.
- **Never grade convergence**; grade the diagnosis.
- G1 is a 29-DOF full humanoid. If training proves too slow, `berkeley_humanoid`
  (12 actuated DOF) and `robotis_op3` are the cheaper fallbacks and also have
  tuned `playground` locomotion envs.
