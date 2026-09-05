# SOC4180 — Robot and AI

Undergraduate course on robotics and robot learning, taught entirely in
simulation. The through-line is a **walking humanoid**: students make the
[Unitree G1](https://github.com/google-deepmind/mujoco_menagerie) walk with
classical control first, then learn a policy that does it instead.

Every lab runs in **Google Colab with zero installation**.

## Weeks

| Wk | Topic | Lab | Runtime |
| --- | --- | --- | --- |
| [00](weeks/w00-robot-stack/) | What a Robot Is — the five-layer stack | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w00-robot-stack/lab.ipynb) | CPU |
| [01](weeks/w01-intro/) | Robots, Simulation, and MuJoCo | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w01-intro/lab.ipynb) | CPU |
| [04](weeks/w04-walking/) | Making a humanoid walk — LIPM & ZMP | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w04-walking/lab.ipynb) | GPU (render) |

Week 00 is the day-one lecture, taught before Week 01.

*Weeks 2–15 are planned; see the syllabus in `CLAUDE.md`.*

## For students

Click the Colab badge for the week. Nothing to install.

**Set the runtime to GPU first** — *Runtime > Change runtime type > T4 GPU*.
The GPU is not for training; MuJoCo renders video through EGL on Colab, and that
needs the GPU runtime.

## For the instructor — local setup

```bash
uv sync                    # lean environment, pinned by uv.lock
uv sync --extra rl         # adds gymnasium, stable-baselines3, torch
uv sync --extra gpu        # adds JAX/MJX/playground (Linux or WSL2 only, see below)
quarto render weeks/w01-intro/slides.qmd
```

`quarto render` produces **both** outputs from the single `slides.qmd`:

- `slides.html` — self-contained reveal.js deck (a build artifact, gitignored)
- `lab.ipynb` — the student notebook, **committed** because Colab loads it from git

A failing code cell aborts the render, so a broken example can never reach a
lecture.

### Prerequisite

[Quarto](https://quarto.org) ≥ 1.10 must be installed separately — it is a
standalone CLI, not a Python package.

## Architecture

Two execution targets, because one machine cannot do both jobs:

| | Local (Windows + RTX 4060) | Colab |
| --- | --- | --- |
| Physics | MuJoCo (CPU) | MuJoCo (CPU) |
| Deep learning | **PyTorch CUDA works** | PyTorch + JAX CUDA |
| Parallel RL training | **not possible** | MJX / Brax on GPU |

**JAX's CUDA plugins are Linux-only** — `jax-cuda12-plugin` and
`jax-cuda13-plugin` publish `manylinux` wheels with no `win_amd64` build. GPU
locomotion training therefore happens on Colab (or WSL2), never natively on
Windows. PyTorch CUDA is unaffected and works locally.

`torch` is resolved from two sources for the same reason — PyPI ships CPU-only
Windows wheels, so Windows pulls the CUDA build from PyTorch's index while Linux
uses PyPI directly. One lockfile covers both:

| Platform | torch | Source |
| --- | --- | --- |
| Windows | `2.14.0+cu130` | download.pytorch.org |
| Linux (Colab) / macOS | `2.14.0` | PyPI |

## Layout

```
src/soc4180/     shared helpers, installed as a package
  render.py      GL backend + ffmpeg selection, rollout rendering, video
  models.py      Menagerie robots, pinned
  sim.py         keyframes, actuation on/off, hold controllers
  kinematics.py  leg FK/IK (damped least squares)
  walking.py     LIPM, footstep planning, the analytic gait
  seeding.py     reproducibility
  checkpoints.py pre-trained policy loading (the RL-week safety net)
weeks/wNN-*/     slides.qmd (source) -> slides.html + lab.ipynb
_quarto.yml      shared deck theme and execution settings
```

Helpers live in one installed package rather than being copy-pasted per week, so
a fix to the render path lands in every lab at once.
