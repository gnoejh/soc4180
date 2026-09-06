# SOC4180 — Robot and AI

Undergraduate course on robotics and robot learning, taught entirely in
simulation. The through-line is a **walking humanoid**: students make the
[Unitree G1](https://github.com/google-deepmind/mujoco_menagerie) walk with
classical control first, then learn a policy that does it instead.

Every lab runs in **Google Colab with zero installation**, and every deck is
readable in a browser at **<https://gnoejh.github.io/soc4180/>**.

## Weeks

| Wk | Topic | Lab | Runtime |
| --- | --- | --- | --- |
| [00](weeks/w00-robot-stack/) | What a Robot Is — the five-layer stack | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w00-robot-stack/lab.ipynb) | CPU |
| [01](weeks/w01-intro/) | Robots, Simulation, and MuJoCo | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w01-intro/lab.ipynb) | CPU |
| [02](weeks/w02-transforms/) | Transforms and forward kinematics | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w02-transforms/lab.ipynb) | GPU (render) |
| [03](weeks/w03-inverse-kinematics/) | Inverse kinematics | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w03-inverse-kinematics/lab.ipynb) | GPU (render) |
| [04](weeks/w04-walking/) | Making a humanoid walk — LIPM & ZMP | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w04-walking/lab.ipynb) | GPU (render) |
| [05](weeks/w05-actuation/) | Actuation, PD control, and rhythm | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w05-actuation/lab.ipynb) | GPU (render) |
| [06](weeks/w06-sensing/) | Sensing and state estimation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w06-sensing/lab.ipynb) | GPU (render) |
| [07](weeks/w07-mdp/) | From control to learning: MDPs and env design | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w07-mdp/lab.ipynb) | GPU (render) |
| [08](weeks/w08-ppo/) | Policy gradients and PPO | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w08-ppo/lab.ipynb) | GPU (render) |
| [09](weeks/w09-reward/) | Reward shaping | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w09-reward/lab.ipynb) | GPU (render) |
| [10](weeks/w10-scaling/) | Scaling: GPU-parallel training | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w10-scaling/lab.ipynb) | **GPU (training)** |

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

## Slides on the web (GitHub Pages)

<https://gnoejh.github.io/soc4180/> carries every rendered deck, so a lecture can
be given from a browser alone — no checkout, no Quarto, no Python.

`.github/workflows/pages.yml` rebuilds it on every push that touches `weeks/`,
`src/` or the build itself, and can also be run by hand from the Actions tab
(*Publish course site → Run workflow*) — which is the whole point: the site can
be rebuilt from a phone.

**The decks are still not committed, and should not be.** `embed-resources: true`
inlines reveal.js, images and base64 video into one file, measured at 3.4–5.1 MB
per deck and ~43 MB for the set. Inlined base64 does not delta-compress, so every
re-render would add a fresh full copy to history — tens of megabytes per render,
permanently. The workflow instead uploads the built site as a Pages *artifact*:
nothing enters git, there is no `gh-pages` branch, and the URL is stable.

Two details worth knowing:

- Weeks render **one at a time**. A week whose cell breaks costs that one deck;
  the other ten still publish and the run turns red naming the broken week.
  `execute.error: false` still aborts that week's render, so nothing broken ships.
- Executed cells are cached in the Actions cache (`_freeze`), not in git, so an
  unchanged week is not re-run. The first build is slow — weeks 8 and 9 actually
  train — and later ones are minutes.

**One-time setup:** repository *Settings → Pages → Build and deployment → Source*
must be set to **GitHub Actions**.

## Interactive 3D viewing

**Locally** MuJoCo has a real interactive viewer — orbit, pan, zoom, and
ctrl-drag to shove the robot around. Far more informative than a rendered video
when a controller is misbehaving:

```bash
uv run scripts/view.py                    # the G1, standing
uv run scripts/view.py --walk             # the week 4 walker, live
uv run scripts/view.py --limp             # motors off; watch it collapse
uv run scripts/view.py --robot robotis_op3 --keyframe home
uv run scripts/view.py --list             # every humanoid available
```

Double-click a body to select it, then **ctrl-drag to push the robot** — the
quickest way to find out whether a controller survives a disturbance, and much
more informative than a rendered video.

`soc4180.launch_viewer(model, data, passive=True)` is the underlying helper if
you want to drive the loop yourself.

**On Colab there is no interactive viewer** — a notebook has no window to draw
into, which is why every lab renders video with `mediapy` instead. For an
MJX/brax rollout, `brax.io.html.render` produces an interactive 3-D scene that
does work inline.

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
  walking.py     LIPM, footstep planning, the analytic gait, CPG
  actuators.py   servo gains and torque limits
  estimation.py  IMU reading, tilt, complementary filter
  envs.py        G1WalkEnv, the Gymnasium walking task
  seeding.py     reproducibility
  checkpoints.py pre-trained policy loading (the RL-week safety net)
weeks/wNN-*/     slides.qmd (source) -> slides.html + lab.ipynb
_quarto.yml      shared deck theme and execution settings
```

Helpers live in one installed package rather than being copy-pasted per week, so
a fix to the render path lands in every lab at once.
