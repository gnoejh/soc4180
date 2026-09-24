# SOC4180 — Robot and AI

Undergraduate course on robotics and robot learning, taught entirely in
simulation. The through-line is a **walking humanoid**: students make the
[Unitree G1](https://github.com/google-deepmind/mujoco_menagerie) walk with
classical control first, then learn a policy that does it instead.

Each week is two classes: a **lecture** from the deck, and a **lab** where
students change code on their own laptops, watch the G1 respond, and show the
result. Every deck is readable in a browser at
**<https://gnoejh.github.io/soc4180/>**, and every notebook also runs in
**Google Colab with zero installation** as the fallback when a laptop cannot.

## Weeks

| Wk | Topic | Slides | Notebook | Laptop lab | Runtime |
| --- | --- | --- | --- | --- | --- |
| [00](weeks/00-robot-stack/) | What a Robot Is — the five-layer stack | [deck](https://gnoejh.github.io/soc4180/00-robot-stack/slides.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/00-robot-stack/lab.ipynb) | [`lab_stack.py`](weeks/00-robot-stack/lab_stack.py), [`stack.py`](weeks/00-robot-stack/stack.py) | CPU |
| [01](weeks/01-intro/) | Robots, simulation, MuJoCo and MJCF | [deck](https://gnoejh.github.io/soc4180/01-intro/slides.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/01-intro/lab.ipynb) | [`lab_mjcf.py`](weeks/01-intro/lab_mjcf.py), [`mjcf_run.py`](weeks/01-intro/mjcf_run.py) | CPU |
| [02](weeks/02-transforms/) | Transforms and forward kinematics | [deck](https://gnoejh.github.io/soc4180/02-transforms/slides.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/02-transforms/lab.ipynb) | [`lab_viewer.py`](weeks/02-transforms/lab_viewer.py), [`fk.py`](weeks/02-transforms/fk.py) | GPU (render) |
| [02b](weeks/02b-robot-as-code/) | The robot as code — body parts, qpos, the package | [deck](https://gnoejh.github.io/soc4180/02b-robot-as-code/slides.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/02b-robot-as-code/lab.ipynb) | [`lab_body.py`](weeks/02b-robot-as-code/lab_body.py), [`anatomy.py`](weeks/02b-robot-as-code/anatomy.py) | GPU (render) |
| [03](weeks/03-inverse-kinematics/) | Inverse kinematics | [deck](https://gnoejh.github.io/soc4180/03-inverse-kinematics/slides.html) | — | [`lab_connected.py`](weeks/03-inverse-kinematics/lab_connected.py) (graded game, edit [`moves.py`](weeks/03-inverse-kinematics/moves.py)) | GPU (render) |
| [04](weeks/04-walking/) | Making a humanoid walk — LIPM & ZMP | [deck](https://gnoejh.github.io/soc4180/04-walking/slides.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/04-walking/lab.ipynb) | [`lab_walk.py`](weeks/04-walking/lab_walk.py), [`walk.py`](weeks/04-walking/walk.py) | GPU (render) |
| [05](weeks/05-actuation/) | Actuation, PD control, and rhythm | [deck](https://gnoejh.github.io/soc4180/05-actuation/slides.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/05-actuation/lab.ipynb) | [`lab_servo.py`](weeks/05-actuation/lab_servo.py), [`servo.py`](weeks/05-actuation/servo.py) | GPU (render) |
| [06](weeks/06-sensing/) | Sensing and state estimation | [deck](https://gnoejh.github.io/soc4180/06-sensing/slides.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/06-sensing/lab.ipynb) | [`lab_imu.py`](weeks/06-sensing/lab_imu.py), [`imu.py`](weeks/06-sensing/imu.py) | GPU (render) |
| [07](weeks/07-mdp/) | From control to learning: MDPs and env design | [deck](https://gnoejh.github.io/soc4180/07-mdp/slides.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/07-mdp/lab.ipynb) | [`lab_env.py`](weeks/07-mdp/lab_env.py), [`env_run.py`](weeks/07-mdp/env_run.py) | GPU (render) |
| [08](weeks/08-ppo/) | Policy gradients and PPO | [deck](https://gnoejh.github.io/soc4180/08-ppo/slides.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/08-ppo/lab.ipynb) | [`lab_train.py`](weeks/08-ppo/lab_train.py), [`train.py`](weeks/08-ppo/train.py) | GPU (render) |
| [09](weeks/09-reward/) | Reward shaping | [deck](https://gnoejh.github.io/soc4180/09-reward/slides.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/09-reward/lab.ipynb) | [`lab_reward.py`](weeks/09-reward/lab_reward.py), [`shape.py`](weeks/09-reward/shape.py) | GPU (render) |
| [10](weeks/10-scaling/) | Scaling: GPU-parallel training | [deck](https://gnoejh.github.io/soc4180/10-scaling/slides.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/10-scaling/lab.ipynb) | [`lab_many.py`](weeks/10-scaling/lab_many.py), [`many.py`](weeks/10-scaling/many.py) | **GPU (training)** |

00 is the day-one lecture, taught before 01. 02b is 02's
**lab class** — the same robot read as code — not a separate week.

**Every week has three artifacts for students.** The *notebook* carries the
lecture's code and rendered video and runs on Colab. The *laptop lab* is a
`lab_*.py` you run with `uv run`: it opens MuJoCo's interactive viewer, the
robot responds live, and the script has a list at the top to extend, a
complete and explained function you change, and a visible difference between
right and wrong. The *pipeline script* beside it is the same idea as a linear
program with flags: it computes, prints what it did, then shows it in the
simulator; read it in class with the slides open. The lab is the second class
of the week.

*11–15 are planned; see the syllabus in `CLAUDE.md`.*

## For students

### Seeing the slides

Every deck is on the web at **<https://gnoejh.github.io/soc4180/>** — nothing to
install, and the per-week *Slides* links in the table above go straight to one.
Each deck is a single self-contained file, so **Ctrl-S** in the browser saves a
copy that still works offline, video and all.

### Installing, once, before the first lab

```bash
# 1. uv (a Python package manager that also fetches Python itself)
#    Windows:  winget install astral-sh.uv
#    macOS:    brew install uv
#    Linux:    curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. the course
git clone https://github.com/gnoejh/soc4180.git
cd soc4180
uv sync --extra rl                        # ~10 min, mostly the 2.9 GB torch download
uv run scripts/view.py --keyframe stand   # a window with the robot = it works
```

`--extra rl` is not needed until week 8, but **do it now, at home**: thirty
laptops pulling 2.9 GB of PyTorch over classroom wifi is a lost lab class. Plain
`uv sync` is much smaller and enough through week 6 if disk space is tight.

This is **the same environment on every laptop and on the instructor's
machine**: `uv.lock` is committed and pins every package version, and `uv sync`
reproduces it exactly, fetching Python 3.12 itself if the laptop lacks it. Do not
`pip install` into it; if something is missing, the fix is `git pull` and
`uv sync` again — run both at the start of every week to pick up the new lab.

### Running a week's notebook on your laptop

```bash
uv run jupyter lab      # then open weeks/NN-*/lab.ipynb
```

`uv run` is what matters: it launches Jupyter from the project environment, so
the notebook's kernel is the pinned Python and the setup cell at the top does
nothing. In VS Code instead, **select `.venv` as the kernel** — with any other
interpreter the notebook silently downloads its own copy of the course package
and you end up running last week's code.

**Save your own copy before you change anything** — *File > Save Notebook As…* >
`lab_mywork.ipynb`. `lab.ipynb` is generated from the slides and is replaced
whenever a week is re-rendered, so edits saved into it are lost at the next
`git pull`. This is the local version of Colab's *Save a copy in Drive*.

Week 10's GPU training section runs through JAX, which has no Windows build —
**that section is Colab-only**. Everything else, including every laptop lab,
runs on the laptop.

### Running a week's laptop lab

Every week 0–10 ships two scripts (week 3 ships one, a graded game), and both are complete and explained line by
line — nothing is left blank; the exercises are experiments whose expected
numbers were measured with the scripts themselves:

- **`lab_*.py`, interactive.** Opens the simulator, physics or kinematics
  live, keys listed in the docstring (they also work typed in the terminal),
  a list at the top to extend and a function to change, and a visible
  difference on screen between right and wrong.
- **The pipeline script** (`stack.py`, `mjcf_run.py`, `fk.py`, `anatomy.py`,
  `walk.py`, `servo.py`, `imu.py`, `env_run.py`, `train.py`,
  `shape.py`, `many.py`). Linear, numbered steps, flags on the command line:
  it computes, **prints** what it did, then **replays or shows** the result in
  the simulator. `--no-viewer` runs it headless. This is the file to read in
  class, with the slides open; [`docs/mujoco-calls.md`](docs/mujoco-calls.md)
  maps every idea of the course to the MuJoCo call and the script that uses it.

```bash
uv sync                                          # once; --extra rl from week 7
uv run weeks/03-inverse-kinematics/lab_connected.py   # ten problems, G grades out of 10; edit moves.py
SOC4180_AUTOCLOSE=4 uv run weeks/04-walking/lab_walk.py        # closes itself: the smoke test
uv run scripts/check_labs.py                     # runs every script briefly and reports
```

In the simulator window, Tab and Shift+Tab hide the side panels; F1 lists its
shortcuts. Snap the window beside the editor (Win+← / Win+→) and keep the
integrated terminal visible: keys typed there reach the lab too.

## For the instructor — local setup

```bash
uv sync                    # lean environment, pinned by uv.lock
uv sync --extra rl         # adds gymnasium, stable-baselines3, torch
uv sync --extra gpu        # adds JAX/MJX/playground (Linux or WSL2 only, see below)
quarto render weeks/01-intro/slides.qmd
```

`quarto render` produces **both** outputs from the single `slides.qmd`:

- `slides.html` — self-contained reveal.js deck (a build artifact, gitignored)
- `lab.ipynb` — the student notebook, **committed** because Colab loads it from git

A failing code cell aborts the render, so a broken example can never reach a
lecture. After a render, check the notebook the way a student's *Run All* would
and look at every figure it produced:

```bash
uv run scripts/check_notebook.py run    weeks/04-walking/lab.ipynb   # execute every cell
uv run scripts/check_notebook.py images weeks/04-walking/lab.ipynb _figs   # dump the PNGs
```

### Answer files

Reference answers live in `weeks/*/instructor/`, which is gitignored: this
repository is public. They are pushed only encrypted, as
`instructor.tar.gz.gpg`:

```bash
uv run scripts/instructor.py open    # passphrase -> restores weeks/*/instructor/
uv run scripts/instructor.py seal    # after editing an answer; then commit the .gpg
```

`open` refuses to overwrite a local answer file that differs from the archive
(`--force` to override). The passphrase is kept outside the repository; without
it the archive cannot be opened.

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
uv run scripts/view.py --static --keyframe stand           # orbit, physics off
uv run scripts/view.py --pose=-0.35,0,0,0.70,-0.35,0       # place the left leg (week 2)
uv run scripts/view.py --robot robotis_op3 --keyframe home
uv run scripts/view.py --list             # every humanoid available
```

Double-click a body to select it, then **ctrl-drag to push the robot** — the
quickest way to find out whether a controller survives a disturbance, and much
more informative than a rendered video.

`soc4180.launch_viewer(model, data, passive=True, key_callback=...)` is the
underlying helper if you want to drive the loop yourself; every `lab_*.py`
uses it, and `SOC4180_AUTOCLOSE=6 uv run weeks/NN-*/lab_*.py` closes the window
after six seconds, which is how the scripts are smoke-tested.

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
weeks/NN-*/     slides.qmd (source) -> slides.html + lab.ipynb; lab_*.py (laptop lab)
scripts/         view.py (viewer), check_notebook.py (execute / compile / dump figures), build_site.py,
                 check_labs.py (run every lab script), instructor.py (seal / open the answers)
instructor.tar.gz.gpg   the encrypted weeks/*/instructor/ answers
_quarto.yml      shared deck theme and execution settings
```

Helpers live in one installed package rather than being copy-pasted per week, so
a fix to the render path lands in every lab at once.
