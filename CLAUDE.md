# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Undergraduate course "Robot and AI" (SOC4180): 15 weeks of slides + runnable
labs, taught through MuJoCo simulation. The spine is a **walking humanoid**
(Unitree G1) — classical control first, learned control second.

**Simulation only. No hardware exists.** Never propose a lab that needs a
physical robot.

**Week 0 is the day-one stack/vocabulary lecture, taught before Week 1.** Every
week names which of its five layers it belongs to.

## Syllabus (revised)

This replaces the older course description, which covered deep learning broadly
(CNNs, RNNs, Transformers, diffusion, CLIP, agentic systems), computational
neuroscience, MicroDuck and NVIDIA IsaacSim. **Those are deliberately out of
scope.** MicroDuck went because there is no hardware; IsaacSim because it cannot
run on Colab, which the zero-install requirement depends on. Generative and
agentic content returns only where it attaches to the robot, in weeks 14–15.

| Wk | Topic | Deck | Status |
| --- | --- | --- | --- |
| 1 | The five-layer robot stack; MuJoCo and MJCF | `w00`, `w01` | built, Colab-verified |
| 2 | Transforms and forward kinematics | `w02` | built, Colab-verified |
| 3 | Inverse kinematics | `w03` | built, Colab-verified |
| 4 | Contact, balance, analytic walking (LIPM/ZMP) | `w04` | built, Colab-verified |
| 5 | Actuation, PD control, and CPG gaits | `w05` | built, Colab-verified |
| 6 | Sensing, state estimation, observation design | `w06` | built, Colab-verified |
| 7 | From control to learning: MDPs and environment design | `w07` | built, Colab-verified |
| 8 | Policy gradients and PPO | `w08` | built, Colab-verified |
| 9 | Reward shaping and diagnosing failed runs | `w09` | built, Colab-verified |
| 10 | Scaling: GPU-parallel locomotion training | `w10` | built; CPU parts verified, **GPU run untimed** |
| 11 | Domain randomization and robustness | — | not written |
| 12 | Sim-to-real, measured | — | not written |
| 13 | Perception and imitation | — | not written |
| 14 | Vision-language-action: grounding instructions | — | **blocked** |
| 15 | Agentic robotics: perception, reasoning, action | — | **blocked** |

Capstone presentations occupy the final-exam slot. Weeks 0 and 1 share the first
session — the stack lecture is short, the MuJoCo lab is hands-on.

Weeks 1–7 are built and confirmed on Colab; week 8 is built but only run locally.

**Week 8 facts, measured.** REINFORCE from scratch on CartPole: 60 -> 489 in
173 s, using a **batch of 8 episodes per update** — with one episode per update
identical code reached 260 and 88 on two runs, so batching is required for the
lab to be reliable. PPO solves InvertedPendulum (26 -> 1000, 99 s). On
`G1WalkEnv`, PPO **gets worse**: untrained scores 774.3 (exactly the
hold-the-crouch baseline, because residual actions start near zero), and after
30k steps scores 94.7.

**The cause is exploration noise, NOT reward hacking.** An earlier version of
these slides claimed PPO had found a reward loophole; that was wrong and the
numbers disproved it (94.7 < 774.3, so nothing was exploited). Evaluating the
reward by hand: walking at the target is worth 1250 over an episode, standing
776, lunging-and-falling 52. The reward is fine.

What actually happens: PPO's initial action std is 1.0 on a `[-1, 1]` action
space. The untrained **deterministic** policy survives all 500 steps; the
**stochastic** policy PPO collects data from survives 38. Over 90% of training
experience is a fall. `log_std_init=-2.0` (std 0.135) raises stochastic survival
to ~180 and the same 30k run then scores 776 instead of 95.

Even fixed, it only matches the standing baseline — it has learned not to fall,
not to walk (which is worth 1250). 30k steps is 0.02% of a real run.

Week 8 is ~6 min; it trains four times.

**Week 9 facts, measured.** The reward's *ranking* is correct — walking at target
1250, slow walk 1023, standing 776, lunge-and-fall 51. But the planned six-way
ablation **failed to discriminate**: every variant stood still for all 500 steps
at 30k steps, because none of our five terms changes whether standing is
attractive. The real G1 reward in MuJoCo Playground has **24 terms**, notably
`feet_air_time=+2.0`, `stand_still=-1.0`, and `alive=0.0` with
`termination=-100.0` (a per-step alive bonus is what makes standing profitable).
Adding `stand_still` or `air_time` alone changes nothing; **both together** break
the standing optimum — 224 steps, −0.524 m, feet lifted 0.08 — i.e. it falls over
backwards. Shaping picks which optimum you land in; it does not buy the search.
`G1WalkEnv` now takes `reward_weights`; see `DEFAULT_REWARD` in `envs.py`.

### Week 10 — scaling

**Measured locally (36-core Windows).** Single env 1308 control steps/s (13,083
physics steps/s), so 150M steps is ~32 hours. `DummyVecEnv` gives **no speedup**
(1173/1240/1250 for 1/4/16 envs) — it steps sequentially, so vectorised is not
parallel. `SubprocVecEnv` does scale (1649/3205/5570 for 2/4/8) but is capped by
core count; a Colab CPU runtime has 2, against a tuned config asking for 8192.

**Read from the library, not memory.** Berkeley Humanoid is 150M timesteps at
8192 envs, Op3 100M. Network: policy (512,256,128) on `state`, value
(512,256,128) on **`privileged_state`** — the week-6 asymmetric actor-critic in
production.

#### The GPU path: five failures, all hit for real

1. `ModuleNotFoundError: mujoco_playground` — not in `soc4180[rl]`; install
   `playground` separately.
2. `AttributeError: type object 'int' has no attribute 'WARP'` — env configs
   default to `impl="warp"` (MJWarp), needing the separate `mujoco-warp`
   package. Use `registry.load(name, config_overrides={"impl": "jax"})`.
3. `'State' object has no attribute 'pipeline_state'` — a playground env is not
   a brax env. Training needs `wrap_env_fn=wrapper.wrap_for_brax_training`.
4. **JAX 0.11 removed two APIs still called by flax
   (`jax.core.get_opaque_trace_state`) and brax (`jax.device_put_replicated`).**
   brax requires only `jax>=0.4.6`, no upper bound, so pip installs a broken
   pair — and **Colab ships 0.11, so it is broken out of the box.**
5. `NameError` in a later cell — caused by putting imports in the same cell as
   the install, so the auto-restart killed it before they ran.

#### The fix: patch, do not pin

`soc4180.jax_compat.patch_jax()` restores both removed functions
(`jax.core.get_opaque_trace_state` moved to `jax.extend.core`;
`jax.device_put_replicated` is a few lines of `jnp.stack`). **Verified: training
runs end to end on jax 0.11.1 with the shims.** Call it before brax/flax are used.

This replaced an earlier `jax[cuda12]==0.9.2` pin. The pin also works, but on
Colab it meant a large download *and* a forced runtime restart on **every fresh
session** — Colab always starts on JAX 0.11 — so every lab began with a
"session crashed for an unknown reason". Do not reintroduce it.

Still required regardless: `impl="jax"` in `registry.load`,
`wrap_env_fn=wrapper.wrap_for_brax_training`, `playground` installed separately
(never `playground[all]`), and dependency setup in a cell containing no imports.

Rendering this week needs `uv sync --extra rl --extra gpu --extra env` first; a
plain `uv run` re-syncs and drops `mujoco_playground`.

**Confirmed on Colab (A100):** the training cell **runs**. At 15M timesteps /
2048 envs it exceeded 10 minutes, so the default is now 5M with `num_evals=10`.
**Always pass `progress_fn`** — brax prints nothing without one, and a silent
ten-minute cell is indistinguishable from a hang. The first progress line is
slow regardless, because brax XLA-compiles the whole loop before stepping.

**Still uncalibrated:** no Track A run has been timed to completion, so
`num_timesteps=5M` is a reduction from a known-too-slow figure rather than a
measured one. Read seconds-per-step from the progress output and scale.

Ordering matters here: the verification cell must come **before** the training
cell. It originally landed after it, because the troubleshooting slides sat
after the "Running it" slide, so a Run All hit training first.

## Commands

```bash
uv sync                                        # lean env: mujoco + rendering only
uv sync --extra rl                             # + gymnasium, SB3, torch (CUDA on Windows)
uv sync --extra gpu                            # + JAX/MJX/playground (Linux/WSL2)
uv run python -c "import soc4180"              # smoke test
quarto render weeks/w01-intro/slides.qmd       # -> slides.html + lab.ipynb
```

There is no test suite or linter configured. Add the tooling before inventing
commands for it.

## Adding a week

The pipeline is proven; follow it rather than improvising.

1. `weeks/wNN-slug/slides.qmd`, with the notebook-only Colab header (below) and
   the `<slug>` updated.
2. Put reusable code in `src/soc4180/`, not in the slides. Slides show the idea;
   the package carries anything a later week needs.
3. **Verify a claim before writing it into a slide.** Several assertions in this
   repo were wrong until measured: that the robot falls with `ctrl=0`, that a
   residual was iteration-limited, that a leg segment was 0.194 m long. Run it.
4. `quarto render weeks/wNN-slug/slides.qmd` — a broken cell fails the render.
5. Execute the generated notebook standalone with `nbclient` before committing.
6. Add `weeks/wNN-slug/README.md` and a row in the top-level README table.
7. **Test it on Colab.** Every environment bug this project hit was invisible on
   Windows: the GL ordering bug, the dependency upgrades that broke the runtime,
   the unguarded renderer. Ask for `soc4180.gl_report()` when rendering fails.

## Authoring model: one source, two outputs

Each week is a single `weeks/wNN-*/slides.qmd` that Quarto renders into:

- `slides.html` — self-contained reveal.js deck (**gitignored build artifact**)
- `lab.ipynb` — student notebook (**committed**; Colab loads it from GitHub)

`execute.error: false` in `_quarto.yml` means a broken cell **fails the render** —
verified. Do not set `error: true`; that protection is the point.

Rendered decks are gitignored because reveal.js assets are ~5 MB/week. Only
`lab.ipynb` is committed.

### Install unconditionally, before the first import

**`try: import soc4180 / except ImportError: %pip install ...` is wrong for this
repo.** The package changes between labs, so on any runtime with an older copy
the import succeeds, the install is skipped, and a function added later is
missing — surfacing as an `AttributeError` for something that plainly exists in
the repository. This bit for real with `soc4180.jax_compat` in week 10.

Use instead, as the first thing in the notebook:

```python
%pip install -q --upgrade "soc4180[rl] @ git+https://github.com/gnoejh/soc4180.git"
import soc4180
```

Upgrade **before** importing: pip rewrites files on disk and cannot replace a
module the interpreter has already loaded. Weeks 0–9 still use the conditional
form; they are stable, but any week that gains new package features should be
switched.

### Standing pattern: every week starts with a notebook-only header

Immediately after the YAML frontmatter, before the first slide:

````
::: {.content-visible when-format="ipynb"}
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/<slug>/lab.ipynb)

**Before you start, two things:**

1. **Runtime -> Change runtime type -> T4 GPU.** ...
2. **File -> Save a copy in Drive.** ...
:::
````

`content-visible when-format="ipynb"` keeps it out of the deck — verified: the
badge appears in `lab.ipynb` and not in `slides.html`. Both notices matter:
without the GPU runtime nothing renders, and a GitHub-opened Colab notebook is
unsaved, so student work vanishes on tab close. Remember to update `<slug>`.

**Never edit `lab.ipynb` directly** — it is generated. Edit `slides.qmd` and
re-render. In particular do not use Colab's *Save a copy in GitHub* on these
notebooks; it would commit executed outputs over the generated file.

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

### Dependencies must not disturb a Colab runtime

`[project.dependencies]` is deliberately small and permissive: `mujoco`,
`mujoco-menagerie`, `mediapy`, `imageio-ffmpeg`, `numpy>=1.24`. That is exactly
what the package imports.

**Do not add torch, gymnasium or SB3 back to the required set.** Pinning
`torch>=2.14` and `numpy>=2.5.2` there broke a live Colab session: pip upgraded
both, which broke the preinstalled `torchvision` (wants `torch==2.11`) and
`numba` (wants `numpy<2.3`), and forced a kernel restart — for libraries the
package never imports. They live in the `rl` extra now. Exact local versions are
pinned in `uv.lock`, which is the right place for them.

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

### MUJOCO_GL ordering (this has already broken once)

MuJoCo resolves its render backend from `MUJOCO_GL` at `import mujoco` time.
Setting it afterwards does nothing — the process is stuck, and on a headless
machine that means GLFW dying on a missing X11 `DISPLAY`.

Selection therefore lives in **`src/soc4180/_gl.py`**, which imports nothing from
mujoco, and `__init__.py` imports it **first**. Do not move that import, and do
not let any module that imports mujoco be imported ahead of it.

**The original bug:** `__init__.py` imported `.models` first, and `models.py`
does `import mujoco` at module level — so the backend was chosen *after* mujoco
had already picked GLFW. It passed on Windows (the default backend renders
offscreen fine) and failed on Colab with
`an OpenGL platform library has not been loaded into this process`.

Regression test — this must stay true:

```bash
uv run python -c "import soc4180; from soc4180._gl import MUJOCO_WAS_PREIMPORTED; assert not MUJOCO_WAS_PREIMPORTED"
```

Backend choice (verified against simulated conditions):

| Environment | Backend |
| --- | --- |
| Colab / headless Linux **with** NVIDIA device | `egl` (writes the NVIDIA EGL ICD file if missing) |
| Colab / headless Linux **without** GPU | `osmesa` |
| Windows, macOS | MuJoCo default |
| `MUJOCO_GL` already set | respected, always |

**Colab needs a GPU runtime for rendering.** The `osmesa` fallback requires
`libosmesa6`, which Colab images do not reliably ship, so a CPU runtime cannot
render even though the physics runs fine. Week READMEs say to pick a T4.

**Every renderer is created through `render._new_renderer`.** It checks
`GL_UNAVAILABLE` and wraps construction failures with an actionable message. Do
not call `mujoco.Renderer` directly anywhere else — `render_poses` was added
without the guard and reproduced the raw
`FatalError: an OpenGL platform library has not been loaded` on Colab, which is
exactly the message the guard exists to prevent.

`soc4180.gl_report()` prints what selection saw (colab, NVIDIA device node,
OSMesa, DISPLAY, MUJOCO_GL, chosen backend). Ask for it first when someone
reports a rendering problem.

When neither EGL nor OSMesa is usable, `_gl` sets **no** environment variable and
records `GL_UNAVAILABLE` instead; `render_rollout` raises that message. This is
deliberate: exporting `PYOPENGL_PLATFORM=osmesa` when OSMesa is absent makes
`import OpenGL` itself die with a bare
`AttributeError: 'NoneType' object has no attribute 'glGetError'`, ten frames
deep and impossible to act on. Physics still works on a CPU runtime; only
rendering fails, and it fails with instructions.

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

## Cross-platform reproducibility

Verified on Colab T4 against local Windows:

| Demo | Windows | Colab T4 |
| --- | --- | --- |
| Servos holding `stand`, 3 s | 0.792 m | 0.792 m |
| Actuation disabled, 3 s | 0.134 m | 0.142 m |

Stable equilibria reproduce exactly across platforms; **chaotic trajectories do
not**. A collapsing robot amplifies floating-point and contact-solver
differences over 1500 steps, so the same seed gives different final numbers on
different hardware.

Consequence: **never assert exact float values from a fall, and never autograde
one.** Assert direction and magnitude instead (`fell more than 0.5 m`,
`stayed above 0.7 m`). `set_seed` makes a run repeatable on one machine; it does
not make it identical across machines.

## Kinematics (weeks 2-3)

`fk_foot` composes the leg chain by hand and matches MuJoCo's `site_xpos` to
**~1e-16**. Keep that as the week-2 acceptance test. The three ways to break it:
omitting the joint anchor (`a - Rj @ a`, since joints rotate about `jnt_pos`, not
the body origin), composing in the wrong order, and scalar-last quaternions —
MuJoCo is `(w, x, y, z)`.

**Leg segment lengths must be measured between joint anchors, not from
`|body_pos|`.** The hip is three separate link bodies whose offsets accumulate,
so `|body_pos|` of `knee_link` gives 0.194 m when the real thigh is **0.341 m**
(shin 0.300 m, so reach is an annulus from 0.041 m to 0.641 m). Getting this
wrong silently produced a reachability table calling 0.50 m unreachable.

The week-3 circle demo tracks to 8.4e-05 m. A taller circle leaves the reachable
set and pins at 2.68e-03 m regardless of iteration count — that invariance is
deliberate teaching material, not a solver defect. Do not "fix" it by raising
`iterations`; it does nothing.

`render_poses` draws a sequence of `qpos` without stepping physics — use it for
anything demonstrating kinematics, so the robot does not fall over mid-lesson.

## Walking (week 4)

`kinematics.py` (damped least-squares leg IK) and `walking.py` (LIPM + footstep
gait) make the G1 walk **~1.0 m in 9 s, open loop, with no learning**. Two bugs
cost real time here; do not reintroduce them.

**The `stand` keyframe is a kinematic singularity.** Every leg joint is exactly
zero, i.e. a perfectly straight leg, so the Jacobian has no direction that
shortens it and IK cannot lower the body at all (the knee range is
`[-0.087, 2.880]`, so it also clips immediately). Seed IK from the bent-knee
crouch in `WalkingController.nominal`, never from `stand`.

**Chain the LIPM boundary value problems.** Each step's CoM trajectory must start
where the previous step ended (`_build_segments`). Computing each step absolutely
instead teleports the commanded pelvis backwards by half a stride at every
support exchange — the robot then falls, and it looks like a balance problem
rather than the bookkeeping error it is.

Verified behaviour, useful as regression checks:

- total vertical ground reaction ≈ body weight (~327 N)
- measured ZMP y reaches ±0.26 m against stance feet at ±0.119 m — **outside the
  support polygon**. The LIPM's assumptions fail (point mass, constant height,
  massless legs, stiff servos), not the ZMP criterion. Teach this; do not "fix" it.
- the gait is genuinely sensitive: several nearby `GaitParams` settings fall over.
  Defaults came from a sweep, so re-sweep before changing one.

`render_rollout(..., track="pelvis")` follows a body with the camera. Any
locomotion video needs it — the robot leaves a fixed frame in about two seconds.

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
