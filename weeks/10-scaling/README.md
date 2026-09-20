# 10 — Scaling

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/10-scaling/lab.ipynb)

| | |
| --- | --- |
| **Runtime** | **GPU required** — the final section needs CUDA, not just rendering. |
| **Wall clock** | ~2 min for the measured sections; Track A adds ~15 min on GPU. |
| **Needs** | `soc4180[rl]` **and** `playground` — both installed by the setup cell. |
| **Platform** | The GPU section **cannot run on Windows** — JAX ships CUDA wheels for Linux only. |

## Objectives

1. Quantify the gap between our training budget and a real one.
2. Distinguish *vectorised* from *parallel*, and measure the difference.
3. Explain what `jit` and `vmap` buy, and what they cost in code style.
4. Read the tuned playground configuration and justify its choices.
5. Diagnose a learning curve from its shape.

## Measured results (local, 36-core Windows machine)

| Setup | Throughput |
| --- | --- |
| single environment | **1308 control steps/s** (13,083 physics steps/s) |
| `DummyVecEnv`, 1 / 4 / 16 envs | 1173 / 1240 / 1250 — **no gain** |
| `SubprocVecEnv`, 2 / 4 / 8 envs | 1649 / 3205 / 5570 — scales with cores |

**150M steps at the single-env rate is ~32 hours.** `DummyVecEnv` batches the
policy forward pass, which was never the bottleneck — the physics still runs one
environment at a time. "Vectorised" does not mean "parallel".

`SubprocVecEnv` genuinely parallelises but is capped by core count, and **a Colab
CPU runtime has two**. The tuned config asks for 8192 environments.

## Verified from the library

Read from `mujoco_playground`, not quoted from memory:

- `BerkeleyHumanoidJoystickFlatTerrain`: **150,000,000** timesteps, **8192**
  parallel envs, batch 256, discount 0.97
- `Op3Joystick`: 100,000,000 timesteps, 8192 envs
- Network: policy `(512, 256, 128)` reading `state`; value `(512, 256, 128)`
  reading **`privileged_state`**

That last line is the **asymmetric actor-critic** from Week 6, in production: the
critic sees what no real robot can measure, and is discarded after training.

Our entire Week 9 experiment was **0.027%** of one tuned run.

## Installing playground on Colab

The setup cell installs **`playground`, not `playground[all]`**. The `[all]`
extra depends on `jax[cuda12]`, which would reinstall JAX over Colab's
preinstalled GPU build and silently drop you onto CPU. Plain `playground`
declares `jax` with no version constraint, so pip leaves the existing install
alone.

This bit once: the first version of this lab installed only `soc4180[rl]`, and
the config-reading cells failed on Colab with `ModuleNotFoundError: No module
named 'mujoco_playground'`.

## The GPU path: four failures and the fix

Hit for real while preparing this lab, all now taught in the slides.

1. **`ModuleNotFoundError: mujoco_playground`** — not in `soc4180[rl]`.
2. **`type object 'int' has no attribute 'WARP'`** — env configs default to
   `impl="warp"` (MJWarp), needing the separate `mujoco-warp` package.
   **Fix: `registry.load(name, config_overrides={"impl": "jax"})`.**
3. **`'State' object has no attribute 'pipeline_state'`** — a playground env is
   not a brax env. **Fix: `wrap_env_fn=wrapper.wrap_for_brax_training`.**
4. **Two removed JAX APIs** — flax calls `jax.core.get_opaque_trace_state` and
   brax calls `jax.device_put_replicated`; JAX 0.11 removed both. brax requires
   only `jax>=0.4.6` with no upper bound, so pip installs a broken pair, and
   **Colab's preinstalled JAX is 0.11, so it is broken out of the box.**

### Never put imports in a cell that may restart the kernel

The dependency setup is its **own cell**, separate from every import. An earlier
version combined them, so the auto-restart killed the kernel partway through and
`from soc4180.envs import G1WalkEnv` never ran — leaving the *next* cell to fail
with `NameError: name 'G1WalkEnv' is not defined`, which looks nothing like the
install problem that caused it.

### Two ways this still bites after the fix

**The pin must be unconditional.** An early version of the setup cell installed
the pin only inside `except ImportError` for `playground`. If a previous attempt
already installed playground, that branch is skipped, the pin never applies, and
you get the cryptic JAX AttributeError much later. Check the *installed version*,
not whether an import succeeds.

**You must restart the runtime.** `pip` rewrites files on disk; it cannot replace
a module the interpreter already imported. If the traceback still names JAX 0.11
after a successful install, the runtime was not restarted — the `ipykernel_NNNN`
process id in the traceback path is the tell, since it stays the same across
re-runs.

### The fix: patch, do not pin

```python
soc4180.jax_compat.patch_jax()      # before brax or flax are used
```

Restores both functions — `get_opaque_trace_state` simply moved to
`jax.extend.core`, and `device_put_replicated` is a few lines of `jnp.stack`.
**Verified: a full tiny training run completes on jax 0.11.1 with the shims.**

An earlier version pinned `jax[cuda12]==0.9.2` instead. That works too, but on
Colab it required a large download **and a forced runtime restart on every fresh
session**, since Colab always starts on JAX 0.11 — so each lab opened with a
"session crashed for an unknown reason". The patch leaves Colab's own
GPU-enabled JAX untouched and needs no restart.

## Verified on Colab

With the pin and auto-restart in place, the check cell prints
`0.9.2 [CudaDevice(id=0)]` on a Colab A100. The dependency path is settled: the
recipe installs, the restart takes effect, and JAX sees the GPU.

## Training runs; budget still needs calibrating

Confirmed on a Colab A100: the training cell **runs**. At `num_timesteps=15M`
with `num_envs=2048` it was still going after **10 minutes**, so that figure is
too large for a 12–15 minute lab. The default is now **5M with `num_evals=10`**,
which reports progress roughly every 500k steps.

**brax prints nothing without a `progress_fn`.** The original cell had none, so a
ten-minute run showed no output at all and was indistinguishable from a hang.
The lab now passes one that prints step count, elapsed seconds, episode reward
and episode length at each evaluation.

Note the first line is slow regardless: brax compiles the whole training loop
through XLA before any steps run.

## Still not verified

**No Track A training run has completed.** The throughput numbers above are
CPU-only, since JAX has no Windows CUDA wheels. One timed run on Colab is needed
to set `num_timesteps` so the lab lands at 12–15 minutes — and note that an A100
will be substantially faster than the T4 the current figure assumes.

## Lab class: on your laptop

```bash
uv run weeks/10-scaling/lab_many.py
uv run weeks/10-scaling/many.py --robots 1 2 4 8 16 --procs 1 2 4 8 16
```

The GPU section is Colab-only, so the laptop lab measures the thing the GPU
buys — throughput — with the tools a laptop has. `N_ROBOTS` copies of the G1
are attached into **one scene** with `MjSpec` (names prefixed `r0_`, `r1_`,
…), so there is one model, one `MjData`, one `mj_step`, and one week 4
controller whose command is tiled across every robot. They walk side by side.

```
+ / -   one more / fewer robot (rebuilds the scene and reopens the window)
F       randomise each robot's foot friction (0.3–1.0) and mass (±20%): they diverge
P       benchmark 1, 2, 4, 8 processes with one robot each, two seconds
R       restart      ENTER   robot-steps/s measured vs predicted, x real time, hours for 150 M
```

**The code is complete and explained**: `predict_rate(n, single)` models one
process's throughput as `single · n^(1 − p)` with `COST_EXPONENT = 1` as
shipped (flat), `build` explains the `MjSpec` attach-and-compile, and the
loop names the tiled `ctrl` and the single `mj_step`.

| Step | Change | Right looks like (measured with `many.py`) |
| --- | --- | --- |
| 1 | `ENTER`; `+` three times with `ENTER` each; correct `COST_EXPONENT` | physics-only throughput is nearly flat (16,164 → 13,064 robot-steps/s from 1 to 16 robots, p ≈ 1.09); the walking number climbs because the one IK call is shared |
| 2 | `P` | 16,822 / 33,382 / 62,410 / 130,429 robot-steps/s for 1/2/4/8 processes here; 199,062 at 16 on 36 cores |
| 3 | arithmetic | 150 M steps: 2.5 h in one process, 0.2 h at 16 processes on this machine |
| 4 | `F` | robots that drift apart within a few steps; what a policy trained on all of them at once would have to learn |
| 5 | `+` until the window drops below 0.25× real time | the cost of one environment: ~62 µs of physics per robot-step here, 0.5 ms for the controller |

### `many.py`: robot-steps per second, two ways

```bash
uv run weeks/10-scaling/many.py
uv run weeks/10-scaling/many.py --robots 1 2 4 8 16 --seconds 3 --no-viewer
uv run weeks/10-scaling/many.py --robots 6 --randomise
```

For each scene size it measures two rates — holding the crouch (physics only,
what `predict_rate` models) and walking (physics plus one controller call per
step, shared by every robot) — fits the cost exponent, then benchmarks
processes with one robot each, turns the best rate into hours for 150 million
steps, and shows the largest scene walking. Run it on a quiet machine: it is a
timing measurement.

| Step | Does | Calls |
| --- | --- | --- |
| 1 | one robot alone | `load_g1`, `mj_step` in a loop |
| 2 | scenes of n robots, held and walking | `MjSpec.from_string/from_file`, `add_frame`, `attach_body`, `compile`, `np.tile(ctrl, n)` |
| 3 | processes, one robot each | `multiprocessing.get_context("spawn").Pool` |
| 4 | the 150 M budget | |
| 6 | the largest scene, walking live | `launch_viewer(passive=True)` |

Measured here (36 cores, quiet):

| n robots | nbody | physics only, robot-steps/s | walking, robot-steps/s |
| --- | --- | --- | --- |
| 1 | 31 | 16,164 | 1,544 |
| 2 | 61 | 14,835 | 2,542 |
| 4 | 121 | 14,686 | 4,448 |
| 8 | 241 | 12,544 | 6,844 |
| 16 | 481 | 13,064 | 8,656 |

The physics column is the point of the week: one process gets no more
physics out of more robots (p ≈ 1.09). The walking column rises only because
the controller — 0.5 ms of IK per step, eight times the physics — is computed
once and tiled. Processes scale until the cores run out.

`SOC4180_AUTOCLOSE=8 uv run weeks/10-scaling/lab_many.py` closes the window by
itself, which is how the script is smoke-tested.

## Rebuilding this week

`mujoco_playground` must be present or the config cells fail, and a plain
`uv run` re-syncs the environment and **drops the extras**. Sync explicitly
first:

```bash
uv sync --extra rl --extra gpu --extra env
quarto render weeks/10-scaling/slides.qmd
```

The setup cell's auto-restart calls `os.kill(os.getpid(), 9)` **only when
`soc4180.is_colab()`**, so rendering locally cannot kill its own kernel.

## Rebuild

```bash
quarto render weeks/10-scaling/slides.qmd
```
