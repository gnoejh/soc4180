# Week 10 — Scaling

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w10-scaling/lab.ipynb)

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

## Still not verified

**No Track A training run has completed.** The throughput numbers above are
CPU-only, since JAX has no Windows CUDA wheels. One timed run on Colab is needed
to set `num_timesteps` so the lab lands at 12–15 minutes — and note that an A100
will be substantially faster than the T4 the current figure assumes.

## Rebuilding this week

`mujoco_playground` must be present or the config cells fail, and a plain
`uv run` re-syncs the environment and **drops the extras**. Sync explicitly
first:

```bash
uv sync --extra rl --extra gpu --extra env
quarto render weeks/w10-scaling/slides.qmd
```

The setup cell's auto-restart calls `os.kill(os.getpid(), 9)` **only when
`soc4180.is_colab()`**, so rendering locally cannot kill its own kernel.

## Rebuild

```bash
quarto render weeks/w10-scaling/slides.qmd
```
