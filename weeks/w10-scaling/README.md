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

### The verified recipe

```bash
pip install -q "jax[cuda12]==0.9.2" playground
```

`jax==0.9.2` is the newest release retaining **both** removed APIs. Verified end
to end in a clean environment: env load, `ppo.train`, and a complete tiny
training run (130 s on CPU, 6054 steps/s).

Install with the **`[cuda12]` extra**. A bare `jax==0.9.2` is CPU-only, runs
without error, and is roughly a hundred times slower — a bug that never raises.

This **reverses** earlier advice in this repo to leave Colab's JAX alone. That
was right about `playground[all]` clobbering a working GPU build, and wrong
about the build being usable: JAX 0.11 breaks brax and flax regardless.

## Still not verified

The GPU cell remains `eval: false`. Everything above is verified **on CPU**; no
CUDA run has happened, because JAX has no Windows CUDA wheels. Track A needs one
timed run on a Colab T4 to set `num_timesteps` for a 12–15 minute lab.

## Rebuild

```bash
quarto render weeks/w10-scaling/slides.qmd
```
