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

## Three failures found while preparing this lab

All three were hit for real, and all three are in the slides.

1. **`ModuleNotFoundError: mujoco_playground`** — not in `soc4180[rl]`. Install
   plain `playground`, never `playground[all]` (its `jax[cuda12]` dependency
   reinstalls JAX over Colab's GPU build and silently drops you to CPU).
2. **`AttributeError: type object 'int' has no attribute 'WARP'`** — the env
   config defaults to `impl="warp"` (MJWarp), which needs the separate
   `mujoco-warp` package. **Fix: `registry.load(name, config_overrides={"impl":
   "jax"})`** — verified working here.
3. **`AttributeError: jax.device_put_replicated is deprecated`** — brax 0.14.2
   (latest) calls an API JAX 0.11 removed, and brax requires only `jax>=0.4.6`
   with no upper bound. Colab's older preinstalled JAX works. **Do not upgrade
   JAX on Colab.**

Also verified: playground's `registry.load` clones its **own** Menagerie copy
(~40 s), separate from the `mujoco-menagerie` package `soc4180` uses. And a
playground env is not a brax env — training needs
`wrap_env_fn=wrapper.wrap_for_brax_training`, confirmed accepted by
`brax...ppo.train`.

## Not verified

**The GPU training cell is `eval: false` and has never completed a run.** JAX's
CUDA wheels are Linux-only, so it cannot be tested on the authoring machine. The
env *loads* here on CPU JAX with `impl="jax"` (action size 12, observation
`{state: (52,), privileged_state: (114,)}`), and the `ppo.train` call signature
is verified — but the CPU smoke test then stops at failure 3 above.

Before teaching this week, run Track A once on a Colab T4, confirm it gets past
all three failures, and set `num_timesteps` so it lands at 12–15 minutes.

## Rebuild

```bash
quarto render weeks/w10-scaling/slides.qmd
```
