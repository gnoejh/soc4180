# Week 7 — From Control to Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w07-mdp/lab.ipynb)

**The hinge week.** Six weeks of deriving controllers end; specifying problems
begins. No learning code appears — only the problem definition.

| | |
| --- | --- |
| **Runtime** | Local: CPU. **Colab: pick a GPU runtime** (Runtime > Change runtime type > T4). |
| **Wall clock** | ~1 min |
| **Convergence risk** | None. Nothing is trained. |
| **Needs** | `soc4180[env]` — gymnasium. Lighter than `[rl]`: no torch, no SB3. |
| **Feeds** | Weeks 8–12, which all train in this environment |

## Objectives

1. State the MDP formalism and identify each element in the walking task.
2. Explain why the problem is really partially observable, and how that is
   normally papered over.
3. Justify residual actions, a 50 Hz control rate, and the termination rules.
4. Validate an environment against `gymnasium.utils.env_checker`.
5. **Test whether the environment can represent a solution you already have.**

## Measured results

| Policy | Return | Outcome |
| --- | --- | --- |
| Do nothing (hold the crouch) | 774.09 | survived all 500 steps |
| Uniform random | 22.47 | fell in 25 steps (0.5 s) |
| Analytic walker (Week 4) | — | **fell after 4.2 s, 0.24 m** |

The Week 4 walker **fails inside its own environment**, with actions clipped on
**35.7%** of steps. The diagnosis separates two causes:

| action_scale | 50 Hz | 100 Hz | 200 Hz |
| --- | --- | --- | --- |
| 0.3 | fell (36% clipped) | fell | fell |
| 0.6 | fell (18% clipped) | fell | fell |
| 1.0 | fell (0% clipped) | **survived, 1.05 m** | **survived, 1.05 m** |

Both variables matter independently. Only `action_scale = 1.0` **and** ≥100 Hz
reproduces the walk (1.05 m against 0.99 m at 500 Hz).

## The two lessons

**An action space defines the set of reachable behaviours.** A solution outside
it does not exist in your problem, and no learning curve will ever say so — the
run simply plateaus. The defaults stay at 0.3 / 50 Hz because that is what the
locomotion literature trains with, and learned policies do succeed inside that
envelope; they just find *different* gaits from ours. An environment is a
commitment about what kind of solution you expect.

**Return is not comparable across control rates.** The 200 Hz runs score ~5×
the 50 Hz ones purely because they contain more steps. Report distance, time
upright, or mean velocity alongside it.

## Rebuild

```bash
quarto render weeks/w07-mdp/slides.qmd
```
