# Week 1 — Robots, Simulation, and MuJoCo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w01-intro/lab.ipynb)

| | |
| --- | --- |
| **Runtime** | CPU only — local or Colab. No GPU needed. |
| **Wall clock** | ~1 min (first run adds ~5 s to fetch the G1 model) |
| **Convergence risk** | None. No learning this week. |

## Objectives

By the end of this lab a student can:

1. Explain why robot learning happens in simulation, and name three ways
   simulation misleads.
2. Read an MJCF body tree and identify the kinematic chain.
3. Load a Menagerie robot, step the physics, and render video.
4. Interpret `nq`, `nv`, and `nu`, and explain why `nq > nv` for a floating-base
   robot.

## Expected output

A video of the Unitree G1 collapsing under gravity. **The robot falling is the
correct result** — it motivates every remaining week.

## Files

- `slides.qmd` — the single source. Renders to both outputs.
- `slides.html` — reveal.js lecture deck (video embedded, self-contained).
- `lab.ipynb` — student notebook, opened by the Colab badge above.

## Rebuild

```bash
quarto render weeks/w01-intro/slides.qmd
```
