# Week 0 — What a Robot Is: The Five-Layer Stack

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w00-robot-stack/lab.ipynb)

**Day one. Taught before Week 1** — this is the vocabulary lecture the rest of
the semester refers back to.

| | |
| --- | --- |
| **Runtime** | CPU only — local or Colab. No GPU needed. |
| **Wall clock** | ~1 min |
| **Convergence risk** | None. No learning. |

## Objectives

A student who has seen this lecture can:

1. Name the five layers and say what each is responsible for.
2. Explain why layering exists — the three-orders-of-magnitude spread in rates,
   from a 500 Hz physics step to a sub-1 Hz task decision.
3. Map a real robot onto a simulated one, and identify **Layer 1 as the only
   layer that does not transfer** — which is exactly where sim-to-real lives.
4. Explain why `ctrl = 0` is *not* an uncontrolled robot.
5. Ask of any robotics claim: *which layer is this, and what does it assume
   about the layers beneath it?*

## The demonstration

Two videos of the same robot, same physics, same starting pose:

- **actuation disabled** — a rag doll; the torso drops 0.790 m → 0.134 m
- **servos holding `stand`** — upright and stable

That contrast *is* Layer 4. Standing is a property of the loop, not of the robot.

## Departures from the classic five-layer diagram

Three deliberate refinements, each of which heads off a common confusion:

1. **Layer 1 is "Physics / World", not "Physics Engine (simulation only)".**
   A real robot's Layer 1 is reality itself. Framing it this way makes the real
   and simulated stacks line up row for row, so the sim-to-real gap becomes a
   visible asymmetry in a single table rather than a separate topic.
2. **Models are not simulation-only.** Real robots carry models too — IK needs
   link lengths, MPC needs dynamics. The model describes the *robot*; the engine
   simulates the *world*.
3. **Timescales are given as the reason layers exist**, rather than layering
   being presented as mere tidiness.

## Files

- `slides.qmd` — single source
- `slides.html` — reveal.js deck, videos embedded (build artifact, gitignored)
- `lab.ipynb` — student notebook, committed for the Colab badge

## Rebuild

```bash
quarto render weeks/w00-robot-stack/slides.qmd
```
