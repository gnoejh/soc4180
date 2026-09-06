# Week 5 — Actuation, PD Control, and Rhythm

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w05-actuation/lab.ipynb)

| | |
| --- | --- |
| **Runtime** | Local: CPU. **Colab: pick a GPU runtime** (Runtime > Change runtime type > T4). |
| **Wall clock** | ~1 min (several 8-step walking sweeps) |
| **Convergence risk** | None. Nothing is trained. |
| **Depends on** | Week 4's walker, reused as the test load |

## Objectives

1. State the actuator law $\tau = k_p(\text{ctrl}-q) - k_v\dot q$ and verify it
   against the simulator.
2. Explain the Week 4 pelvis sag as spring deflection under load, not a bug.
3. Discover that the model has **no torque limit**, and find where imposing one
   breaks the gait.
4. Show that the gait survives only a narrow gain window — and that holding the
   damping ratio constant does not widen it.
5. Explain why a central pattern generator produces rhythm but not balance.

## Measured results

- Actuator law predicts torque **exactly** (−150.000 predicted, −150.000 actual).
- `kp = 500` on every joint; `kv` ranges **4.55 – 43.01**, larger nearer the trunk.
- Walking sag: **11.0 mm mean, 27.1 mm worst**, peak joint torque **124 N·m**.
- **`torque_limit(model)` is `None`** — the Menagerie G1's motors are infinitely
  strong. The gait needs **> 50 N·m**: it falls at 50 and walks at 55.
- Gain sweep: only the nominal `kp = 500` walks. 0.25× and 0.5× collapse; 2× and
  4× fall. Scaling `kv` as $\sqrt{k_p}$ changes nothing.
- CPG: **all nine parameter settings fall** (3 frequencies × 3 amplitudes).

## The two lessons

**The textbook gain rule fails here, and that is correct.** Constant damping
ratio is a statement about one joint tracking a setpoint. The failure is a
whole-body gait whose *timing* was tuned against this plant. This is exactly what
domain randomization addresses in Week 11.

**Rhythm is not balance.** The CPG's legs move in a perfectly good walking
pattern and the robot falls regardless, because nothing keeps the centre of mass
over the support polygon. It states Week 4's value negatively: the LIPM was the
entire reason the robot stayed up. Real CPGs are entrained by sensory feedback —
which is Week 6.

## Rebuild

```bash
quarto render weeks/w05-actuation/slides.qmd
```
