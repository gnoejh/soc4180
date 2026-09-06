# Week 6 — Sensing and State Estimation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w06-sensing/lab.ipynb)

| | |
| --- | --- |
| **Runtime** | Local: CPU. **Colab: pick a GPU runtime** (Runtime > Change runtime type > T4). |
| **Wall clock** | ~1 min |
| **Convergence risk** | None. Nothing is trained. |
| **Feeds** | Week 7 onward — the observation vector built here is the policy input |

## Objectives

1. Distinguish what a real robot can measure from what the simulator merely knows.
2. Explain specific force, and why an accelerometer is a tilt sensor.
3. Show why gravity gives roll and pitch but never yaw.
4. Implement a complementary filter and justify the choice of $\alpha$.
5. Design an observation vector containing only deployable quantities.

## Measured results

- The G1's entire sensory world is **12 numbers**: two IMUs (torso, pelvis), each
  a 3-axis gyroscope and accelerometer.
- At rest the accelerometer reads **9.81 upward**; in **free fall it reads zero**.
- Sensors are **exactly deterministic** — two identical runs give identical
  `sensordata`. The MJCF declares noise (0.0005, 0.01) but this MuJoCo build
  applies none, so noise and bias must be injected by hand.
- During the Week 4 walk, `|accel|` swings **3.36 – 19.34 m/s²**. Tilt-from-gravity
  therefore fails: **6.09° mean error, 20.59° worst**.
- Gyroscope integration is near-exact with a perfect sensor (0.13° final) and
  drifts to **2.85°** with a 0.01 rad/s bias.
- Complementary filter at $\alpha = 0.995$: **1.46° mean, 1.37° final**.

## The metric that matters

$\alpha = 1.0$ (pure gyroscope) has a marginally *lower mean* error than the
filter, and twice the final error. That gap is drift, and drift only grows —
which is why **mean error is the wrong way to judge an estimator you intend to
run for an hour.** Students should be pushed to notice this in the table before
being told.

## Observation design

The lab ends by assembling a **64-number** observation — gravity in body frame,
angular velocity, joint angles, joint velocities — every element of which exists
on real hardware. Body height and world position are explicitly excluded as
privileged, with asymmetric actor-critic introduced as the principled way to use
privileged data without making the policy undeployable.

## Rebuild

```bash
quarto render weeks/w06-sensing/slides.qmd
```
