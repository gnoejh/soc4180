# 06 — Sensing and State Estimation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/06-sensing/lab.ipynb)

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

## Lab class: on your laptop

```bash
uv run weeks/06-sensing/lab_imu.py
uv run weeks/06-sensing/imu.py --walk --noise --sweep
```

Three arrows hang from the torso IMU, each an estimate of *down* made from the
IMU alone and drawn back in the world, so **right means vertical**: orange the
accelerometer, cyan the integrated gyroscope, green the complementary filter,
white the truth. `W` walks, `N` injects a gyro bias (0.02 rad/s) and
accelerometer noise (σ 0.5 m/s²), `[` `]` sweep α, `ENTER` prints each
estimate's mean and final error over the last three seconds, and ctrl-drag
pushes the robot — the best experiment here.

**The code is complete and explained**: `my_filter` is the complementary
filter with predict / correct / blend spelled out and τ = −Δt/ln α in its
docstring; `down_in_world` and `true_tilt` turn a (roll, pitch) into an arrow
and into the answer key; the loop names `read_imu` (six numbers out of
`sensordata`) and `mj_step`.

| Step | Do | Right looks like (measured with `imu.py`) |
| --- | --- | --- |
| 1 | ctrl-drag the standing robot | orange swings during the push and is right after; green barely moves |
| 2 | `N`, `ENTER` twice | cyan leans further every time: 0.02 rad/s is 1.1° per second, for ever |
| 3 | break `my_filter`: α = 1, then α = 0 | green becomes cyan, then orange: a low-pass plus a high-pass, nothing more |
| 4 | `W`, then `[` `]` | on the noisy walk: accelerometer 7.8° mean, gyro 3.4°, filter 1.8° at α = 0.995; the sweep bottoms out at α ≈ 0.998 (1.15°) |
| 5 | `SITE = "pelvis"` | the pelvis accelerometer is worse (8.2°) and its gyro better (0.9°); the same α is no longer the best — explained either way |

### `imu.py`: three estimators, scored

```bash
uv run weeks/06-sensing/imu.py                       # standing, clean sensors
uv run weeks/06-sensing/imu.py --walk --noise
uv run weeks/06-sensing/imu.py --walk --noise --sweep --no-viewer
uv run weeks/06-sensing/imu.py --site pelvis --walk
```

Runs the stand or the walk, updates the three estimators every physics step
from `read_imu`, scores them against `site_xmat` (which a real robot never
has), prints a row per second and the mean and final errors, optionally sweeps
twelve values of α, then replays with the four arrows.

| Step | Does | Calls |
| --- | --- | --- |
| 1 | the robot and its IMU (`nsensor = 4` is two IMUs) | `mj_name2id(imu_in_torso)`, `read_imu` |
| 2 | the three estimators, each a few lines | `tilt_from_accel`, `integrate_gyro`, `complementary` |
| 3 | the loop, a row per second | `mj_step`, `sensordata`, `site_xmat` |
| 4 | mean and final error | |
| 5 | `--sweep`: α from 0.9 to 1 | the same loop, twelve times |
| 6 | replay with arrows | `launch_viewer(passive=True)`, `mju_quatZ2Vec` |

Measured, walking with the biased gyro and noisy accelerometer, 6 s:

| α | τ | mean error | final error |
| --- | --- | --- | --- |
| 0.99 | 0.20 s | 2.77° | 3.25° |
| 0.995 | 0.40 s | 1.79° | 2.06° |
| 0.998 | 1.00 s | **1.15°** | 0.36° |
| 0.9985 | 1.33 s | 1.24° | **0.36°** |
| 0.999 | 2.00 s | 1.53° | 1.24° |
| 1 (bare gyro) | ∞ | 3.43° | 6.58° |

Standing with clean sensors every estimate is within 0.5°; the filter's only
job then is the push.

## Rebuild

```bash
quarto render weeks/06-sensing/slides.qmd
```
