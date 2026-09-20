# 05 — Actuation, PD Control, and Rhythm

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/05-actuation/lab.ipynb)

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
- `kp = 500` on every joint; `kv` ranges **4.55 – 43.01**, larger nearer the trunk —
  because **every leg joint is critically damped**: $\zeta = k_v / 2\sqrt{k_p M} = 1.00$
  against the mass-matrix diagonal, on all twelve. The arms sit at 0.7–1.9.
- Step response of the knee (gravity off): $k_p \times 4$ reaches 1% of target in
  0.034 s instead of 0.116 s with 2% overshoot and 4x the torque; $k_v / 4$
  overshoots 9%; $k_v \times 4$ is still 1% short after 0.5 s.
- Sag against stiffness: 11.0 mm at 500, 9.4 at 600, 6.0 at 750 — the $1/k_p$
  spring law — then a fall at 1000. Below 500 it does not sag more, it falls (400 collapses).
- The knee exceeds 50 N·m for only **4.0%** of the walk, 55 N·m for 2.1%.
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

## Lab class: on your laptop

```bash
uv run weeks/05-actuation/lab_servo.py
uv run weeks/05-actuation/servo.py --kp-scale 4
```

The walker live, with a sphere on every leg joint that turns from green to red
as its torque approaches the reference — the torque limit when one is set.
`SERVOS` lists (stiffness, damping, limit) settings; keys `1`–`6` select one and
restart. **The code is complete and explained**: `torque_from_pd` is the
actuator law with both terms explained, and `ENTER` reports how far it is from
MuJoCo's `actuator_force` (1e-12). **`H` hands the servos over**: the model's
gains are zeroed and the law's torques go straight into `qfrc_applied` at
1 kHz, because an explicit spring this stiff is unstable at 500 Hz (week 1,
exercise 14; the engine integrates its own servo implicitly). A correct law
changes nothing; break it and the robot is the week 0 rag doll.

| Step | Do | Right looks like (measured with `servo.py`) |
| --- | --- | --- |
| 1 | `ENTER` | gap ~1e-12; the two terms explained |
| 2 | `H`, then delete the damping term and `R` `H` | identical walk, then ringing |
| 3 | `2` (50 N·m) then `3` (55 N·m) | knees go red at support exchange; 50 falls (sag 365 mm), 55 walks 0.65 m |
| 4 | `4` and `5` (half and double $k_p$) | half sinks and falls; double is thrown at 5.0 s with 327 N·m at the knee |
| 5 | `SERVO_HZ = 500`, `H` | it blows up; explained with week 1 |
| 6 | the lowest $k_p$ that walks eight steps, in `SERVOS` | `ENTER` prints its sag; sag against $k_p$ on paper (11 mm at 500) |

### `servo.py`: one servo, measured

```bash
uv run weeks/05-actuation/servo.py                          # left knee, step 0.5 rad, gravity off
uv run weeks/05-actuation/servo.py --kp-scale 4
uv run weeks/05-actuation/servo.py --kv-scale 0.25
uv run weeks/05-actuation/servo.py --walk --limit 50
```

Step-response mode holds `stand`, commands one joint to jump by `--step`, and
prints the law by hand against `actuator_force`, the damping ratio from the
mass matrix, the time to 1 % of the step, the overshoot and the peak torque,
then replays it. `--walk` runs the week 4 gait under the scaled gains and an
optional limit, printing the pelvis sag and the loudest joint every second.

| Step | Does | Calls |
| --- | --- | --- |
| 1 | scale the gains, set the limit | `scale_gains` (`actuator_gainprm`, `actuator_biasprm`), `set_torque_limit` (`actuator_forcerange`) |
| 2 | ζ from the mass matrix | `mj_fullM`, `jnt_dofadr[actuator_trnid]` |
| 3 | the step (or the walk), a row every 50 ms (or 1 s) | `ctrl`, `mj_step`, `actuator_force`, `targets_at` |
| 4 | rise time, overshoot, peak torque, the gap | |
| 6 | replay | `launch_viewer(passive=True)` |

Measured with it, left knee, 0.5 rad step, gravity off:

| gains | ζ | 1 % reached | overshoot | peak torque |
| --- | --- | --- | --- | --- |
| as shipped (kp 500, kv 15.85) | 1.00 | 0.132 s | 0.1 % | 250 N·m |
| kp × 4 | 0.50 | 0.036 s | 2.1 % | 1000 N·m |
| kv / 4 | 0.25 | 0.034 s | 4.6 % | 250 N·m |
| kv × 4 | 4.00 | 0.600 s | 0 % | 250 N·m |
| limit 50 N·m | 1.00 | 0.136 s | 0.1 % | 50 N·m |

The law by hand matches `actuator_force` to 1e-13 in every unclipped case.
Walking: as shipped 0.66 m in 7 s with 11 mm sag and knees peaking at 57 N·m;
limit 50 falls, 55 walks; half stiffness sinks and falls, double stiffness is
thrown at 5.0 s, quarter damping falls.

## Rebuild

```bash
quarto render weeks/05-actuation/slides.qmd
```
