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
- Step response of the knee (gravity off, robot lifted 0.5 m so nothing touches
  the floor): $k_p \times 4$ reaches 1% of target in 0.032 s instead of 0.110 s
  with 4.0% overshoot and 4x the torque; $k_v / 4$ overshoots 16.9%; $k_v \times 4$
  is not within 1% after 0.5 s. The one-joint formula ($M_{ii} = 0.1256$,
  $\omega_n = 63.1$ rad/s) predicts 0.106 s nominal (measured 0.110) but 44.4%
  overshoot for $k_v/4$: the floating body's "free" inertia $1/(M^{-1})_{ii} = 0.0412$
  predicts 21.8%, and the robot sits between the two limits. (An earlier version
  quoted 9% — that run let a foot touch the floor mid-step; `servo.py` does too,
  hence its 4.6%.)
- P / PD / PI / PID on the knee (0.5 rad step, constant 10 N·m load, joint
  damping ζ 0.15; slides only, no script): P 19 ms rise, 55.6 % overshoot, 20 mrad
  final error; PD 78 ms, 0.0 %, 20 mrad; PI 19 ms, 57.0 %, 1.7 mrad at 4 s; PID
  72 ms, 0.0 %, 0.8 mrad. Only the I term removes the sag, and the G1's servo is
  PD. PD's rise is slower than P's: P is quick because it rings.
- Sag, one joint (left arm straight out, shoulder only re-tuned): gravity torque
  4.73 N·m, sag 9.45 mrad at $k_p = 500$ against $\tau_g/k_p = 9.46$; the $1/k_p$ law
  holds to 0.05 mrad from 125 to 2000.
- Saturation, same arm: gravity's largest pull is 5.09 N·m. A 6 N·m limit holds;
  4 and 3 N·m give way and stop inside the band where $\tau_g = \tau_{\max} \pm 0.3$
  (joint friction); 2 N·m stops 0.4° short of its band.
- Coupled phase oscillators ($\dot e = \Delta\omega - 2w\sin e$): $w = 2$ locks a
  0.1 Hz mismatch at $e^* = 0.158$ rad as predicted; $w = 0.25$ drifts. The
  package's `CPG` has no coupling term — it is the locked solution written down.
- Sag against stiffness: 11.0 mm at 500, 9.4 at 600, 6.0 at 750 — the $1/k_p$
  spring law — then a fall at 1000. Below 500 it does not sag more, it falls (400 collapses).
- The knee exceeds 50 N·m for only **4.0%** of the walk, 55 N·m for 2.1%.
- Walking sag: **11.0 mm mean, 27.1 mm worst**, peak joint torque **124 N·m**.
- **`torque_limit(model)` is `None`** — the Menagerie G1's motors are infinitely
  strong. The gait needs **> 50 N·m**: it falls at 50 and walks at 55.
- Gain sweep: only the nominal `kp = 500` walks. 0.25× and 0.5× collapse; 2× and
  4× fall. Scaling `kv` as $\sqrt{k_p}$ changes nothing.
- CPG: **all nine parameter settings fall** (3 frequencies × 3 amplitudes),
  between 1.9 and 3.2 s in — now computed in the deck rather than tabulated.

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

**`L` tries variations of the law**: it restarts the walk with the next entry
of `LAWS` (`PD`, `P`, `PI`, `PID`, `D`), already handed over; `--law PID`
starts with one. `"PD"` is the student's `torque_from_pd`; the others are built
from its terms, one per letter, with `ki = KI_RATIO * kp` (0.05).

| Step | Do | Right looks like (measured with `servo.py`) |
| --- | --- | --- |
| 1 | `ENTER` | gap ~1e-12; the two terms explained |
| 2 | `H`, then delete the damping term and `R` `H` | identical walk, then ringing |
| 3 | `2` (50 N·m) then `3` (55 N·m) | knees go red at support exchange; 50 falls (sag 365 mm), 55 walks 0.65 m |
| 4 | `4` and `5` (half and double $k_p$) | half sinks and falls; double is thrown at 5.0 s with 327 N·m at the knee |
| 5 | `SERVO_HZ = 500`, `H` | it blows up; explained with week 1 |
| 6 | the lowest $k_p$ that walks eight steps, in `SERVOS` | `ENTER` prints its sag; sag against $k_p$ on paper (11 mm at 500) |
| 7 | `L` through `LAWS`, `ENTER` at the end | PD walks 0.65 m (sag 10.8 mm); P falls at 2.4 s, PI at 2.5 s; PID walks 0.63 m (sag 9.8 mm); D alone falls at 1.0 s; `KI_RATIO = 0.2` falls |

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
