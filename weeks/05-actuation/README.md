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
```

The walker live in the viewer, with a sphere at each of the twelve leg joints
coloured green→red by that actuator's torque against a reference (the torque
limit when one is set, else 100 N·m) and sized the same way. `SERVOS` at the top
is a list of (name, kp scale, kv scale, torque limit); keys `1`–`9` select and
restart.

```
1..9    pick a servo setting and restart     R   restart     SPACE   pause
H       hand the servos over to your own torque law (toggle)
ENTER   sag, peak torque and its joint, and your law's gap from MuJoCo's
```

`torque_from_pd(kp, kv, ctrl, q, qdot)` is the actuator law
$\tau = k_p(\text{ctrl} - q) - k_v \dot q$ over all 29 actuators, and it returns
`None` as shipped. While MuJoCo's servos run, the script compares the student's
prediction to `data.actuator_force` every step and `ENTER` prints the largest
gap over the last second. **`H` zeroes the model's gains and feeds the student's
torques through `qfrc_applied`**, so their function *is* the servo: a correct
law changes nothing visible, an empty one gives the week 0 rag doll, one without
damping rings.

| Step | Change | Right looks like |
| --- | --- | --- |
| 1 | write the law | `ENTER` reports a gap of ~1e-12 N·m |
| 2 | `H`; then remove the damping term, `R`, `H` | the walk is unchanged; then it oscillates and falls |
| 3 | `2` (limit 50) then `3` (limit 55) | knees flash red at every support exchange; 50 drops the robot, 55 walks |
| 4 | `4` (half kp) and `5` (double kp) | half sinks and collapses; double is thrown over by its own timing |
| 5 | `SERVO_HZ = 500`, then `H` | the simulation blows up; the student explains it with week 1 |
| 6 | lowest kp that walks, added to `SERVOS` | `ENTER` prints its sag; the student plots sag against kp |

**The hand-over runs the student's servo loop at 1 kHz, not 500 Hz — and that
is a lesson, not a workaround.** Measured: the correct law fed through
`qfrc_applied` at 500 Hz blows up (`BADQACC` after 14 steps), because an
explicit spring of this stiffness is unstable at that timestep; at 1 kHz it
walks 0.65 m against 0.66 m with MuJoCo's own servos, and 0.5 kHz/0.25 kHz
give the same. MuJoCo's position actuator survives 500 Hz only because the
engine integrates its affine bias implicitly. `SERVO_HZ = 500` is step 5 of the
lab, and it is week 1's exercise 14 seen from the other side. Also verified:
the correct law reproduces `actuator_force` to ~1e-13 while MuJoCo's servos
run, and the empty law gives the rag doll.

`SOC4180_AUTOCLOSE=6 uv run weeks/05-actuation/lab_servo.py` closes the window
by itself, which is how the script is smoke-tested.

## Rebuild

```bash
quarto render weeks/05-actuation/slides.qmd
```
