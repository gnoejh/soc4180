# 03 — Inverse Kinematics

| | |
| --- | --- |
| **Runtime** | Local laptop, CPU. No notebook this week: the lab is `lab_connected.py`. |
| **Wall clock** | ~1 min |
| **Convergence risk** | None. No learning. |
| **Feeds** | 04 — the walker calls this solver 500 times a second |

## Objectives

1. Solve planar two-link IK in closed form and read off its three lessons:
   reach is an **annulus**, there are **two** solutions, and sensitivity diverges
   as the leg straightens.
2. Explain why closed-form IK does not scale to a 6-DOF foot pose.
3. Derive and use damped least squares, and say what $\lambda$ trades away.
4. **Diagnose a singularity** from the Jacobian's singular values.
5. Justify warm starting and per-iteration joint clamping.

## The two numbers that carry the week

Measured on the G1's left leg:

- **Damping.** Asking the solver to lower the foot 5 cm at the `stand` pose gives
  `|dq| = 55778 rad` with $\lambda = 0$, and `0.04 rad` with $\lambda = 10^{-3}$.
- **Seeding.** The identical request for an 8 cm squat: residual error
  **7.9e-01 from the straight-leg seed** versus **2.2e-06 from the crouch**. Same
  solver, same target, and one of them simply cannot be solved.

The `stand` keyframe has every leg joint at exactly zero, so the Jacobian's
smallest singular value is ~9e-07 and its condition number ~2e+06. This is why
every gait in the course starts crouched.

## Reachability, shown not asserted

The circle demo tracks to **8.4e-05 m**. Stretching it 2 cm taller pushes its
lowest point outside the leg's reach, and the residual jumps to **2.68e-03 m at
279°** — and stays there at 12, 40 or 120 iterations.

That invariance is the diagnostic worth teaching: **a residual that will not
shrink with more iterations is a geometry problem, not a solver problem.**

## Note on the demo

The circle-tracing demo renders with `render_poses` — **kinematics only, physics
never stepped**. IK is a statement about geometry; a pose it solves perfectly may
be one the robot cannot hold. Keeping that separate here is what lets Week 4
combine them deliberately.

## Lab class: on your laptop

### `lab_connected.py`: the lab game, scored out of 10

```bash
uv run weeks/03-inverse-kinematics/lab_connected.py            # play; students edit moves.py only
uv run weeks/03-inverse-kinematics/lab_connected.py --joints   # the 29 joint names, ranges, stand angles
uv run weeks/03-inverse-kinematics/lab_connected.py --grade --no-viewer   # the same referee, headless
```

Ten problems (wave, sway, twist, squat, bow, clap, both hands up, squat-wave,
twist-wave, boss). Students answer each with a **move** in `moves.py`: a list of
`(seconds, pose)` steps, played under physics. A pose names joints in degrees,
or uses **IK**: `"pelvis": [0, 0, -0.2]` lowers the body 20 cm while
`ik_legs` keeps both feet planted. It passes if the goal is reached, the robot
never falls, and it ends standing. There is no time limit on a move: slow
motions are often the ones that balance, so students may take as long as they
need.

**What students start from.** `moves.py` ships every move written out, with
its key numbers left as `...` (nine blanks: shoulder pitches, waist pitch and
yaw, and the pelvis offsets). The task is to fill each `...` with a number.
Until then the problem shows "not filled in yet" and names the lines, so the
shipped file scores 0 / 10. The blanks sit in named poses (`LEFT_UP`, `SQUAT`,
`TURN`, ...) shared by several problems, so one answer fills several
problems. The other numbers are given, and students may change them. Joints
are in degrees, `"pelvis"` offsets in metres.

How they solve it by looking, not guessing: `E` freezes the robot and the joint
sliders pose it, `ENTER` prints the pose as a dict to paste, `K` steps through
the current move frozen, and `1`–`0` play a problem with a live gauge
("left hand 1.12 m, need > 1.20"). `moves.py` is re-read on every key press.

**Grading.** Students press `G` whenever they like; all ten play back to back
(about 45 s for the reference answers, longer for slower moves) and the score stays on screen: name, `SCORE n / 10`, `O`/`X` per problem, a
four-letter **scorer code** and the time. The instructor walks the room reading
screens. The scorer code is a hash of `lab_connected.py`; if it differs from the
one on the board, the referee was edited. `moves.py` is parsed as data, never
executed, so it cannot change the rules. Mouse pushes are zeroed while a move
plays.

What it teaches, measured with its own `--grade` before it was written: every
pose here is one IK solves perfectly, and half of them fall over under gravity.
Both arms up at once falls forward; one arm, then the other, reaches 1.36 m. A
clap needs the pelvis 4 cm back as a counterweight. A bow needs the pelvis
pushed back while the hips bend. The boss needs the robot to stand up *before*
lowering its arms. One-foot balance is left out on purpose: 72 open-loop
attempts all fell.

Reference answers (10/10) are in `instructor/moves_solution.py`, which is
gitignored and reaches the repository only encrypted, inside
`instructor.tar.gz.gpg` (`uv run scripts/instructor.py open` restores it). Check them with
`--grade --moves weeks/03-inverse-kinematics/instructor/moves_solution.py`.

## Rebuild

```bash
quarto render weeks/03-inverse-kinematics/slides.qmd
```
