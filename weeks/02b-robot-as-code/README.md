# 02b — The Robot as Code

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/02b-robot-as-code/lab.ipynb)

| | |
| --- | --- |
| **Runtime** | Laptop: CPU, plus a window for the viewer. Colab fallback: **T4 required** — the deck renders the robot. |
| **Wall clock** | ~1 min |
| **Convergence risk** | None. No learning, no physics. |
| **Lecture class** | Week 2 (`02-transforms`) — transforms, FK, the Jacobian |
| **Lab class** | **this deck**, then `lab_body.py` on the laptop |
| **Feeds** | Week 3 (IK), Week 4 (walking), and every week that indexes `qpos` |

## Why this exists

Week 2 answers *"where is the foot?"* for **one leg** and never says which six
of the robot's thirty-six numbers that leg is. Twenty-three joints — both arms,
the waist, the other leg — are never mentioned, and students finish the week able
to do kinematics but unable to make the robot wave.

This is the **second class of week 2**: the same robot, read as code.

- Week 2 is the mathematics. It stays a 46-slide lecture and is unchanged.
- Week 2b is the system. 21 slides, six generated diagrams, then the laptop.

The split is deliberate. Do not merge them: one session of 67 slides is not
teachable, and the two halves want different rooms — one a lecture, one a lab.

## Shape of the session

1. **The robot is a tree.** 31 bodies, each with exactly one parent.
2. **Five chains off one pelvis** — left leg 6, right leg 6, waist 3, left arm 7,
   right arm 7. That is 29, which is exactly `nu`.
3. **`qpos` is that tree flattened**, and a body part is a contiguous slice.
4. **`mj_forward` is the whole of this week** — `qpos` in, `xpos` out, no time.
5. **What each joint moves**, measured over the whole body.
6. **What the package does**, module by module.

## Six figures, all computed from the model

Nothing here is drawn by hand, so a diagram cannot drift from the robot.

| Figure | What it shows |
| --- | --- |
| The body tree | all 29 joints, in five colour-coded columns, each labelled with its `qpos` slot |
| `qpos` / `qvel` strip | 36 cells above 35, so the off-by-one is a visible shift |
| `mj_forward` vs `mj_step` | what each one reads and what it writes |
| Four named poses | `stand`, `wave`, `bow`, `squat`, rendered |
| The effect heat map | mm each of five landmarks moves per joint — the block structure **is** the tree |
| The package map | `src/soc4180/` in four layers, with the week each module serves |

## Facts this week measures, and does not assert

Every number on a slide is computed in the cell beside it. The ones worth
knowing in advance:

- **6 + 6 + 3 + 7 + 7 = 29 = `nu`.** There is no head joint, no neck, no hand.
- **`qpos` index − `qvel` index = 1, for all 29 joints.** The floating base
  spends 7 numbers on position and 6 on velocity. Both indices are valid, so
  using the wrong one silently addresses the neighbouring joint.
- **A waist joint moves both hands and neither foot.** A leg joint moves one
  foot and nothing else. The blank cells in the heat map are the tree.
- **Within a limb, effect falls from root to tip**: hip pitch 65 mm, knee 32 mm,
  ankle pitch 2 mm per 0.1 rad. The root of a chain is always the loud one.
- **Three joints move their landmark by exactly zero** — `ankle_roll`,
  `wrist_roll`, `wrist_yaw` — because the landmark sits on the joint's own axis.
  They still rotate it 28.6° per 0.5 rad. Week 2 met this once, as the
  `ankle_roll` column of zeros in the position Jacobian.
- **The squat lifts the feet 0.111 m off the floor**, because `set_pose` pins the
  pelvis. Kinematics places a pose; it does not settle a robot.
- **The G1 is mirror-symmetric to 10 µm, not to machine precision.** The left
  shoulder sits at `y = +0.100220` and the right at `y = −0.100210`. It is one
  CAD rounding artefact in the MJCF, and exercise 10 is to find it.

## The setup cell is a third form — read it before copying it

Weeks 0–9 use `try: import soc4180 / except ImportError: %pip install`, which
never delivers an update. Week 10 upgrades unconditionally, which would clobber
the editable install during a local `quarto render`. This week needs new package
code (`soc4180.bodies`) **and** must render locally, so it does both:

```python
if importlib.util.find_spec("google.colab") is not None:
    %pip install -q --upgrade "soc4180 @ git+https://github.com/gnoejh/soc4180.git"
import soc4180
```

On Colab it always upgrades, before the first import. Locally it installs
nothing. Use this shape for any future week that adds package code.

## Lab class: on your laptop

```bash
uv run weeks/02b-robot-as-code/lab_body.py
uv run weeks/02b-robot-as-code/anatomy.py --nudge left_arm
```

`lab_body.py` holds the G1 in each pose of `POSES`, never steps physics, and
drives all five chains: keys `1`–`5` highlight a chain (yellow spheres along
`chain_bodies`, white at the landmark), `M` mirrors the pose, `ENTER` prints it
as a pasteable `set_pose(...)` call, `R` returns to `stand`. The Control sliders
are wired straight onto the joint angles, as in week 2. The code is complete;
poses are dictionaries of joint names, never slot indices.

| Step | Do | Right looks like |
| --- | --- | --- |
| 1 | `1`–`5` | each chain named aloud before the highlight appears |
| 2 | drag one slider per chain | the landmarks that move, predicted before looking; the heat map is the answer key |
| 3 | three poses added to `POSES` by joint name | the robot takes them; no slot indices anywhere |
| 4 | a left-arm pose, then `M` | roll and yaw flipped sign, pitch did not — measured: the `stand` keyframe's own arms are `[0.2, ±0.2, 0, 1.28, …]` |
| 5 | the left hand above the right foot, `ENTER` | the printed call pasted into the notebook |

### `anatomy.py`: the robot as data, printed

```bash
uv run weeks/02b-robot-as-code/anatomy.py                 # the five chains and the qpos / qvel map
uv run weeks/02b-robot-as-code/anatomy.py --nudge waist   # one row of the heat map, measured live
uv run weeks/02b-robot-as-code/anatomy.py --pose left_shoulder_roll=0.8 left_elbow=1.2 --mirror
```

With no flags it prints every joint's chain, `qpos` slot, `qvel` slot (one
less, always) and range. `--nudge CHAIN` moves each joint of the chain 0.1 rad
from `stand`, one at a time, and prints how far five landmarks moved — the
deck's heat map, one row per joint, measured: a left-arm joint moves the left
wrist (elbow 205 mm, shoulder pitch 38, wrist roll 0.0 because the landmark is
on its axis) and nothing else; a waist joint moves both wrists (22 mm for yaw)
and the torso IMU (4–15 mm) and neither foot. `--pose` places named joints,
`--mirror` reflects them, and the simulator shows the result with the chain
highlighted.

| Step | Does | Calls |
| --- | --- | --- |
| 1 | the robot at `stand` | `load_g1`, `keyframe_data`, `mj_forward` |
| 2 | the chain table | `bodies.CHAINS`, `joint_index`, `dof_index`, `jnt_range` |
| 3 | nudge and measure | `set_pose` (reset + `mj_forward`), `site_xpos`, `xpos` |
| 4 | a named pose, mirrored | `set_pose`, `mirror`, `chain_of` |
| 5 | show it | `launch_viewer(passive=True)`, `chain_bodies`, `user_scn` |

`SOC4180_AUTOCLOSE=4` closes either script's window by itself.

## Why the viewer is not in the notebook

`lab.ipynb` must render headless under Quarto, on the Pages runner, and on Colab,
where `launch_viewer` raises — there is no window to draw into. So viewer work
lives in `lab_body.py` and the notebook carries the exercises and the rendered
figures. This is the same split as week 2.

## Common problems

| Symptom | Cause |
| --- | --- |
| `ValueError: no joint named 'left_ellbow'` | a typo; the message lists the chains — this is what `_full_name` exists for |
| the wrong joint moved | a `qvel` index used on `qpos`. Both are valid, so nothing raised |
| the robot floats above the floor | correct: `set_pose` never moves the pelvis. See slide 11 |
| rendering fails on Colab | CPU runtime. Runtime → Change runtime type → T4, then ask for `soc4180.gl_report()` |
| the Control sliders do nothing | you are in a script that calls `mj_forward` but not `mj_step`; `ctrl` only reaches the joints through physics. `lab_body.py` wires them onto `qpos` by hand |

## Objectives

1. Describe the G1 as a tree, and name the five chains without looking.
2. Given a joint name, say which `qpos` slot it uses and which landmarks it
   moves — and given a slot, say which joint it is.
3. Explain why `qpos` and `qvel` indices differ by one, and what goes wrong when
   they are confused.
4. Pose any part of the robot by name, with `soc4180.set_pose`.
5. Say what `mj_forward` reads and writes, and how it differs from `mj_step`.
6. Read `soc4180/bodies.py` and `soc4180/kinematics.py` and say what each line
   is for.
