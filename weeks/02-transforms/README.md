# 02 — Transforms and Forward Kinematics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/02-transforms/lab.ipynb)

| | |
| --- | --- |
| **Runtime** | Laptop: CPU, plus a window for the viewer. Colab fallback: **T4 required** — the deck renders the robot (see below). |
| **Wall clock** | ~1 min |
| **Convergence risk** | None. No learning. |
| **Lecture class** | `slides.qmd` → `slides.html` |
| **Lab class** | [`02b-robot-as-code`](../02b-robot-as-code/) — the same robot read as code, then `lab_viewer.py` and `lab_body.py` on the laptop |
| **Feeds** | Week 3 (IK), Week 4 (walking) |

## This week now needs a GL backend

Week 2 used to be pure arithmetic and would run fine on a CPU runtime despite the
header's advice. It no longer does: the third slide calls `render_poses` to show
the robot in three poses, so a CPU Colab runtime will fail there with the
`GL_UNAVAILABLE` message. Ask for `soc4180.gl_report()` if a student hits it.

`render_poses` is capped at **480 px high** by the G1 scene's offscreen
framebuffer; asking for more raises before any GL work happens.

## Shape of the session

The deck assumes **no prior robotics and no prior linear algebra beyond matrix
multiplication**. It climbs in four steps, and each one is checked against the
simulator before the next is allowed to build on it:

1. **One link in 2D**, rotated by hand — `tip = (-L sin θ, -L cos θ)`, verified
   with a calculator.
2. **Two links in 2D**, chained — the real G1 thigh and shin, still on paper.
3. **The full 6-joint 3D chain**, composed body by body in twelve lines.
4. **The Jacobian**, measured in millimetres, then divided by the nudge and named
   as the derivative it is.

Rotations are shown not to commute with a book on the desk *and* with two 3x3
matrices, because students will not believe the algebra until they have felt it.

**Eight figures carry the explanation**, every one generated from the measured
model rather than drawn by hand, so a diagram cannot drift from the text:

| Figure | What it shows |
| --- | --- |
| Three rendered poses | what six angles do; the right leg is left alone as a ruler |
| Three arrows, one foot | world/pelvis/hip vectors to the same star |
| One link, three angles | where $-L\sin\theta$ and $-L\cos\theta$ live |
| The book, both ways | two turns in each order, blue cover and orange spine |
| The leg as the formula sees it | $L_1$, $L_2$, both angle arcs, the real foot, and the 3.7 cm air gap |
| Workspace scatter | the reachable set, x–z and y–z |
| Anchor vs body origin | why the missing term costs 0.215 m |
| Jacobian bar chart | mm of foot-site travel per 0.57° of each joint |
| Columns of J as vectors | the six columns from a common origin, side and top view |

## Objectives

1. Say what a frame is, and why a position without one is meaningless.
2. Compose rotations and translations, and say why order matters.
3. Compute a two-link planar leg's foot position **by hand**, and check it.
4. Write one rotation in all four representations, and state each one's failure
   mode — including MuJoCo's scalar-first `(w,x,y,z)` quaternions.
5. Explain `nq > nv`, and why it means `qpos += qvel * dt` is not a valid step.
6. **Implement forward kinematics from the model tree** and validate it against
   `mj_forward`.
7. Read a workspace plot, and connect its boundary to next week's singularities.

## The two results students must reproduce

**The paper model.** The planar formula agrees with MuJoCo to **2.3e-06 m** —
about the width of a bacterium — across four poses. It is deliberately *not*
exact: at the zero pose MuJoCo puts the foot 2.3 µm behind the hip anchor rather
than exactly beneath it, because the G1's own offsets do not quite cancel. That
gap is the lesson that a flat two-link picture is a **model** of the leg.

**The full chain.** Hand-written FK agrees with `site_xpos` to **~1e-16** —
machine precision, because it is the same computation. Anything larger is a bug,
and the two causes that can actually bite on the G1 are named on the slide:
composition in the wrong order, or scalar-last quaternions.

The third textbook cause — a missing joint anchor — **cannot** produce an error
here: all 30 of the G1's joints have `jnt_pos = 0` (so does every other humanoid
in Menagerie). The slide demonstrates it on a nine-line MJCF instead, where
dropping the term moves the tip 0.215 m.

## The MuJoCo API is introduced before it is used

The deck used to reach for `mj_forward` in its very first cell and `site_xmat`
400 lines later, both unexplained. Two reference slides now sit right after the
setup cell, before anything depends on them.

The rule students are given: **an `x` in front means "computed, in world
coordinates"; no `x` means "declared in the MJCF, relative to the parent"**.

| Declared — in `model` | Computed — in `data` |
| --- | --- |
| `body_pos`, `body_quat` | `xpos`, `xmat` |
| `site_pos` | `site_xpos`, `site_xmat` |
| `jnt_pos`, `jnt_axis` | `qpos` |

And the sentence that makes the whole lab make sense: **`mj_forward` is MuJoCo's
own forward kinematics** — it reads `qpos` and fills every `x` quantity. The lab
reimplements it and demands agreement to the last decimal.

The second slide covers the `soc4180.kinematics` lookups, and uses them to expose
a real hazard: `leg_qpos_indices` returns `7..12` while `leg_dof_indices` returns
`6..11`. Same six joints, addresses off by one, because the floating base spends 7
numbers on position and 6 on velocity. That is week 1's `nq`/`nv` gap reappearing
as an indexing bug waiting to happen.

`mj_jacSite` also gets an inline note where it first appears, because it **writes
into** the arrays you pass and returns nothing — and they are `3 x nv`, columns for
every joint in the robot, not just the leg.

## Why three compact representations exist at all

The deck states the purpose before it compares anything, because "four options,
no free lunch" teaches nothing on its own. The reason is the **constraint count**,
not the number count:

| Form | Numbers | Constraints | Freedoms |
| --- | --- | --- | --- |
| matrix | 9 | 6 | 3 |
| quaternion | 4 | 1 | 3 |
| axis-angle / Euler | 3 | 0 | 3 |

A matrix you keep multiplying drifts off the set of valid rotations and needs
re-orthonormalising against six rules; a quaternion needs `q /= norm(q)`.

**The traffic is one-way this week and reverses next week**, and the deck says so
rather than leaving a rule that later looks broken. Forward kinematics only ever
goes representation → R. But week 3's `pose_error` goes back — `mju_mat2Quat`
then `mju_quat2Vel` — because the Jacobian's angular rows want a 3-vector, and an
error must be something you can **scale and add**: half a rotation matrix is not a
rotation, half an axis-angle vector is "go half way there". The slide runs it:
ask for 2 cm and 10°, get back `[0.02, 0, 0, 0, 0.1745, 0]`.

So: compress to **store**, expand to **act**, compress again to **correct**.

## The worked rotation

One rotation carries the whole representations section: **120° about
$(1,1,1)/\sqrt3$**, the axis out of a cube's corner. It is chosen because every
form of it is memorable and hand-checkable:

| Form | Value |
| --- | --- |
| axis-angle | 120° about `(0.5774, 0.5774, 0.5774)` |
| quaternion | `(0.5, 0.5, 0.5, 0.5)` — all four components equal |
| matrix | a permutation matrix of 0s and 1s; it sends x→y→z→x |
| Euler | roll 90, pitch 90, yaw 0 (`xyz`) |

**Every conversion to $R$ is done by hand and then checked**, because the matrix
is the only form that acts on a vector, so the other three have to reach it:

| Conversion | Formula on the slide | Agreement with MuJoCo |
| --- | --- | --- |
| axis-angle -> R | Rodrigues, with the cross-product matrix written out | 2.2e-16 |
| quaternion -> R | the quadratic 3x3 in `w,x,y,z` | 1.1e-16 |
| Euler -> R | `Rx(roll) @ Ry(pitch) @ Rz(yaw)` | 1.1e-16 |

**MuJoCo's lowercase `'xyz'` is intrinsic and equals `Rx @ Ry @ Rz`** — verified
against a non-degenerate triple (20, 35, 50), where the reversed product does
*not* match. Uppercase `'XYZ'` is extrinsic and equals `Rz @ Ry @ Rx`. Getting
that backwards is the "dozen conventions" trap in person.

The operator slide closes it: `R @ [1,2,3] = [3,1,2]`, the x->y->z->x cycle as
arithmetic, and the same `rot @ body_pos` that `my_fk` runs once per link.

The three failure modes are then measured, not asserted:

- **sign** — `q` and `-q` give matrices differing by exactly `0.0`
- **scalar-last** — the validated pose's foot quaternion is tilted 7.6°; read
  backwards it becomes 174.8°, flipping the foot's down-vector from `-1.0` to
  `+0.99`. It is still unit length and still a legal rotation, which is the point
- **gimbal lock** — at pitch 90° four different (roll, yaw) pairs with the same
  sum give a bit-identical quaternion, and the degeneracy fades slowly: 0.52° of
  difference at pitch 89°, 5.17° at 80°, 15.36° at 60°

## The Jacobian is taught as a derivative, not a black box

The deck measures first (nudge 0.01 rad, read millimetres), then **divides by the
nudge** and writes the mathematics:

$$J = \partial x / \partial q, \qquad J e_j \approx [f(q + \epsilon e_j) - f(q)] / \epsilon$$

The full 3x6 finite-difference matrix at `eps = 1e-6` matches `mj_jacSite` to
**3.1e-07**, and then a sweep shows why "as small as possible" is wrong:

| eps | max error vs analytic |
| --- | --- |
| 1e-1 | 3.077e-02 |
| 1e-2 | 3.078e-03 |
| 1e-4 | 3.078e-05 |
| 1e-6 | 3.078e-07 |
| **1e-8** | **1.111e-08** (best; sqrt of machine epsilon is 1.49e-08) |
| 1e-10 | 1.564e-06 |
| 1e-12 | 6.612e-05 |

Exactly ten-fold per decade down to the minimum — visible proof the approximation
is first-order — then round-off takes over and `1e-12` is worse than `1e-4`. That
is the argument for `mj_jacSite`: MuJoCo differentiates analytically, so there is
no epsilon to choose.

A final figure draws the six columns as **velocity vectors from a common origin**,
side view and top view. Do not draw them on the leg: positions and velocities are
different spaces, and hip_pitch and ankle_pitch are nearly parallel so their
labels collide. Both ankle joints are absent from the drawing for different
reasons — ankle_pitch is a 0.02 m/rad stub, ankle_roll is exactly zero.

## The foot site is not the sole

`left_foot` is a site at the **ankle-roll link's origin**, 3.5 cm above the four
contact spheres — at `stand` it reads `z = 0.0331` while the foot's underside is
on the floor at `z = -0.0019`. Do not call it the sole; a student who does will
not understand why the leg diagram shows the foot above the floor line.

It shows the foot above the floor for a second reason too: the drawing pins the
pelvis at standing height while the leg bends, so the foot rises **3.7 cm**. That
is correct — kinematics places a pose, it does not settle a robot onto the ground
— and the deck says so rather than hiding it.

## Two lengths for the same thigh

Week 2's planar model uses the **in-plane projection, 0.3366 m**. Week 3's law of
cosines uses the **true 3D length, 0.3409 m**. Both are right; the hip-to-knee
vector splays 5.4 cm sideways. Substituting one for the other costs 4 mm.

## Rebuild

```bash
quarto render weeks/02-transforms/slides.qmd
```

Roughly 25 s. Produces `slides.html` (the deck, gitignored) and `lab.ipynb` (the
student notebook, committed).

## How to try it

**Locally.** Render as above, open `slides.html` in a browser for the deck, and
`lab.ipynb` to see exactly what students get. To verify without Quarto, execute
the generated notebook with `nbclient` — that runs *every* cell, including any
marked `eval: false`, which is how a broken cell gets caught before a student
finds it.

**On Colab.** Open the badge at the top of this file, set **Runtime → Change
runtime type → T4**, then Run All. The whole notebook is about 9 s of compute.

> **Push first.** The badge loads `lab.ipynb` **from GitHub**, not from your
> working copy. Testing on Colab before pushing silently checks the *previous*
> version of the week, and everything looks fine while the change you wanted to
> test is not there. Confirm with `git status -sb` that you are not ahead of
> `origin/main`.

**With the interactive viewer** (any laptop with the repo installed):

```bash
uv run scripts/view.py --static --keyframe stand         # orbit the leg geometry
uv run scripts/view.py --pose=-0.35,0,0,0.70,-0.35,0     # place the left leg, frozen
uv run scripts/view.py --limp                            # motors off, watch it collapse
```

`--pose` takes the six leg angles in the week 2 order, freezes physics, and prints
the foot site in world and pelvis frames — the number a student's FK must
reproduce. Write it with the `=`: argparse reads a value starting with `-` as an
option otherwise. Double-click a body, then ctrl-drag to push the robot. This is
the fastest way to sanity-check a geometric claim before committing it to a slide.

## The lab class

The week has two classes: the deck is the lecture, and the second class is
hands-on on student laptops. Students run and change
[`lab_viewer.py`](lab_viewer.py) and read [`fk.py`](fk.py).

```bash
uv run weeks/02-transforms/lab_viewer.py
uv run weeks/02-transforms/fk.py --angles -0.35 0 0 0.70 -0.35 0
```

`lab_viewer.py` holds the G1 in each pose of `POSES` (SPACE / arrows to move
between them), never steps physics, and draws two spheres at the left foot:
**green** at MuJoCo's `site_xpos`, **red** at the student's forward kinematics.
**Both versions of that FK are complete and explained in the file**: `chain_fk`
composes the six bodies exactly as the slides derive it (1.8e-16 against
MuJoCo), `paper_fk` is the planar three-angle formula (2.2e-06 with roll and
yaw at zero, 1.2 mm with them on), and `F` switches which one draws the red
sphere. Each pose prints the truth, the prediction and the max error; ENTER
prints them for whatever the sliders say.

**The Control sliders move joints directly here.** In a normal viewer session
they set actuator targets, which only move the robot through `mj_step`, and this
lab never steps — so the first thing a student tries, dragging a slider, would do
nothing. The script therefore copies `ctrl` into each actuator's joint angle on
every tick (every G1 actuator drives exactly one hinge, and `ctrlrange` equals
`jnt_range`, so the slider limits are the joint limits). Drag `left_knee_joint`
and the leg bends, with the red sphere following the FK live. `view.py
--static` does the same.

The five steps students demonstrate, each visible on screen:

| Step | Do | What they should see |
| --- | --- | --- |
| 1 | add three poses to `POSES` | the leg goes where they predicted, or not; one pose explained from `fk.py`'s printed chain |
| 2 | ENTER on the crouch, then `F` | chain error 1.8e-16; paper error 2.2e-06, both inside the green sphere; the 2e-6 explained as the model's constant offset |
| 3 | third pose, roll and yaw on, `F` | the red sphere leaves the foot by 1.2 mm: the planar model is a model |
| 4 | break `chain_fk` on purpose | swapping offset and rotation moves the sphere; dropping `(a − Rj a)` changes nothing on the G1 because every joint sits at its body origin |
| 5 | put the foot 10 cm forward of `stand`, still flat | pitch angles summing to zero; measured: hip −0.30, knee +0.30, ankle 0.00 works, so does hip −0.45, knee +0.60, ankle −0.15 (`R[2,2] = 1.000000` for both) |

### `fk.py`: the chain, one body at a time

```bash
uv run weeks/02-transforms/fk.py
uv run weeks/02-transforms/fk.py --angles -0.30 0.10 0.05 0.70 -0.35 0.02 --side right
uv run weeks/02-transforms/fk.py --angles 0 0 0 0 0 0 --no-viewer
```

Places the leg (`mj_forward`, no physics), then walks pelvis → six bodies →
foot site by hand, printing for every body the offset it adds from its parent
(`body_pos`), the joint it turns about and by how much, and the world position
that leaves us at — so the composition can be read line by line — and compares
the result with `site_xpos`. The planar paper model is evaluated beside it.
Then the simulator shows the pose with green (MuJoCo) and red (the chain)
spheres at the foot.

| Step | Does | Calls |
| --- | --- | --- |
| 1 | place the six angles | `keyframe_data`, `leg_qpos_indices`, `mj_forward`, `site_xpos` |
| 2 | the chain by hand, printed body by body | `leg_chain`, `body_pos`, `body_quat`, `jnt_pos`, `jnt_axis`, `mju_quat2Mat`, `mju_axisAngle2Quat` |
| 3 | the paper model from measured `L1`, `L2`, the site drop | `xpos` of the hip, knee and ankle links |
| 4 | the verdict: 1e-16 vs 2e-6 vs "the plane is broken" | |
| 5 | show it | `launch_viewer(passive=True)`, `user_scn` |

Then exercises 4, 5 and 8 from the deck's exercise slide, in the notebook.

**The scripts are deliberately not notebook cells.** `lab.ipynb` must render
headless and run on Colab, and a viewer call blocks and has no window there.
The viewer lives in `.py` files students run; the notebook keeps the exercises.

## The knob worth turning in class

```python
crouch = [-0.35, 0.0, 0.0, 0.70, -0.35, 0.0]     # first cell of the notebook
```

Those six numbers feed **five figures**: the three rendered poses, the leg
diagram, the Jacobian bar chart, the column-vector diagram, and the singular
values. Change them and the whole second half of the lab redraws around the new
pose — the cheapest live demonstration in the week.

Two other useful knobs: the validated test pose in *Check it against MuJoCo*
(`[-0.30, 0.10, 0.05, 0.70, -0.35, 0.02]`, deliberately has nothing at zero), and
`eps` in the finite-difference sweep.
