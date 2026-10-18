# SOC4180 Robot and AI — Midterm Study Guide

**Scope: Weeks 0 – 6** (the five-layer stack, MuJoCo and MJCF, transforms and
forward kinematics, the robot as code, inverse kinematics, walking with the
LIPM, actuation and PD control, sensing and state estimation).

**Format:** 33 multiple-choice questions, one answer each, 60 minutes. No notes,
no calculator, no laptop. Every calculation on the exam can be done by hand in
under a minute: square roots of easy numbers, a product, a ratio.

This guide is meant to be *studied*, not skimmed. Each week has four parts:

- **The ideas**: what the week is about, in plain words.
- **The maths**: the equations you must be able to write down, read, and use.
- **Numbers worth knowing**: measured on our G1. You do not need every decimal,
  but you should know the *size* and what each number *means*.
- **Traps**: the misconceptions the lectures warned about. Exam distractors
  are built from them.

At the end you will find a one-page formula sheet, a glossary, and practice
questions to test yourself.

---

## How the course fits together

The course has one spine: **a walking humanoid, the Unitree G1, simulated in
MuJoCo.** The first half (weeks 0–6) builds control *by hand*, so that when
learning replaces it in the second half you know exactly what it replaced.

| Week | Question it answers | Layer |
| --- | --- | --- |
| 0 | What is a robot made of? | all five |
| 1 | How is a robot written down and simulated? | 1 + 2 |
| 2 | Given the joint angles, where is the foot? | 2 + 4 |
| 2b | Which of the 36 numbers is which body part? | 2 |
| 3 | Given where the foot should be, what are the angles? | 4 |
| 4 | How does a robot that can fall over walk? | 4 |
| 5 | What does a motor actually do? Why does the robot sag? | 2 / 4 seam |
| 6 | What can the robot actually know about itself? | 2 + 4 |

A thread runs through all of it, and it is the single most useful thing to
carry into the exam: **measure, do not believe.** Several claims in this
course were wrong until they were measured: the robot falls with `ctrl = 0`
(it does not), a straight leg is a good IK seed (it is the worst possible), the
LIPM keeps the ZMP inside the feet (it does not, on the real robot). When an
answer option sounds right but contradicts a measurement, the measurement wins.

---

# Week 0 — The five-layer robot stack

## The ideas

Every robot runs one loop: **sense → decide → act → the world changes → sense
again.** It is split into layers for one engineering reason: **the parts run at
wildly different speeds.** A motor must be corrected thousands of times a
second; a decision about where to walk need not change more than once a second.
One program doing both would be "both too slow and too twitchy".
*Layers are a consequence of timescales, not of tidiness.*

| # | Layer | Nickname | Answers | Typical rate |
| --- | --- | --- | --- | --- |
| 1 | Physics / World | the body's reality | What happens when forces act? | 500 Hz (ours) – 1 kHz+ |
| 2 | Robot Model | the blueprint | What *is* this robot? | static |
| 3 | Middleware / Robot OS | the nervous system | How do the parts talk? | — |
| 4 | Control & Planning | the brainstem | What should the motors do *now*? | 100 – 1000 Hz |
| 5 | AI / Behaviour | the cortex | What should the robot *achieve*? | ~1 – 50 Hz |

"Bottom is fast and dumb. Top is slow and smart." From the 500 Hz physics step
to a sub-1 Hz task plan is **about three orders of magnitude**.

**Layer 1 (Physics / World).** On a real robot there is *no* physics layer:
there is only reality. In simulation an engine (ours is **MuJoCo**; others are
PyBullet, Isaac Sim/Lab, Gazebo, ODE/Bullet/PhysX) stands in for reality. It
computes integration, contact, friction, inertia, joint limits, actuator
dynamics and sensor readings, and every one of these is an approximation.
"Contact is the whole game in legged robotics."

**Layer 2 (Robot Model).** A file describing the robot: geometry, masses,
joints, actuators, sensors. Formats: **MJCF** (MuJoCo, ours), **URDF** (ROS,
the interchange format), **SDF** (Gazebo), **USD** (NVIDIA Omniverse/Isaac).
A model is a **tree of bodies**; nesting is the kinematic chain.
*Correction worth making early:* the model is **not** "the simulation part".
Real robots need models too: IK needs link lengths, MPC needs dynamics,
estimation needs to know where the sensors are. **The model describes the
robot; the engine simulates the world.**

**Layer 3 (Middleware).** ROS 2 (DDS publish/subscribe), ROS 1 (legacy), vendor
SDKs, LCM, YARP, Drake. It provides transport, interface contracts, time,
lifecycle, logging and introspection, which together buy *modularity*. **This
course skips it on purpose**: ROS 2 is effectively Linux-only, cannot run in
Colab, and solves a problem we do not have. In our stack, Layer 3 is a single
Python process.

**Layer 4 (Control & Planning).** PID/PD, inverse kinematics, trajectory
generation, ZMP/LIPM walking, whole-body control, MPC. **Weeks 2–6 build this
layer from scratch.**

**Layer 5 (AI / Behaviour).** Reinforcement learning, imitation, vision,
navigation, task planning, vision-language-action models. Weeks 7–15. Learning
has largely won at locomotion, but **it sits on top of classical control; it
does not delete it**. A learned walking policy typically outputs joint
*targets* at about 50 Hz, and classical PD servos track them 10–20× faster.

## The maths: the loop as three equations

$$x_{k+1} = f(x_k, u_k) \qquad y_k = h(x_k) \qquad u_k = \pi(y_k)$$

| Symbol | Meaning | On the G1 | Size |
| --- | --- | --- | --- |
| $x$ | state: everything needed to predict the future | `qpos`, `qvel` | 36 + 35 = **71** |
| $u$ | action sent to the motors | `ctrl` | **29** |
| $y$ | observation: what the sensors report | `sensordata` | **12** |
| $f$ | the world (Layer 1) | `mj_step` | |
| $h$ | the sensors (Layer 2) | declared in MJCF | |
| $\pi$ | the policy (Layers 4–5), the only part you write | your code | |

"The robot is described by 71 numbers but *senses* only 12 of them." No box can
skip another: the policy never sees the state directly (only through $h$) and
never moves the robot directly (only through $f$).

**Ten steps per decision.** Physics at 500 Hz has period 2 ms; a policy at 50 Hz
has period 20 ms; so $N = f_\text{phys}/f_\text{pol} = 500/50 = 10$. Between
two decisions the same `ctrl` is **held**, and the servo law keeps correcting
the joints at every 2 ms tick. "The slow layer sets the goal; the fast layer
chases it."

**One timestep.** An integrator turns a derivative into a jump:
$v_{k+1} = v_k + \Delta t\,a_k$, then $x_{k+1} = x_k + \Delta t\,v_{k+1}$
(velocity first, as MuJoCo does). It pretends nothing changes *during*
$\Delta t$. A ball dropped from 0.79 m: at $\Delta t = 0.05$ s the error after
0.4 s is 98 mm; at $\Delta t = 0.002$ s it is 3.9 mm. **25× smaller step, 25×
smaller error, 25× more work.**

## Numbers worth knowing

- Timestep **0.002 s → 500 Hz**, integrator `implicitfast`, gravity
  $(0, 0, -9.81)$.
- G1: **31 bodies, 30 joints, nq = 36, nv = 35, nu = 29, nsensor = 4**, one
  keyframe `stand`. Mass 33.3 kg, weight **327 N**.
- Sensors: **two IMUs** (torso and pelvis), each a 3-axis gyroscope + 3-axis
  accelerometer, so 4 sensors and 12 numbers. **No camera, no joint-torque
  sensors, no foot contact switches.** "A robot cannot sense what its blueprint
  does not declare."
- Actuators are **position servos**: $\tau = k_p(\text{ctrl} - q) - k_v\dot q$
  with **$k_p = 500$** on every joint. Knee $k_v \approx 15.8$. A 0.1 rad error
  gives 50 N·m.
- Torso height after 3 s: **servos off 0.134 m, servos on 0.792 m.**
  "Standing is not a property of the robot. It is a property of the loop."

## The real-versus-simulated table

| Layer | Real G1 | Our MuJoCo G1 |
| --- | --- | --- |
| 1 Physics | reality itself | MuJoCo, 500 Hz, approximate |
| 2 Model | onboard kinematic/dynamic model | MJCF from Menagerie, pinned |
| 3 Middleware | vendor SDK + ROS 2 bridge | one Python process |
| 4 Control | onboard servos, WBC, MPC | controllers we write |
| 5 AI | policies deployed onboard | policies we train |

**Layers 2–5 transfer between the columns. Layer 1 never does.** That one
asymmetry *is* the sim-to-real problem. Reality disagrees with the engine:
friction changes with dust and wear, motors saturate and lag, sensors are
noisy, biased and *late*, gearboxes have backlash, real mass is never the
file's mass.

## Traps

- **`ctrl = 0` is NOT an uncontrolled robot.** With position servos it commands
  every joint to angle zero, a straight-legged stance the servos hold (0.79 m).
  "Zero command is a command." To see a robot with no controller you must
  **disable actuation** (`soc4180.actuation_disabled`).
- The limp robot does not fall at free-fall speed (0.35 s to 0.2 m); it takes
  about 1.3 s, because it folds on legs whose feet stay on the floor.
- "You cannot put a neural network in a 1 kHz loop, and you do not need to."
- The question to carry all semester: **"Which layer is this, and what is it
  assuming about the layers beneath it?"**

---

# Week 1 — Robots, simulation, MuJoCo and MJCF

## The ideas

**Why simulate?** A real humanoid is expensive, fragile and slow to reset;
learning needs millions of trials; simulation is repeatable, faster than real
time, and parallel. **The cost:** contact and friction are hard, actuator
limits and delays are often ignored, sensors are noiseless. The distance
between simulation and reality is the **sim-to-real gap**.

**MuJoCo** = *Mu*lti-*Jo*int dynamics with *Co*ntact. **MJCF** is its XML
format. Sections: `<compiler>`, `<option>`, `<default>`, `<asset>`,
`<worldbody>`, `<actuator>`, `<sensor>`, `<keyframe>`. A file with only
`<worldbody>` compiles; "the other seven sections are the difference between a
shape and a robot."

**Body, joint, geom, site.**

- `<body>`: a coordinate frame carrying mass. Its `pos` is measured **from its
  parent**, not from the world.
- `<joint>`: how a body may move *relative to its parent*. A body with no joint
  is welded to its parent.
- `<geom>`: shape, for drawing and/or collision.
- `<site>`: a named point with no physics; sensors attach to sites.
- **Nesting `<body>` tags *is* the kinematic chain.** A body's position depends
  only on the joints on its path to the root, so a knee can never move a hand.

**An MJCF is a diff against a default robot.** The G1's `<option>` sets only
`integrator="implicitfast"`; the 0.002 s timestep and the gravity are
**MuJoCo's defaults**. "A number you never wrote is shaping your results."
`<default class="g1">` holds `kp = 500`, inherited through `childclass="g1"` on
the pelvis; there is no gain written in the 29 actuator lines.
`inheritrange="1"` copies each joint's range onto its actuator, so a command
outside the joint's travel cannot even be expressed.

**Visual vs collision geometry.** 72 geoms: 1 floor, **35 in group 2** (the
pretty meshes, `contype=0 conaffinity=0`, invisible to physics) and **36 in
group 3** (what collides). Each foot touches the floor through **four spheres
of 5 mm radius**, not a mesh: sphere contacts are cheap and stable, mesh-on-mesh
contact is neither.

**Conventions.** **+z up** (gravity along −z), **+y the robot's left** (left
hip at $y = +0.064$), **+x forward** ($x = y \times z$, right-handed). Lengths
in metres, angles in radians.

| Name | Turns about | Your head | The robot |
| --- | --- | --- | --- |
| roll | x (forward) | ear to shoulder | tipping sideways |
| pitch | y (left) | nod "yes" | leaning forward/back |
| yaw | z (up) | shake "no" | turning to face a new way |

A hip needs three joints (pitch to step, roll to shift weight, yaw to turn); a
knee needs one (pitch), with range $[-0.087, 2.880]$: it bends one way only.
Every one of the G1's joint names matches its axis.

**`model` vs `data`.** `model` is the robot, compiled from XML and constant
while simulating (`body_pos`, `jnt_range`, `actuator_gainprm`). `data` is "one
robot, at one instant": `qpos`, `qvel`, `ctrl`, `xpos`, `time`. **One model can
serve many data** (that is how week 10 runs thousands of robots).
`mj_step` advances time by one timestep; **`mj_forward` computes positions from
`qpos` without advancing time.**

## The maths

**Joints decide nq and nv.**

| Joint | nq | nv |
| --- | --- | --- |
| free | 7 (3 position + 4 quaternion) | 6 |
| ball | 4 | 3 |
| hinge / slide | 1 | 1 |
| none (welded) | 0 | 0 |

G1 = 1 free joint + 29 hinges: **nq = 7 + 29 = 36, nv = 6 + 29 = 35.** Only
free and ball joints have nq > nv, because a quaternion uses 4 numbers for 3
rotational freedoms. *A free joint cannot share a body with a hinge* (the
compiler refuses: "more than 6 dofs in body"); wrap the leg in a parent body
carrying the free joint. A two-link leg on a free base then has nq = 9, nv = 8.

**The quaternion.**
$$\mathbf u = (w, x, y, z) = \left(\cos\tfrac\theta2,\ \mathbf n\sin\tfrac\theta2\right), \qquad w^2+x^2+y^2+z^2 = 1$$
Scalar first; $(1,0,0,0)$ = not turned. One constraint on four numbers leaves
three freedoms. **A quaternion holds half the angle.** The integrator cannot do
"quaternion + h × velocity" (sizes do not match); it *turns* the quaternion.

**Frames.** $\mathbf p_\text{child} = \mathbf p_\text{parent} + R_\text{parent}\,\mathbf{pos}_\text{child}$:
start at the parent's origin, walk the typed offset along the parent's own axes.

**The servo droop.** $\tau = k_p(\text{ctrl}-q) - k_v\dot q$. At rest
$\dot q = 0$ and the servo torque balances gravity's $\tau_g$, so
$$q - \text{ctrl} = -\tau_g / k_p$$
**"A non-zero load needs a non-zero error."** Droop ∝ $1/k_p$: ten times the
gain, a tenth of the error, and never zero. Removing it needs an integral term
or feedforward (week 5). Two-link leg at $k_p = 200$: hip droop −0.080 rad;
at $k_p = 5$: −0.858 rad.

**Inside `mj_step`.**
$$M(q)\,\dot v + c(q, v) = \tau + \tau_\text{contact}$$
$M$ is the 35×35 mass matrix (it depends on pose), $c$ is gravity + Coriolis
(`qfrc_bias`), $\tau$ the actuators, $\tau_\text{contact}$ floor and limits.
`mj_step` (1) solves for $\dot v$, then (2) integrates $q, v$ by $h$.

**Integrators.** Explicit Euler: $q_{n+1} = q_n + h v_n$, $v_{n+1} = v_n + h\dot v_n$.
It is **first order**: halve the step, halve the error.
**MuJoCo's "Euler" is semi-implicit**: velocity first, then position with the
*new* velocity, $v_{n+1} = v_n + h\dot v_n$, $q_{n+1} = q_n + h v_{n+1}$.

**Stability of a stiff spring** ($\ddot x = -\omega^2 x$, $\omega = \sqrt{k/m}$):

- Explicit Euler: eigenvalues $1 \pm ih\omega$, $|\lambda| = \sqrt{1 + h^2\omega^2} > 1$
  for **every** $h > 0$. It cannot hold an undamped spring at any timestep.
- Semi-implicit: $\det A = 1$, $\operatorname{tr}A = 2 - h^2\omega^2$; stable iff
  $$h < \frac{2}{\omega} = 2\sqrt{m/k}$$
  More than π ≈ 3 steps per oscillation. **Four times the stiffness halves
  the largest safe step.** Example: 1 kg on k = 10,000 N/m, ω = 100 rad/s,
  bound h < 0.02 s.
- Damping ($\zeta$) moves the bound: with an explicit damper
  $h\omega < 2(\sqrt{1+\zeta^2} - \zeta)$ (0.83 at ζ = 1, **less** stable);
  with an implicit damper (`implicitfast`) $h\omega < 2(\sqrt{1+\zeta^2} + \zeta)$
  (4.83 at ζ = 1). On the G1, plain Euler shakes the robot to the floor at
  0.004 s; `implicitfast` stands at every step tried. "The integrator line in
  the file is a safety margin."

**Contact forces.** Standing, $m\ddot z = \sum F_n - mg = 0$, so the normal forces
sum to the weight: **8 contacts, 327.1 N against 327.1 N.** The motors cannot
change this sum: they push the robot against itself. Heels carry ~225 N and
toes ~102 N, because the CoM (x = +0.3 cm) is nearer the heels (−5 cm) than the
toes (+12 cm).

## Numbers worth knowing

- ~15,000–16,000 physics steps/s on one core, about **30× real time** (machine
  dependent).
- The `stand` keyframe: pelvis at 0.79 m, quaternion (1,0,0,0), only the arms
  bent (elbows 1.28 rad ≈ 73°). **Every leg joint is exactly zero**: a
  perfectly straight leg, which week 3 shows is a singularity.
- One `mj_step` from `stand`: the pelvis rises 0.018 mm, because the foot
  spheres start ~2 mm inside the floor and soft contacts push them out.

## Traps

- **Stable is not the same as accurate.** At hω = 1.99 a 1 cm spring release
  swings to 10 cm: bounded but wrong.
- **Silent failure:** past the bound MuJoCo raises `BADQACC` and **resets the
  state to zero**, which reads like success. At `dt = 0.02` with servos the leg
  "holds" at zero; with no servos it gives a different, wrong swing with no
  warning at all (silent approximation).
- **Determinism vs chaos.** Two runs on one machine are bit-identical (difference
  exactly 0.0). A falling robot amplifies a $10^{-12}$ rad nudge about
  $10^5$-fold in 3 s ($\delta \approx \delta_0 e^{\lambda t}$, λ ≈ 3.7/s).
  Across machines the arithmetic is reordered, so the fall ends elsewhere
  (0.134 m here, 0.142 m on Colab). **Never assert or grade an exact number from
  a fall; assert direction and size.**
- **Test sensitivity along a direction that is not a symmetry.** Nudging the
  base *x* position is a pure shift on a flat floor and amplifies 1×; it proves
  nothing. Nudge a joint angle.

---

# Week 2 — Transforms and forward kinematics

## The ideas

**Forward kinematics (FK):** given the joint angles, where is the foot? A real
robot's encoders report only angles; nowhere among six leg angles is a
position. FK turns them into one.

**Naming rule in MuJoCo:** an `x` prefix means *computed, in world
coordinates*; no `x` means *declared in the MJCF, relative to the parent*:
`body_pos` vs `xpos`, `site_pos` vs `site_xpos`. **`mj_forward` IS MuJoCo's
forward kinematics**: it reads `qpos` and fills every `x` quantity.

**Frames.** A frame is an origin plus three axes. "The foot is at (0, 0.12,
0.03)" is only true in a stated frame. The same foot at `stand` is at
z = 0.033 in the world, z = −0.757 in the pelvis frame, z = −0.654 relative to
the hip. "Most robotics bugs are one number used in the wrong frame."

## The maths

**A rotation matrix is where the axes land.** Its columns are the turned axes:
$R = [\,R\mathbf e_x \mid R\mathbf e_y \mid R\mathbf e_z\,]$.

$$R_z(\theta)=\begin{bmatrix}c&-s&0\\ s&c&0\\0&0&1\end{bmatrix},\quad
R_x(\theta)=\begin{bmatrix}1&0&0\\0&c&-s\\0&s&c\end{bmatrix},\quad
R_y(\theta)=\begin{bmatrix}c&0&s\\0&1&0\\-s&0&c\end{bmatrix}$$

$R_y$'s minus sign *looks* misplaced and is not. A link hanging to $(0,0,-L)$
rotated by θ about y has its tip at $(-L\sin\theta, -L\cos\theta)$: **a positive
pitch swings the leg backwards**, so the week-4 crouch commands hip pitch −0.35.

**Subscripts read right to left.** $R_{ab}, \mathbf t_{ab}$ describe frame b
as seen from a; in $R_{ab}\mathbf p_b$ the inner letters match and cancel.

**A transform:** $\mathbf p_a = R_{ab}\,\mathbf p_b + \mathbf t_{ab}$
(turn first, then shift).

**Chaining** (the whole of forward kinematics):
$$R_{ac} = R_{ab}R_{bc}, \qquad \mathbf t_{ac} = R_{ab}\,\mathbf t_{bc} + \mathbf t_{ab}$$
**The parent's rotation acts on the child's offset.** Forget it and the leg
swings from the wrong place.

**Homogeneous 4×4 transforms.**
$T_{ab} = \begin{bmatrix}R_{ab}&\mathbf t_{ab}\\ \mathbf 0^\top&1\end{bmatrix}$,
points get a 1 appended, *directions* get a 0 (they turn but do not shift).
Chaining is plain multiplication: a leg is $T_1T_2\cdots T_6$.
**Inverse:** $\mathbf p_b = R^\top(\mathbf p_a - \mathbf t)$, because
$R^{-1} = R^\top$ for a rotation.

**Order matters.** $T_{ab}T_{bc} \ne T_{bc}T_{ab}$. Matrix multiplication does
not commute, and neither does turning a book. Read a product left to right,
root to tip; in code, `rot = rot @ Rj` (the joint goes on the right).

**Two links on paper.**
$\text{knee} = (-L_1\sin\theta_1,\ z_h - L_1\cos\theta_1)$,
$\text{ankle} = (\text{knee}_x - L_2\sin(\theta_1+\theta_2),\ \text{knee}_z - L_2\cos(\theta_1+\theta_2))$.
The angles **add**, because the knee angle is measured from the thigh. This
paper model matches MuJoCo to ~2 µm (a constant offset, not noise: it is a
*model*). The full 3-D chain matches to $10^{-16}$.

**Four ways to write a rotation.**

| Representation | Numbers | Constraints | Good at | Bad at |
| --- | --- | --- | --- | --- |
| Rotation matrix | 9 | 6 | acting on a vector, composing | storing; drifts off SO(3) |
| Quaternion | 4 | 1 | storing, interpolating | reading; sign ambiguity |
| Axis–angle | 3 | 0 | angular velocity and *error* | composing |
| Euler angles | 3 | 0 | saying out loud | gimbal lock, many conventions |

All have 3 freedoms; **the constraint count is the reason they differ.** That
is why `qpos` stores the base as 4 numbers. A matrix is a rotation iff
$RR^\top = I$ and $\det R = +1$ (−1 would be a mirror).

- **Rodrigues:** $R = I\cos\theta + (1-\cos\theta)\mathbf u\mathbf u^\top + [\mathbf u]_\times\sin\theta$ (unit axis!).
- **Quaternion → R:** every entry is quadratic in $(w,x,y,z)$, so **$q$ and $-q$
  give the same R.**
- **Euler → R:** MuJoCo's lowercase `'xyz'` is intrinsic, $R = R_x R_y R_z$.
  The order of the product *is* the convention.
- Worked example: 120° about $(1,1,1)/\sqrt3$ is quaternion $(½,½,½,½)$, a
  permutation matrix sending x→y→z→x.

**Why the reverse conversion matters (week 3):** an orientation *error* must be
something you can scale and add. Half a rotation matrix is not a rotation; half
an axis-angle vector is "go half way". So IK converts matrix → quaternion →
3-vector. Motto: *compress to store, expand to act, compress again to correct.*

**The Jacobian.** $J = \partial\mathbf x / \partial\mathbf q$, columns are joints,
rows are foot coordinates, units m/rad. Finite difference:
$J\mathbf e_j \approx [f(\mathbf q + \varepsilon\mathbf e_j) - f(\mathbf q)]/\varepsilon$.
$$\dot{\mathbf x} = J(\mathbf q)\,\dot{\mathbf q}$$
A column is a *velocity*, not a place. For a hinge with axis $\mathbf a_j$
through anchor $\mathbf o_j$:
$$J_{\text{pos},j} = \mathbf a_j \times (\mathbf p - \mathbf o_j), \qquad J_{\text{rot},j} = \mathbf a_j$$
Its size is the distance from the *axis line*, not from the anchor point. The
hip has the long lever (0.62 m/rad), knee 0.32, ankle pitch 0.018, ankle roll
0 (the foot site sits on its axis).

**How small should ε be?** error ≈ $C\varepsilon + u/\varepsilon$
(truncation + round-off), so $\varepsilon_\text{best} \approx \sqrt u \approx 1.5\times10^{-8}$.
The error falls 10× per decade down to $10^{-8}$ (first order), then round-off
takes over: $10^{-12}$ is worse than $10^{-4}$. That is why real code uses
analytic `mj_jacSite`.

**Singular values** of the 6×6 leg Jacobian measure response along six
independent directions. Crouched, the smallest is 0.089; at `stand` it is
$9\times10^{-7}$, about 100,000× smaller. **That is a singularity.**

## Numbers worth knowing

- Thigh: true 3-D length **0.3409 m**, in-plane projection 0.3366 m (it splays
  5.4 cm sideways). Shin 0.3000 m. Ankle to foot site 0.0176 m.
- `leg_qpos_indices` = 7..12, `leg_dof_indices` = 6..11: same joints, off by one.
- `mj_jacSite` **writes into** the arrays you pass, which are 3 × nv.

## Traps

- **The `left_foot` site is not the sole.** It sits at the ankle-roll link
  origin, 3.5 cm above the contact spheres.
- **A kinematics figure that pins the pelvis floats.** Bend the knee with the
  pelvis held and the foot rises off the floor (3.7 cm for the crouch).
  "Kinematics places a pose; it does not settle a robot."
- **q and −q are the same rotation**, so do not compare orientations number by
  number.
- **Scalar-first vs scalar-last.** MuJoCo is (w, x, y, z); SciPy and ROS are
  scalar-last. A misread quaternion is still unit length and legal; nothing
  raises; the foot is simply upside down (7.6° becomes 174.8°).
- **Gimbal lock.** At pitch 90°, roll and yaw stop being independent; only
  their sum survives. It fades slowly (0.5° apart at 89°, 5° at 80°). This is
  why we never hand a controller three Euler angles.
- **`qpos += qvel * dt` is the wrong shape** (36 vs 35). Use `mj_integratePos`.
- **"As small as possible" is the wrong ε.**
- The joint-anchor term `a - Rj @ a` does nothing on the G1 (all joints have
  `jnt_pos = 0`) but is right in general: on a toy model with the hinge at
  0.3 m, dropping it moves the tip 21 cm.

---

# Week 2b — The robot as code

## The ideas

**The robot is a tree.** Every body has exactly one parent, so the whole
skeleton is one integer array, `model.body_parentid` ($\pi(0) = 0$: the world
is its own parent). The G1 has 31 bodies and four leaves (two feet, two hands).

**Five chains, one pelvis.**

| Chain | Joints | qpos | qvel |
| --- | --- | --- | --- |
| left leg | 6 | 7..12 | 6..11 |
| right leg | 6 | 13..18 | 12..17 |
| waist | 3 | 19..21 | 18..20 |
| left arm | 7 | 22..28 | 21..27 |
| right arm | 7 | 29..35 | 28..34 |

6 + 6 + 3 + 7 + 7 = **29 = nu.** Each chain is contiguous in `qpos`, root to tip.
**The legs hang off the pelvis; the arms hang off the torso, three waist joints
up.** So the waist moves both arms and neither leg.

**What a joint moves.** A joint can move a landmark only if the landmark's body
is the joint's body or downstream of it:
joint $j$ moves landmark $k$ ⟺ $\beta(j) = k$ or $\beta(j) \in A(k)$ (ancestors).
Measured with a 0.1 rad nudge: a leg joint moves one foot and nothing else; a
waist joint moves both hands and neither foot; within a limb **the effect falls
from root to tip** (hip pitch 65 mm, knee 32, ankle pitch 2). "The root of a
chain is always the loud one; an ankle is for trimming."

**Three joints move their landmark by exactly zero**: ankle roll, wrist roll,
wrist yaw, because the landmark sits on the joint's own axis. They still
*rotate* it.

## The maths

**Counting slots.** $i_q(j) - i_v(j) = \sum_{k<j}(n_q(k) - n_v(k))$. Every hinge
contributes 0; only the free joint (first in the file) contributes 7 − 6 = 1.
So **for every one of the 29 joints, qpos index − qvel index = 1.**

**Why a rotation costs 4 in qpos but 3 in qvel:**
$\dot{\mathbf q} = \tfrac12\mathbf q \otimes (0, \boldsymbol\omega)$ keeps
$\|\mathbf q\| = 1$. Spinning the base at 0.5 rad/s for 1 s gives
$(\cos 0.25, 0, 0, \sin 0.25)$: half the angle.

**Mirroring.** With $M = \operatorname{diag}(1, -1, 1)$ the mirrored rotation is
$MRM$. $R_x(\theta) \to R_x(-\theta)$, $R_z(\theta) \to R_z(-\theta)$,
$R_y(\theta) \to R_y(\theta)$. **Roll and yaw negate, pitch keeps its sign.**
The `stand` keyframe proves it: left arm $[0.2, 0.2, 0, 1.28, \dots]$, right arm
$[0.2, -0.2, 0, 1.28, \dots]$. The rule works because both sides use the same
axes.

**`mj_forward` vs `mj_step`.** `mj_forward` (kinematics): `qpos` in, every `x`
quantity and subtree CoM out. No time passes; nothing falls. `mj_step`
(dynamics): uses `ctrl`, `qvel`, contacts; advances time by 2 ms; the robot can
fall. In a kinematic script, `ctrl` does nothing, which is why the viewer's
Control sliders look dead until wired onto `qpos`.

## Numbers worth knowing

- nbody 31 (one is the world), njnt 30 (1 free + 29 hinges), nsite 4 (two feet,
  two IMUs, **no hand sites**: a hand is read from `xpos` of the wrist-yaw body).
- The G1 is mirror-symmetric to **10 µm**, not to machine precision: the left
  shoulder sits at y = +0.100220, the right at −0.100210. A CAD artefact in the
  model, not a bug in your code.
- `set_pose` pins the pelvis, so a symmetric squat lifts the feet 0.111 m off
  the floor.

## Traps

- **The off-by-one that never raises.** A `qvel` index used on `qpos` is a valid
  index: it silently reads the *neighbouring* joint. Habit: **write names, not
  slots** (`soc4180.set_pose(model, data, left_elbow=1.2)`).
- Ids move when the MJCF changes; names do not.
- Use the torso **IMU site**, not the `torso_link` origin, to measure the torso:
  that origin is on the waist axes and barely moves.

---

# Week 3 — Inverse kinematics

## The ideas

IK reverses FK: foot position in, angles out. FK always has exactly one answer.
IK can have **none** (out of reach), **two** (knee forward or back),
**infinitely many** (more joints than constraints), or an answer that
**explodes** near some poses. IK is geometry, not balance: a perfectly solved
pose may be one the robot cannot hold.

**Closed form does not scale.** A 6-joint foot pose can be solved in closed form,
but the derivation is specific to one robot, brittle to model changes, and hard
to extend with limits. The course uses **differential IK** instead.

## The maths

**Two links in a plane (law of cosines).** With the knee angle θ = 0 for a
straight leg (MuJoCo's convention):
$$\cos\theta_\text{knee} = \frac{d^2 - \ell_1^2 - \ell_2^2}{2\ell_1\ell_2}$$
- Reach is an **annulus**, $|\ell_1 - \ell_2| \le d \le \ell_1 + \ell_2$:
  0.0409 m to 0.6409 m for the G1. **Too close is as impossible as too far.**
- arccos returns one angle but $-\theta$ also works: **two solutions**.
- Sensitivity $|d\theta/dd| = d / (\ell_1\ell_2\sin\theta)$ **diverges as the leg
  straightens** (sin θ → 0): one millimetre asks for degrees of knee.
- Measured: knee 144° at d = 0.20 m, 41° at 0.60, 6° at 0.64; unreachable at
  0.02 and 0.70.

**Differential IK.** Solve for a small change, then repeat: $J\,\Delta q = \mathbf e$.

**The plain inverse** $\Delta q = J^{-1}\mathbf e$. With the SVD
$J = U\Sigma V^\top$, each foot direction is divided by its own $\sigma_i$.
Lowering the foot 5 cm: crouched (σ_min 0.089) costs 0.56 rad; at `stand`
(σ_min 9e-7) it costs **~55,000 rad**, and the residual is still zero.
**A zero residual does not mean a sensible solution**: "the algebra succeeded
and commanded the impossible."

**Damped least squares (DLS).** Minimise
$C = \|J\Delta q - \mathbf e\|^2 + \lambda^2\|\Delta q\|^2$ (error left plus joint
motion spent). Setting the gradient to zero:
$$\Delta q = (J^\top J + \lambda^2 I)^{-1}J^\top\mathbf e = J^\top(JJ^\top + \lambda^2 I)^{-1}\mathbf e$$
The inverse **always exists** for λ > 0 (the matrix is positive definite). The
two forms are equal (push-through identity); the right one inverts a 6×6, which
is cheaper when there are more joints than constraints.

**What λ does, direction by direction:**
$$\frac{\sigma_i}{\sigma_i^2 + \lambda^2} \approx \begin{cases} 1/\sigma_i & \sigma_i \gg \lambda \text{ (true inverse)}\\ \sigma_i/\lambda^2 \to 0 & \sigma_i \ll \lambda \text{ (switched off)}\end{cases}$$
**λ is a threshold.** It trades accuracy for stability. At `stand` with
λ = 10⁻³ the 5 cm request costs 0.04 rad instead of 55,778.

**Condition number** $\kappa = \sigma_\max/\sigma_\min$: the aspect ratio of the
ellipsoid $J$ maps the unit sphere of joint motions onto. Relative errors are
amplified by up to κ, and $\log_{10}\kappa$ ≈ digits lost. Crouch κ ≈ 20 (1
digit lost); stand κ ≈ 2×10⁶ (6 digits). At a straight leg the ellipse
**collapses to a line**: there is no direction that shortens the leg.

**The weak direction** is −0.38 hip pitch + 0.81 knee − 0.44 ankle pitch: bend
the knee, pitch hip and ankle back, keep the foot level. It moves the foot
straight up/down. Same mix in both poses; only the gain changes, from 9e-7 to
0.089: **a factor of 100,000 restored by bending the knee.**

## Numbers worth knowing

- Lower the pelvis 8 cm with both feet held, 40 iterations: from a straight-leg
  seed residual **0.79**, from a crouched seed **2.2e-6**. Same request, same
  solver, different seed.
- A 90-target circle with warm starts: worst error 8.4e-5 m. Stretch it 2 cm
  taller: worst 2.68e-3 m, **identical at 12, 40 or 120 iterations.**

## Traps

- **Damping makes the request sane but cannot invent a direction the pose
  lacks.** The fix for a straight leg is a different *starting pose*, not a
  different λ. That is why every gait in the course starts crouched, and why
  humans do not walk with locked knees.
- **A residual that does not shrink with more iterations is a geometry problem**
  (the target is unreachable). Only a residual that shrinks slowly is a solver
  problem. Reachability is a property of the *target*; check it with a ruler.
- **Warm starting**: seed each solve from the last answer; it converges faster
  and keeps the leg from flipping between knee-forward and knee-back.
- **Clamp joint limits every iteration**, or IK returns a mathematically perfect
  pose the mechanical stops forbid.
- The lab game's lesson: **every pose IK solves frozen; under gravity half of
  them fall over.** Both arms up at once falls on its face; one then the other
  works. "Kinematics places a pose; it does not balance a robot. The order of
  the poses is part of the answer."

---

# Week 4 — Contact, balance and analytic walking

## The ideas

**Why walking is hard: count the motors.** nv = 35, nu = 29: **six freedoms
have no motor**, the floating pelvis. The robot is **underactuated**. Motors
push link against link inside the robot; internal forces come in equal and
opposite pairs and cannot move the whole robot.

**Newton for the whole robot:** $m\ddot{\mathbf c} = \mathbf f_\text{floor} + m\mathbf g$.
Only the floor can move the body horizontally, and the floor can **push, never
pull**, only **where a foot touches**, and sideways only as hard as friction
allows. Walking is **deliberately falling and catching yourself**.

**Static vs dynamic walking.** Static: CoM carried over each foot before the
other lifts; frozen at any instant it stays up; slow. Dynamic: the CoM spends
most of its time *between* the feet, falling toward the next one (it sways only
~6 cm); fast and efficient, but frozen mid-step it falls.

**Centre of pressure (CoP)** is the force-weighted average of the contact
points:
$$p = \frac{\sum_i f_i x_i}{\sum_i f_i} = \sum_i w_i x_i, \qquad w_i \ge 0,\ \sum w_i = 1$$
Because the floor only pushes, the weights are non-negative, so **p always lies
inside the support polygon.**

**Zero moment point (ZMP):** the point on the floor about which gravity plus the
body's inertia produce no tipping moment. On flat ground **ZMP = CoP**.
**Balance rule: the ZMP must be inside the support polygon.** A motion that
would need it outside cannot happen with flat feet: the foot rolls on its edge
instead. Support polygon = one foot (about 17 × 6 cm on the G1) in single
support, the convex hull of both in double support.

## The maths

**The Linear Inverted Pendulum Model (LIPM).** Assume all mass at one point (the
CoM), a massless leg to the floor at p, and **constant CoM height** $z_c$.

1. Constant height: $f_z = mg$.
2. A massless leg carries no moment, so the force points along the leg:
   $f_x/f_z = (x - p)/z_c$.
3. $m\ddot x = f_x = mg(x-p)/z_c$, so
$$\ddot x = \omega^2(x - p), \qquad \omega = \sqrt{g/z_c}$$

**The CoM accelerates away from the ZMP, harder the further away it is.** That
is a hill, not a valley. (Equivalently, $mz_c\ddot x = mg(x-p)$: zero moment
about p, which is where the name comes from.) Units: ω in 1/s.

**Closed-form solution** (p fixed, $x(0) = x_0$, $\dot x(0) = v_0$):
$$x(t) = p + (x_0 - p)\cosh\omega t + \frac{v_0}{\omega}\sinh\omega t$$
No numerical integration needed. With $z_c = 0.60$ m: **ω = 4.04 /s, time
constant 1/ω = 0.247 s.** The distance from p grows about ×e every 0.25 s; a
0.65 s step is 2.6 time constants, so a starting error grows ×cosh 2.6 ≈ **7**
per step. "That is why a walker must act every step."

**Phase portrait.** On the line $\dot x = -\omega(x-p)$ the CoM comes to rest
over p; on $\dot x = +\omega(x-p)$ it runs away fastest; every other motion
eventually runs away: the robot falls to the side of the "rest" line it is on.

**The capture point.**
$$\xi = x + \frac{\dot x}{\omega}, \qquad \dot\xi = \omega(\xi - p) \Rightarrow \xi(t) - p = (\xi_0 - p)e^{\omega t}$$
A second-order equation becomes first order, and all the danger is in it.
**Put the foot (the ZMP) on ξ and the CoM slides onto it and stops**: the
capture point is where you must step to come to rest in one step. Step short
and you keep falling forward; step beyond and you are pushed back. At
0.3 m/s with ω = 4.04, ξ is 7.4 cm ahead; twice the speed needs ~15 cm, more
than the foot reaches, so you must take a step.

**One step = one boundary value problem.** Choose $v_0$ so the CoM ends at
$x_T$:
$v_0 = \omega\,\dfrac{x_T - p - (x_0-p)\cosh\omega T}{\sinh\omega T}$. It is
linear in $v_0$: exactly one answer. From −0.07 to +0.07 m in 0.65 s:
$v_0 = 0.327$ m/s. "Run it backwards" means *reason from the end you want to
the start you need*, not integrate backwards in time. **Chain the BVPs**: each
step must start where the last ended.

**The sideways problem.** Stance foot at $\pm w$ ($w = 0.119$ m), mirrored
steps. Solving gives $y_0 = 0$: **the CoM crosses the midline exactly at each
support exchange**, with $v_0 = \omega w\tanh(\omega T/2) = 0.416$ m/s. The sway
peaks at 5.9 cm: it never reaches the foot it leans toward.

**The swing leg.** With σ the swing fraction (after removing double support):
$x_\text{foot} = x_A + (x_B - x_A)(\sigma - \tfrac{\sin 2\pi\sigma}{2\pi})$, whose
velocity factor $1 - \cos 2\pi\sigma$ is **zero at both ends** (leave gently,
land at rest), and $z_\text{foot} = z_0 + h\tfrac{1-\cos 2\pi\sigma}{2}$ (peak
h = 10 cm at mid-swing). A straight-line swing would jump to full speed.

## The controller

At 500 Hz, **open loop**: time → which step → LIPM CoM target → foot targets
(plan + swing) → week-3 DLS IK for 12 leg angles → 29 position servos →
physics. Nothing comes back from the physics. Defaults: step 0.14 m, 0.65 s,
clearance 0.10 m, 20 % double support at each end; the first step is a half
stride.

**Double support** matters because the ZMP is a weighted average of contact
points: it **cannot jump** from one foot to the other; it can only travel
between them while both touch. Set `double_support = 0` and the robot falls at
**2.0 s**.

## Numbers worth knowing

- G1 mass 33.3 kg; foot about 17 × 6 cm through four spheres at
  $x \in \{-0.05, +0.12\}$, $y = \pm 0.025\ldots0.03$.
- **Walks 0.99 m in 8.8 s (~0.11 m/s)** with no learning, ~200 lines of code.
  Pelvis stays in 0.69–0.74 m.
- Failures: rushed (0.45 s steps) falls at 3.2 s; no double support at 2.0 s;
  Moon gravity (ω drops to √(1.62/0.6) ≈ 1.6) at 5.3 s; ice (friction 0.2)
  changes nothing for this gait.
- σ_min at a straight leg 9e-7, at the crouch 0.089: **every gait starts
  crouched.**

## Traps

- **The measured ZMP leaves the support polygon** (y over −0.23…+0.11 m against
  feet at ±0.119 m). The criterion is right; **the LIPM's assumptions failed**:
  the real robot has a swinging leg with momentum, a bobbing CoM, ~30 % of its
  mass below the hips, a rolling weight transfer, and servos that sag. "A model
  does not have to be true to be useful, only wrong in ways that do not matter
  yet."
- Computing each step's CoM trajectory absolutely instead of chaining it
  teleports the body back half a stride at every support exchange. It *looks*
  like a balance failure and is a bookkeeping error.
- The walker is **open loop**: it never reads the IMU, drifts sideways, cannot
  survive a push, walks on flat ground only. The field moved to feedback.
- "A learned policy may need 150 million steps. **Learning buys robustness, not
  the gait itself.**"

---

# Week 5 — Actuation, PD control, and CPG gaits

## The ideas

**The actuator is a controller in disguise.** MuJoCo's position actuator is a
PD law, declared in Layer 2 but doing Layer 4's job:
$$\tau = k_p(q_\text{ref} - q) - k_v\dot q$$
The P term pulls toward the command; the D term opposes motion in *either*
direction (the damper is fixed to the ground, the spring to the command).
In MuJoCo's general form $\tau = g_0\,\text{ctrl} + b_0 + b_1 q + b_2\dot q$, a
position actuator sets $g_0 = k_p$, $b_1 = -k_p$, $b_2 = -k_v$; so **$k_p$
appears twice**, and changing only the gain makes the servo pull toward some
other angle. On the G1, $k_p = 500$ everywhere and $k_v$ ranges 4.55–43.

## The maths

**One joint is a mass–spring–damper** (gravity off, others held):
$$M\ddot q + k_v\dot q + k_p q = k_p q_\text{ref}$$
Characteristic equation $Mr^2 + k_v r + k_p = 0$, i.e.
$r^2 + 2\zeta\omega_n r + \omega_n^2 = 0$ with
$$\omega_n = \sqrt{k_p/M}, \qquad \zeta = \frac{k_v}{2\sqrt{k_p M}}, \qquad r = \omega_n\left(-\zeta \pm \sqrt{\zeta^2 - 1}\right)$$

| Regime | Roots | What you see |
| --- | --- | --- |
| ζ < 1 under-damped | complex pair $-\zeta\omega_n \pm i\omega_d$ | overshoots and rings |
| ζ = 1 critical | double root $-\omega_n$ | fastest arrival **without** overshoot |
| ζ > 1 over-damped | two real negative roots | creeps in; the slow root dominates |

With $k_v > 0$ both roots have negative real part: **the joint always arrives**;
the question is how. A complex pair always has magnitude $\omega_n$.

**Overshoot depends on ζ only:**
$\text{overshoot} = \exp\!\left(-\pi\zeta/\sqrt{1-\zeta^2}\right)$ — 44 % at
ζ = 0.25, 16 % at 0.5, none at 1.
**"Stiffness sets the speed; the damping ratio sets the shape."**
Quadrupling $k_p$ doubles $\omega_n$ (∝ √k_p) **and halves ζ** (∝ 1/√k_p).

**The G1's design choice:** all twelve leg joints have **ζ = 1.00** (critically
damped); solving gives $k_v = 2\sqrt{k_p M}$, which is why $k_v$ varies tenfold
while $k_p$ does not. The arms are looser (ζ 0.7–1.9). Knee: M = 0.126 kg·m²,
$\omega_n$ = 63 rad/s (10 Hz), $1/\omega_n$ ≈ 16 ms; critical damping reaches
1 % in ~0.105 s (≈ 6.6/ω_n).

**The formula over-predicts ringing on the real robot** (16.9 % measured vs
44 % predicted for $k_v/4$), because the other joints and the floating pelvis
move back. Use the one-joint formula to choose gains; measure to know what you
got.

**Sag under gravity.** At rest, $0 = k_p(q_\text{ref} - q) + \tau_g$, so
$$q - q_\text{ref} = \tau_g / k_p$$
Double $k_p$, halve the sag; **never zero** for a finite spring. "A spring at
rest is a spring under load, and load means deflection." Arm straight out
(4.7 N·m): 9.5 mrad sag at $k_p$ = 500, matching $\tau_g/k_p$ within 0.05
mrad. Whole walking robot: **~11 mm** of steady pelvis sag.

**P, PD, PI, PID.** $\tau = k_p e + k_i\int e\,dt + k_d\dot e$.
P reacts to the present, I to the past (it accumulates until the error is
zero), D to the trend (it brakes). Measured on one knee with a 10 N·m load:

| Law | Rise | Overshoot | Final error |
| --- | --- | --- | --- |
| P | 19 ms | 56 % | 20 mrad |
| PD | 78 ms | 0 % | 20 mrad |
| PI | 19 ms | 57 % | 1.7 mrad |
| PID | 72 ms | 0 % | 0.8 mrad |

**Only I removes the sag** (the I term ends up carrying the whole 10 N·m).
**D is the brake.** The G1 servo is **PD**: it has no I term.

**Saturation.** $\tau = \operatorname{clip}(k_p e - k_v\dot q, -\tau_\max, \tau_\max)$.
The law is linear only while $|e| \le \tau_\max/k_p$ (50/500 = 0.1 rad ≈ 5.7°).
Outside that band the effective stiffness is zero: **more error buys no more
torque, and no value of $k_p$ helps.** If gravity asks for more than
$\tau_\max$, the joint gives way until the load falls to $\tau_\max$.

**Central pattern generators (CPGs).** Coupled phase oscillators,
$\dot\phi_L = \omega_L + w\sin(\phi_R - \phi_L - \pi)$ and symmetrically for R.
The phase error $e = \phi_R - \phi_L - \pi$ obeys
$\dot e = (\omega_R - \omega_L) - 2w\sin e$. Equal rates: the legs lock into
anti-phase ($e \to 0$ as $e^{-2wt}$). Unequal rates lock only if
$|\Delta\omega| \le 2w$; weaker coupling lets the legs drift apart for ever.

## Numbers worth knowing

- **The motors in the model are infinitely strong**: the MJCF limits the
  commanded angle, not the torque. The walker peaks at ~124 N·m and nothing
  objects. A sim-to-real gap "in plain sight".
- Torque limit on the walk: **50 N·m falls, 55 walks.** The knee is above
  50 N·m for only 4 % of the walk, and that is enough to drop the robot.
- $k_p$ sweep: **only 500 walks**; 250 collapses, 1000 throws it over. Rescaling
  $k_v$ to keep ζ = 1 does not save it: ζ keeps the *shape* but changes the
  speed and the sag the gait was tuned against.
- CPG: nine frequency × amplitude settings, **all nine fall within ~2–3 s**.

## Traps

- **PD rose *slower* than P here** (78 vs 19 ms), against the textbook table.
  P is quick only because it rings: "the quickness is the ringing".
- **D changes how the joint arrives, not where.** P and PD keep the same 20 mrad
  sag.
- **Rhythm is not balance.** A CPG produces a plausible stepping pattern and
  "has no idea where the floor is": nothing keeps the CoM over the support
  polygon. That is week 4's value stated negatively. Biological CPGs are
  entrained by sensory feedback; "an open-loop oscillator is a CPG with its
  spinal cord cut off from its body."
- Both the LIPM walker and the CPG are **open loop**. Every remaining week of
  the course is about closing that loop.
- An explicit servo law written by hand through `qfrc_applied` is **unstable at
  500 Hz and fine at 1 kHz**; MuJoCo's own actuator survives 500 Hz because the
  engine treats it implicitly.

---

# Week 6 — Sensing, state estimation, observation design

## The ideas

**The cheating we have been doing.** Every controller so far read `qpos[:7]`
(pelvis position and orientation). A real robot knows neither: there is no
sensor for "where am I in the room". The G1 has **two IMUs and joint
encoders**, nothing else. Using `qpos[7:]` (encoder angles) is fair; using
`qpos[:7]` is not.

**Sensors in simulation are perfect.** Two runs give identical `sensordata`,
and declared noise is not applied. "Convenient and dishonest." Noise and bias
must be injected by hand, because an estimator tested on perfect sensors
teaches nothing.

## The maths

**The accelerometer does not measure acceleration.** A proof mass on a spring:
$m\mathbf a = \mathbf f_\text{spring} + m\mathbf g$, and the chip reads only the
spring, so it reports the **specific force**
$$\mathbf f = \mathbf a - \mathbf g, \qquad \mathbf f_b = R^\top(\mathbf a - \mathbf g)$$
At rest $\mathbf f = -\mathbf g$: **9.81 m/s² pointing up**. In free fall
$\mathbf f = 0$. In a lift accelerating up, more than 9.81. At rest the reading's
*direction* is "up" in body axes: **an accelerometer is a tilt sensor.**

**Tilt from the accelerometer** (at rest, $R = R_z(\psi)R_y(\theta)R_x(\phi)$):
$\mathbf f_b = g(-\sin\theta,\ \sin\phi\cos\theta,\ \cos\phi\cos\theta)$, so
$$\phi_\text{acc} = \operatorname{atan2}(f_y, f_z), \qquad \theta_\text{acc} = \operatorname{atan2}\!\left(-f_x, \sqrt{f_y^2 + f_z^2}\right)$$
g cancels; only the direction matters. **Yaw drops out**:
$R_z(\psi)^\top\mathbf e_z = \mathbf e_z$, so no accelerometer reading can ever
contain heading. (Readings at yaw 0°, 45° and 170° are identical.) That is why
policies observe **gravity in the body frame**: "the largest piece of attitude a
real robot can honestly know."

**The assumption breaks while walking.** A sideways acceleration adds straight
into $f_y$, and atan2 cannot tell it from gravity: every 1 m/s² reads as
atan(1/9.81) ≈ **5.8° of false roll**. The accelerometer is kicked at every
footfall (|f| ranged 3.4–19.3 m/s² during the walk).

**The gyroscope** reads the rate plus a bias: $\omega_k = \dot\phi_k + b$.
Integrating, $e(t) = bt$: **the error grows linearly, for ever.** 0.01 rad/s is
0.57°/s, **34° after a minute.** "A gyro is right now and wrong later; the
accelerometer is the opposite." These are **complementary failure modes**.

**Why not integrate twice for position?** A bias b in acceleration gives
velocity error $bt$ and **position error $\tfrac12 bt^2$, a parabola.** The
worst bias is gravity itself: a 1° tilt error leaves $g\sin 1° \approx 0.17$
m/s², which is 8.6 m after 10 s and 34 m after 20 s of standing still.
**Position needs a sensor that sees the world**; the G1 has none, so position
is privileged.

**The complementary filter.** "Trust the gyro over short intervals; let gravity
correct it slowly."
$$\hat\theta_k = \alpha(\hat\theta_{k-1} + \omega_k\Delta t) + (1 - \alpha)\,\theta_{\text{acc},k}$$
α = 1 is a pure gyro (drifts); α = 0 is a pure accelerometer (kicked by every
step). As transfer functions, the estimate is a **low-pass** of the
accelerometer angle plus a **high-pass** of the gyro angle, and
$L + H = 1$: with perfect sensors you get the truth back.

**Time constant.** The low-pass memory decays as $\alpha^k = e^{-t/\tau}$, so
$$\tau = -\frac{\Delta t}{\ln\alpha} \approx \frac{\Delta t}{1 - \alpha}, \qquad f_c = \frac{1}{2\pi\tau}$$
At $\Delta t = 0.002$ s and α = 0.995: **τ ≈ 0.4 s, f_c ≈ 0.4 Hz.** Below f_c the
accelerometer wins, above it the gyro. A gyro bias now costs a **bounded**
error, $e_\infty \approx \tau b$ (a quarter degree at b = 0.01 rad/s), instead
of growing as $bt$. "Trust the gyro for τ seconds and you inherit τ seconds'
worth of its bias, never more."

**Choosing α.** Lowering $f_c$ (α closer to 1) lets less footfall through and
pays more bias; raising it does the reverse. The optimum on our walk is around
α ≈ 0.998 (τ ≈ 1 s). **"α is not a constant, it is a claim about your
sensors."** A worse bias pushes it toward smaller τ; a rougher gait toward
larger τ.

| Estimator (walk, biased gyro) | Mean error | Final error |
| --- | --- | --- |
| accelerometer only | ~6° | ~4.5° (worst 21°) |
| gyro only | 1.35° | 2.50° and growing |
| filter, α = 0.995 | 1.43° | 1.02° |

**Observable vs privileged.** $o = h(s) + \text{noise}$. $s$ is the full
simulator state (36 + 35 numbers, contacts, friction, terrain, masses); $o$ is
what the robot measures (2 IMUs + 29 encoders). **Observable**: computable from
$o$, perhaps through a filter. **Privileged**: in $s$ but not recoverable from
any function of $o$: world position, heading (yaw), height above the floor,
contact forces, friction.

| Quantity | Real robot? | Use in a policy? |
| --- | --- | --- |
| joint angles `qpos[7:]` | encoders | yes |
| joint velocities `qvel[6:]` | differentiated encoders | yes |
| gravity in body frame | IMU + filter | yes |
| angular velocity | gyroscope | yes |
| previous action | the robot commanded it | yes |
| body height `qpos[2]` | no sensor | no |
| world position `qpos[:2]` | double integration diverges | no |
| yaw | gravity is blind; gyro drifts | no |
| foot contact forces | not on this robot | no |

Gravity 3 + gyro 3 + 29 angles + 29 velocities = **64 numbers**, all measurable.
Week 7's leg-only environment uses 42 inputs and outputs 12 leg targets.

**Asymmetric actor-critic** (preview of week 8): the **actor** (policy) runs on
the robot and sees only $o$; the **critic** (value function) is used only
during training and then discarded, so it may see all of $s$. Privileged
information is used where it is legal.

## Traps

- **Mean error is the wrong way to judge an estimator you intend to run for an
  hour.** The pure gyro had a slightly *lower mean* (1.35° vs 1.43°) but 2.5×
  the final error, and its error keeps growing.
- "Accelerometer reads zero at rest": wrong, it reads 9.81 up. "Zero in free
  fall": right.
- A filter's truth must come from the **IMU site's** own orientation, not the
  pelvis: they differ by up to 0.7° while walking, the size of the filter
  errors themselves.

---

# Formula sheet (memorise these)

| Topic | Formula |
| --- | --- |
| The loop | $x_{k+1} = f(x_k, u_k)$, $y_k = h(x_k)$, $u_k = \pi(y_k)$ |
| Steps per decision | $N = f_\text{phys}/f_\text{pol}$ |
| Servo (PD) | $\tau = k_p(\text{ctrl} - q) - k_v\dot q$ |
| Droop / sag | $q - q_\text{ref} = \tau_g/k_p$ (sign: in the direction gravity pulls) |
| Equation of motion | $M(q)\dot v + c(q,v) = \tau + \tau_\text{contact}$ |
| Semi-implicit Euler | $v_{n+1} = v_n + h\dot v_n$, $q_{n+1} = q_n + h v_{n+1}$ |
| Spring stability | $h < 2/\omega$, $\omega = \sqrt{k/m}$ |
| Quaternion | $(\cos\tfrac\theta2, \mathbf n\sin\tfrac\theta2)$, unit length, scalar first |
| nq / nv | free 7/6, ball 4/3, hinge 1/1 |
| Transform | $\mathbf p_a = R_{ab}\mathbf p_b + \mathbf t_{ab}$ |
| Chain | $R_{ac} = R_{ab}R_{bc}$, $\mathbf t_{ac} = R_{ab}\mathbf t_{bc} + \mathbf t_{ab}$ |
| Inverse | $\mathbf p_b = R^\top(\mathbf p_a - \mathbf t)$ |
| Jacobian | $\dot{\mathbf x} = J\dot{\mathbf q}$; hinge column $\mathbf a_j \times (\mathbf p - \mathbf o_j)$ |
| Finite-difference ε | $\varepsilon_\text{best} \approx \sqrt{u} \approx 10^{-8}$ |
| Two-link reach | $\lvert\ell_1 - \ell_2\rvert \le d \le \ell_1 + \ell_2$ |
| Knee angle | $\cos\theta = (d^2 - \ell_1^2 - \ell_2^2)/(2\ell_1\ell_2)$ |
| DLS | $\Delta q = J^\top(JJ^\top + \lambda^2 I)^{-1}\mathbf e$; gain $\sigma/(\sigma^2 + \lambda^2)$ |
| Condition number | $\kappa = \sigma_\max/\sigma_\min$ |
| CoP | $p = \sum f_i x_i / \sum f_i$ |
| LIPM | $\ddot x = \omega^2(x - p)$, $\omega = \sqrt{g/z_c}$ |
| LIPM solution | $x = p + (x_0 - p)\cosh\omega t + (v_0/\omega)\sinh\omega t$ |
| Capture point | $\xi = x + \dot x/\omega$, $\dot\xi = \omega(\xi - p)$ |
| Natural frequency, damping | $\omega_n = \sqrt{k_p/M}$, $\zeta = k_v/(2\sqrt{k_pM})$ |
| Overshoot | $\exp(-\pi\zeta/\sqrt{1 - \zeta^2})$ |
| Saturation band | $\lvert e\rvert \le \tau_\max/k_p$ |
| CPG lock | $\lvert\Delta\omega\rvert \le 2w$ |
| Specific force | $\mathbf f = \mathbf a - \mathbf g$ |
| Tilt | $\phi = \operatorname{atan2}(f_y, f_z)$, $\theta = \operatorname{atan2}(-f_x, \sqrt{f_y^2 + f_z^2})$ |
| Gyro drift | $e = bt$; double integration $\tfrac12 bt^2$ |
| Complementary filter | $\hat\theta_k = \alpha(\hat\theta_{k-1} + \omega_k\Delta t) + (1-\alpha)\theta_\text{acc}$ |
| Filter time constant | $\tau = -\Delta t/\ln\alpha \approx \Delta t/(1-\alpha)$; bias error ≈ $\tau b$ |

---

# Glossary

- **Actuator**: what you command. On the G1, a position servo (PD law), one per
  hinge, 29 in all. Not the same thing as a joint.
- **Annulus**: a ring; the reachable set of a two-link leg.
- **BADQACC**: MuJoCo's warning that accelerations blew up; it then resets the
  state.
- **Capture point (ξ)**: where to step to stop in one step.
- **CoM / CoP / ZMP**: centre of mass; centre of pressure (force-weighted
  average of contacts); zero moment point (equals the CoP on flat ground).
- **CPG**: central pattern generator, a rhythm produced by coupled oscillators.
- **Critical damping (ζ = 1)**: the fastest arrival without overshoot.
- **DLS**: damped least squares, the IK step that stays sane near a
  singularity.
- **Double support**: both feet on the floor; the ZMP can move between them.
- **Forward / inverse kinematics**: angles → foot; foot → angles.
- **Free joint**: 6 freedoms, 7 `qpos` numbers; the floating base.
- **IMU**: inertial measurement unit = gyroscope + accelerometer.
- **Jacobian**: the matrix of derivatives mapping joint velocities to foot
  velocity.
- **LIPM**: linear inverted pendulum model.
- **MJCF**: MuJoCo's XML model format.
- **Open loop / closed loop**: commands computed without / with measurements of
  the result.
- **Privileged information**: in the simulator state but not measurable on the
  robot.
- **Quaternion**: 4 numbers, unit length, representing a rotation (scalar first
  in MuJoCo).
- **Semi-implicit Euler**: velocity updated first, position moved with the new
  velocity.
- **Sim-to-real gap**: the difference between Layer 1 in simulation and
  reality.
- **Singularity**: a pose where the Jacobian loses a rank (σ_min ≈ 0); the
  straight leg.
- **Specific force**: what an accelerometer measures, $\mathbf a - \mathbf g$.
- **Support polygon**: the convex hull of the contact points.
- **Underactuated**: fewer motors than degrees of freedom.

---

# Practice questions (with answers at the bottom)

These are in the exam's style but are *not* the exam questions. Try each one
before you look.

1. A robot has one free joint, one ball joint and twelve hinges. What are nq
   and nv?
2. Physics runs at 1 kHz and a policy at 100 Hz. How many physics steps per
   decision, and what happens to `ctrl` between decisions?
3. A servo with $k_p = 400$ holds a joint against 8 N·m of gravity. How large
   is the steady-state error, and what happens if $k_p$ is doubled?
4. A spring–mass system has ω = 50 rad/s. What is the largest stable timestep
   for semi-implicit Euler?
5. Write $\mathbf t_{ac}$ in terms of $R_{ab}$, $\mathbf t_{ab}$, $\mathbf t_{bc}$.
6. Why may you not compare two orientations by comparing their quaternion
   components?
7. The left knee is at `qpos[10]`. Which index holds its velocity in `qvel`?
8. Which landmarks does `right_hip_yaw` move? Which does `waist_roll` move?
9. A two-link leg has $\ell_1 = 0.5$, $\ell_2 = 0.3$. Is a target 0.1 m from the
   hip reachable?
10. IK from a straight leg leaves a residual of 0.79 after 40 iterations, and
    the same after 400. Should you raise λ, raise the iteration count, or change
    the seed?
11. A LIPM has $z_c = 0.4$ m (take g = 10). What is ω? What is the time
    constant?
12. The CoM is at x = 0 moving at 0.2 m/s, ω = 4 /s. Where is the capture point?
13. Why can the ZMP not jump instantly from one foot to the other?
14. A joint has $k_p = 900$, M = 1 kg·m², $k_v = 30$. What are $\omega_n$ and ζ?
    Does it overshoot?
15. A motor's limit is 30 N·m with $k_p = 600$. Over what error is the servo
    linear?
16. An accelerometer at rest reads (0, 0, 9.81). The robot is then dropped.
    What does it read during the fall?
17. A gyro has bias 0.02 rad/s. Roughly how far off is the integrated angle
    after 10 s, in degrees?
18. With Δt = 0.001 s and α = 0.999, what is τ?
19. Is the robot's heading (yaw) observable from the IMU alone? Why?

**Answers.** 1: nq = 7 + 4 + 12 = 23, nv = 6 + 3 + 12 = 21. 2: 10; the same
`ctrl` is held and the servos keep correcting every step. 3: 8/400 = 0.02 rad;
it halves to 0.01 rad, never zero. 4: h < 2/50 = 0.04 s. 5:
$\mathbf t_{ac} = R_{ab}\mathbf t_{bc} + \mathbf t_{ab}$. 6: q and −q are the
same rotation (and conventions differ: scalar first vs last). 7: `qvel[9]`.
8: only the right foot; both hands and the torso, neither foot. 9: no — the
annulus starts at |0.5 − 0.3| = 0.2 m. 10: change the seed (crouch): a
residual that ignores iterations is geometry, and damping cannot add the
missing direction. 11: ω = √(10/0.4) = 5 /s; 1/ω = 0.2 s. 12: ξ = 0 + 0.2/4 =
0.05 m ahead. 13: it is a weighted average of the contact points with
non-negative weights, so it can only lie between the feet while both touch.
14: ω_n = 30 rad/s, ζ = 30/(2·√900) = 0.5; yes, about 16 %. 15:
|e| ≤ 30/600 = 0.05 rad. 16: zero (specific force a − g = 0). 17: 0.2 rad ≈
11.5°, and growing. 18: τ ≈ 0.001/0.001 = 1 s. 19: no; gravity is blind to
rotation about the vertical, and the gyro's integral drifts.

---

## Final advice

- Re-run the lab scripts for weeks 0–6 and predict each printed number before
  you see it. The exam asks about *what the numbers mean*, not about decimals.
- For every formula on the sheet, be able to say what each symbol is and what
  happens when you double one of them.
- For every trap above, be able to say *why* the tempting answer is wrong.
- When in doubt, ask the course question: **which layer is this, and what is it
  assuming about the layers beneath it?**
