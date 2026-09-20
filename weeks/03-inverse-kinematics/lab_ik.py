"""Week 3 lab, on your laptop: move a target, and watch three solvers reach for it.

    uv run weeks/03-inverse-kinematics/lab_ik.py

A window opens with the G1 frozen in the week 4 crouch. Physics is never
stepped: this is inverse kinematics, geometry only, so the robot holds poses it
could never balance in and the pelvis never moves.

    arrows                 move the target forward/back (left/right) and up/down
    , and .                move the target sideways (y)
    [ and ]                damping lambda, ten times smaller / larger
    M                      next solver: dls -> inverse -> transpose -> dls
    I                      iterations per frame: 3 -> 10 -> 1 -> 3
    O                      position only (3 rows) <-> position and orientation (6 rows)
    S                      swap the seed: straight leg (the `stand` singularity) <-> crouch
    R                      put the target back on the foot
    SPACE                  next entry of TARGETS
    ENTER                  print target, foot, residual, |dq|, condition number, angles
    left-drag / right-drag / wheel   orbit / pan / zoom
    Keys go to the MuJoCo window, not the terminal -- click the window first.
    If no key does anything in the window, press the same key in this terminal
    instead -- single keys, no enter, arrows included. `q` stops it.

Three spheres. The RED one is the target you move. The GREEN one is the left
foot site, where the leg actually is. The big TRANSLUCENT one is the reach of
the leg -- thigh plus shin plus the ankle-to-site drop, measured from the
model -- centred on the hip. When the red sphere leaves it, no solver in the
world can put the green one on it.

Everything in this file is complete and explained; nothing is left blank. The
three solver functions in section 2 are the lecture's mathematics, each a few
lines. Read them first, run them, then change them. Section 4 shows what the
script does around them: measuring the error, asking MuJoCo for the Jacobian,
clamping to the joint limits, and iterating a few times per frame so you can
watch a solver converge, creep, or blow up. Every number quoted below was
measured with this script's own functions, headless, before it was written.

Experiments, in order, and what to show the instructor:

1. Damped least squares (the default: lambda = 1e-2, 3 iterations per frame).
   Press the up arrow five times, a 5 cm move. The green sphere follows and the
   residual on ENTER is about 4e-6 m after one frame, 1e-10 after two. Press
   `I` twice for one iteration per frame: 1.4e-2 after the first frame, 5e-4
   after the third -- it converges in a few frames, not one, because the
   problem is nonlinear and each step solves a linearised copy of it.
2. The plain inverse. Press `M` once. Crouched, it converges like dls (1.5e-2,
   then 5e-4 after three iterations: this is Newton's method). Now press `S`
   for the straight leg and the UP arrow once: 1 cm up, which only needs the
   knee to bend -- a target any bent leg reaches instantly. The first iteration
   asks for |dq| = 1.1e4 rad, because that direction has sigma = 9e-7. Only
   the clamp to the joint limits stops it: the leg folds onto its limits and
   never recovers (residual 1.2 m after 300 iterations). `S` back to the
   crouch and the same 1 cm is answered at once.
3. Damped least squares at the singularity. `M` to dls, `S` to the straight
   leg, up arrow once, and sweep lambda with `[` and `]`:
     1e-6        the same fold as the inverse (|dq| = 5e3).
     1e-3, 1e-2  a tiny first step (1e-4 rad), then as the knee leaves zero the
                 lost direction comes back with a gain of 1/(2 lambda) and it
                 blows up anyway: |dq| = 60 rad by the tenth iteration, a
                 garbage pose on the limits, residual 4.5 cm for ever.
     1e-1        |dq| ~ 1e-5: the foot creeps 0.6 mm in 300 iterations, and the
                 knee creeps to its BACKWARD limit (-0.087). At exactly zero the
                 linearisation cannot tell knee-forward from knee-back -- both
                 raise the foot only at second order -- and it picks the wrong
                 sign. The lecture's "no direction" made visible.
     1           nothing moves at all: the direction is switched off.
   No lambda brings the foot up from a straight leg. Bending the knee first
   (`S`) is the only fix, and it is what week 4's walker does with every step.
4. The Jacobian transpose. `M` twice, crouch, up arrow five times. It creeps:
   after 100 iterations 8 mm of the 50 remain (press `I` for 10 per frame to
   speed it up). At the straight leg it asks for |dq| = 1.5e-5 and nothing
   happens -- it never divides by anything, so it cannot blow up; it just stops.
5. Reach. Right arrow thirty times (0.3 m forward): the red sphere leaves the
   translucent one, and ENTER prints hip->target 0.69 m of a 0.66 m reach. The
   residual stops shrinking and hovers around 5-6 cm; the angles show the knee
   on its lower limit (-0.087) and the ankle on its upper (0.524): the leg is as
   long as it gets. Press `O` for position only and pull back to 0.25 m: the
   residual pins at exactly 6.8e-3 on every ENTER. Geometry, not a solver
   running out of iterations.
6. Damping as a price. dls, crouch, `I` to one iteration per frame, `]` up to
   lambda = 1: after 50 frames 3.1 cm of a 5 cm move remain. At 0.3 it closes
   (1.6e-5) within 50 frames, at 0.1 within 10. Find the largest lambda that
   closes a 5 cm move within one second (50 frames) -- it is between 0.1 and 1.
7. Orientation. Press `O`: three rows instead of six, J is 3 x 6, and the foot
   is free to tilt. Move the target 15 cm to the robot's left (`.` fifteen
   times) and 5 cm up: the foot tilts 16 degrees, and the ankle-roll angle on
   ENTER stays 0 because nothing asked it to move. `O` again: the six-row
   solver levels the foot (0.2 degrees) by turning ankle roll to -0.26 rad.
8. Add a pose to TARGETS with the foot 10 cm forward and 5 cm up, and one you
   believe is unreachable. Press SPACE to cycle through them, and be right.

Then break something on purpose and explain what you see: return `-dq` from
`dls_step` (the residual runs to 0.57 m and sticks there, on the joint limits),
or return `J.T @ err` without the step length in `transpose_step` (the residual
goes 0.050, 0.049, 0.049, ... 0.32, 0.59 in ten iterations: it overshoots,
then diverges).
"""

from __future__ import annotations

# --- 0. what each import is for ---------------------------------------------
import os          # SOC4180_AUTOCLOSE, the environment variable that ends the smoke test
import time        # frame pacing, and the auto-close deadline

import mujoco      # the simulator: models, forward kinematics, Jacobians, the viewer's geoms
import numpy as np

import soc4180                          # the course package: model loading, viewer, keys
from soc4180 import kinematics as kin   # week 2/3 helpers: leg indices, foot site, pose error


# --- 1. settings you are meant to change --------------------------------------

# Offsets from the foot's starting position (metres, world frame: x forward,
# y left, z up). SPACE cycles through them. Add your own.
TARGETS = [
    ("start: on the foot",        [0.00, 0.00, 0.00]),
    ("5 cm up",                   [0.00, 0.00, 0.05]),
    ("10 cm forward",             [0.10, 0.00, 0.00]),
    ("forward and up",            [0.12, 0.00, 0.08]),
    # add your own below
]

SIDE = "left"                    # which leg the solver drives ("right" works too)
STEP = 0.01                      # metres the target moves per key press
ITERATION_CHOICES = (1, 3, 10)   # what `I` cycles through; the script starts at 3


# --- 2. the three solvers: this is the lecture, in code -------------------------
#
# Each takes the same three things and returns one joint step `dq` (6 numbers):
#   J    the Jacobian: how fast the foot moves per unit of each joint's motion.
#        6 x 6 with orientation (three position rows, three rotation rows) or
#        3 x 6 in position-only mode. Section 4 gets it from MuJoCo.
#   err  the foot motion that would put the foot on the target: [dx, dy, dz]
#        in metres, then [droll, dpitch, dyaw] in radians (if orientation is on).
#   lam  the damping. Only dls uses it; the others take it so `M` can swap them.
# Each solves, in its own way, the lecture's equation   J dq = err.

def dls_step(J: np.ndarray, err: np.ndarray, lam: float) -> np.ndarray:
    """Damped least squares:  dq = J^T (J J^T + lambda^2 I)^-1 err.

    This is the minimiser of  |J dq - err|^2 + lambda^2 |dq|^2 : reduce the
    error, but pay lambda^2 per unit of joint motion squared. Read the code
    against the slide "Where the formula comes from":

      J @ J.T + lam**2 * np.eye(m)   the m x m matrix (J J^T + lambda^2 I). For
                                     lambda > 0 it is positive definite, so it
                                     ALWAYS has an inverse -- even at a
                                     singularity, where J itself does not.
      np.linalg.solve(A, err)        computes A^-1 err without forming A^-1
                                     (cheaper and more accurate than inv(A) @ err).
      J.T @ (...)                    the transpose on the outside: the "right
                                     form" of the push-through identity, which
                                     inverts an m x m matrix rather than 6 x 6
                                     when m = 3 rows are in use.

    In singular-value terms each foot direction is scaled by sigma / (sigma^2 +
    lambda^2): the true inverse 1/sigma where sigma >> lambda, and switched off
    where sigma << lambda. That is why the straight leg stays put at lambda =
    0.1 (experiment 3) instead of folding: the lost direction is switched off.
    """
    m = J.shape[0]                                   # 6 rows, or 3 in position-only mode
    A = J @ J.T + lam**2 * np.eye(m)                 # (J J^T + lambda^2 I), always invertible
    return J.T @ np.linalg.solve(A, err)             # J^T A^-1 err


def inverse_step(J: np.ndarray, err: np.ndarray, lam: float) -> np.ndarray:
    """The obvious answer:  dq = J^-1 err  (a least-squares solve when J is 3 x 6).

    `np.linalg.lstsq` returns the exact J^-1 err whenever J is square and
    invertible, and the minimum-norm least-squares answer otherwise. There is
    no damping: every direction of `err` is divided by its own singular value,
    so a direction with sigma = 9e-7 (the straight leg) costs 1/9e-7 = 1e6 rad
    per metre. A 1 cm request becomes ~1e4 rad -- experiment 2. `lam` is
    ignored; it is in the signature only so `M` can swap solvers freely.
    """
    dq, *_ = np.linalg.lstsq(J, err, rcond=None)     # (solution, residuals, rank, sigmas)
    return dq


def transpose_step(J: np.ndarray, err: np.ndarray, lam: float) -> np.ndarray:
    """Jacobian transpose:  dq = alpha J^T err, with alpha the best step length.

    No inverse at all. Why it works: the squared error |err|^2 falls fastest
    along the direction J^T err (it is minus the gradient of |err - J dq|^2 at
    dq = 0), so stepping that way always helps a little. `alpha` is the length
    along that direction that minimises |err - J dq|, found in closed form:
    with g = J^T err and v = J g, the best alpha is (err . v) / (v . v).
    Nothing is ever divided by a small singular value, so it cannot blow up --
    at the straight leg it simply stops moving in the lost direction
    (experiment 4). The price is speed: it creeps.
    """
    g = J.T @ err                                    # the descent direction in joint space
    v = J @ g                                        # what that direction does to the foot
    alpha = float(err @ v) / max(float(v @ v), 1e-12)   # best step length; the max() guards v = 0
    return alpha * g


SOLVERS = [("dls", dls_step), ("inverse", inverse_step), ("transpose", transpose_step)]


# --- 3. two helpers ----------------------------------------------------------------

def leg_lengths(model, data):
    """Thigh, shin and the ankle-to-site drop, measured between joint anchors.

    The hip is three separate bodies whose offsets accumulate, so the distance
    from the pelvis to `knee_link` is not the thigh. Measure between the points
    the joints actually pivot about: a joint's world anchor is its body's
    origin plus the joint's own offset, rotated into the world. The foot site
    sits 0.0176 m below the ankle joint, so the farthest the SITE can be from
    the hip is thigh + shin + that drop: 0.3409 + 0.3000 + 0.0176 = 0.6585 m.
    Returns (hip anchor, thigh, shin, drop).
    """
    def anchor(body):
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        return data.xpos[b] + data.xmat[b].reshape(3, 3) @ model.jnt_pos[model.body_jntadr[b]]
    hip, knee, ankle = (anchor(f"{SIDE}_{n}_link") for n in ("hip_pitch", "knee", "ankle_pitch"))
    site = data.site_xpos[kin.foot_site_id(model, SIDE)]
    return hip, np.linalg.norm(knee - hip), np.linalg.norm(ankle - knee), np.linalg.norm(site - ankle)


def sphere(geom, pos, rgba, radius):
    """Fill one of the viewer's user geoms with a sphere (the three markers)."""
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0, 0], dtype=float),   # size: radius, unused, unused
                        np.asarray(pos, dtype=float),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=float))


# --- 4. the leg as the solvers see it ----------------------------------------------

class Leg:
    """One leg's pose, Jacobian, error and update, so a solver only does the maths.

    MuJoCo keeps every joint angle in one long array, `data.qpos` (36 numbers
    for the G1: 3 position + 4 quaternion for the floating pelvis, then 29
    joints). The six leg joints are a contiguous slice of it; `qidx` is that
    slice. Velocities live in `data.qvel` (35 numbers: the pelvis needs only 6),
    so the SAME six joints sit one index earlier there; `didx` is that slice.
    The Jacobian is a velocity object, so its columns are addressed by `didx`,
    while the angles we update are addressed by `qidx`. Mixing them is a silent
    off-by-one bug -- the reason both are kept here, with names.
    """

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.site = kin.foot_site_id(model, SIDE)         # the foot site: the point we place
        self.qidx = kin.leg_qpos_indices(model, SIDE)      # the six angles in data.qpos
        self.didx = kin.leg_dof_indices(model, SIDE)       # the same six joints in data.qvel
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{SIDE}_{j}")
                  for j in kin.LEG_JOINTS]
        self.lo, self.hi = model.jnt_range[joints, 0], model.jnt_range[joints, 1]   # joint limits
        self.flat = data.site_xmat[self.site].reshape(3, 3).copy()   # the level foot's rotation
        self._jp, self._jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))   # mj_jacSite output

    @property
    def foot(self):
        """Where the foot site is now, in the world (filled by mj_forward)."""
        return self.data.site_xpos[self.site]

    def jacobian(self, orientation=True):
        """The site Jacobian for the six leg joints: 6 x 6, or 3 x 6 position-only.

        `mj_jacSite` WRITES INTO the two arrays passed to it and returns
        nothing; they are 3 x nv -- every joint in the robot -- so the leg's
        six columns are picked out with `didx`. Rows: position, then rotation.
        """
        mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, self.site)
        J = np.vstack([self._jp, self._jr])[:, self.didx]
        return J if orientation else J[:3]

    def error(self, target, orientation=True):
        """The foot motion that would carry the foot onto the target.

        `kin.pose_error` returns six numbers: the position gap in metres and
        the rotation from the current foot orientation to `flat` (the level
        foot) as a rotation vector in radians -- something you can scale and
        add, which a rotation matrix is not. Position-only mode keeps the first
        three and lets the foot tilt as it likes.
        """
        err = kin.pose_error(self.data, self.site, target, self.flat)
        return err if orientation else err[:3]

    def step(self, solver, target, lam, orientation=True):
        """One iteration: measure, ask the solver for dq, apply it. Returns (err, |dq|).

        The clip keeps every angle inside its joint range; without it the
        solver would happily bend the knee backwards. `mj_forward` is MuJoCo's
        forward kinematics: it reads the new qpos and recomputes every world
        position, including the foot site and the Jacobian's basis, so the
        next iteration measures the pose the solver just produced.
        """
        err = self.error(target, orientation)
        J = self.jacobian(orientation)
        dq = np.asarray(solver(J, err, lam), float).reshape(6)
        self.data.qpos[self.qidx] = np.clip(self.data.qpos[self.qidx] + dq, self.lo, self.hi)
        mujoco.mj_forward(self.model, self.data)
        return err, float(np.linalg.norm(dq))

    def condition_number(self, orientation=True):
        """sigma_max / sigma_min of the current Jacobian: 20 crouched, 2e6 straight."""
        s = np.linalg.svd(self.jacobian(orientation), compute_uv=False)
        return s[0] / max(s[-1], 1e-300)


# --- 5. the interactive loop --------------------------------------------------------

def main() -> int:
    if soc4180.is_colab():
        print("This lab needs a desktop window; run it on your laptop.")
        return 1

    # The model and two seed poses. The crouch is the week 4 walker's nominal
    # pose (bent knees, pelvis at 0.74 m); the straight leg is every leg joint
    # at zero, which is the `stand` keyframe and the kinematic singularity.
    model = soc4180.load_g1()
    controller = soc4180.WalkingController(model)
    data = controller.initial_data()
    crouch = data.qpos.copy()
    straight = crouch.copy()
    for s in ("left", "right"):
        straight[kin.leg_qpos_indices(model, s)] = 0.0

    leg = Leg(model, data)
    slider_qpos = model.jnt_qposadr[model.actuator_trnid[:, 0]]   # for the viewer's sliders
    hip0, l1, l2, drop = leg_lengths(model, data)
    reach = l1 + l2 + drop

    foot0 = leg.foot.copy()
    # Everything the keys change lives in one dict, so the key callback (which
    # the viewer calls from its own thread) and the loop below share it.
    state = {"target": foot0.copy(), "lam": 1e-2, "seed": "crouch", "i": 0,
             "solver": 0, "iters": 1, "orientation": True, "print": False, "moved": True}

    print(f"thigh {l1:.4f} + shin {l2:.4f} + ankle-to-site {drop:.4f} = reach {reach:.4f} m from the hip")
    print(f"foot site starts at {foot0.round(4)}")

    def on_key(keycode):
        """Called with a GLFW keycode from the window, or from the terminal thread."""
        t = state["target"]
        if keycode == 265:                                   # up arrow
            t[2] += STEP
        elif keycode == 264:                                 # down arrow
            t[2] -= STEP
        elif keycode == 262:                                 # right arrow: forward (+x)
            t[0] += STEP
        elif keycode == 263:                                 # left arrow: back
            t[0] -= STEP
        elif keycode == 44:                                  # ','  to the robot's right (-y)
            t[1] -= STEP
        elif keycode == 46:                                  # '.'  to the robot's left (+y)
            t[1] += STEP
        elif keycode == 91:                                  # '['  less damping
            state["lam"] = max(state["lam"] / 10, 1e-9)
        elif keycode == 93:                                  # ']'  more damping
            state["lam"] = min(state["lam"] * 10, 1e3)
        elif keycode == 77:                                  # 'M'  next solver
            state["solver"] = (state["solver"] + 1) % len(SOLVERS)
            print(f"\n[solver: {SOLVERS[state['solver']][0]}]")
        elif keycode == 73:                                  # 'I'  iterations per frame
            state["iters"] = (state["iters"] + 1) % len(ITERATION_CHOICES)
            print(f"\n[{ITERATION_CHOICES[state['iters']]} iterations per frame]")
        elif keycode == 79:                                  # 'O'  orientation rows on/off
            state["orientation"] = not state["orientation"]
            print("\n[position and orientation: 6 rows]" if state["orientation"]
                  else "\n[position only: 3 rows, the foot may tilt]")
        elif keycode == 83:                                  # 'S'  swap the seed pose
            state["seed"] = "straight" if state["seed"] == "crouch" else "crouch"
            data.qpos[:] = straight if state["seed"] == "straight" else crouch
            mujoco.mj_forward(model, data)
        elif keycode == 82:                                  # 'R'  target back onto the foot
            t[:] = leg.foot
        elif keycode == 32:                                  # SPACE  next TARGETS entry
            state["i"] = (state["i"] + 1) % len(TARGETS)
            name, off = TARGETS[state["i"]]
            t[:] = foot0 + np.asarray(off, float)
            print(f"\n[{name}]")
        elif keycode in (257, 335):                          # ENTER (main or keypad)
            state["print"] = True
        else:
            return
        state["moved"] = True

    def report(err, dq_norm):
        """What ENTER prints: everything a diagnosis needs, in one block."""
        name = SOLVERS[state["solver"]][0]
        dist = np.linalg.norm(state["target"] - hip0)
        print(f"  target {state['target'].round(4)}  foot {leg.foot.round(4)}")
        print(f"  residual {np.linalg.norm(err[:3]):.2e} m   |dq| {dq_norm:.2e} rad   "
              f"solver {name}   lambda {state['lam']:.0e}   seed {state['seed']}   "
              f"{ITERATION_CHOICES[state['iters']]} it/frame   "
              f"{'6 rows' if state['orientation'] else '3 rows'}")
        outside = "   <- OUTSIDE: unreachable" if dist > reach else ""
        print(f"  hip->target {dist:.4f} m of reach {reach:.4f} m{outside}   "
              f"condition number {leg.condition_number(state['orientation']):.1e}")
        print(f"  angles {np.round(data.qpos[leg.qidx], 3).tolist()}"
              "   (hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll)")

    print("\n\n".join(__doc__.split("\n\n")[2:4]))
    soc4180.terminal_keys(on_key)        # the same callback, fed from the terminal
    print("  [terminal] keys dead in the window? press them here instead; 'q' stops.",
          flush=True)
    # SOC4180_AUTOCLOSE=<seconds> closes the window by itself: the smoke test.
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)

    # The passive viewer draws whatever is in `data` each time we call sync();
    # we own the loop. Physics is never stepped -- only mj_forward, inside
    # Leg.step -- so nothing falls and the pelvis stays where the seed put it.
    with soc4180.launch_viewer(model, data, passive=True, key_callback=on_key) as viewer:
        while viewer.is_running() and time.time() < deadline:
            solver = SOLVERS[state["solver"]][1]
            err, dq_norm = None, None
            for _ in range(ITERATION_CHOICES[state["iters"]]):      # a few solver steps per frame
                err, dq_norm = leg.step(solver, state["target"], state["lam"], state["orientation"])
            data.ctrl[:] = data.qpos[slider_qpos]          # the viewer's sliders show the solution
            if state["print"] or state["moved"]:
                state["print"] = state["moved"] = False
                report(err, dq_norm)
            with viewer.lock():                            # the three markers, redrawn every frame
                scn = viewer.user_scn
                sphere(scn.geoms[0], hip0, (0.3, 0.5, 1.0, 0.12), reach)          # reach
                sphere(scn.geoms[1], state["target"], (0.9, 0.1, 0.1, 0.9), 0.02)  # target
                sphere(scn.geoms[2], leg.foot, (0.1, 0.9, 0.2, 0.6), 0.03)         # foot
                scn.ngeom = 3
            viewer.sync()
            time.sleep(0.02)                               # about 50 frames per second
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
