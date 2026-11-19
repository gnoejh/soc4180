"""G1 actions: make the robot move its body, arms, hands and legs by name.

    from soc4180.actions import G1

    g1 = G1()                                   # the G1 with hands, crouched, standing still
    g1.squat(depth=0.10)
    g1.wave("right", count=3)
    g1.reach("left", (0.30, 0.25, 0.95))        # metres, in the robot's own frame (below)
    g1.hands(close=1.0)                         # make fists
    g1.turn(0.8)                                # radians, + = to the left
    print(g1.report())                          # one line per action: stood or fell, ...

Every action is a method of `G1`. It runs the physics until the motion is finished and
returns a `Result`: fell or not, the largest pelvis tilt, and the action's own numbers
(how far the palm ended from its target, how far the ball rolled, ...).

THE ROBOT'S FRAME. Every position you give is in the robot's own frame:

    origin  on the floor, half-way between the two ankles
    +x      forward (the way the feet point)
    +y      the robot's left
    +z      up

It moves with the robot: after a step or a turn it is between the new feet.

THREE WAYS TO MOVE A ROBOT; every action uses one of them:

    joint space    give the angles (arms, waist, fingers)          wave, raise_arms, twist, hands
    task space     give a point; IK finds the angles (weeks 2-3)   reach, point_at, clap
    balance-aware  keep the centre of mass over the feet (week 4)  squat, lean, one_leg, step, kick

THE BALANCE LAYER is on by default. Whatever the arms and waist do, the pelvis slides so
the whole body's centre of mass (CoM) stays over `shift` -- the middle of the soles, or one
sole. Three measured ingredients keep it standing (`G1(balance=False)` turns them off):

    gravity feedforward   each servo target leads by the sag it will have under load
                          (torque / kp, from inverse dynamics). Without it the loaded leg
                          sags, the body rolls onto it, and every one-leg stance falls.
    CoM feedback          COM_GAIN, TILT_GAIN: correct the command by the measured error.
    ankle reflex          ANKLE_GAIN, ANKLE_DAMP: week 7's ankle strategy on planted feet.
                          Without it (or COM_GAIN 1.0) the second of two steps falls.

Slow on purpose: moving the weight onto one foot takes SHIFT_SECONDS (1.5 s; 1.0 s
still stands, 0.8 s falls), and steps are quasi-static (about 4.6 s each). For dynamic walking see
week 4's LIPM walker, `soc4180.WalkingController`.

SNAPSHOTS AND DANCES. A snapshot is a named whole-body pose (`SNAPSHOTS`, below);
join names with '+' to combine them. A dance is a list of (snapshot, seconds):

    DANCE = [("arms_up+sway_left", 1.0), ("arms_up+sway_right", 1.0), ("home", 1.0)]
    print(g1.check(DANCE))                     # problems found before running it
    print(g1.play(DANCE, tempo=1.0))           # tempo 2 = twice as fast

MAKING YOUR OWN ACTION. Everything above is built from one method:

    g1.move(seconds, **targets)      # glide from the current pose to `targets`

targets: any joint (`left_elbow=1.2`, radians), a chain (`left_arm=[7 angles]`,
`waist=[yaw, roll, pitch]`), a hand (`left_hand=0.8`, 0 open .. 1 fist), the pelvis
(`height=0.65`, `pitch=0.1`, `roll=`, `yaw=`), the weight (`shift=(x, y)` or
`weight="left"`), or a foot (`left_lift=0.05` above its planted spot, `left_offset=(dx, dy)`
from it, or `left_foot=(x, y, z, yaw)`). `g1.hold(seconds)` waits.

Measured on the pinned G1 with hands (mass 34.4 kg), the soles 5-6 cm wide: every action
and all 36 snapshots stand on their own, and 50 random chains of 8 actions stood; stepping
outward is limited by the ankle roll's +-15 degrees, so side steps are 4 cm a pair.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ._gl import GL_BACKEND  # noqa: F401  (ensures MUJOCO_GL is set first)

import mujoco

from . import kinematics as kin
from .bodies import CHAINS

__all__ = ["G1", "Result", "HOME", "SNAPSHOTS", "centre"]


# The home stance: the week-4 walker's crouch (knees bent, so leg IK is well posed).
HOME = dict(height=0.72, width=0.119, foot_z=0.033)

# The middle of a sole, in front of the ankle: the foot runs from 0.05 m behind the
# ankle to 0.12 m in front. A CoM over the ankle itself is near the heel and falls back.
FOOT_CENTRE_X = 0.03

# A point on the palm, in the wrist_yaw_link frame (between the wrist and the finger roots).
PALM = np.array([0.10, 0.0, 0.0])

# Arm joints IK may move for reach (the wrist keeps its angles).
REACH_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")

# Arms hanging naturally, slightly forward (the `stand` keyframe's arms).
ARMS_DOWN = [0.2, 0.2, 0.0, 1.28, 0.0, 0.0, 0.0]      # left; the right one is mirrored

# The G1's elbow at 0 is a RIGHT ANGLE (forearm 82 degrees from the upper arm); the arm is
# straightest at about 1.45 (15 degrees: the upper arm is offset). Measured, not assumed.
STRAIGHT = 1.45


def centre(foot) -> np.ndarray:
    """The middle of a sole (x, y) for a foot given as [x, y, z, yaw]: where weight should go."""
    return np.asarray(foot[:2], float) + FOOT_CENTRE_X * np.array([math.cos(foot[3]), math.sin(foot[3])])


def _smooth(s: float) -> float:
    """0 -> 1 with zero speed at both ends (a smoothstep), so motions start and stop gently."""
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def _rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class Result:
    """What one action did."""

    action: str
    seconds: float
    fell: bool
    max_tilt_deg: float
    info: dict = field(default_factory=dict)

    def __str__(self) -> str:
        extra = "".join(f", {k} {v}" for k, v in self.info.items())
        state = "FELL" if self.fell else "stood"
        return f"{self.action}: {state}, max tilt {self.max_tilt_deg:.1f} deg, {self.seconds:.1f} s{extra}"


@dataclass
class _Pose:
    """The commanded pose, in the robot's frame. `move` glides between two of these."""

    q: np.ndarray                      # target angle for every hinge, indexed like qpos (legs ignored)
    height: float                      # pelvis height
    rpy: np.ndarray                    # pelvis roll, pitch, yaw
    shift: np.ndarray                  # CoM target (balance on) or pelvis x, y (balance off)
    feet: dict                         # side -> [x, y, z, yaw]

    def copy(self) -> "_Pose":
        return _Pose(self.q.copy(), self.height, self.rpy.copy(), self.shift.copy(),
                     {s: f.copy() for s, f in self.feet.items()})

    @staticmethod
    def blend(a: "_Pose", b: "_Pose", s: float) -> "_Pose":
        return _Pose(a.q + s * (b.q - a.q), a.height + s * (b.height - a.height),
                     a.rpy + s * (b.rpy - a.rpy), a.shift + s * (b.shift - a.shift),
                     {k: a.feet[k] + s * (b.feet[k] - a.feet[k]) for k in a.feet})


class G1:
    """The Unitree G1 with hands, driven by named actions. See the module docstring.

    hands     use the model with seven finger joints per hand (default) or without
    balance   keep the centre of mass over `shift` (default True)
    on_tick   called after every control step with this object: the lab uses it
              to sync the viewer and keep real time. None runs as fast as it can.
    ball      add a ball (radius 0.06 m, 0.4 kg) for `kick`; it waits off to the side
    """

    CONTROL_HZ = 100         # commands per second (physics runs at 500)
    # The balance layer's settings. Change them on the class or on one robot (g1.COM_GAIN = 0).
    GRAVITY_FF = True        # servo targets lead by their sag under load (torque / kp)
    COM_GAIN = 0.5           # move the CoM target against the measured CoM error
    TILT_GAIN = 1.0          # command extra pelvis tilt against the measured tilt error
    ANKLE_GAIN = 2.0         # planted ankles: rad per metre of CoM error (week 7's ankle strategy)
    ANKLE_DAMP = 0.3         # planted ankles: rad per m/s of CoM velocity
    ANKLE_SIGN = (1.0, -1.0) # (pitch, roll), measured: the other three sign pairs all fall
    # Timing and step sizes.
    SHIFT_SECONDS = 1.5      # weight onto one foot (measured: 1.0 stands, 0.8 falls)
    SIDE_PER_STEP = 0.04     # side_step: metres per pair of steps
    TURN_PER_STEP = 0.40     # turn: radians per pair of steps

    def __init__(self, *, hands: bool = True, balance: bool = True, on_tick=None,
                 ball: bool = True):
        from .models import load_g1

        self.hands_model = hands
        self.balance = balance
        self.on_tick = on_tick
        self.model = self._build(load_g1(with_hands=hands), ball)
        self.data = mujoco.MjData(self.model)
        self._scratch = mujoco.MjData(self.model)
        m = self.model

        self._act_qpos = np.array([m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        self._legs = {s: kin.leg_qpos_indices(m, s) for s in ("left", "right")}
        self._leg_dof = {s: kin.leg_dof_indices(m, s) for s in ("left", "right")}
        self._leg_lo = {s: m.jnt_range[[m.joint(f"{s}_{j}").id for j in kin.LEG_JOINTS], 0]
                        for s in ("left", "right")}
        self._leg_hi = {s: m.jnt_range[[m.joint(f"{s}_{j}").id for j in kin.LEG_JOINTS], 1]
                        for s in ("left", "right")}
        self._jp, self._jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self.track = False       # play() sets it: also compute the commanded joint angles
        name2act = {m.joint(m.actuator_trnid[a, 0]).name: a for a in range(m.nu)}
        self._ankle_act = {s: (name2act[f"{s}_ankle_pitch_joint"], name2act[f"{s}_ankle_roll_joint"])
                           for s in ("left", "right")}
        self._act_dof = np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        self._kp = m.actuator_gainprm[:, 0].copy()
        self._pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self._feet_site = {s: kin.foot_site_id(m, s) for s in ("left", "right")}
        self._ball = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self._ball_q = m.jnt_qposadr[m.body_jntadr[self._ball]] if self._ball >= 0 else None
        self.decimation = int(round(1.0 / (self.CONTROL_HZ * m.opt.timestep)))
        self.reset()

    # ------------------------------------------------------------------ set-up
    @staticmethod
    def _build(model, ball):
        if not ball:
            return model
        from .models import robot_path      # re-read the scene as a spec so a ball can be added

        spec = mujoco.MjSpec.from_file(str(robot_path(
            "unitree_g1", "scene_with_hands" if model.nu > 29 else "scene")))
        body = spec.worldbody.add_body(name="ball", pos=[3.0, 3.0, 0.06])
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.06, 0, 0], mass=0.4,
                      rgba=[1.0, 0.45, 0.1, 1.0], condim=6, priority=1,
                      friction=[0.7, 0.02, 0.002])
        for key in spec.keys:  # the keyframe must cover the ball's 7 numbers too
            key.qpos = list(key.qpos) + [3.0, 3.0, 0.06, 1.0, 0.0, 0.0, 0.0]
        return spec.compile()

    def reset(self) -> None:
        """Start again: crouched in the home stance, arms down, hands open, standing still."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:] = m.key_qpos[0]
        w = HOME["width"]
        self.pose = _Pose(m.key_qpos[0].copy(), HOME["height"], np.zeros(3), np.array([FOOT_CENTRE_X, 0.0]),
                          {"left": np.array([0.0, +w, HOME["foot_z"], 0.0]),
                           "right": np.array([0.0, -w, HOME["foot_z"], 0.0])})
        self._seed = m.key_qpos[0].copy()
        for side in ("left", "right"):  # bent knees: the straight `stand` legs are an IK singularity
            i = self._legs[side]
            self._seed[i[0]], self._seed[i[3]], self._seed[i[4]] = -0.35, 0.70, -0.35
        self._pxy = np.zeros(2)
        self.planted = {k: v.copy() for k, v in self.pose.feet.items()}
        self._plan_pose = None
        self._ff_level = self._ff_goal = 1.0
        ctrl, qfull = self._solve(self.pose)
        d.qpos[:] = qfull
        d.qpos[2] += 0.005
        if self._ball_q is not None:
            d.qpos[self._ball_q:self._ball_q + 7] = [3.0, 3.0, 0.06, 1, 0, 0, 0]
        d.qvel[:] = 0
        d.ctrl[:] = ctrl
        mujoco.mj_forward(m, d)
        self.fallen = False
        self.history: list[Result] = []
        self._run(0.5, lambda s: self.pose)          # settle onto the floor

    # -------------------------------------------------------------- the solver
    def _solve(self, pose: _Pose):
        """Pose -> (ctrl, full qpos of the commanded robot in the robot frame)."""
        m, sc = self.model, self._scratch
        quat = np.empty(4)
        mujoco.mju_euler2Quat(quat, pose.rpy[::-1].copy(), "zyx")   # yaw, then pitch, then roll
        feet = {s: (pose.feet[s][:3], _rz(pose.feet[s][3])) for s in ("left", "right")}
        pxy = self._pxy.copy() if self.balance else pose.shift.copy()
        for it in range(4 if self.balance else 1):
            sc.qpos[:] = pose.q
            sol = self._ik_legs(sc, [pxy[0], pxy[1], pose.height], quat, feet)
            if not self.balance or it == 3:
                break
            mujoco.mj_comPos(m, sc)                     # the CoM of the solved pose
            pxy = pxy + (pose.shift - sc.subtree_com[self._pelvis][:2])
        q = pose.q.copy()
        q[:3] = [pxy[0], pxy[1], pose.height]
        q[3:7] = quat
        for s in ("left", "right"):
            q[self._legs[s]] = sol[s]
        self._seed = q.copy()
        self._pxy = pxy
        self.ik_error = sol["error"]
        ctrl = q[self._act_qpos]
        if self.GRAVITY_FF:
            ctrl = ctrl + self._sag_ahead(q, pose)
        return ctrl, q

    def _ik_legs(self, sc, base_pos, base_quat, feet, iterations=12, tolerance=1e-4):
        """Both legs' angles that put each foot site on its target, the pelvis placed at
        base_pos/base_quat: damped least squares, as `kinematics.ik_legs` (week 4), with
        every index looked up once instead of on every iteration."""
        m = self.model
        sc.qpos[:3], sc.qpos[3:7] = base_pos, base_quat
        for s in ("left", "right"):
            sc.qpos[self._legs[s]] = self._seed[self._legs[s]]
        worst = np.inf
        for _ in range(iterations):
            mujoco.mj_kinematics(m, sc)
            mujoco.mj_comPos(m, sc)          # mj_jacSite reads subtree_com: stale -> wrong Jacobian
            worst = 0.0
            for s in ("left", "right"):
                site = self._feet_site[s]
                err = kin.pose_error(sc, site, feet[s][0], feet[s][1])
                worst = max(worst, float(np.linalg.norm(err[:3])))
                mujoco.mj_jacSite(m, sc, self._jp, self._jr, site)
                J = np.vstack([self._jp, self._jr])[:, self._leg_dof[s]]
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
                i = self._legs[s]
                sc.qpos[i] = np.clip(sc.qpos[i] + dq, self._leg_lo[s], self._leg_hi[s])
            if worst < tolerance:
                break
        mujoco.mj_kinematics(m, sc)
        out = {s: sc.qpos[self._legs[s]].copy() for s in ("left", "right")}
        out["error"] = worst
        return out

    def _sag_ahead(self, q, pose):
        """How far each servo will sag holding pose `q` still, so the target can lead by that much.

        Inverse dynamics with zero velocity: gravity's generalised force g(q); the planted
        feet carry the body, split by where the CoM target sits between them; the joints
        supply the rest, tau = g - J^T w. A position servo gives tau = kp (ctrl - q), so
        ctrl = q + tau / kp holds the pose exactly."""
        m, sc = self.model, self._scratch
        sc.qpos[:] = q
        sc.qvel[:] = 0
        mujoco.mj_kinematics(m, sc)
        mujoco.mj_comPos(m, sc)
        g = np.zeros(m.nv)
        mujoco.mj_rne(m, sc, 0, g)
        low = min(pose.feet[s][2] for s in ("left", "right"))
        planted = [s for s in ("left", "right") if pose.feet[s][2] <= low + 0.005]
        if len(planted) == 2:
            a, b = centre(pose.feet["left"]), centre(pose.feet["right"])
            t = float(np.clip(np.dot(pose.shift - b, a - b) / max(np.dot(a - b, a - b), 1e-9), 0, 1))
            share = {"left": t, "right": 1 - t}
        else:
            share = {planted[0]: 1.0}
        Js, W = [], []
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for s in planted:
            mujoco.mj_jacSite(m, sc, jp, jr, self._feet_site[s])
            Js.append(np.vstack([jp, jr]))
            W.append(np.full(6, max(share[s], 1e-3)))
        J = np.vstack(Js)                          # (6 per foot) x nv
        Wd = np.diag(np.concatenate(W))
        Jb = J[:, :6]                              # the floating base's columns
        w = Wd @ Jb @ np.linalg.solve(Jb.T @ Wd @ Jb, g[:6])
        tau = g - J.T @ w
        return self._ff_level * tau[self._act_dof] / self._kp

    def com(self, pose: _Pose | None = None) -> np.ndarray:
        """Centre of mass of the commanded pose, in the robot frame."""
        _, q = self._solve(pose or self.pose)
        self._scratch.qpos[:] = q
        mujoco.mj_kinematics(self.model, self._scratch)
        mujoco.mj_comPos(self.model, self._scratch)
        return self._scratch.subtree_com[self._pelvis].copy()

    # ------------------------------------------------------------- feedback
    def _frame_now(self, pose):
        """Where the robot frame is in the world now (x, y, yaw), read from the planted feet."""
        d = self.data
        low = min(pose.feet[s][2] for s in ("left", "right"))
        offs, yaws = [], []
        for s in ("left", "right"):
            if pose.feet[s][2] > low + 0.005:
                continue                                  # a lifted foot says nothing
            R = d.site_xmat[self._feet_site[s]].reshape(3, 3)
            yaw = math.atan2(R[1, 0], R[0, 0]) - pose.feet[s][3]
            offs.append(d.site_xpos[self._feet_site[s]][:2] - _rz(yaw)[:2, :2] @ pose.feet[s][:2])
            yaws.append(yaw)
        return np.mean(offs, axis=0), math.atan2(np.mean(np.sin(yaws)), np.mean(np.cos(yaws)))

    def measured(self, pose=None):
        """(pelvis roll, pelvis pitch, CoM x, CoM y) as the robot really is, in the robot frame."""
        pose = pose or self.pose
        R = self.data.xmat[self._pelvis].reshape(3, 3)
        roll, pitch = math.atan2(R[2, 1], R[2, 2]), math.asin(max(-1.0, min(1.0, -R[2, 0])))
        o, yaw = self._frame_now(pose)
        com = _rz(-yaw)[:2, :2] @ (self.data.subtree_com[self._pelvis][:2] - o)
        return roll, pitch, com

    def _ankles(self, pose):
        """Ankle pitch/roll corrections on the planted feet against the CoM's error and speed."""
        m, d = self.model, self.data
        roll, pitch, com = self.measured(pose)
        o, yaw = self._frame_now(pose)
        mujoco.mj_subtreeVel(m, d)
        v = _rz(-yaw)[:2, :2] @ d.subtree_linvel[self._pelvis][:2]
        out = np.zeros(m.nu)
        low = min(pose.feet[s][2] for s in ("left", "right"))
        for s in ("left", "right"):
            if pose.feet[s][2] > low + 0.005:
                continue
            R = _rz(-pose.feet[s][3])[:2, :2]                  # into the foot's own axes
            e, ve = R @ (com - pose.shift), R @ v
            corr = self.ANKLE_GAIN * e + self.ANKLE_DAMP * ve
            out[self._ankle_act[s][0]] = self.ANKLE_SIGN[0] * corr[0]
            out[self._ankle_act[s][1]] = self.ANKLE_SIGN[1] * corr[1]
        return out

    def _feedback(self, pose):
        """The commanded pose, corrected by what the sensors say (TILT_GAIN, COM_GAIN)."""
        if not self.balance or (self.TILT_GAIN == 0 and self.COM_GAIN == 0):
            return pose
        roll, pitch, com = self.measured(pose)
        out = pose.copy()
        out.rpy[0] += self.TILT_GAIN * (pose.rpy[0] - roll)
        out.rpy[1] += self.TILT_GAIN * (pose.rpy[1] - pitch)
        out.shift = pose.shift + self.COM_GAIN * (pose.shift - com)
        return out

    # ------------------------------------------------------------- the engine
    def tilt_deg(self) -> float:
        """How far the pelvis leans from upright, in degrees (a bow tilts the torso, not this)."""
        z = self.data.xmat[self._pelvis].reshape(3, 3)[2, 2]
        return math.degrees(math.acos(max(-1.0, min(1.0, z))))

    def _run(self, seconds, pose_at):
        """Step the physics for `seconds`; pose_at(s in 0..1) gives the commanded pose."""
        m, d = self.model, self.data
        n = max(int(round(seconds * self.CONTROL_HZ)), 1)
        worst = 0.0
        for k in range(n):
            pose = pose_at((k + 1) / n)
            if self.track:                                  # the pose itself, as joint angles
                self.commanded = self._solve(pose)[1][self._act_qpos]
            ctrl = self._solve(self._feedback(pose))[0]
            if self.balance and (self.ANKLE_GAIN or self.ANKLE_DAMP):
                ctrl += self._ankles(pose)
            d.ctrl[:] = ctrl
            for _ in range(self.decimation):
                mujoco.mj_step(m, d)
            step = 2.0 / self.CONTROL_HZ                   # the feedforward fades in 0.5 s
            self._ff_level += float(np.clip(self._ff_goal - self._ff_level, -step, step))
            tilt = self.tilt_deg()
            worst = max(worst, tilt)
            if tilt > 45.0 or d.xpos[self._pelvis][2] < 0.35:
                self.fallen = True
            if self.on_tick is not None:
                self.on_tick(self)
        return worst

    def _act(self, name, steps, info=None):
        """Run a list of (seconds, target pose) segments and report them as one action."""
        t0, worst = self.data.time, 0.0
        if self.fallen:
            r = Result(name, 0.0, True, self.tilt_deg(), {"note": "already fallen: reset first"})
            self.history.append(r)
            return r
        for seconds, target in steps:
            start = self.pose.copy()
            worst = max(worst, self._run(seconds, lambda s: _Pose.blend(start, target, _smooth(s))))
            self.pose = target.copy()
            if self.fallen:
                break
        out = info() if callable(info) else (info or {})
        r = Result(name, self.data.time - t0, self.fallen, worst, out)
        self.history.append(r)
        return r

    # ---------------------------------------------------------- the one verb
    def target(self, base: _Pose | None = None, **targets) -> _Pose:
        """A new pose: `base` (default: the current one) with `targets` changed. See `move`."""
        p = (base or self.pose).copy()
        m = self.model
        for key, value in targets.items():
            if value is None:
                continue
            if key in ("height",):
                p.height = float(value)
            elif key in ("roll", "pitch", "yaw"):
                p.rpy[("roll", "pitch", "yaw").index(key)] = float(value)
            elif key == "shift":
                p.shift = np.asarray(value, float)[:2].copy()
            elif key in ("left_foot", "right_foot"):
                p.feet[key[:-5]] = np.asarray(value, float).copy()
            elif key in ("left_lift", "right_lift"):            # metres above its planted spot
                side = key[:-5]
                p.feet[side][2] = self.planted[side][2] + float(value)
            elif key in ("left_offset", "right_offset"):        # (dx, dy) from its planted spot
                side = key[:-7]
                p.feet[side][:2] = self.planted[side][:2] + np.asarray(value, float)[:2]
            elif key == "weight":                               # "left", "right" or "middle"
                p.shift = self._weight_point(str(value))
            elif key in ("left_hand", "right_hand"):
                for j, a in self._finger_angles(key[:-5], float(value)).items():
                    p.q[m.jnt_qposadr[m.joint(j).id]] = a
            elif key in CHAINS:
                for j, a in zip(CHAINS[key], np.asarray(value, float).ravel()):
                    p.q[m.jnt_qposadr[m.joint(j).id]] = a
            else:
                name = key if key.endswith("_joint") else key + "_joint"
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid < 0:
                    raise ValueError(f"move() does not know {key!r}: use a joint name such as "
                                     f"'left_elbow', a chain {sorted(CHAINS)}, or one of height, "
                                     f"roll, pitch, yaw, shift, left_foot, right_foot, left_hand, "
                                     f"right_hand")
                lo, hi = m.jnt_range[jid]
                p.q[m.jnt_qposadr[jid]] = float(np.clip(value, lo, hi))
        return p

    def move(self, seconds: float = 1.0, **targets) -> Result:
        """Glide to a new pose in `seconds`. The building block of every action.

        Keywords: any joint (`left_elbow=1.2`, radians), a chain (`left_arm=[...]`,
        `waist=[yaw, roll, pitch]`), a hand (`left_hand=0..1`), `height`, `roll`,
        `pitch`, `yaw` (the pelvis), `shift=(x, y)` (where the CoM goes), or a foot
        (`left_foot=(x, y, z, yaw)`). Anything not named keeps its current value.
        """
        return self._act("move", [(seconds, self.target(**targets))])

    def hold(self, seconds: float = 1.0) -> Result:
        """Keep the current pose for `seconds`."""
        return self._act("hold", [(seconds, self.pose.copy())])

    def idle(self, seconds: float = 0.05) -> None:
        """Keep the current pose for `seconds` without recording an action (a viewer loop)."""
        if not self.fallen:
            self._run(seconds, lambda s: self.pose)

    # --------------------------------------------------------- body (legs)
    def home(self, seconds: float = 1.5) -> Result:
        """Back to the home stance -- crouched, weight in the middle, arms down, hands open,
        waist straight -- with the feet where they are planted (feet never slide)."""
        return self._act("home", [(seconds, self._home_pose())])

    def squat(self, depth: float = 0.12, seconds: float = 1.5, hold: float = 0.5) -> Result:
        """Lower the pelvis by `depth` metres, pause, and come back up."""
        h = self.pose.height
        down, up = self.target(height=h - depth), self.target(height=h)
        return self._act("squat", [(seconds, down), (hold, down), (seconds, up)],
                         info={"lowest pelvis": f"{h - depth:.2f} m"})

    def stand_tall(self, height: float = 0.76, seconds: float = 1.5) -> Result:
        """Raise the pelvis (straighter knees). Near 0.79 m the legs are straight: IK's singularity."""
        return self._act("stand_tall", [(seconds, self.target(height=height))],
                         info=lambda: {"IK error": f"{self.ik_error * 1000:.1f} mm"})

    def lean(self, pitch: float = 0.25, roll: float = 0.0, seconds: float = 1.5,
             hold: float = 1.0) -> Result:
        """Tilt the pelvis (pitch + forward, roll + toward the right), hold, come back.

        With balance on, the hips slide back as the chest goes forward, so the CoM stays put."""
        p0 = self.pose.rpy.copy()
        t = self.target(pitch=p0[1] + pitch, roll=p0[0] + roll)
        back = self.target(pitch=p0[1], roll=p0[0])
        return self._act("lean", [(seconds, t), (hold, t), (seconds, back)])

    def sway(self, amount: float = 0.04, count: int = 2, seconds: float = 0.8) -> Result:
        """Move the weight left and right `count` times (`amount` metres each way)."""
        x, y = self.pose.shift
        steps = []
        for _ in range(count):
            steps += [(seconds, self.target(shift=(x, y + amount))),
                      (seconds, self.target(shift=(x, y - amount)))]
        steps.append((seconds / 2, self.target(shift=(x, y))))
        return self._act("sway", steps)

    # ------------------------------------------------------- body (waist)
    def bow(self, angle: float = 0.45, seconds: float = 1.2, hold: float = 0.8) -> Result:
        """Bend forward at the waist (waist pitch, at most 0.52 rad) and come back."""
        a0 = self._get("waist_pitch")
        t = self.target(waist_pitch=angle)
        return self._act("bow", [(seconds, t), (hold, t), (seconds, self.target(waist_pitch=a0))])

    def twist(self, angle: float = 0.8, seconds: float = 1.2, hold: float = 0.5) -> Result:
        """Turn the upper body (waist yaw, + = to the left), then back. The G1 has no neck:
        this is how it looks around."""
        a0 = self._get("waist_yaw")
        t = self.target(waist_yaw=angle)
        return self._act("twist", [(seconds, t), (hold, t), (seconds, self.target(waist_yaw=a0))])

    # ---------------------------------------------------- arms and hands
    def raise_arms(self, which: str = "both", angle: float = 2.9, seconds: float = 1.2,
                   hold: float = 0.8) -> Result:
        """Lift straight arm(s) forward and up (shoulder pitch -angle), hold, lower.
        1.5 points the arms forward, 2.9 straight up."""
        sides = ("left", "right") if which == "both" else (which,)
        up = {}
        for s in sides:
            up[f"{s}_shoulder_pitch"] = -angle
            up[f"{s}_elbow"] = STRAIGHT
            up[f"{s}_shoulder_roll"] = 0.15 if s == "left" else -0.15
        t = self.target(**up)
        down = self.target(**{f"{s}_arm": ARMS_DOWN if s == "left" else self._mirror(ARMS_DOWN)
                              for s in sides})
        return self._act("raise_arms", [(seconds, t), (hold, t), (seconds, down)])

    def wave(self, arm: str = "right", count: int = 3, speed: float = 1.0,
             amplitude: float = 0.6) -> Result:
        """Upper arm out to the side, forearm up, and wave `count` times (`speed` waves per
        second, `amplitude` radians of elbow swing). The forearm points up because the
        shoulder yaw turns the upper arm 90 degrees and the G1's elbow 0 is a right angle."""
        up = self.target(**_arm(arm, -0.3, 1.3, 1.57, 0.0), **{f"{arm}_hand": 0.0})
        half = 0.5 / speed
        a = self.target(up, **{f"{arm}_elbow": -amplitude / 2})
        b = self.target(up, **{f"{arm}_elbow": +amplitude / 2})
        steps = [(1.0, up)]
        for _ in range(count):
            steps += [(half, a), (half, b)]
        down = self.target(**{f"{arm}_arm": ARMS_DOWN if arm == "left" else self._mirror(ARMS_DOWN)})
        steps += [(half, up), (1.0, down)]
        return self._act("wave", steps)

    def hands(self, close: float = 1.0, which: str = "both", seconds: float = 0.6,
              hold: float = 0.6, release: bool = True) -> Result:
        """Close the fingers (`close` 0 open .. 1 fist), hold, and open again if `release`."""
        sides = ("left", "right") if which == "both" else (which,)
        if not self.hands_model:
            raise RuntimeError("this G1 has no fingers: make it with G1(hands=True)")
        t = self.target(**{f"{s}_hand": close for s in sides})
        steps = [(seconds, t), (hold, t)]
        if release:
            steps.append((seconds, self.target(**{f"{s}_hand": 0.0 for s in sides})))
        return self._act("hands", steps)

    def reach(self, hand: str = "right", point=(0.35, -0.20, 0.95), seconds: float = 1.5,
              hold: float = 1.0, back: bool = True) -> Result:
        """Put the palm on `point` (x, y, z in the robot frame) by arm IK, hold, and return.

        Reports how far the palm ended from the point (in the commanded pose)."""
        start = self.pose.copy()
        t, err = self._reach_pose(hand, np.asarray(point, float))
        steps = [(seconds, t), (hold, t)]
        if back:
            steps.append((seconds, start))
        return self._act("reach", steps, info={"palm error": f"{err * 1000:.0f} mm"})

    def point_at(self, hand: str = "right", point=(1.5, -0.5, 0.8), seconds: float = 1.2,
                 hold: float = 1.0) -> Result:
        """Point the straight arm at a far `point` (robot frame), hold, and lower it."""
        side = 1 if hand == "left" else -1
        shoulder = np.array([0.0, side * 0.17, 1.10])        # roughly, in the robot frame
        d = np.asarray(point, float) - shoulder
        d /= np.linalg.norm(d)
        t, err = self._reach_pose(hand, shoulder + 0.42 * d)
        down = self.target(**{f"{hand}_arm": ARMS_DOWN if hand == "left" else self._mirror(ARMS_DOWN)})
        return self._act("point_at", [(seconds, t), (hold, t), (seconds, down)],
                         info={"aim error": f"{err * 1000:.0f} mm"})

    def clap(self, count: int = 3, speed: float = 1.5, x: float = 0.30, z: float = 0.95) -> Result:
        """Bring both palms together in front of the chest `count` times."""
        start = self.pose.copy()
        o, _ = self._reach_pose("left", np.array([x, 0.16, z]))
        o, _ = self._reach_pose("right", np.array([x, -0.16, z]), base=o)
        c, _ = self._reach_pose("left", np.array([x, 0.045, z]), base=o)
        c, err = self._reach_pose("right", np.array([x, -0.045, z]), base=c)
        half = 0.5 / speed
        steps = [(1.0, o)]
        for _ in range(count):
            steps += [(half, c), (half, o)]
        steps.append((1.0, start))
        return self._act("clap", steps)

    # --------------------------------------------------------------- legs
    def one_leg(self, stand_on: str = "left", lift: float = 0.08, seconds: float | None = None,
                hold: float = 2.0) -> Result:
        """Move the weight over one foot, lift the other `lift` metres, hold, put it down."""
        other = "right" if stand_on == "left" else "left"
        seconds = seconds or self.SHIFT_SECONDS
        f = self.pose.feet
        over = self.target(shift=centre(f[stand_on]), height=min(self.pose.height, HOME["height"]))
        up_foot = f[other].copy()
        up_foot[2] += lift
        up = self.target(over, **{f"{other}_foot": up_foot})
        mid = (centre(f["left"]) + centre(f["right"])) / 2
        steps = [(seconds, over), (seconds * 0.6, up), (hold, up), (seconds * 0.6, over),
                 (seconds, self.target(over, shift=mid))]
        return self._act("one_leg", steps)

    def step(self, foot: str = "left", dx: float = 0.10, dy: float = 0.0, dyaw: float = 0.0,
             lift: float = 0.06, seconds: float = 1.0) -> Result:
        """One slow (quasi-static) step: weight over the other foot, move `foot` by (dx, dy)
        metres and turn it dyaw radians, put it down, weight back to the middle."""
        r = self._act("step", self._step_segments(foot, dx, dy, dyaw, lift, seconds))
        self._reanchor()
        return r

    def walk(self, steps: int = 3, step_length: float = 0.10) -> Result:
        """Walk slowly: `steps` pairs of quasi-static steps, left foot ahead then the right
        foot alongside (negative step_length walks backwards). About 9 s per pair. For
        dynamic walking see week 4's LIPM walker (soc4180.WalkingController)."""
        x0 = self._feet_mid()
        return self._chain("walk", [[("left", step_length, 0, 0), ("right", step_length, 0, 0)]] * steps,
                           lambda: {"moved": self._moved(x0)})

    def side_step(self, distance: float = 0.08) -> Result:
        """Walk sideways `distance` metres (+ = to the left), SIDE_PER_STEP per pair of
        steps: the leading foot steps out, the other follows."""
        n = max(1, math.ceil(abs(distance) / self.SIDE_PER_STEP))
        dy = distance / n
        lead, follow = ("left", "right") if dy > 0 else ("right", "left")
        x0 = self._feet_mid()
        return self._chain("side_step", [[(lead, 0, dy, 0), (follow, 0, dy, 0)]] * n,
                           lambda: {"moved": self._moved(x0)})

    def turn(self, angle: float = 0.8) -> Result:
        """Turn on the spot by `angle` radians (+ = to the left), TURN_PER_STEP per pair:
        the leading foot turns about the middle of the feet, the other comes alongside."""
        n = max(1, math.ceil(abs(angle) / self.TURN_PER_STEP))
        da = angle / n
        lead, follow = ("left", "right") if da > 0 else ("right", "left")
        yaw0 = self._world_yaw()

        def info():
            turned = math.degrees(math.remainder(self._world_yaw() - yaw0, 2 * math.pi))
            return {"turned": f"{turned:.0f} deg"}

        return self._chain("turn", [[("turn", lead, da), ("follow", follow)]] * n, info)

    def _chain(self, name, pairs, info):
        """Run pairs of quasi-static steps as one action. A step is (foot, dx, dy, dyaw) in the
        current robot frame, ("turn", foot, dyaw) or ("follow", foot)."""
        t0, worst = self.data.time, 0.0
        w = HOME["width"]
        for pair in pairs:
            for st in pair:
                f = self.pose.feet
                if st[0] == "turn":
                    _, foot, da = st
                    mid = (f["left"][:2] + f["right"][:2]) / 2
                    new = mid + _rz(da)[:2, :2] @ (f[foot][:2] - mid)
                    st = (foot, *(new - f[foot][:2]), da)
                elif st[0] == "follow":
                    foot = st[1]
                    other = "right" if foot == "left" else "left"
                    side = +1 if foot == "left" else -1
                    new = f[other][:2] + _rz(f[other][3])[:2, :2] @ np.array([0.0, side * 2 * w])
                    st = (foot, *(new - f[foot][:2]), f[other][3] - f[foot][3])
                r = self.step(*st)
                self.history.pop()
                worst = max(worst, r.max_tilt_deg)
                if r.fell:
                    break
            if self.fallen:
                break
        r = Result(name, self.data.time - t0, self.fallen, worst, info())
        self.history.append(r)
        return r

    def set_balance(self, on: bool) -> None:
        """Switch the balance layer on or off without moving: `shift` is re-read as a CoM
        target (on) or as the pelvis position (off). The feedforward fades in or out."""
        if on and not self.balance:
            self.balance = True
            self.pose.shift = self.com()[:2]
        elif not on and self.balance:
            self.pose.shift = self._pxy.copy()
            self.balance = False
        self._ff_goal = 1.0 if on else 0.0

    def kick(self, foot: str = "right", strength: float = 1.0, ball: bool = True) -> Result:
        """Stand on the other foot and kick forward. With `ball`, a ball is put in front of
        the foot first; the result says how far it rolled. strength 0.5 .. 1.5 scales the swing."""
        support = "left" if foot == "right" else "right"
        f = self.pose.feet
        base = f[foot].copy()
        if ball and self._ball_q is not None:
            p = self.to_world(np.array([base[0] + 0.20, base[1], 0.06]))
            self.data.qpos[self._ball_q:self._ball_q + 7] = [p[0], p[1], 0.06, 1, 0, 0, 0]
            self.data.qvel[self.model.jnt_dofadr[self.model.body_jntadr[self._ball]]:][:6] = 0
            mujoco.mj_forward(self.model, self.data)
        b0 = self.ball_position()
        over = self.target(shift=centre(f[support]), height=min(self.pose.height, HOME["height"]))
        lift = self.target(over, **{f"{foot}_foot": base + [0.0, 0, 0.05, 0]})
        back = self.target(over, **{f"{foot}_foot": base + [-0.08 * strength, 0, 0.08, 0]})
        swing = self.target(over, **{f"{foot}_foot": base + [0.16 * strength, 0, 0.06, 0]})
        mid = (centre(f["left"]) + centre(f["right"])) / 2
        steps = [(1.5, over), (0.5, lift), (0.5, back), (max(0.12, 0.25 / strength), swing),
                 (0.6, lift), (0.5, self.target(over)), (1.2, self.target(over, shift=mid))]
        r = self._act("kick", steps)
        if ball and b0 is not None:
            self._run(1.0, lambda s: self.pose)
            r.info["ball rolled"] = f"{np.linalg.norm(self.ball_position()[:2] - b0[:2]):.2f} m"
        return r

    # ------------------------------------------------------------ snapshots
    def snapshot(self, spec) -> _Pose:
        """A snapshot pose from a name in SNAPSHOTS, several names joined by '+'
        ("arms_up+squat"), a dict of `move` keywords, or a pose. Anything a snapshot leaves
        out is the home stance's, so a snapshot is the same pose wherever a dance uses it.
        Planted feet never move; a foot is only lifted (`left_lift`, `left_offset`)."""
        if isinstance(spec, _Pose):
            return spec.copy()
        if isinstance(spec, str):
            kw = {}
            for name in spec.split("+"):
                name = name.strip()
                if name not in SNAPSHOTS:
                    raise ValueError(f"no snapshot {name!r}; choose from {', '.join(SNAPSHOTS)}")
                kw.update(SNAPSHOTS[name])
        else:
            kw = dict(spec)
        return self.target(self._home_pose(), **kw)

    def check(self, sequence) -> list:
        """Problems a sequence will have, found before running it (an empty list: none found).

        Found: a snapshot standing on one foot with its weight not over that foot; a foot
        leaving the floor before the weight has moved off it; and a glide that moves the
        arms while a foot goes up or down (measured: the sample dance did that and fell at
        every tempo below 1.05, and stood from 0.5 to 1.35 once the two were separated)."""
        problems = []
        prev = self.pose.copy()
        arms = np.array([self.model.jnt_qposadr[self.model.joint(j).id]
                         for c in ("left_arm", "right_arm") for j in CHAINS[c]])
        for i, (spec, _) in enumerate(sequence):
            snap = self.snapshot(spec)
            down = [s for s in ("left", "right") if snap.feet[s][2] <= self.planted[s][2] + 1e-6]
            if not down:
                problems.append(f"step {i} {spec!r}: both feet are off the floor")
            elif len(down) == 1:
                stand, up = down[0], ("right" if down[0] == "left" else "left")
                d = np.linalg.norm(snap.shift - centre(snap.feet[stand]))
                if d > 0.02:
                    problems.append(f"step {i} {spec!r}: stands on the {stand} foot but the weight "
                                    f"is {d * 100:.0f} cm away from it (add weight='{stand}')")
                d0 = np.linalg.norm(prev.shift - centre(prev.feet[stand]))
                if d0 > 0.02 and prev.feet[up][2] <= self.planted[up][2] + 1e-6:
                    problems.append(f"step {i} {spec!r}: lifts the {up} foot before the weight is "
                                    f"on the {stand} foot (put 'on_{stand}' before it)")
            foot_moves = any(abs(snap.feet[s][2] - prev.feet[s][2]) > 1e-3 for s in snap.feet)
            if foot_moves and np.abs(snap.q[arms] - prev.q[arms]).max() > 0.3:
                problems.append(f"step {i} {spec!r}: moves the arms while a foot goes up or down "
                                f"(one thing at a time on one foot)")
            prev = snap
        return problems

    def play(self, sequence, tempo: float = 1.0, name: str = "dance") -> Result:
        """Glide through `sequence`, a list of (snapshot, seconds), `tempo` times faster.

        Each glide eases in and out (zero speed at each snapshot). The result also says how
        far, on average, the real joints were from the commanded ones ("tracking")."""
        segs = [(sec / tempo, self.snapshot(spec)) for spec, sec in sequence]
        err = []
        before = self.on_tick

        def tick(g):
            err.append(float(np.mean(np.abs(g.commanded - g.data.qpos[g._act_qpos]))))
            if before is not None:
                before(g)

        self.on_tick, self.track = tick, True
        try:
            r = self._act(name, segs)
        finally:
            self.on_tick, self.track = before, False
        r.info["tempo"] = f"x{tempo:g}"
        r.info["tracking"] = f"{math.degrees(np.mean(err)) if err else 0.0:.1f} deg"
        return r

    def _home_pose(self) -> _Pose:
        p = self.target(self.pose.copy(), height=HOME["height"], roll=0, pitch=0, yaw=0,
                        shift=self._weight_point("middle"), waist=[0, 0, 0], left_arm=ARMS_DOWN,
                        right_arm=self._mirror(ARMS_DOWN), left_hand=0, right_hand=0)
        p.feet = {k: v.copy() for k, v in self.planted.items()}
        return p

    def _weight_point(self, where: str) -> np.ndarray:
        f = self.planted
        if where in ("left", "right"):
            return centre(f[where])
        if where == "middle":
            return (centre(f["left"]) + centre(f["right"])) / 2
        raise ValueError("weight is 'left', 'right' or 'middle'")

    # -------------------------------------------------------------- helpers
    def ball_position(self):
        """The ball's world position, or None if there is no ball."""
        if self._ball_q is None:
            return None
        return self.data.qpos[self._ball_q:self._ball_q + 3].copy()

    def to_world(self, p) -> np.ndarray:
        """A point in the robot frame -> world coordinates (where it is on the floor now)."""
        mid, yaw = self._actual_feet_frame()
        out = np.array(p, float)
        out[:2] = mid + _rz(yaw)[:2, :2] @ out[:2]
        return out

    def palm(self, hand: str) -> np.ndarray:
        """Where a palm is now, in world coordinates."""
        b = self.model.body(f"{hand}_wrist_yaw_link").id
        return self.data.xpos[b] + self.data.xmat[b].reshape(3, 3) @ PALM

    def report(self) -> str:
        """Every action so far, one line each."""
        return "\n".join(str(r) for r in self.history)

    def _get(self, joint: str) -> float:
        m = self.model
        return float(self.pose.q[m.jnt_qposadr[m.joint(joint + "_joint").id]])

    @staticmethod
    def _mirror(arm):
        """Left arm angles -> right arm angles (roll and yaw change sign)."""
        signs = [1, -1, -1, 1, -1, 1, -1]
        return [a * s for a, s in zip(arm, signs)]

    def _finger_angles(self, side: str, close: float) -> dict:
        """Finger joint angles for a hand closed by `close` (0 open, 1 fist)."""
        m = self.model
        out = {}
        for f in ("thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"):
            name = f"{side}_hand_{f}_joint"
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                continue
            lo, hi = m.jnt_range[jid]
            end = lo if abs(lo) > abs(hi) else hi       # the closing direction is the long one
            if f == "thumb_1":
                end *= 0.6
            out[name] = close * end
        return out

    def _reach_pose(self, hand, point, base=None, iterations=60):
        """Pose with `hand`'s palm at `point` (robot frame), by damped least squares on the arm."""
        m, sc = self.model, self._scratch
        pose = (base or self.pose).copy()
        names = [f"{hand}_{j}_joint" for j in REACH_JOINTS]
        jids = [m.joint(n).id for n in names]
        qa = [m.jnt_qposadr[j] for j in jids]
        va = [m.jnt_dofadr[j] for j in jids]
        lo, hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]
        body = m.body(f"{hand}_wrist_yaw_link").id
        jp = np.zeros((3, m.nv))
        err = np.inf
        for _ in range(3):                    # balance moves the pelvis, which moves the shoulder
            _, q = self._solve(pose)
            sc.qpos[:] = q
            for _ in range(iterations):
                mujoco.mj_kinematics(m, sc)
                mujoco.mj_comPos(m, sc)
                palm = sc.xpos[body] + sc.xmat[body].reshape(3, 3) @ PALM
                e = point - palm
                err = float(np.linalg.norm(e))
                if err < 1e-3:
                    break
                mujoco.mj_jac(m, sc, jp, None, palm, body)
                J = jp[:, va]
                dq = J.T @ np.linalg.solve(J @ J.T + 0.05**2 * np.eye(3), e)
                sc.qpos[qa] = np.clip(sc.qpos[qa] + dq, lo, hi)
            pose.q[qa] = sc.qpos[qa]
        return pose, err

    def _feet_mid(self):
        return self.to_world(np.zeros(3))[:2]

    def _moved(self, x0) -> str:
        d = self._feet_mid() - x0
        return f"{np.linalg.norm(d):.2f} m"

    def _actual_feet_frame(self):
        """Mid-point of the real feet and the direction they point (from the foot sites)."""
        d = self.data
        L, R = d.site_xpos[self._feet_site["left"]], d.site_xpos[self._feet_site["right"]]
        yaws = []
        for s in ("left", "right"):
            x = d.site_xmat[self._feet_site[s]].reshape(3, 3)[:, 0]
            yaws.append(math.atan2(x[1], x[0]))
        yaw = math.atan2(sum(math.sin(a) for a in yaws), sum(math.cos(a) for a in yaws))
        return (L[:2] + R[:2]) / 2, yaw

    def _world_yaw(self):
        return self._actual_feet_frame()[1]

    def _step_segments(self, foot, dx, dy, dyaw, lift, seconds, plan_only=False):
        """The five glides of one quasi-static step, starting from the planned pose."""
        base = (self._plan_pose if plan_only else None) or self.pose
        other = "right" if foot == "left" else "left"
        f = base.feet
        over = self.target(base, shift=centre(f[other]), height=min(base.height, HOME["height"]))
        new = f[foot].copy()
        up = new.copy(); up[2] += lift
        moved = new + [dx, dy, 0, dyaw]
        moved_up = moved.copy(); moved_up[2] += lift
        yaw = (f[other][3] + moved[3]) / 2
        s1 = self.target(over, **{f"{foot}_foot": up})
        s2 = self.target(s1, **{f"{foot}_foot": moved_up}, yaw=yaw)
        s3 = self.target(s2, **{f"{foot}_foot": moved})
        mid = (centre(moved) + centre(f[other])) / 2
        s4 = self.target(s3, shift=mid)
        if plan_only:
            self._plan_pose = s4
        return [(self.SHIFT_SECONDS, over), (seconds * 0.5, s1), (seconds * 0.6, s2),
                (seconds * 0.5, s3), (self.SHIFT_SECONDS, s4)]

    def _reanchor(self):
        """Re-centre the robot frame between the commanded feet (the motion does not change)."""
        self._plan_pose = None
        f = self.pose.feet
        mid = (f["left"][:2] + f["right"][:2]) / 2
        yaw = math.atan2(math.sin(f["left"][3]) + math.sin(f["right"][3]),
                         math.cos(f["left"][3]) + math.cos(f["right"][3]))
        R = _rz(-yaw)[:2, :2]
        for s in ("left", "right"):
            f[s][:2] = R @ (f[s][:2] - mid)
            f[s][3] -= yaw
        self.pose.shift = R @ (self.pose.shift - mid)
        self.pose.rpy[2] -= yaw
        self._pxy = R @ (self._pxy - mid)
        self._seed[:2] = R @ (self._seed[:2] - mid)
        self.planted = {k: v.copy() for k, v in f.items()}


# ----------------------------------------------------------------------------- snapshots
def _arm(side, pitch, roll, yaw, elbow):
    """One arm's angles; roll and yaw are given for the LEFT arm and mirrored for the right."""
    sgn = 1.0 if side == "left" else -1.0
    return {f"{side}_shoulder_pitch": pitch, f"{side}_shoulder_roll": sgn * roll,
            f"{side}_shoulder_yaw": sgn * yaw, f"{side}_elbow": elbow,
            f"{side}_wrist_roll": 0.0, f"{side}_wrist_pitch": 0.0, f"{side}_wrist_yaw": 0.0}


def _both(pitch, roll, yaw, elbow):
    return {**_arm("left", pitch, roll, yaw, elbow), **_arm("right", pitch, roll, yaw, elbow)}


# Named snapshots: each is a dict of `move` keywords, applied to the home stance.
# Join names with '+' to combine parts that do not overlap ("arms_up+squat").
SNAPSHOTS = {
    # whole body
    "home": {},
    "squat": dict(height=0.62),
    "half_squat": dict(height=0.67),
    "tall": dict(height=0.76),
    "lean_forward": dict(pitch=0.20),
    "lean_back": dict(pitch=-0.12),
    "sway_left": dict(shift=(FOOT_CENTRE_X, 0.05)),
    "sway_right": dict(shift=(FOOT_CENTRE_X, -0.05)),
    "on_left": dict(weight="left"),
    "on_right": dict(weight="right"),
    # waist
    "twist_left": dict(waist_yaw=0.7),
    "twist_right": dict(waist_yaw=-0.7),
    "bow": dict(waist_pitch=0.45),
    "tilt_left": dict(waist_roll=0.3),
    "tilt_right": dict(waist_roll=-0.3),
    # arms
    "arms_down": _both(0.2, 0.2, 0.0, 1.28),
    "arms_up": _both(-2.9, 0.15, 0.0, STRAIGHT),
    "arms_out": _both(0.0, 1.45, 0.0, STRAIGHT),
    "arms_forward": _both(-1.5, 0.1, 0.0, STRAIGHT),
    "arms_bent": _both(0.0, 1.4, 1.57, 0.0),         # upper arms out, forearms up
    "left_up": _arm("left", -2.9, 0.15, 0.0, STRAIGHT),
    "right_up": _arm("right", -2.9, 0.15, 0.0, STRAIGHT),
    "left_out": _arm("left", 0.0, 1.45, 0.0, STRAIGHT),
    "right_out": _arm("right", 0.0, 1.45, 0.0, STRAIGHT),
    "left_wave_a": _arm("left", -0.3, 1.3, 1.57, -0.3),
    "left_wave_b": _arm("left", -0.3, 1.3, 1.57, 0.3),
    "right_wave_a": _arm("right", -0.3, 1.3, 1.57, -0.3),
    "right_wave_b": _arm("right", -0.3, 1.3, 1.57, 0.3),
    # hands
    "fists": dict(left_hand=1.0, right_hand=1.0),
    "open_hands": dict(left_hand=0.0, right_hand=0.0),
    # legs: always with the weight on the other foot
    "left_knee_up": dict(weight="right", left_lift=0.10),
    "right_knee_up": dict(weight="left", right_lift=0.10),
    "left_tap_forward": dict(weight="right", left_lift=0.03, left_offset=(0.10, 0.0)),
    "right_tap_forward": dict(weight="left", right_lift=0.03, right_offset=(0.10, 0.0)),
    "left_tap_side": dict(weight="right", left_lift=0.03, left_offset=(0.0, 0.04)),
    "right_tap_side": dict(weight="left", right_lift=0.03, right_offset=(0.0, -0.04)),
}
