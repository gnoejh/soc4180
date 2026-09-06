"""Analytic walking: LIPM centre-of-mass trajectories and a footstep gait.

No learning anywhere in this file. The gait comes from a linear model of
balance, solved in closed form, and the legs are placed by inverse kinematics.

The Linear Inverted Pendulum Model treats the robot as a point mass at constant
height over a pivot at the centre of pressure (the ZMP). Its equation of motion
is linear::

    x_ddot = omega^2 (x - p),        omega = sqrt(g / z_com)

with the closed-form solution::

    x(t) = p + (x0 - p) cosh(omega t) + (v0 / omega) sinh(omega t)

Everything else here is a choice of boundary conditions for that solution.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

__all__ = ["CPG", "GaitParams", "LIPM", "WalkingController", "footstep_plan"]


class LIPM:
    """Linear inverted pendulum, and the two periodic gaits it admits."""

    def __init__(self, com_height: float, gravity: float = 9.81):
        if com_height <= 0:
            raise ValueError("com_height must be positive")
        self.com_height = com_height
        self.omega = math.sqrt(gravity / com_height)

    def evolve(self, x0: float, v0: float, zmp: float, t: float):
        """State at time ``t`` given an initial state and a fixed ZMP."""
        w = self.omega
        c, s = math.cosh(w * t), math.sinh(w * t)
        x = zmp + (x0 - zmp) * c + (v0 / w) * s
        v = (x0 - zmp) * w * s + v0 * c
        return x, v

    def bvp_velocity(self, x0: float, xT: float, zmp: float, T: float) -> float:
        """Launch velocity taking the mass from ``x0`` to ``xT`` in time ``T``.

        The two-point boundary value problem for the LIPM. Solving this per step,
        with each step starting where the last one ended, is what keeps the
        commanded centre of mass continuous — recomputing an absolute trajectory
        per step instead teleports the body backwards at every support exchange.
        """
        w = self.omega
        c, s_ = math.cosh(w * T), math.sinh(w * T)
        return w * (xT - zmp - (x0 - zmp) * c) / s_

    def lateral_periodic_velocity(self, zmp_y: float, step_time: float) -> float:
        """Sideways launch velocity giving a gait that repeats every step.

        Requiring the step map to satisfy ``(y, v) -> (-y, -v)`` (the next step
        is the mirror of this one) forces ``y0 = 0``: the centre of mass is
        always at the midline when support switches. It sways toward the stance
        foot, is turned around by gravity, and arrives back at the midline with
        the mirrored velocity, ready for the other leg.
        """
        w = self.omega
        c, s = math.cosh(w * step_time), math.sinh(w * step_time)
        return w * zmp_y * s / (1.0 + c)

    def sagittal_periodic_velocity(self, step_length: float, step_time: float) -> float:
        """Forward velocity at support exchange for a steady stride.

        The centre of mass enters the step half a stride behind the stance foot
        and leaves it half a stride ahead, with the same speed at both ends, so
        the pattern repeats indefinitely.
        """
        w = self.omega
        c, s = math.cosh(w * step_time), math.sinh(w * step_time)
        return w * (step_length / 2.0) * (1.0 + c) / s


@dataclasses.dataclass
class GaitParams:
    """Everything that defines the gait. Tune these; they are the lab's knobs."""

    step_length: float = 0.14      # metres advanced per step
    step_time: float = 0.65        # seconds per step
    step_height: float = 0.10      # commanded swing foot clearance
    stance_width: float = 0.119    # lateral foot offset from midline
    com_height: float = 0.60       # LIPM pendulum height
    pelvis_height: float = 0.72    # commanded pelvis height (crouched)
    foot_height: float = 0.033     # foot site height when planted
    double_support: float = 0.20   # fraction of each end of the step with both feet down
    settle_time: float = 1.0       # stand still before walking
    n_steps: int = 12


def footstep_plan(params: GaitParams):
    """Where each foot lands, and which foot is supporting when.

    Returns a list of ``(support_side, support_xy, swing_from_xy, swing_to_xy)``,
    one entry per step. The first step is a half-stride so the gait starts from
    a symmetric stance rather than lurching.
    """
    w = params.stance_width
    L = params.step_length
    plan = []
    pos = {"left": np.array([0.0, +w]), "right": np.array([0.0, -w])}

    for k in range(params.n_steps):
        support = "right" if k % 2 == 0 else "left"
        swing = "left" if support == "right" else "right"
        # First step is half length: the swing foot starts level with the
        # stance foot, so a full stride would over-reach.
        advance = L / 2.0 if k == 0 else L
        target = np.array([pos[swing][0] + advance, pos[swing][1]])
        plan.append((support, pos[support].copy(), pos[swing].copy(), target.copy()))
        pos[swing] = target
    return plan


def _swing_xy(start, end, s: float):
    """Horizontal swing path, with zero velocity at lift-off and touchdown.

    A cycloid-like blend: moving smoothly at both ends matters because a foot
    still travelling when it lands will slip or trip.
    """
    blend = s - math.sin(2 * math.pi * s) / (2 * math.pi)
    return start + (end - start) * blend


def _swing_z(height: float, foot_height: float, s: float) -> float:
    """Lift and lower, peaking at mid-swing."""
    return foot_height + height * 0.5 * (1.0 - math.cos(2 * math.pi * s))


class WalkingController:
    """Turns time into joint position targets. Layer 4, in one object."""

    def __init__(self, model, params: GaitParams | None = None):
        import mujoco

        from . import kinematics as kin

        self._mujoco = mujoco
        self._kin = kin
        self.model = model
        self.params = params or GaitParams()
        self.lipm = LIPM(self.params.com_height)
        self.plan = footstep_plan(self.params)

        self._scratch = mujoco.MjData(model)
        self._flat = np.eye(3)

        # Nominal crouch: a bent-knee pose. The 'stand' keyframe has every leg
        # joint at zero, which is a straight-leg singularity that IK cannot
        # escape, so it is unusable as a seed.
        self.nominal = np.array(model.key_qpos[0], copy=True)
        for side in ("left", "right"):
            idx = kin.leg_qpos_indices(model, side)
            self.nominal[idx[0]] = -0.35   # hip pitch
            self.nominal[idx[3]] = +0.70   # knee
            self.nominal[idx[4]] = -0.35   # ankle pitch
        self._seed = self.nominal.copy()

        self._act_qpos = self._actuator_to_qpos()
        self._segments = self._build_segments()
        self.total_time = self.params.settle_time + self.params.step_time * len(self.plan)

    def _build_segments(self):
        """Per-step centre-of-mass boundary conditions, chained for continuity.

        Each step ends with the centre of mass half an advance beyond its stance
        foot and back on the midline; the next step begins exactly there.
        """
        p, T = self.params, self.params.step_time
        segments = []
        x0, y0 = 0.0, 0.0

        for k, (_, support_xy, _, _) in enumerate(self.plan):
            nxt = self.plan[k + 1][1] if k + 1 < len(self.plan) else None
            advance = (nxt[0] - support_xy[0]) if nxt is not None else p.step_length
            x_end = support_xy[0] + advance / 2.0
            y_end = 0.0
            vx = self.lipm.bvp_velocity(x0, x_end, support_xy[0], T)
            vy = self.lipm.bvp_velocity(y0, y_end, support_xy[1], T)
            segments.append((x0, vx, y0, vy, support_xy[0], support_xy[1]))
            x0, y0 = x_end, y_end
        return segments

    def _actuator_to_qpos(self) -> np.ndarray:
        """qpos index driven by each actuator, so we can build a ctrl vector."""
        mujoco = self._mujoco
        out = np.zeros(self.model.nu, dtype=int)
        for a in range(self.model.nu):
            jid = self.model.actuator_trnid[a, 0]
            out[a] = self.model.jnt_qposadr[jid]
            if self.model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                raise RuntimeError("expected joint-transmission actuators")
        return out

    def targets_at(self, t: float):
        """Desired pelvis position and both foot poses at time ``t``."""
        p = self.params
        w = p.stance_width

        if t < p.settle_time:
            return (
                np.array([0.0, 0.0, p.pelvis_height]),
                {
                    "left": (np.array([0.0, +w, p.foot_height]), self._flat),
                    "right": (np.array([0.0, -w, p.foot_height]), self._flat),
                },
            )

        walk_t = t - p.settle_time
        k = min(int(walk_t / p.step_time), len(self.plan) - 1)
        s = min((walk_t - k * p.step_time) / p.step_time, 1.0)
        support, support_xy, swing_from, swing_to = self.plan[k]
        swing = "left" if support == "right" else "right"

        # --- centre of mass: this step's boundary value problem
        x0, vx0, y0, vy0, zmp_x, zmp_y = self._segments[k]
        tau = s * p.step_time
        com_x, _ = self.lipm.evolve(x0, vx0, zmp_x, tau)
        com_y, _ = self.lipm.evolve(y0, vy0, zmp_y, tau)
        pelvis = np.array([com_x, com_y, p.pelvis_height])

        # --- feet
        # Double support: the foot stays planted at both ends of the step, so
        # weight can transfer before it lifts and after it lands.
        ds = p.double_support
        s_sw = float(np.clip((s - ds) / max(1.0 - 2 * ds, 1e-6), 0.0, 1.0))
        sw_xy = _swing_xy(swing_from, swing_to, s_sw)
        sw_z = _swing_z(p.step_height, p.foot_height, s_sw)
        feet = {
            support: (np.array([*support_xy, p.foot_height]), self._flat),
            swing: (np.array([*sw_xy, sw_z]), self._flat),
        }
        return pelvis, feet

    def control(self, t: float) -> np.ndarray:
        """Full actuator command at time ``t``."""
        pelvis, feet = self.targets_at(t)
        sol = self._kin.ik_legs(
            self.model,
            self._scratch,
            pelvis,
            self.nominal[3:7],
            feet,
            seed_qpos=self._seed,
        )
        target = self.nominal.copy()
        for side in ("left", "right"):
            target[self._kin.leg_qpos_indices(self.model, side)] = sol[side]
        self._seed = target  # warm start the next solve
        self.last_ik_error = sol["error"]
        return target[self._act_qpos]

    def initial_data(self):
        """`MjData` posed in the nominal crouch, ready to walk.

        Starting from the `stand` keyframe instead would begin at a straight-leg
        singularity, which the IK cannot work from.
        """
        data = self._mujoco.MjData(self.model)
        data.qpos[:] = self.nominal
        data.qpos[2] = self.params.pelvis_height + 0.02
        self._mujoco.mj_forward(self.model, data)
        self._seed = self.nominal.copy()
        return data

    def zmp(self, data):
        """Measured centre of pressure: ground reaction forces, force-weighted.

        Returns ``(x, y)``, or ``(nan, nan)`` in flight when there is no contact.
        """
        mujoco = self._mujoco
        fz = mx = my = 0.0
        wrench = np.zeros(6)
        for i in range(data.ncon):
            mujoco.mj_contactForce(self.model, data, i, wrench)
            force_world = data.contact[i].frame.reshape(3, 3).T @ wrench[:3]
            fz += force_world[2]
            mx += data.contact[i].pos[0] * force_world[2]
            my += data.contact[i].pos[1] * force_world[2]
        if fz < 1.0:
            return float("nan"), float("nan")
        return mx / fz, my / fz

    def ctrl_fn(self):
        """A `ctrl_fn` for `soc4180.render_rollout`."""

        def fn(model, data):
            data.ctrl[:] = self.control(data.time)

        return fn


class CPG:
    """A central pattern generator: rhythm with no model of balance.

    Two anti-phase oscillators drive the hips and knees around a nominal crouch.
    There is no centre-of-mass plan, no footstep plan, and no notion of the
    support polygon — which is exactly the point of comparing it against the
    LIPM walker.
    """

    def __init__(self, model, nominal=None, *, frequency=1.2,
                 hip_amplitude=0.25, knee_amplitude=0.40, settle_time=1.0):
        import mujoco

        from . import kinematics as kin

        self.model = model
        self.frequency = frequency
        self.hip_amplitude = hip_amplitude
        self.knee_amplitude = knee_amplitude
        self.settle_time = settle_time

        if nominal is None:
            nominal = WalkingController(model).nominal
        self.nominal = np.array(nominal, copy=True)
        self._idx = {s: kin.leg_qpos_indices(model, s) for s in ("left", "right")}
        self._act_qpos = np.array(
            [model.jnt_qposadr[model.actuator_trnid[a, 0]] for a in range(model.nu)],
            dtype=int,
        )
        self._mujoco = mujoco

    def control(self, t: float) -> np.ndarray:
        phase = 2 * math.pi * self.frequency * max(t - self.settle_time, 0.0)
        target = self.nominal.copy()
        for side, offset in (("left", 0.0), ("right", math.pi)):
            p = phase + offset
            i = self._idx[side]
            target[i[0]] += self.hip_amplitude * math.sin(p)
            target[i[3]] += self.knee_amplitude * max(0.0, math.sin(p))
            target[i[4]] -= 0.5 * self.hip_amplitude * math.sin(p)
        return target[self._act_qpos]

    def ctrl_fn(self):
        def fn(model, data):
            data.ctrl[:] = self.control(data.time)

        return fn

    def initial_data(self):
        data = self._mujoco.MjData(self.model)
        data.qpos[:] = self.nominal
        data.qpos[2] = GaitParams().pelvis_height + 0.02
        self._mujoco.mj_forward(self.model, data)
        return data
