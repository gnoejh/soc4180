"""Model-predictive control by sampling (MPPI), with MuJoCo itself as the model.

Week 11. Every controller so far was either written down in advance (weeks
4-5) or learned in advance (weeks 8-9). MPPI decides at run time instead: at
every decision it copies the current state into the simulator, tries K
perturbed futures, scores each with a cost function, and executes the first
step of their cost-weighted average. Then it throws the plan away and does
it again 25 times a second.

    plan  = argmin over u[0:H] of  sum_t cost(state_t)   (approximately, by sampling)

The pieces, each a few lines below:

- ``planning_model()``   the G1 with sensors added for what a cost reads
                          (feet, hands, centre of mass), so rollouts return them
- ``Rollouts``           K sampled futures as arrays, with named accessors
- ``MPPI.plan(data)``    one decision: sample, roll out in parallel threads
                          (``mujoco.rollout``), weight by exp(-cost / lambda),
                          average, shift the plan forward one knot

The rollouts run through ``mujoco.rollout.rollout``, which steps many copies
of the state on a thread pool in C. That is the whole reason this is fast
enough to run live on a laptop.
"""

from __future__ import annotations

import os

import numpy as np

from ._gl import GL_BACKEND  # noqa: F401  (sets MUJOCO_GL before mujoco loads)

import mujoco
from mujoco import rollout as _rollout

__all__ = ["MPPI", "Rollouts", "SENSORS", "actuators_for", "crouch", "planning_model", "standing_cost"]

#: Sensors added to the G1 so a cost function can read them from every rollout.
#: name -> (sensor type, object type, object name)
SENSORS = {
    "left_foot": ("FRAMEPOS", "SITE", "left_foot"),
    "right_foot": ("FRAMEPOS", "SITE", "right_foot"),
    "left_hand": ("FRAMEPOS", "BODY", "left_wrist_yaw_link"),
    "right_hand": ("FRAMEPOS", "BODY", "right_wrist_yaw_link"),
    "head": ("FRAMEPOS", "SITE", "imu_in_torso"),
    "com": ("SUBTREECOM", "BODY", "pelvis"),
}


def planning_model() -> mujoco.MjModel:
    """The G1 scene with the ``SENSORS`` above appended (after its own IMUs).

    ``MjSpec`` edits the model before compiling it: the same robot, plus six
    3-number sensors. A rollout records ``sensordata`` at every step, so these
    are what a cost function can see of the future without recomputing
    kinematics in Python.
    """
    from .models import robot_path

    spec = mujoco.MjSpec.from_file(str(robot_path()))
    for name, (stype, otype, oname) in SENSORS.items():
        spec.add_sensor(name=name, type=getattr(mujoco.mjtSensor, f"mjSENS_{stype}"),
                        objtype=getattr(mujoco.mjtObj, f"mjOBJ_{otype}"), objname=oname)
    return spec.compile()


def actuators_for(model, chains=("left_leg", "right_leg")) -> np.ndarray:
    """Actuator indices driving the joints of the named ``bodies.CHAINS``, in chain order."""
    from .bodies import group_indices

    act_qpos = [int(model.jnt_qposadr[model.actuator_trnid[a, 0]]) for a in range(model.nu)]
    return np.array([act_qpos.index(int(q)) for c in chains for q in group_indices(model, c)], dtype=int)


def crouch(model):
    """The week-4 bent-knee crouch: (qpos, ctrl). The planner's residuals are added to this ctrl."""
    from .walking import WalkingController

    q = WalkingController(model).nominal.copy()
    act_qpos = np.array([model.jnt_qposadr[model.actuator_trnid[a, 0]] for a in range(model.nu)])
    return q, q[act_qpos].copy()


class Rollouts:
    """K sampled futures, T steps each, read-only. Every accessor is (K, T) or (K, T, 3)."""

    def __init__(self, model, state, sensordata, sensor_adr):
        self.model = model
        nq, nv = model.nq, model.nv
        self.qpos = state[:, :, 1:1 + nq]            # FULLPHYSICS state: time, qpos, qvel, ...
        self.qvel = state[:, :, 1 + nq:1 + nq + nv]
        self._s = sensordata
        self._adr = sensor_adr

    def sensor(self, name: str) -> np.ndarray:
        a = self._adr[name]
        return self._s[:, :, a:a + 3]

    # the quantities costs are written in
    @property
    def pelvis_height(self):
        return self.qpos[:, :, 2]

    @property
    def upright(self):
        """Cosine of the pelvis tilt: 1 upright, 0 lying down. (Body z-axis . world z.)"""
        x, y = self.qpos[:, :, 4], self.qpos[:, :, 5]
        return 1.0 - 2.0 * (x * x + y * y)

    @property
    def com(self):
        return self.sensor("com")

    def foot(self, side: str):
        return self.sensor(f"{side}_foot")

    def hand(self, side: str):
        return self.sensor(f"{side}_hand")

    @property
    def head(self):
        return self.sensor("head")

    @property
    def yaw(self):
        """Pelvis heading, radians."""
        w, x, y, z = (self.qpos[:, :, i] for i in range(3, 7))
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def standing_cost(r: Rollouts) -> np.ndarray:
    """Stay tall, stay upright, keep the centre of mass over the feet. Returns (K,).

    The four terms are the whole specification of "balance" this planner
    gets; nothing in it says HOW (ankle, hip or a step). Measured with
    `plan.py`: holding the crouch falls at a 70 N push for 0.2 s; this cost
    survives 300 N, by stepping.
    """
    mid = 0.5 * (r.foot("left") + r.foot("right"))
    c = (10.0 * (r.pelvis_height - 0.72) ** 2
         + 5.0 * (1.0 - r.upright)
         + 20.0 * np.sum((r.com[..., :2] - mid[..., :2]) ** 2, axis=-1)
         + 0.01 * np.sum(r.qvel[..., :6] ** 2, axis=-1)
         + 100.0 * (r.pelvis_height < 0.5))
    return c.sum(axis=1)


class MPPI:
    """Model-predictive path integral control over a subset of the G1's servos.

    The plan is ``H`` knots of residuals (radians, added to ``nominal_ctrl``)
    for the actuators in ``actuators``, each held for ``knot`` seconds.

    samples      K futures tried per decision (row 0 is always the current plan, unperturbed)
    horizon      seconds looked ahead
    knot         seconds per plan step = seconds between decisions
    sigma        standard deviation of the perturbations, radians
    temperature  lambda: small -> trust only the best sample; large -> average them all
    """

    def __init__(self, model, nominal_ctrl, actuators, *, samples=64, horizon=0.4, knot=0.04,
                 sigma=0.15, temperature=0.05, nthread=None, seed=0):
        self.model = model
        self.nominal_ctrl = np.asarray(nominal_ctrl, float).copy()
        self.actuators = np.asarray(actuators, int)
        self.K = int(samples)
        self.H = max(int(round(horizon / knot)), 1)
        self.substeps = max(int(round(knot / model.opt.timestep)), 1)
        self.knot = self.substeps * model.opt.timestep
        self.sigma, self.temperature = float(sigma), float(temperature)
        self.U = np.zeros((self.H, len(self.actuators)))
        nthread = nthread or max(1, min(16, os.cpu_count() or 1))
        self._datas = [mujoco.MjData(model) for _ in range(nthread)]
        self._spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self._state = np.empty(mujoco.mj_stateSize(model, self._spec))
        self._adr = {}
        for name in SENSORS:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid >= 0:
                self._adr[name] = int(model.sensor_adr[sid])
        self.rng = np.random.default_rng(seed)
        self.cost = standing_cost
        self.last_costs = None

    def reset(self):
        self.U[:] = 0.0

    def plan(self, data) -> np.ndarray:
        """One decision. Returns the full ``ctrl`` vector to apply for the next knot."""
        mujoco.mj_getState(self.model, data, self._state, self._spec)          # 1. where we are
        eps = self.rng.normal(0.0, self.sigma, (self.K, self.H, len(self.actuators)))
        eps[0] = 0.0                                                            # keep the old plan as a candidate
        U = self.U[None] + eps                                                  # 2. K perturbed plans
        ctrl = np.tile(self.nominal_ctrl, (self.K, self.H, 1))
        ctrl[:, :, self.actuators] += U
        ctrl = np.repeat(ctrl, self.substeps, axis=1)                           #    each knot held for its substeps
        states, sens = _rollout.rollout(self.model, self._datas,                 # 3. K futures, in parallel
                                        np.tile(self._state, (self.K, 1)), ctrl,
                                        persistent_pool=True)
        costs = np.asarray(self.cost(Rollouts(self.model, states, sens, self._adr)), float)   # 4. score
        w = np.exp(-(costs - costs.min()) / self.temperature)                  # 5. weight: exp(-cost / lambda)
        w /= w.sum()
        self.U = np.einsum("k,khn->hn", w, U)                                   # 6. the weighted average plan
        u0 = self.U[0].copy()
        self.U = np.roll(self.U, -1, axis=0); self.U[-1] = 0.0                  # 7. shift: next decision starts here
        self.last_costs = costs
        out = self.nominal_ctrl.copy()
        out[self.actuators] += u0
        return out
