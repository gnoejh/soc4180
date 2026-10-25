"""What the brain may sense, and what it may not. Model file.

The proprioceptive vector is laid out by the robot's spec: its IMU sensors in
order, then joint positions, then joint velocities (for the G1:
4 x 3 + 29 + 29 = 70; robot_mujoco/CLAUDE.md, the sensor inventory).
Contact forces per foot (6 each) are a PRIVILEGE a real humanoid lacks:
exported only when asked for, and flagged as such in the reply.

PRIVILEGED state -- masses, friction, applied external forces, the world's own
pose of things -- is named here and refused by the bridge by construction: no
message can request it. The engine's constitution forbids an orchestrator whose
benefit is assumed; a brain that reads the world's table is the same sin.
"""

from __future__ import annotations

import numpy as np
import mujoco

from body.robot import RobotSpec
from body.sim import MujocoSim

__all__ = [
    "PRIVILEGED", "proprio_layout", "proprio_dim", "contact_dim",
    "proprioception", "contact_forces",
]

# Named so that a test can assert none of it ever crosses the bridge.
PRIVILEGED = ("body_mass", "geom_friction", "xfrc_applied", "qpos_free_base", "world_pose")


def proprio_layout(spec: RobotSpec) -> tuple[tuple[str, int], ...]:
    return tuple((name, dim) for name, _sensor, dim in spec.imu_sensors) + (
        ("joint_pos", spec.n_joints),
        ("joint_vel", spec.n_joints),
    )


def proprio_dim(spec: RobotSpec) -> int:
    return sum(d for _n, d in proprio_layout(spec))


def contact_dim(spec: RobotSpec) -> int:
    return 6 * len(spec.foot_sites)   # force xyz, torque xyz, contact frame, per foot


def _sensor_slices(sim: MujocoSim) -> list[tuple[int, int]]:
    model, out = sim.model, []
    for _name, sensor, dim in sim.spec.imu_sensors:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor)
        if sid < 0:
            raise RuntimeError(f"model lacks sensor {sensor!r} (standard {sim.spec.menagerie} entry expected)")
        adr, sdim = int(model.sensor_adr[sid]), int(model.sensor_dim[sid])
        if sdim != dim:
            raise RuntimeError(f"sensor {sensor!r} has dim {sdim}, expected {dim}")
        out.append((adr, adr + dim))
    return out


def proprioception(sim: MujocoSim) -> np.ndarray:
    """The proprioceptive vector, in proprio_layout(spec) order."""
    parts = [sim.data.sensordata[a:b] for a, b in _sensor_slices(sim)]
    parts.append(sim.joint_pos)
    parts.append(sim.joint_vel)
    v = np.concatenate(parts).astype(float)
    assert v.shape == (proprio_dim(sim.spec),), v.shape
    return v


def _descends_from(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


def contact_forces(sim: MujocoSim) -> np.ndarray:
    """Summed 6-vector (contact-frame force, torque) per foot, feet in spec
    order. A privilege: a real humanoid has no foot force sensors."""
    model, data = sim.model, sim.data
    foot_bodies = []
    for site in sim.spec.foot_sites:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if sid < 0:
            raise RuntimeError(f"model lacks site {site!r}")
        foot_bodies.append(int(model.site_bodyid[sid]))
    out = np.zeros((len(foot_bodies), 6))
    f = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        for k, fb in enumerate(foot_bodies):
            if _descends_from(model, b1, fb) or _descends_from(model, b2, fb):
                mujoco.mj_contactForce(model, data, i, f)
                out[k] += f
    return out.reshape(contact_dim(sim.spec))
