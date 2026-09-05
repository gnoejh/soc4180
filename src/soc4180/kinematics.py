"""Leg kinematics: forward kinematics and damped least-squares inverse kinematics.

The G1 has a 6-DOF leg (hip pitch/roll/yaw, knee, ankle pitch/roll), so a foot
pose is generally reachable exactly and IK is well posed.

IK here is *differential*: repeatedly ask "which joint change reduces the pose
error?", using the site Jacobian MuJoCo already computes. Damping keeps the
solution finite near singularities — a straight leg being the obvious one.
"""

from __future__ import annotations

import numpy as np

from ._gl import GL_BACKEND  # noqa: F401  (ensures MUJOCO_GL is set first)

import mujoco

__all__ = [
    "LEG_CHAIN",
    "LEG_JOINTS",
    "fk_foot",
    "leg_chain",
    "foot_site_id",
    "ik_legs",
    "leg_dof_indices",
    "leg_qpos_indices",
    "pose_error",
]

# Ordered hip -> ankle, matching the kinematic chain.
LEG_JOINTS = (
    "hip_pitch_joint",
    "hip_roll_joint",
    "hip_yaw_joint",
    "knee_joint",
    "ankle_pitch_joint",
    "ankle_roll_joint",
)


# Bodies from the pelvis down to the foot, in order.
LEG_CHAIN = (
    "hip_pitch_link",
    "hip_roll_link",
    "hip_yaw_link",
    "knee_link",
    "ankle_pitch_link",
    "ankle_roll_link",
)


def leg_chain(model, side: str) -> list[int]:
    """Body ids from the first leg link down to the foot."""
    ids = []
    for link in LEG_CHAIN:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_{link}")
        if bid < 0:
            raise ValueError(f"no body named {side}_{link}")
        ids.append(bid)
    return ids


def fk_foot(model, qpos, side: str):
    """Forward kinematics of a foot site, composed by hand from the model tree.

    A reference implementation to check student code against. Each body carries a
    fixed offset from its parent (``body_pos``/``body_quat``) followed by a
    rotation about its joint axis, anchored at ``jnt_pos``. Composing those down
    the chain is all forward kinematics is.

    Returns ``(position, rotation)`` of the foot site in world coordinates.
    """
    qpos = np.asarray(qpos, float)

    def quat2mat(q):
        out = np.zeros(9)
        mujoco.mju_quat2Mat(out, np.asarray(q, float))
        return out.reshape(3, 3)

    def axis_angle(axis, angle):
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), float(angle))
        return quat2mat(q)

    pos = qpos[:3].copy()
    rot = quat2mat(qpos[3:7])

    for bid in leg_chain(model, side):
        pos = pos + rot @ model.body_pos[bid]
        rot = rot @ quat2mat(model.body_quat[bid])
        jnt = model.body_jntadr[bid]
        if jnt >= 0:
            anchor = model.jnt_pos[jnt]
            rot_j = axis_angle(model.jnt_axis[jnt], qpos[model.jnt_qposadr[jnt]])
            # rotate about an axis through `anchor`, not through the origin
            pos = pos + rot @ (anchor - rot_j @ anchor)
            rot = rot @ rot_j

    site = foot_site_id(model, side)
    return pos + rot @ model.site_pos[site], rot @ quat2mat(model.site_quat[site])


def _joint_id(model, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"no joint named {name!r}")
    return jid


def leg_dof_indices(model, side: str) -> np.ndarray:
    """Velocity-space (Jacobian column) indices for one leg."""
    return np.array(
        [model.jnt_dofadr[_joint_id(model, f"{side}_{j}")] for j in LEG_JOINTS],
        dtype=int,
    )


def leg_qpos_indices(model, side: str) -> np.ndarray:
    """Position-space (`qpos`) indices for one leg."""
    return np.array(
        [model.jnt_qposadr[_joint_id(model, f"{side}_{j}")] for j in LEG_JOINTS],
        dtype=int,
    )


def foot_site_id(model, side: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot")
    if sid < 0:
        raise ValueError(f"no site named {side}_foot")
    return sid


def pose_error(data, site_id: int, target_pos, target_mat) -> np.ndarray:
    """6-vector [position error; rotation error] taking the site to the target.

    The rotational part is the axis-angle of the residual rotation, which is what
    the site Jacobian's angular rows expect.
    """
    pos_err = np.asarray(target_pos, float) - data.site_xpos[site_id]

    cur = data.site_xmat[site_id].reshape(3, 3)
    residual = np.asarray(target_mat, float).reshape(3, 3) @ cur.T
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, residual.flatten())
    rot_err = np.empty(3)
    mujoco.mju_quat2Vel(rot_err, quat, 1.0)

    return np.concatenate([pos_err, rot_err])


def ik_legs(
    model,
    scratch,
    base_pos,
    base_quat,
    targets: dict,
    *,
    seed_qpos=None,
    iterations: int = 12,
    damping: float = 1e-2,
    tolerance: float = 1e-4,
) -> dict:
    """Solve both legs so each foot site reaches its target pose.

    The pelvis is *placed*, not solved for: `base_pos`/`base_quat` fix the
    floating base, and only the twelve leg joints are free. That is what makes
    this a walking controller rather than a whole-body solver — we decide where
    the body should be, then ask the legs to make it so.

    ``targets`` maps ``"left"``/``"right"`` to ``(position, rotation_matrix)``.
    ``scratch`` is an ``MjData`` used purely for the solve; the live simulation
    state is never touched. Returns ``{side: joint angles}`` plus ``"error"``.
    """
    scratch.qpos[:3] = base_pos
    scratch.qpos[3:7] = base_quat
    if seed_qpos is not None:
        for side in targets:
            idx = leg_qpos_indices(model, side)
            scratch.qpos[idx] = np.asarray(seed_qpos)[idx]
    scratch.qvel[:] = 0

    dof = {s: leg_dof_indices(model, s) for s in targets}
    qpos_idx = {s: leg_qpos_indices(model, s) for s in targets}
    sites = {s: foot_site_id(model, s) for s in targets}

    jac_p, jac_r = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    worst = np.inf

    for _ in range(iterations):
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        worst = 0.0

        for side, (t_pos, t_mat) in targets.items():
            err = pose_error(scratch, sites[side], t_pos, t_mat)
            worst = max(worst, float(np.linalg.norm(err[:3])))

            mujoco.mj_jacSite(model, scratch, jac_p, jac_r, sites[side])
            jac = np.vstack([jac_p, jac_r])[:, dof[side]]

            # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 e
            jjt = jac @ jac.T + damping**2 * np.eye(6)
            dq = jac.T @ np.linalg.solve(jjt, err)

            q = scratch.qpos[qpos_idx[side]] + dq
            lo = model.jnt_range[[_joint_id(model, f"{side}_{j}") for j in LEG_JOINTS], 0]
            hi = model.jnt_range[[_joint_id(model, f"{side}_{j}") for j in LEG_JOINTS], 1]
            scratch.qpos[qpos_idx[side]] = np.clip(q, lo, hi)

        if worst < tolerance:
            break

    result = {side: scratch.qpos[qpos_idx[side]].copy() for side in targets}
    result["error"] = worst
    return result
