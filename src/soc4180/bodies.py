"""The robot's body parts, by name: which joints they are and how to pose them.

Week 2 answers "where is the foot?" for one leg. This module answers the
question that comes before it -- **which numbers are the leg at all?** -- for
every part of the robot.

The G1 is five kinematic chains hanging off one floating pelvis::

    pelvis --+-- left leg    6 joints
             +-- right leg   6 joints
             +-- waist       3 joints -- torso --+-- left arm   7 joints
                                                 +-- right arm  7 joints

6+6+3+7+7 = 29, which is exactly ``model.nu``. ``qpos`` stores them in that
order after the floating base's seven numbers, so a body part is a *contiguous
slice* of the state vector -- ``CHAINS`` below is that layout written down.

Nothing here computes anything MuJoCo could not; it removes the index
bookkeeping that otherwise hides which six numbers are a leg.
"""

from __future__ import annotations

import numpy as np

from ._gl import GL_BACKEND  # noqa: F401  (ensures MUJOCO_GL is set first)

import mujoco

__all__ = [
    "CHAINS",
    "MIRRORED",
    "chain_bodies",
    "chain_of",
    "describe",
    "dof_index",
    "group_indices",
    "joint_index",
    "mirror",
    "set_pose",
]


# Joint names per chain, ordered root -> tip, exactly as they appear in `qpos`.
CHAINS: dict[str, tuple[str, ...]] = {
    "left_leg": (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ),
    "right_leg": (
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ),
    "waist": (
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    ),
    "left_arm": (
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ),
    "right_arm": (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ),
}


# Joints whose sign flips when a pose is mirrored left <-> right.
#
# A humanoid is mirror-symmetric about the x-z plane, and mirroring negates y.
# Rotations about x (roll) and z (yaw) reverse; rotations about y (pitch) do
# not. The G1's joint ranges show it: `left_hip_roll` runs [-0.52, 2.97] and
# `right_hip_roll` runs [-2.97, 0.52] -- the same interval, negated.
MIRRORED = ("roll", "yaw")


def _joint_id(model, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"no joint named {name!r}")
    return jid


def _full_name(model, name: str) -> str:
    """Accept ``left_elbow`` for ``left_elbow_joint``, and check it exists."""
    for candidate in (name, f"{name}_joint"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, candidate) >= 0:
            return candidate
    raise ValueError(
        f"no joint named {name!r}. Chains: {', '.join(CHAINS)}; "
        f"the joints in one are soc4180.bodies.CHAINS['left_arm']"
    )


def joint_index(model, name: str) -> int:
    """`qpos` slot of one joint, by short or full name.

    Position space. Its velocity-space twin is `dof_index`, and the two differ
    by one on every joint of this robot -- see that function.
    """
    return int(model.jnt_qposadr[_joint_id(model, _full_name(model, name))])


def dof_index(model, name: str) -> int:
    """`qvel` / Jacobian-column index of one joint, by short or full name.

    Not the same number as `joint_index`. The floating base spends **7** numbers
    on position (3 translation + 4 quaternion) and **6** on velocity, so every
    joint after it sits one slot earlier in `qvel` than in `qpos`. Mixing the
    two reads a neighbouring joint and raises nothing.
    """
    return int(model.jnt_dofadr[_joint_id(model, _full_name(model, name))])


def group_indices(model, chain: str, *, dof: bool = False) -> np.ndarray:
    """`qpos` (or `qvel`, with ``dof=True``) indices of a whole chain."""
    if chain not in CHAINS:
        raise ValueError(f"no chain named {chain!r}; try one of {', '.join(CHAINS)}")
    pick = dof_index if dof else joint_index
    return np.array([pick(model, j) for j in CHAINS[chain]], dtype=int)


def chain_of(model, name: str) -> str | None:
    """Which chain a joint belongs to, or None for the floating base."""
    full = _full_name(model, name)
    for chain, joints in CHAINS.items():
        if full in joints:
            return chain
    return None


def chain_bodies(model, chain: str) -> list[int]:
    """Body ids carrying a chain's joints, ordered root -> tip.

    Useful for highlighting a limb: these are the bodies that move when, and
    only when, that chain's joints move.
    """
    if chain not in CHAINS:
        raise ValueError(f"no chain named {chain!r}; try one of {', '.join(CHAINS)}")
    return [int(model.jnt_bodyid[_joint_id(model, j)]) for j in CHAINS[chain]]


def mirror(angles: dict) -> dict:
    """Reflect a pose left <-> right, negating roll and yaw.

    ``mirror({"left_shoulder_roll": 0.8})`` is ``{"right_shoulder_roll": -0.8}``.
    Pitch joints and the knee keep their sign; roll and yaw reverse, because
    mirroring negates y and those axes point along x and z. A waist joint has no
    opposite number, so it mirrors in place -- sign flipped, name unchanged.
    """
    out: dict = {}
    for name, value in angles.items():
        flipped = float(value)
        if any(axis in name for axis in MIRRORED):
            flipped = -flipped
        if name.startswith("left"):
            out["right" + name[len("left"):]] = flipped
        elif name.startswith("right"):
            out["left" + name[len("right"):]] = flipped
        else:
            out[name] = flipped
    return out


def set_pose(model, data, *, reset: str | None = "stand", degrees: bool = False,
             **parts):
    """Place named joints and run forward kinematics. No physics is stepped.

    Each keyword is either a **joint** taking one number::

        set_pose(model, data, left_elbow=1.2, waist_yaw=-0.5)

    or a **chain** from `CHAINS` taking one number per joint::

        set_pose(model, data, left_arm=[0, 0.3, 0, 1.2, 0, 0, 0])

    ``reset`` names a keyframe to start from, so every call describes a pose in
    full instead of accumulating on the last one; pass ``reset=None`` to layer
    changes onto the current state. ``degrees=True`` reads every number as
    degrees, which is how a person thinks about an elbow.

    Returns ``data``, with every ``xpos``/``xmat``/``site_xpos`` already filled
    in by ``mj_forward`` -- so the caller can read a hand's position at once.
    """
    if reset is not None:
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, reset)
        if key < 0:
            raise ValueError(f"no keyframe named {reset!r}")
        data.qpos[:] = model.key_qpos[key]

    for part, value in parts.items():
        if part in CHAINS:
            angles = np.asarray(value, dtype=float).ravel()
            if angles.size != len(CHAINS[part]):
                raise ValueError(
                    f"{part} has {len(CHAINS[part])} joints, got {angles.size}: "
                    f"{', '.join(CHAINS[part])}"
                )
            data.qpos[group_indices(model, part)] = (
                np.radians(angles) if degrees else angles
            )
        else:
            angle = float(value)
            data.qpos[joint_index(model, part)] = (
                float(np.radians(angle)) if degrees else angle
            )

    mujoco.mj_forward(model, data)
    return data


def describe(model) -> str:
    """One line per chain: how many joints, and which `qpos` slots they occupy."""
    lines = [f"{'chain':10s} {'joints':>6s}   qpos slots   qvel slots"]
    for chain in CHAINS:
        q = group_indices(model, chain)
        v = group_indices(model, chain, dof=True)
        lines.append(
            f"{chain:10s} {len(q):6d}   {q[0]:3d}..{q[-1]:<4d}    {v[0]:3d}..{v[-1]:<4d}"
        )
    return "\n".join(lines)
