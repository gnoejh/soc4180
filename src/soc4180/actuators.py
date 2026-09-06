"""Reading and reshaping the joint servos.

The G1's "position actuators" are PD controllers written in the model file:

    force = kp * (ctrl - q) - kv * qdot

with ``kp`` in ``actuator_gainprm[:, 0]`` and ``-kp``, ``-kv`` in
``actuator_biasprm[:, 1:3]``. Changing a gain therefore means changing both
arrays consistently, which is what these helpers exist to get right.
"""

from __future__ import annotations

import numpy as np

__all__ = ["gains", "scale_gains", "set_torque_limit", "torque_limit"]


def gains(model):
    """Current ``(kp, kv)`` arrays, one entry per actuator."""
    kp = np.array(model.actuator_gainprm[:, 0], copy=True)
    kv = -np.array(model.actuator_biasprm[:, 2], copy=True)
    return kp, kv


def scale_gains(model, kp_scale: float = 1.0, kv_scale: float = 1.0):
    """Scale every servo's stiffness and damping in place.

    ``kp`` appears twice in the model — as the gain and as the negated position
    bias — and they must stay consistent or the servo no longer holds its
    setpoint. Returns the new ``(kp, kv)``.
    """
    kp, kv = gains(model)
    new_kp, new_kv = kp * kp_scale, kv * kv_scale
    model.actuator_gainprm[:, 0] = new_kp
    model.actuator_biasprm[:, 1] = -new_kp
    model.actuator_biasprm[:, 2] = -new_kv
    return new_kp, new_kv


def torque_limit(model):
    """The per-actuator torque limit, or ``None`` when the model has none."""
    if not np.any(model.actuator_forcelimited):
        return None
    return float(np.abs(model.actuator_forcerange).max())


def set_torque_limit(model, limit: float | None):
    """Impose a symmetric torque limit on every actuator, or remove it.

    The Menagerie G1 ships with **no** force limit, so its simulated motors are
    infinitely strong. Imposing a realistic one is the cheapest sim-to-real
    experiment available.
    """
    if limit is None:
        model.actuator_forcelimited[:] = 0
        model.actuator_forcerange[:] = 0.0
        return None
    model.actuator_forcelimited[:] = 1
    model.actuator_forcerange[:, 0] = -abs(limit)
    model.actuator_forcerange[:, 1] = abs(limit)
    return abs(limit)
