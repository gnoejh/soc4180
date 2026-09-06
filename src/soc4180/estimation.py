"""Reading the IMUs, and estimating body orientation from them.

The G1 carries two inertial measurement units, at the torso and the pelvis. Each
is a gyroscope (angular velocity) plus an accelerometer (specific force). The
accelerometer senses gravity, which is what makes it a tilt sensor — and also
why it reads approximately zero in free fall.

Neither sensor alone gives good orientation while walking: the gyroscope is
smooth but its integral drifts, and the accelerometer is drift-free but is
corrupted by the robot's own acceleration. Blending them is the point.
"""

from __future__ import annotations

import math

import numpy as np

from ._gl import GL_BACKEND  # noqa: F401  (sets MUJOCO_GL before mujoco loads)

import mujoco

__all__ = ["ComplementaryFilter", "gravity_body", "read_imu", "tilt_from_accel"]

GRAVITY = 9.81


def read_imu(model, data, site: str = "torso"):
    """``(gyro, accelerometer)`` for one IMU, as 3-vectors in the sensor frame."""
    out = []
    for kind in ("angular-velocity", "linear-acceleration"):
        name = f"imu-{site}-{kind}"
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise ValueError(f"no sensor named {name!r}")
        adr = model.sensor_adr[sid]
        out.append(np.array(data.sensordata[adr : adr + model.sensor_dim[sid]], copy=True))
    return out[0], out[1]


def gravity_body(data) -> np.ndarray:
    """Unit gravity direction expressed in the body frame — the ground truth.

    This is the quantity locomotion policies observe. It is directly available
    from an IMU on real hardware, unlike absolute position or heading, which is
    why policies are built around it rather than around world pose.
    """
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(data.qpos[3:7], float))
    return rot.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])


def tilt_from_accel(accel) -> tuple[float, float]:
    """``(roll, pitch)`` in radians, assuming the accelerometer sees only gravity.

    That assumption is exactly what fails while walking: any real acceleration
    of the body is indistinguishable from a change in tilt.
    """
    ax, ay, az = np.asarray(accel, float)
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.hypot(ay, az))
    return roll, pitch


class ComplementaryFilter:
    """Blend a drifting gyroscope with a noisy accelerometer.

    Each step predicts orientation by integrating the gyroscope, then pulls that
    prediction a little way toward the accelerometer's estimate::

        angle = alpha * (angle + omega * dt) + (1 - alpha) * angle_from_accel

    ``alpha`` near 1 trusts the gyroscope and follows fast motion but inherits
    its drift; lower values track gravity but pick up acceleration noise. It is a
    one-line Kalman filter with the gains chosen by hand.
    """

    def __init__(self, dt: float, alpha: float = 0.995, roll: float = 0.0,
                 pitch: float = 0.0):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        self.dt = dt
        self.alpha = alpha
        self.roll = roll
        self.pitch = pitch

    def update(self, gyro, accel) -> tuple[float, float]:
        """Advance one step and return the current ``(roll, pitch)``."""
        gyro = np.asarray(gyro, float)
        acc_roll, acc_pitch = tilt_from_accel(accel)
        self.roll = self.alpha * (self.roll + gyro[0] * self.dt) + (1 - self.alpha) * acc_roll
        self.pitch = self.alpha * (self.pitch + gyro[1] * self.dt) + (1 - self.alpha) * acc_pitch
        return self.roll, self.pitch
