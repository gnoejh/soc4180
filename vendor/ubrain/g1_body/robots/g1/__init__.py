"""The Unitree G1 as a RobotSpec: the menagerie `unitree_g1` model (entry
`scene`, the robot on a floor with lights), its `stand` keyframe, 29 actuated
joints (legs 6+6, waist 3, arms 7+7; no head or hand joints), two IMUs, the
foot sites, a forward head camera on the torso, and the codebook.

The menagerie G1 declares position actuators with kp = 500 and critical
damping on all 29 joints (robot_mujoco/CLAUDE.md, the course-derived facts).
Proprioception is 4 x 3 IMU + 29 qpos + 29 qvel = 70 numbers.

Five primitives. `stand`, `pose` and `sway` hold or nudge the keyframe. `walk` is
the frozen trained locomotion policy (`gait.py`, Unitree's open-source G1 policy;
Phase XI.0, 2026-10-12): the body's pattern generator. `walk_offset` is the same
gait with a 29-vector added to its targets, the channel through which the
cerebellum's corrections reach the joints (XI.2, XI.3). A hand-built LIPM/ZMP walker
was tried first and dropped as too fragile; it is in git history (b508f6b) and
robot_mujoco/CLAUDE.md. `reach` (IK to a target) is still to come.

The targets function is the robot's: it knows which actuator is the waist yaw. For
`walk` it also senses (pelvis gyro, IMU attitude, leg encoders; nothing
privileged), and it puts the policy's PD gains on the leg servos while walking.
Any other primitive restores the body's own gains.
"""

from __future__ import annotations

import mujoco
import numpy as np

from body.robot import Primitive, RobotSpec
from robots.g1.gait import PolicyGait

__all__ = ["SPEC", "PRIMITIVES"]

PRIMITIVES: tuple[Primitive, ...] = (
    Primitive(0, "stand", 0, "hold the stand keyframe's joint targets"),
    Primitive(1, "pose", 29, "stand targets plus a per-joint offset (radians), clipped to the actuator range"),
    Primitive(2, "sway", 2, "stand plus a sinusoidal waist-yaw offset: params = (amplitude radians, period ticks)"),
    Primitive(3, "walk", 3, "the frozen locomotion policy: params = (vx m/s, vy m/s, yaw rate rad/s)"),
    Primitive(4, "walk_offset", 32, "walk plus a per-joint offset (radians): params = (vx, vy, wz, 29 offsets)"),
)

LEGS = np.arange(12)          # the policy's 12 actuators come first, in its order


def _gait(ctl, tick: int) -> PolicyGait:
    """The controller's gait, reset whenever the walk is not continuing from the
    previous tick (a reset of the sim, or a switch from another primitive)."""
    g = getattr(ctl, "_g1_gait", None)
    if g is None:
        g = ctl._g1_gait = PolicyGait(LEGS, ctl.sim.nu)
        m = ctl.sim.model
        ctl._g1_gyro = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu-pelvis-angular-velocity")])
        ctl._g1_imu = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu_in_pelvis")
        ctl._g1_last = None
    if ctl._g1_last is None or tick != ctl._g1_last + 1:
        g.reset()
    ctl._g1_last = tick
    return g


def targets(ctl, p: Primitive, params: np.ndarray, tick: int) -> np.ndarray:
    """Joint targets before clipping. `ctl` is the body's Controller: it gives
    the home targets and resolves actuator names."""
    if p.name in ("walk", "walk_offset"):
        sim = ctl.sim
        g = _gait(ctl, tick)
        g.engage(sim.model)
        target = g.targets(params[:3], sim.data.sensordata[ctl._g1_gyro:ctl._g1_gyro + 3].copy(),
                           sim.data.site_xmat[ctl._g1_imu].reshape(3, 3).copy(),
                           sim.joint_pos[LEGS], sim.joint_vel[LEGS])
        if p.name == "walk_offset":
            target = target + params[3:]
        return target
    g = getattr(ctl, "_g1_gait", None)
    if g is not None:
        g.release(ctl.sim.model)
        ctl._g1_last = None
    target = ctl.home.copy()
    if p.name == "pose":
        target = target + params
    elif p.name == "sway":
        amp, period = params
        if period <= 0:
            raise ValueError("sway period must be positive")
        target[ctl.actuator("waist_yaw_joint")] += amp * np.sin(2 * np.pi * tick / period)
    return target


SPEC = RobotSpec(
    name="g1",
    menagerie="unitree_g1",
    entry="scene",
    keyframe="stand",
    n_joints=29,
    imu_site="imu_in_torso",
    # Declared by the standard menagerie G1 model (g1.xml, g1_with_hands.xml).
    imu_sensors=(
        ("imu_torso_gyro", "imu-torso-angular-velocity", 3),
        ("imu_torso_acc", "imu-torso-linear-acceleration", 3),
        ("imu_pelvis_gyro", "imu-pelvis-angular-velocity", 3),
        ("imu_pelvis_acc", "imu-pelvis-linear-acceleration", 3),
    ),
    foot_sites=("left_foot", "right_foot"),
    head_camera_body="torso_link",
    head_camera_pos=(0.06, 0.0, 0.42),   # metres, roughly where the G1's head sensors sit
    head_camera_fovy=90.0,
    primitives=PRIMITIVES,
    targets=targets,
    package="robots.g1",
)
