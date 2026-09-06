"""The walking task as a Gymnasium environment.

This is where the course stops deriving controllers and starts specifying
problems. Every design choice here is a decision a human makes, and each one is
argued in the week 7 slides:

- **Observation** contains only quantities a real robot can measure. No body
  height, no world position. (Week 6.)
- **Actions are residuals** around a fixed nominal crouch, not absolute angles.
  The policy nudges a decent posture instead of inventing one from scratch, and
  it starts near something sensible rather than in a heap.
- **Control runs at 50 Hz** while physics runs at 500 Hz. A policy does not need
  to think ten times faster than a leg can move.
- **Only the twelve leg joints are controlled.** Arms and waist hold their
  nominal pose, which removes 17 dimensions the walking problem does not need.
"""

from __future__ import annotations

import numpy as np

from ._gl import GL_BACKEND  # noqa: F401  (sets MUJOCO_GL before mujoco loads)

import mujoco

__all__ = ["G1WalkEnv", "walker_actions"]

try:  # gymnasium lives in the `env` extra
    import gymnasium as gym
    from gymnasium import spaces

    _BASE = gym.Env
except ImportError:  # pragma: no cover - exercised only without the extra
    gym = None
    spaces = None
    _BASE = object


class G1WalkEnv(_BASE):
    """Forward walking with the Unitree G1.

    Reward is deliberately simple this week — track a forward velocity, pay for
    effort, stay alive. Week 9 takes it apart.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        target_velocity: float = 0.5,
        control_hz: float = 50.0,
        action_scale: float = 0.3,
        episode_seconds: float = 10.0,
        min_height: float = 0.5,
        max_tilt: float = 0.7,
    ):
        if gym is None:
            raise ImportError(
                "G1WalkEnv needs gymnasium: pip install 'soc4180[env]'"
            )
        from . import load_g1
        from .walking import WalkingController

        self.model = load_g1()
        self.data = mujoco.MjData(self.model)

        self.nominal = WalkingController(self.model).nominal.copy()
        self.leg_qpos = np.concatenate(
            [self._leg_indices(side) for side in ("left", "right")]
        )
        self.leg_dof = self.leg_qpos - 1  # qpos has one extra entry for the quaternion
        self.act_qpos = np.array(
            [self.model.jnt_qposadr[self.model.actuator_trnid[a, 0]]
             for a in range(self.model.nu)],
            dtype=int,
        )

        self.target_velocity = target_velocity
        self.action_scale = action_scale
        self.min_height = min_height
        self.max_tilt = max_tilt
        self.decimation = max(int(round(1.0 / (control_hz * self.model.opt.timestep))), 1)
        self.control_dt = self.decimation * self.model.opt.timestep
        self.max_steps = int(episode_seconds / self.control_dt)

        n_act = len(self.leg_qpos)
        self.action_space = spaces.Box(-1.0, 1.0, (n_act,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (3 + 3 + 3 * n_act,), dtype=np.float32
        )

        self._prev_action = np.zeros(n_act, dtype=np.float32)
        self._step = 0

    def _leg_indices(self, side):
        from .kinematics import leg_qpos_indices

        return leg_qpos_indices(self.model, side)

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.nominal
        self.data.qpos[2] = 0.72
        if seed is not None:
            noise = self.np_random.uniform(-0.02, 0.02, len(self.leg_qpos))
            self.data.qpos[self.leg_qpos] += noise
        mujoco.mj_forward(self.model, self.data)
        self._prev_action[:] = 0.0
        self._step = 0
        return self._observation(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32),
                         self.action_space.low, self.action_space.high)

        target = self.nominal.copy()
        target[self.leg_qpos] += self.action_scale * action
        ctrl = target[self.act_qpos]

        for _ in range(self.decimation):
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

        self._step += 1
        obs = self._observation()
        reward, parts = self._reward(action)
        terminated = self._fallen()
        truncated = self._step >= self.max_steps
        self._prev_action[:] = action
        return obs, reward, terminated, truncated, parts

    # ------------------------------------------------------------- internals
    def _observation(self) -> np.ndarray:
        from .estimation import gravity_body, read_imu

        gyro, _ = read_imu(self.model, self.data, "torso")
        return np.concatenate([
            gravity_body(self.data),                       # 3, from the IMU
            gyro,                                          # 3, from the IMU
            self.data.qpos[self.leg_qpos] - self.nominal[self.leg_qpos],
            self.data.qvel[self.leg_dof],
            self._prev_action,
        ]).astype(np.float32)

    def _reward(self, action):
        forward = float(self.data.qvel[0])
        tracking = np.exp(-((forward - self.target_velocity) ** 2) / 0.25)
        upright = float(-gravity_z(self.data))
        effort = float(np.sum(np.square(action)))
        smooth = float(np.sum(np.square(action - self._prev_action)))

        reward = (
            1.5 * tracking
            + 0.5 * upright
            - 0.01 * effort
            - 0.05 * smooth
            + 0.5                       # alive bonus
        )
        return reward, {
            "forward_velocity": forward,
            "tracking": tracking,
            "upright": upright,
            "effort": effort,
        }

    def _fallen(self) -> bool:
        from .estimation import gravity_body

        if self.data.qpos[2] < self.min_height:
            return True
        tilt = float(np.linalg.norm(gravity_body(self.data)[:2]))
        return tilt > self.max_tilt


def gravity_z(data) -> float:
    from .estimation import gravity_body

    return float(gravity_body(data)[2])


def walker_actions(env, controller, sim_time: float) -> np.ndarray:
    """Convert the analytic walker's joint targets into an env action.

    Proof that the environment is expressive enough to contain the solution we
    already have: the week 4 controller, expressed as a policy this env accepts.
    """
    target = controller.control(sim_time)          # actuator-space command
    full = np.array(controller.nominal, copy=True)
    full[env.act_qpos] = target
    residual = (full[env.leg_qpos] - env.nominal[env.leg_qpos]) / env.action_scale
    return np.clip(residual, -1.0, 1.0).astype(np.float32)
