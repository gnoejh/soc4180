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

__all__ = ["DEFAULT_REWARD", "G1PushEnv", "G1WalkEnv", "NOMINAL_WORLD", "RANDOM_WORLD", "walker_actions"]

#: Reward term weights. Week 9 removes these one at a time.
DEFAULT_REWARD = {
    "tracking": 1.5,   # match the commanded forward velocity -- the actual goal
    "upright": 0.5,    # shaping: stay vertical
    "effort": -0.01,   # penalise large actions
    "smooth": -0.05,   # penalise jerky changes between actions
    "alive": 0.5,      # constant bonus per surviving step
    "stand_still": 0.0,  # penalty for not moving when asked to (playground: -1.0)
    "air_time": 0.0,     # reward for having a foot off the ground (playground: +2.0)
}

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
        reward_weights: dict | None = None,
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

        self.reward_weights = dict(DEFAULT_REWARD)
        if reward_weights:
            unknown = set(reward_weights) - set(DEFAULT_REWARD)
            if unknown:
                raise ValueError(f"unknown reward terms: {sorted(unknown)}")
            self.reward_weights.update(reward_weights)
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
        terms = {
            "tracking": np.exp(-((forward - self.target_velocity) ** 2) / 0.25),
            "upright": float(-gravity_z(self.data)),
            "effort": float(np.sum(np.square(action))),
            "smooth": float(np.sum(np.square(action - self._prev_action))),
            "alive": 1.0,
            "stand_still": 1.0 if abs(forward) < 0.1 else 0.0,
            "air_time": float(self._feet_airborne()),
        }
        w = self.reward_weights
        reward = float(sum(w[k] * v for k, v in terms.items()))
        info = {"forward_velocity": forward, **terms}
        return reward, info

    def _feet_airborne(self) -> int:
        """How many feet are clear of the ground right now (0, 1 or 2).

        Rewarding this is how real locomotion rewards stop a policy from
        settling into a shuffle: taking a step has to be worth something.
        """
        from .kinematics import foot_site_id

        if not hasattr(self, "_foot_sites"):
            self._foot_sites = [foot_site_id(self.model, s) for s in ("left", "right")]
        return sum(
            1 for sid in self._foot_sites
            if self.data.site_xpos[sid][2] > 0.033 + 0.02
        )

    def _fallen(self) -> bool:
        from .estimation import gravity_body

        if self.data.qpos[2] < self.min_height:
            return True
        tilt = float(np.linalg.norm(gravity_body(self.data)[:2]))
        return tilt > self.max_tilt


#: What G1PushEnv can randomise, name -> (low, high) of a uniform draw per episode.
#: (1, 1) and (0, 0) mean "fixed at nominal".
NOMINAL_WORLD = {
    "friction": (1.0, 1.0),     # scale on every geom's sliding friction
    "mass": (0.0, 0.0),         # kilograms added to the torso
    "kp": (1.0, 1.0),           # scale on every servo's stiffness (and matching damping)
}
RANDOM_WORLD = {"friction": (0.3, 1.2), "mass": (-5.0, 10.0), "kp": (0.7, 1.3)}


class G1PushEnv(G1WalkEnv):
    """Stand in the crouch and survive a shove. Weeks 12 and 13.

    Same observation, action and reward as ``G1WalkEnv`` (with the target
    velocity at zero, so standing still is the goal), but every episode:

    - a push of a random size up to ``push_max`` newtons, in a random
      horizontal direction, hits the torso for ``push_seconds`` at a random
      moment inside ``push_window``;
    - the world is drawn from ``world`` -- friction, added torso mass and
      servo stiffness, each uniform in its range (``NOMINAL_WORLD`` fixes them,
      ``RANDOM_WORLD`` is week 12's domain randomisation);
    - with ``privileged=True`` the observation gains nine numbers no real
      robot can measure: the pelvis's linear velocity (3) and height (1), the
      push force being applied now (2), and this episode's friction, mass and
      stiffness (3). That is a *teacher's* observation (week 13).

    Measured (week 12, `robust.py`, sideways shoves, 10 episodes): holding
    the crouch survives 60 N; PPO trained 3M steps in the nominal world 80 N;
    on randomised worlds, 100 N -- and 120 N with a torso 15 kg heavier,
    where holding the crouch falls with no shove at all.
    """

    PRIVILEGED = 9

    def __init__(self, *, push_max: float = 200.0, push_window=(1.0, 3.0), push_seconds: float = 0.2,
                 world: dict | None = None, privileged: bool = False, episode_seconds: float = 5.0,
                 control_hz: float = 50.0, action_scale: float = 0.3, reward_weights: dict | None = None):
        super().__init__(target_velocity=0.0, control_hz=control_hz, action_scale=action_scale,
                         episode_seconds=episode_seconds, reward_weights=reward_weights)
        self.world = dict(NOMINAL_WORLD)
        if world:
            unknown = set(world) - set(NOMINAL_WORLD)
            if unknown:
                raise ValueError(f"unknown world parameters: {sorted(unknown)}")
            self.world.update(world)
        self.push_max, self.push_window, self.push_seconds = push_max, push_window, push_seconds
        self.privileged = privileged
        self.torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self._friction0 = self.model.geom_friction[:, 0].copy()
        self._mass0 = float(self.model.body_mass[self.torso])
        self._gain0 = self.model.actuator_gainprm[:, 0].copy()
        self._bias0 = self.model.actuator_biasprm[:, 1:3].copy()
        n = self.observation_space.shape[0] + (self.PRIVILEGED if privileged else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, (n,), dtype=np.float32)
        self.drawn = {k: v[0] for k, v in self.world.items()}
        self.push_force = np.zeros(2)
        self.push_time = 0.0

    def set_world(self, **values):
        """Apply one world: friction scale, added torso mass (kg), servo stiffness scale."""
        self.drawn.update(values)
        self.model.geom_friction[:, 0] = self._friction0 * self.drawn["friction"]
        self.model.body_mass[self.torso] = self._mass0 + self.drawn["mass"]
        k = self.drawn["kp"]
        self.model.actuator_gainprm[:, 0] = self._gain0 * k
        self.model.actuator_biasprm[:, 1] = self._bias0[:, 0] * k
        self.model.actuator_biasprm[:, 2] = self._bias0[:, 1] * np.sqrt(k)   # keep the damping ratio

    def reset(self, *, seed=None, options=None):
        _BASE.reset(self, seed=seed)
        rng = self.np_random
        self.set_world(**{k: float(rng.uniform(*v)) for k, v in self.world.items()})
        angle = rng.uniform(0.0, 2.0 * np.pi)
        size = rng.uniform(0.0, self.push_max) if self.push_max else 0.0
        self.push_force = size * np.array([np.cos(angle), np.sin(angle)])
        self.push_time = float(rng.uniform(*self.push_window))
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.nominal
        self.data.qpos[2] = 0.72
        self.data.qpos[self.leg_qpos] += rng.uniform(-0.02, 0.02, len(self.leg_qpos))
        mujoco.mj_forward(self.model, self.data)
        self._prev_action[:] = 0.0
        self._step = 0
        return self._observation(), {}

    def step(self, action):
        # The push is on for whole decisions: at 50 Hz, 0.2 s is ten of them.
        t = self.data.time
        on = self.push_time <= t < self.push_time + self.push_seconds
        self.data.xfrc_applied[self.torso, :2] = self.push_force if on else 0.0
        return super().step(action)

    def _observation(self) -> np.ndarray:
        obs = super()._observation()
        if not getattr(self, "privileged", False):
            return obs
        t = self.data.time
        on = self.push_time <= t < self.push_time + self.push_seconds
        R = self.data.xmat[1].reshape(3, 3)                    # pelvis orientation
        extra = np.concatenate([
            R.T @ self.data.qvel[0:3],                          # pelvis velocity, body frame
            [self.data.qpos[2]],                                # height
            self.push_force / 100.0 if on else np.zeros(2),     # the shove, in 100 N
            [self.drawn["friction"], self.drawn["mass"] / 10.0, self.drawn["kp"]],
        ])
        return np.concatenate([obs, extra]).astype(np.float32)


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
