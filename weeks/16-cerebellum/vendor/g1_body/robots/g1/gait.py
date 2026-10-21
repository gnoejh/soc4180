"""The G1's walking pattern generator: a frozen, trained locomotion policy. Model file.

    walk(vx, vy, wz)  ->  29 joint targets per tick, plus the leg servo gains

The engine's constitution admits backprop-trained parts only as frozen ends, as with
the MiniLM/CLIP encoders at the front. This is the same rule at the back: the spinal
and brainstem pattern generator of the body, trained once by someone else and never
updated here. The cerebellum's job (Phase XI.1-XI.3) is to calibrate the gait this
policy produces, not to replace it.

**Provenance.** Unitree Robotics, `unitree_rl_gym`, commit 276801e (2025-07-25),
`deploy/pre_train/g1/motion.pt`, BSD-3-Clause; sha256 in `policy/PROVENANCE.json`.
The TorchScript file was read once in a throwaway environment, and its weights were
written to `policy/unitree_g1_motion.npz`. The network is an LSTM (47 -> 64)
followed by an MLP (64 -> 32, ELU, -> 12), evaluated here in numpy. The .npz also
holds 50 torch input/output pairs, so the parity test can check the evaluation
exactly.

**The interface it was trained on** (`deploy/deploy_mujoco/configs/g1.yaml`). Every
50 Hz tick it reads 47 numbers:
- the base angular velocity (x 0.25);
- gravity projected into the base frame;
- the command (vx, vy, wz), scaled by (2, 2, 0.25);
- the 12 leg joint positions minus their defaults;
- the 12 leg joint velocities (x 0.05);
- the last action;
- the sin and cos of a 0.8 s gait phase.

It emits 12 leg targets = default + 0.25 x action. The leg PD gains are its own
(kp 100/100/100/150/40/40, kd 2/2/2/4/2/2). Training welded the upper body at zero
joint angles, so the arms and waist are held at zero here.

**What is not privileged.** The angular velocity is the pelvis gyro and gravity
comes from the pelvis IMU's attitude. The joint states are the encoders. No base
position, no world state.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

__all__ = ["LocomotionPolicy", "PolicyGait", "POLICY_FILE", "LEG_KP", "LEG_KD", "DEFAULT_LEG"]

POLICY_FILE = Path(__file__).resolve().parent / "policy" / "unitree_g1_motion.npz"

LEG_KP = np.array([100, 100, 100, 150, 40, 40] * 2, dtype=float)
LEG_KD = np.array([2, 2, 2, 4, 2, 2] * 2, dtype=float)
DEFAULT_LEG = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2, dtype=float)
ANG_VEL_SCALE = 0.25
DOF_VEL_SCALE = 0.05
ACTION_SCALE = 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.25])
PERIOD = 0.8
DT = 0.02


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class LocomotionPolicy:
    """The frozen network, in float32 as it was trained. `step(obs)` advances the
    LSTM state and returns the 12 actions."""

    def __init__(self, path: Path = POLICY_FILE):
        z = np.load(path)
        self.W_ih, self.W_hh = z["memory_weight_ih_l0"], z["memory_weight_hh_l0"]
        self.b = z["memory_bias_ih_l0"] + z["memory_bias_hh_l0"]
        self.A0, self.a0 = z["actor_0_weight"], z["actor_0_bias"]
        self.A2, self.a2 = z["actor_2_weight"], z["actor_2_bias"]
        self.parity = (z["parity_x"], z["parity_y"])
        self.reset()

    def reset(self) -> None:
        self.h = np.zeros(64, dtype=np.float32)
        self.c = np.zeros(64, dtype=np.float32)

    def step(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        g = self.W_ih @ x + self.W_hh @ self.h + self.b
        i, f, gg, o = np.split(g, 4)                   # torch's gate order
        self.c = _sigmoid(f) * self.c + _sigmoid(i) * np.tanh(gg)
        self.h = _sigmoid(o) * np.tanh(self.c)
        y = self.A0 @ self.h + self.a0
        y = np.where(y > 0, y, np.expm1(y))            # ELU
        return self.A2 @ y + self.a2


def gravity_in_base(R_base: np.ndarray) -> np.ndarray:
    """World gravity direction (0, 0, -1) expressed in the base frame."""
    return -R_base[2, :]


class PolicyGait:
    """The stateful gait for one sim. `targets(cmd, sense)` returns 29 joint
    targets. `engage(model)` puts the policy's PD gains on the leg servos and
    `release(model)` restores the body's own."""

    def __init__(self, leg_act: np.ndarray, nu: int, policy: LocomotionPolicy | None = None):
        self.policy = policy or LocomotionPolicy()
        self.leg_act = np.asarray(leg_act)
        self.nu = nu
        self._saved = None
        self.reset()

    def reset(self) -> None:
        self.policy.reset()
        self.action = np.zeros(12, dtype=np.float32)
        self.k = 0

    def engage(self, model: mujoco.MjModel) -> None:
        if self._saved is None:
            self._saved = (model.actuator_gainprm[self.leg_act, 0].copy(),
                           model.actuator_biasprm[self.leg_act, 1:3].copy())
        model.actuator_gainprm[self.leg_act, 0] = LEG_KP
        model.actuator_biasprm[self.leg_act, 1] = -LEG_KP
        model.actuator_biasprm[self.leg_act, 2] = -LEG_KD

    def release(self, model: mujoco.MjModel) -> None:
        if self._saved is not None:
            model.actuator_gainprm[self.leg_act, 0] = self._saved[0]
            model.actuator_biasprm[self.leg_act, 1:3] = self._saved[1]
            self._saved = None

    def observe(self, cmd, gyro: np.ndarray, R_base: np.ndarray,
                q_leg: np.ndarray, dq_leg: np.ndarray) -> np.ndarray:
        t = self.k * DT
        phase = (t % PERIOD) / PERIOD
        return np.concatenate([
            np.asarray(gyro) * ANG_VEL_SCALE,
            gravity_in_base(R_base),
            np.asarray(cmd, dtype=float) * CMD_SCALE,
            np.asarray(q_leg) - DEFAULT_LEG,
            np.asarray(dq_leg) * DOF_VEL_SCALE,
            self.action,
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
        ]).astype(np.float32)

    def targets(self, cmd, gyro, R_base, q_leg, dq_leg) -> np.ndarray:
        """One tick. The caller passes the sensed state after the previous tick."""
        self.k += 1
        self.action = self.policy.step(self.observe(cmd, gyro, R_base, q_leg, dq_leg)).astype(np.float32)
        target = np.zeros(self.nu)
        target[self.leg_act] = DEFAULT_LEG + ACTION_SCALE * self.action
        return target
