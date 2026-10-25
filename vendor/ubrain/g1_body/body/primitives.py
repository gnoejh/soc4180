"""The motor codebook's controller: an index and parameters in, joint targets
out. Model file.

This is the body's FROZEN BACK-END, symmetrical to the frozen front-end encoders:
the brain emits a primitive index (and its parameters), never a torque, exactly
as it receives a code vector and never a pixel. Nothing here learns.

The codebook itself belongs to the robot (`RobotSpec.primitives` and
`RobotSpec.targets`, see `robots/g1`); this module validates, resolves and
clips, and knows no robot's joint names.
"""

from __future__ import annotations

import mujoco
import numpy as np

from body.robot import Primitive, RobotSpec
from body.sim import MujocoSim

__all__ = ["Primitive", "Controller", "by_name"]


def by_name(name: str, spec: RobotSpec) -> Primitive:
    return spec.primitive(name)


class Controller:
    """Turns (primitive, params, tick) into joint targets for `MujocoSim.step`."""

    def __init__(self, sim: MujocoSim):
        self.sim = sim
        self.spec = sim.spec
        self.home = sim.home_ctrl
        m = sim.model
        lim = m.actuator_ctrllimited.astype(bool)
        self.lo = np.where(lim, m.actuator_ctrlrange[:, 0], -np.inf)
        self.hi = np.where(lim, m.actuator_ctrlrange[:, 1], np.inf)
        self._actuators: dict[str, int] = {}

    def actuator(self, name: str) -> int:
        """Actuator index by name, cached; raises if the model lacks it."""
        if name not in self._actuators:
            aid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise RuntimeError(f"model lacks actuator {name!r}")
            self._actuators[name] = int(aid)
        return self._actuators[name]

    def ctrl(self, primitive: int, params, tick: int) -> np.ndarray:
        prims = self.spec.primitives
        if not 0 <= primitive < len(prims):
            raise ValueError(f"unknown primitive index {primitive}")
        p = prims[primitive]
        params = np.asarray(params, dtype=float).reshape(-1)
        if params.shape != (p.n_params,):
            raise ValueError(f"{p.name} takes {p.n_params} params, got {params.shape[0]}")
        target = np.asarray(self.spec.targets(self, p, params, tick), dtype=float)
        if target.shape != (self.sim.nu,):
            raise RuntimeError(f"{self.spec.name}.targets returned shape {target.shape}, expected ({self.sim.nu},)")
        return np.clip(target, self.lo, self.hi)
