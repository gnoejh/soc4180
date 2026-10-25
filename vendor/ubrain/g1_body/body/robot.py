"""RobotSpec: what makes one robot a body in this simulator. Model file.

The simulator (`body/`) is generic over the robot; everything that names a
model, a keyframe, a site, a sensor, a joint count or a primitive lives in one
`RobotSpec`, built by the robot's own package under `robots/` (`robots/g1`
first). `MujocoSim`, the sensors, the controller and the bridge take a spec and
never spell a robot's name themselves. A second robot is a second package under
`robots/` and nothing else; a second simulator (`robot_isaac/`) is a sibling
folder implementing the same contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = ["Primitive", "RobotSpec"]


@dataclass(frozen=True)
class Primitive:
    """One entry of the motor codebook: an index and a parameter count in,
    joint targets out. The robot's spec owns the list and the targets."""
    index: int
    name: str
    n_params: int
    doc: str


@dataclass(frozen=True)
class RobotSpec:
    name: str                       # the short id this folder uses: "g1"
    menagerie: str                  # the menagerie model id: "unitree_g1"
    entry: str                      # the scene entry: "scene"
    keyframe: str                   # the home keyframe the body resets into
    n_joints: int                   # actuated joints (position actuators)
    imu_site: str                   # the site whose frame reads "up"
    imu_sensors: tuple[tuple[str, str, int], ...]   # (field, model sensor name, dim)
    foot_sites: tuple[str, ...]     # for the (privileged) contact forces
    head_camera_body: str           # where the forward camera is attached
    head_camera_pos: tuple[float, float, float]
    head_camera_fovy: float
    primitives: tuple[Primitive, ...]
    # (controller, primitive, params, tick) -> joint targets BEFORE clipping
    targets: Callable[..., np.ndarray]
    package: str                    # the robot package, "robots.g1", for the digest

    def primitive(self, name: str) -> Primitive:
        for p in self.primitives:
            if p.name == name:
                return p
        raise KeyError(name)

    @property
    def param_counts(self) -> list[int]:
        return [p.n_params for p in self.primitives]
