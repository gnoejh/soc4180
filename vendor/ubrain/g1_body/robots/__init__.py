"""The robots this simulator can embody: one package per robot, each exporting
`SPEC`, a `body.robot.RobotSpec`. `get("g1")` resolves a name; a spec passes
through. Adding a robot is adding a package here and its name to ROBOTS --
`body/` does not change.

The menagerie pinned in uv.lock carries 68 models (humanoids: unitree_h1,
booster_t1, fourier_n1, apptronik_apollo, berkeley_humanoid, pal_talos,
robotis_op3; quadrupeds: unitree_go1/go2/a1, boston_dynamics_spot,
anybotics_anymal_b/c; arms and hands: franka_fr3, universal_robots_ur5e,
kinova_gen3, unitree_z1, shadow_hand, wonik_allegro, leap_hand; mobile
manipulators: hello_robot_stretch_3, pal_tiago, stanford_tidybot). Each is a
`RobotSpec` away.
"""

from __future__ import annotations

import importlib

from body.robot import RobotSpec

__all__ = ["ROBOTS", "get"]

ROBOTS = ("g1",)


def get(robot: str | RobotSpec) -> RobotSpec:
    if isinstance(robot, RobotSpec):
        return robot
    if robot not in ROBOTS:
        raise KeyError(f"unknown robot {robot!r}; have {ROBOTS}")
    return importlib.import_module(f"robots.{robot}").SPEC
