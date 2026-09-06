"""Robot models, sourced from MuJoCo Menagerie.

Menagerie is consumed as a pinned pip package rather than a git checkout: the
wheel is ~28 KB and downloads model assets lazily on first use, which keeps a
Colab session fast and needs no git. The package pins its own upstream commit,
so pinning ``mujoco-menagerie`` in ``pyproject.toml`` pins the robot models too.
"""

from __future__ import annotations

import functools

import mujoco
import mujoco_menagerie as mm

__all__ = [
    "MENAGERIE_COMMIT",
    "MENAGERIE_VERSION",
    "humanoids",
    "license_of",
    "load_g1",
    "load_robot",
    "robot_path",
]

MENAGERIE_VERSION = mm.__version__
# The upstream mujoco_menagerie commit this package version vendors.
MENAGERIE_COMMIT = mm.commit()


def load_robot(name: str = "unitree_g1", entry: str = "scene") -> mujoco.MjModel:
    """Load a Menagerie robot as a compiled ``MjModel``.

    ``entry`` selects one of the robot's entry points. ``"scene"`` includes a
    floor and lighting and is what you want for rendering; the bare robot entry
    (e.g. ``"g1"``) has no ground plane.
    """
    robot = mm.get(name)
    known = {e.name for e in robot.entry_points}
    if entry not in known:
        raise ValueError(
            f"'{entry}' is not an entry point of '{name}'. Available: {sorted(known)}"
        )
    return robot.model(entry)


def robot_path(name: str = "unitree_g1", entry: str = "scene"):
    """Filesystem path to a robot's XML, for when you need the file itself.

    Use ``.xml(entry)``; ``Robot.path()`` takes a cache and returns the robot's
    *directory*, not a named entry.
    """
    return mm.get(name).xml(entry)


def load_g1(*, mjx: bool = False, with_hands: bool = False) -> mujoco.MjModel:
    """Load the Unitree G1 humanoid — the course robot.

    ``mjx=True`` selects the MJX-optimised scene used for GPU training on Colab.
    The default is the standard scene used for the classical-robotics weeks.
    """
    if mjx and with_hands:
        raise ValueError("No combined MJX + hands scene exists for the G1.")
    entry = "scene_mjx" if mjx else "scene_with_hands" if with_hands else "scene"
    return load_robot("unitree_g1", entry)


@functools.cache
def by_category(category: str = "humanoid") -> tuple[str, ...]:
    """Names of every Menagerie robot in a category, for picking a fallback.

    Categories include ``humanoid`` (11 models), ``biped`` (Cassie), and
    ``quadruped``. Cassie is filed as ``biped``, not ``humanoid``.
    """
    robots = mm.bundled().robots
    return tuple(sorted(r.name for r in robots.values() if r.category == category))


def humanoids() -> tuple[str, ...]:
    """Names of every humanoid in Menagerie."""
    return by_category("humanoid")


def license_of(name: str = "unitree_g1") -> str:
    """The model's license — worth citing on a slide."""
    return mm.get(name).license
