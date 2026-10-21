"""What identifies this body. The brain folds `manifest_digest` into its cache
keys so a cell computed against one body is never served for another.

The digest covers: the simulator's name, the body package's own model files,
the robot package's files, the MuJoCo version, the menagerie package version
and the upstream menagerie commit it vendors, the robot and scene entry,
numpy's version. It deliberately excludes the platform string, which is
reported but must not move a key (a body is the same body on another machine
if every listed version matches). A second robot or a second simulator is a
different digest by construction.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
from pathlib import Path

import mujoco
import mujoco_menagerie as mm
import numpy as np

import robots
from body import __version__ as BODY_VERSION
from body.robot import RobotSpec
from body.sim import SIMULATOR

PACKAGE_DIR = Path(__file__).resolve().parent

# Every file whose bytes can change what a tick returns. Append-only, as in the
# engine's MODEL_FILES: reordering would move every digest.
MODEL_FILES = (
    "manifest.py",
    "sim.py",
    "sensors.py",
    "primitives.py",
    "protocol.py",
    "bridge.py",
    "robot.py",
)

# What the digest is keyed on. `platform` is reported but excluded on purpose.
KEYED = (
    "body_version",
    "simulator",
    "package_digest",
    "robot",
    "robot_digest",
    "mujoco",
    "menagerie_version",
    "menagerie_commit",
    "menagerie_robot",
    "entry",
    "numpy",
    "gallery",
)


def package_digest() -> str:
    """sha256 over MODEL_FILES concatenated in order, first 12 hex characters."""
    h = hashlib.sha256()
    for name in MODEL_FILES:
        h.update(name.encode())
        h.update(b"\0")
        h.update((PACKAGE_DIR / name).read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:12]


# The robot package's files whose bytes can change a tick: its code and its data
# (from 2026-10-12, the G1's frozen locomotion policy weights under policy/).
ROBOT_SUFFIXES = (".py", ".npz", ".json")


def robot_files(spec: RobotSpec) -> tuple[Path, ...]:
    """The robot package's code and data files, sorted by path relative to the
    package. Caches are excluded."""
    pkg = importlib.import_module(spec.package)
    root = Path(pkg.__file__).resolve().parent
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix in ROBOT_SUFFIXES
             and "__pycache__" not in p.parts]
    return tuple(sorted(files, key=lambda p: p.relative_to(root).as_posix()))


def robot_digest(spec: RobotSpec) -> str:
    """sha256 over the robot package's files, first 12 hex characters."""
    h = hashlib.sha256()
    root = Path(importlib.import_module(spec.package).__file__).resolve().parent
    for p in robot_files(spec):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:12]


def gallery_digest(images) -> str:
    """The gallery scene's identity: sha256 over its images' bytes in order, or
    "none" for the default scene. What the head camera sees is part of the body."""
    if not images:
        return "none"
    h = hashlib.sha256()
    for f in images:
        h.update(Path(f).read_bytes())
        h.update(b"\x00")
    return h.hexdigest()[:12]


def manifest(robot: str | RobotSpec = "g1", entry: str | None = None, gallery=None) -> dict:
    """The body's identity as a plain dict (JSON-serialisable)."""
    spec = robots.get(robot)
    return {
        "body_version": BODY_VERSION,
        "simulator": SIMULATOR,
        "package_digest": package_digest(),
        "robot": spec.name,
        "robot_digest": robot_digest(spec),
        "mujoco": mujoco.__version__,
        "menagerie_version": mm.__version__,
        "menagerie_commit": mm.commit(),
        "menagerie_robot": spec.menagerie,
        "entry": entry or spec.entry,
        "numpy": np.__version__,
        "gallery": gallery_digest(gallery),
        "platform": platform.platform(),
    }


def manifest_digest(m: dict) -> str:
    """sha256 over the KEYED fields in canonical JSON, first 12 hex characters."""
    keyed = {k: m[k] for k in KEYED}
    payload = json.dumps(keyed, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="print this body's manifest and digest")
    ap.add_argument("--robot", default="g1", choices=robots.ROBOTS)
    a = ap.parse_args(argv)
    m = manifest(a.robot)
    print(json.dumps(m, indent=2))
    print("digest:", manifest_digest(m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
