"""Shared helpers for the Robot and AI course.

Importing this package selects a working MuJoCo GL backend for the current
machine, so labs behave the same locally and in Colab.
"""

# MUST be first: selects MUJOCO_GL before any submodule imports mujoco.
from ._gl import GL_BACKEND, is_colab
from .models import (
    MENAGERIE_COMMIT,
    MENAGERIE_VERSION,
    by_category,
    humanoids,
    license_of,
    load_g1,
    load_robot,
    robot_path,
)
from .render import render_rollout, save_video, show_video
from .seeding import set_seed
from .walking import GaitParams, LIPM, WalkingController, footstep_plan
from .sim import actuation_disabled, hold, keyframe_data, keyframe_names

__version__ = "0.1.0"

__all__ = [
    "GL_BACKEND",
    "MENAGERIE_COMMIT",
    "actuation_disabled",
    "MENAGERIE_VERSION",
    "__version__",
    "by_category",
    "humanoids",
    "license_of",
    "load_g1",
    "load_robot",
    "robot_path",
    "is_colab",
    "render_rollout",
    "save_video",
    "hold",
    "keyframe_data",
    "keyframe_names",
    "set_seed",
    "GaitParams",
    "LIPM",
    "WalkingController",
    "footstep_plan",
    "show_video",
]
