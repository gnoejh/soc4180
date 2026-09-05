"""Rendering helpers that behave identically on a local machine and in Colab.

The GL backend must be chosen *before* ``mujoco`` is imported, so this module
selects it at import time. Import ``soc4180.render`` before ``mujoco`` and the
right backend is picked automatically.
"""

from __future__ import annotations

import os
import sys

__all__ = [
    "FFMPEG_PATH",
    "GL_BACKEND",
    "is_colab",
    "render_rollout",
    "save_video",
    "show_video",
]


# Backend selection lives in _gl, imported before any mujoco import.
from ._gl import GL_BACKEND, GL_UNAVAILABLE, is_colab  # noqa: F401

import mediapy as media  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402


def _resolve_ffmpeg() -> str:
    """Point mediapy at a usable ffmpeg.

    mediapy shells out to a system ffmpeg, which Colab has preinstalled and a
    typical Windows machine does not. imageio-ffmpeg ships a bundled binary, so
    fall back to that and video works the same on both.
    """
    import shutil

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg
    except ImportError:
        return "ffmpeg"  # let mediapy raise its own error at write time

    bundled = imageio_ffmpeg.get_ffmpeg_exe()
    media.set_ffmpeg(bundled)
    return bundled


FFMPEG_PATH = _resolve_ffmpeg()


def render_rollout(
    model: "mujoco.MjModel",
    data: "mujoco.MjData | None" = None,
    *,
    duration: float = 5.0,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    camera: int | str = -1,
    ctrl_fn=None,
    track: str | None = None,
    distance: float = 3.0,
    azimuth: float = 120.0,
    elevation: float = -15.0,
) -> list[np.ndarray]:
    """Step the model forward and return one RGB frame per video frame.

    ``ctrl_fn(model, data) -> None`` is called before each physics step, which is
    where a controller writes into ``data.ctrl``. With no ``ctrl_fn`` the robot
    simply falls under gravity, which is the honest week-1 starting point.

    ``track`` names a body the camera should follow. A walking robot leaves a
    fixed frame within a couple of seconds, so any locomotion video needs this.
    """
    if GL_UNAVAILABLE:
        raise RuntimeError(GL_UNAVAILABLE)

    if data is None:
        data = mujoco.MjData(model)

    cam = camera
    track_id = -1
    if track is not None:
        track_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, track)
        if track_id < 0:
            raise ValueError(f"no body named {track!r} to track")
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation

    frames: list[np.ndarray] = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        while data.time < duration:
            if ctrl_fn is not None:
                ctrl_fn(model, data)
            mujoco.mj_step(model, data)
            if len(frames) < data.time * fps:
                if track_id >= 0:
                    cam.lookat[:] = data.xpos[track_id]
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render())
    return frames


def show_video(frames, fps: int = 30):
    """Display frames inline in a notebook."""
    return media.show_video(frames, fps=fps)


def save_video(frames, path, fps: int = 30):
    """Write frames to a video file and return the path."""
    media.write_video(path, frames, fps=fps)
    return path
