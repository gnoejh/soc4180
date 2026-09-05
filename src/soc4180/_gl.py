"""GL backend selection. **This module must be imported before ``mujoco``.**

MuJoCo resolves its rendering backend from ``MUJOCO_GL`` at ``import mujoco``
time. Setting the variable afterwards has no effect — the process is stuck with
whatever was chosen, and on a headless machine that means GLFW failing to find
an X11 display.

This module therefore imports nothing from mujoco, and ``soc4180/__init__.py``
imports it first, before any submodule that touches mujoco.
"""

from __future__ import annotations

import os
import pathlib
import sys
import warnings

__all__ = ["GL_BACKEND", "MUJOCO_WAS_PREIMPORTED", "is_colab"]

# Recorded before we touch anything: if mujoco is already in sys.modules then
# our backend choice arrives too late to matter, and we say so loudly.
MUJOCO_WAS_PREIMPORTED = "mujoco" in sys.modules

_EGL_VENDOR_CONFIG = pathlib.Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
_EGL_VENDOR_JSON = """{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
"""


def is_colab() -> bool:
    """True when running inside a Google Colab runtime."""
    return "google.colab" in sys.modules or os.path.isdir("/content")


def _has_nvidia_gpu() -> bool:
    """True when an NVIDIA device node is present (no subprocess needed)."""
    return (
        os.path.exists("/dev/nvidiactl")
        or os.path.exists("/dev/nvidia0")
        or os.path.exists("/proc/driver/nvidia/version")
    )


def _ensure_egl_vendor_config() -> None:
    """Write the NVIDIA EGL ICD file if it is missing.

    Colab images frequently ship without it, and EGL then fails to initialise —
    often silently, producing black frames rather than an error. MuJoCo's own
    Colab notebooks write this file for the same reason. Best-effort: a
    read-only filesystem or lack of permission is not fatal.
    """
    if _EGL_VENDOR_CONFIG.exists():
        return
    try:
        _EGL_VENDOR_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        _EGL_VENDOR_CONFIG.write_text(_EGL_VENDOR_JSON, encoding="utf-8")
    except OSError:
        pass


def _select_backend() -> str:
    """Choose a MuJoCo GL backend for this machine and export ``MUJOCO_GL``.

    - An explicit ``MUJOCO_GL`` always wins.
    - Colab / headless Linux with an NVIDIA GPU -> ``egl``.
    - Colab / headless Linux without a GPU -> ``osmesa`` (software rendering).
    - Windows and macOS -> MuJoCo's default, which renders offscreen fine.
    """
    explicit = os.environ.get("MUJOCO_GL")
    if explicit:
        return explicit

    headless_linux = sys.platform.startswith("linux") and not os.environ.get("DISPLAY")
    if not (is_colab() or headless_linux):
        return "default"

    if _has_nvidia_gpu():
        _ensure_egl_vendor_config()
        backend = "egl"
    else:
        # No GPU: software rendering. Needs libosmesa6 present in the image.
        backend = "osmesa"

    os.environ["MUJOCO_GL"] = backend
    os.environ.setdefault("PYOPENGL_PLATFORM", backend)

    if MUJOCO_WAS_PREIMPORTED:
        warnings.warn(
            f"mujoco was imported before soc4180, so MUJOCO_GL={backend!r} arrives "
            "too late and rendering will use the wrong backend. Restart the kernel "
            "and import soc4180 before mujoco.",
            RuntimeWarning,
            stacklevel=2,
        )

    return backend


GL_BACKEND = _select_backend()
