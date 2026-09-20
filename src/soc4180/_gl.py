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

__all__ = ["GL_BACKEND", "GL_UNAVAILABLE", "MUJOCO_WAS_PREIMPORTED", "TRITON_BLOCKED",
           "gl_report", "is_colab"]

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


def _osmesa_available() -> bool:
    """Whether software rendering is actually installed."""
    import ctypes.util

    return ctypes.util.find_library("OSMesa") is not None


def _block_triton() -> None:
    """Make ``import triton`` fail cleanly, because OSMesa and triton cannot
    share a process.

    Measured on the GitHub Actions runner and reproduced under WSL (Mesa 25.1,
    torch 2.14, triton 3.8): once libOSMesa is loaded — which ``import mujoco``
    does at import time when ``MUJOCO_GL=osmesa`` — loading triton's native
    library segfaults. torch imports triton lazily, the first time an optimizer
    is constructed, so the kernel died at the first *training* cell of weeks 8
    and 9 and nowhere else; week 10 imports torch and survived because it never
    trains. The reverse order (triton first, then OSMesa) works, and so does
    blocking triton: torch treats a failed ``import triton`` as "not installed"
    and runs eagerly. Software rendering means there is no GPU, so nothing is
    lost by doing that here.
    """
    global TRITON_BLOCKED
    if "triton" in sys.modules:
        return  # already loaded (the working order) or already blocked
    sys.modules["triton"] = None  # type: ignore[assignment]
    TRITON_BLOCKED = True


def _select_backend() -> str:
    """Choose a MuJoCo GL backend for this machine and export ``MUJOCO_GL``.

    - An explicit ``MUJOCO_GL`` always wins.
    - Colab / headless Linux with an NVIDIA GPU -> ``egl``.
    - Colab / headless Linux without a GPU -> ``osmesa`` (software rendering).
    - Windows and macOS -> MuJoCo's default, which renders offscreen fine.

    Whenever the answer is ``osmesa``, chosen or explicit, triton is blocked
    (see ``_block_triton``).
    """
    explicit = os.environ.get("MUJOCO_GL")
    if explicit:
        if explicit.strip().lower() == "osmesa":
            _block_triton()
        return explicit

    headless_linux = sys.platform.startswith("linux") and not os.environ.get("DISPLAY")
    if not (is_colab() or headless_linux):
        return "default"

    if _has_nvidia_gpu():
        _ensure_egl_vendor_config()
        backend = "egl"
    elif _osmesa_available():
        backend = "osmesa"
    else:
        # No GPU and no software renderer. Setting MUJOCO_GL here would poison
        # `import OpenGL` itself — PyOpenGL fails at import with a bare
        # AttributeError when its platform cannot load, which is far harder to
        # read than the message we raise from render_rollout instead. So leave
        # the environment alone: physics still works, only rendering cannot.
        global GL_UNAVAILABLE
        GL_UNAVAILABLE = (
            "No GPU and no software renderer (libOSMesa) are available, so "
            "MuJoCo cannot render here. "
            + (
                "On Colab: Runtime > Change runtime type > T4 GPU, then "
                "Runtime > Restart session."
                if is_colab()
                else "Install libosmesa6, or set MUJOCO_GL yourself."
            )
        )
        return "unavailable"

    os.environ["MUJOCO_GL"] = backend
    os.environ.setdefault("PYOPENGL_PLATFORM", backend)
    if backend == "osmesa":
        _block_triton()

    if MUJOCO_WAS_PREIMPORTED:
        warnings.warn(
            f"mujoco was imported before soc4180, so MUJOCO_GL={backend!r} arrives "
            "too late and rendering will use the wrong backend. Restart the kernel "
            "and import soc4180 before mujoco.",
            RuntimeWarning,
            stacklevel=2,
        )

    return backend


# Set when no usable rendering backend exists; render_rollout raises with it.
GL_UNAVAILABLE: str | None = None

# True when this module made `import triton` fail on purpose (OSMesa only).
TRITON_BLOCKED = False

GL_BACKEND = _select_backend()


def gl_report() -> str:
    """What the backend selection saw. Paste this when rendering misbehaves."""
    lines = [
        f"platform            : {sys.platform}",
        f"colab               : {is_colab()}",
        f"NVIDIA device node  : {_has_nvidia_gpu()}",
        f"OSMesa library      : {_osmesa_available()}",
        f"DISPLAY             : {os.environ.get('DISPLAY') or '(unset)'}",
        f"MUJOCO_GL           : {os.environ.get('MUJOCO_GL') or '(unset)'}",
        f"chosen backend      : {GL_BACKEND}",
        f"mujoco preimported  : {MUJOCO_WAS_PREIMPORTED}",
        f"triton blocked      : {TRITON_BLOCKED}",
        f"rendering available : {GL_UNAVAILABLE is None}",
    ]
    if GL_UNAVAILABLE:
        lines.append(f"reason              : {GL_UNAVAILABLE}")
    return "\n".join(lines)
