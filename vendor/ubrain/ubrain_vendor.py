"""The course's one copy of ubrain, run inside this repository's environment.

ubrain is not public yet and not on PyPI, so it is not in uv.lock; this folder carries
a local copy of its wheels, shared by every Part II week:

    vendor/ubrain/wheels/ubrain-0.14.0.dev1-cp312-*.whl   (the brain, compiled)
    vendor/ubrain/g1_body/body, robots                    (the G1 body server)

Each week imports it through a three-line `ubrain_local.py` beside its scripts. Two
things this module does, so that `uv run weeks/NN-*/lab_*.py` needs nothing installed
beyond `uv sync`:

1. `load()` unpacks the one wheel that matches this laptop (a wheel is a zip of the
   package) into vendor/.unpacked/ once, and puts it on sys.path.
2. `start_body()` starts the body server with *this* Python. ubrain's own launcher
   builds a second environment for the body with `uv run --project`; it is not needed
   here, because the body pins exactly what uv.lock already installs (mujoco 3.12.0,
   numpy 2.5.2, mujoco-menagerie 2026.9.0).
"""

from __future__ import annotations

import platform
import subprocess
import sys
import zipfile
from pathlib import Path

VENDOR = Path(__file__).resolve().parent
WHEELS = VENDOR / "wheels"
BODY = VENDOR / "g1_body"

# the platform part of each wheel's file name
_TAGS = {
    ("win32", "AMD64"): "win_amd64",
    ("darwin", "arm64"): "macosx_11_0_arm64",
    ("darwin", "x86_64"): "macosx_10_13_x86_64",
    ("linux", "x86_64"): "manylinux",
}


def _missing(what: str) -> SystemExit:
    return SystemExit(
        f"{what} not found under {VENDOR}.\n"
        "Pull the repository again: vendor/ubrain/ should hold wheels/ and g1_body/.")


def load() -> None:
    """Make `import ubrain` work: unpack the matching wheel once, add it to sys.path."""
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"ubrain's wheels here are built for Python 3.12; this is "
                         f"{platform.python_version()}. Run it with `uv run` from the repository.")
    tag = _TAGS.get((sys.platform, platform.machine()))
    if tag is None:
        raise SystemExit(f"no ubrain wheel for {sys.platform} {platform.machine()}; "
                         "tell the instructor")
    wheels = sorted(WHEELS.glob(f"ubrain-*-cp312-cp312-{tag}*.whl"))
    if not wheels:
        raise _missing(f"the ubrain wheel for {tag}")
    wheel = wheels[-1]
    target = VENDOR / ".unpacked" / wheel.stem
    done = target / "UNPACKED"
    if not done.is_file():
        with zipfile.ZipFile(wheel) as z:
            z.extractall(target)
        done.write_text(wheel.name, encoding="utf-8")
    sys.path.insert(0, str(target))


def start_body(port: int | None = None, *, viewer: bool = False, echo: bool = True,
               head_camera: bool = False, gallery: bool = False, record_chase: str | None = None,
               chase_every: int = 2) -> tuple[subprocess.Popen, int]:
    """Start the G1 body server; returns (process, port) once it serves. Its stdout
    stays open: in the viewer, the body prints `KEY <code>` for every key press.
    head_camera adds the robot's head camera; gallery serves the gallery scene (21 CC0
    pictures along the walking line, g1_body/scenes/gallery_cc0); record_chase (a path
    without extension) records a camera following the robot, for videos."""
    if not (BODY / "body" / "bridge.py").is_file():
        raise _missing("the G1 body")
    from ubrain.bodies import free_port
    port = port or free_port()
    cmd = [sys.executable, "-m", "body.bridge", "--port", str(port)]
    if viewer:
        cmd.append("--viewer")
    if head_camera:
        cmd.append("--head-camera")
    if gallery:
        cmd += ["--gallery", str(BODY / "scenes" / "gallery_cc0")]
    if record_chase:
        cmd += ["--record-chase", str(record_chase), "--chase-every", str(int(chase_every))]
    proc = subprocess.Popen(cmd, cwd=BODY, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    seen = []
    for line in proc.stdout:
        if "serving on" in line:
            if echo:
                print(line.strip(), flush=True)
            return proc, port
        seen.append(line)
        if proc.poll() is not None:
            break
    raise RuntimeError("the body did not start; its output was:\n" + "".join(seen[-30:]))
