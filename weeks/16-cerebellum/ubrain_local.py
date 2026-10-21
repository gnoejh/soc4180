"""ubrain from vendor/, run inside this repository's environment.

ubrain is not on PyPI, so it is not in uv.lock; this folder carries it instead:

    weeks/16-cerebellum/vendor/wheels/ubrain-0.13.0.dev1-cp312-*.whl   (the brain, compiled)
    weeks/16-cerebellum/vendor/g1_body/body, robots                    (the G1 body server)

Two things this module does, so that `uv run weeks/16-cerebellum/lab_*.py` needs nothing
installed beyond `uv sync`:

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

VENDOR = Path(__file__).resolve().parent / "vendor"
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
        "Pull the repository again: weeks/16-cerebellum/vendor/ should hold "
        "wheels/ and g1_body/ (see weeks/16-cerebellum/README.md).")


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


def start_body(port: int | None = None, *, viewer: bool = False,
               echo: bool = True) -> tuple[subprocess.Popen, int]:
    """Start the G1 body server; returns (process, port) once it serves. Its stdout
    stays open: in the viewer, the body prints `KEY <code>` for every key press."""
    if not (BODY / "body" / "bridge.py").is_file():
        raise _missing("the G1 body")
    from ubrain.bodies import free_port
    port = port or free_port()
    cmd = [sys.executable, "-m", "body.bridge", "--port", str(port)]
    if viewer:
        cmd.append("--viewer")
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
