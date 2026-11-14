"""ubrain for this week: the course's one copy is vendor/ubrain/ (see ubrain_vendor.py there).

Part II (weeks 16-18) is written for ubrain 0.46, the whole brain.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor" / "ubrain"))

from ubrain_vendor import BODY, VENDOR, start_body  # noqa: E402,F401
from ubrain_vendor import load as _load  # noqa: E402

VERSION = "0.46.0.dev2"


def load() -> None:
    """Make `import ubrain` give this week's release."""
    _load(VERSION)
