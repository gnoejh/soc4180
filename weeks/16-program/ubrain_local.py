"""ubrain for this week: the course's one copy is vendor/ubrain/ (see ubrain_vendor.py there).

Week 28 on is written for ubrain 0.45 (the whole brain, `Brain()`); weeks 16-27 stay on 0.14,
the release their numbers were measured on.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor" / "ubrain"))

from ubrain_vendor import BODY, VENDOR, start_body  # noqa: E402,F401
from ubrain_vendor import load as _load  # noqa: E402

VERSION = "0.45.0.dev1"


def load() -> None:
    """Make `import ubrain` give this week's release."""
    _load(VERSION)
