"""ubrain for this week: the course's one copy is vendor/ubrain/ (see ubrain_vendor.py there)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor" / "ubrain"))

from ubrain_vendor import BODY, VENDOR, load, start_body  # noqa: E402,F401
