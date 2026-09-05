"""Pre-trained policy loading.

The safety net for the RL weeks: a lab must never dead-end because a training run
failed to converge inside the session. Students who do not get a working policy
download one and carry on with the analysis.
"""

from __future__ import annotations

import pathlib
import urllib.request

from .models import cache_dir

__all__ = ["download", "checkpoint_path"]

# Populated as each RL week is built: week -> URL of a known-good policy.
REGISTRY: dict[str, str] = {}


def checkpoint_path(name: str) -> pathlib.Path:
    return cache_dir() / "checkpoints" / name


def download(name: str, url: str | None = None, *, force: bool = False) -> pathlib.Path:
    """Fetch a pre-trained checkpoint, caching it locally. Returns its path."""
    url = url or REGISTRY.get(name)
    if url is None:
        raise KeyError(
            f"No checkpoint registered for '{name}'. "
            f"Known: {sorted(REGISTRY) or 'none yet'}"
        )

    dest = checkpoint_path(name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest

    urllib.request.urlretrieve(url, dest)
    return dest
