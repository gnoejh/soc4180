"""Deterministic seeding so every student reproduces the lecture's numbers."""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["set_seed"]

DEFAULT_SEED = 4180


def set_seed(seed: int = DEFAULT_SEED, *, deterministic_torch: bool = False) -> int:
    """Seed Python, NumPy and (if installed) PyTorch. Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        # Slower, but makes GPU results bit-reproducible across runs.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
