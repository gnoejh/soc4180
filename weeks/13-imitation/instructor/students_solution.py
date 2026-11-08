"""Reference answers for lab_imitate.py (instructor copy; gitignored, sealed by scripts/instructor.py)."""

NAME = "Reference"

WALKER_FEATURES = ["clock", "time"]

COPY = {"features": ["gravity", "gyro", "joints", "velocities", "previous"], "history": 1, "demos": 40, "dagger": 0}

FEW = {"features": ["gravity", "gyro", "joints", "velocities", "previous"], "history": 1, "demos": 2, "dagger": 2}

BLIND = {"features": ["gravity", "gyro", "joints", "previous"], "history": 5, "demos": 40, "dagger": 3}
