"""Instructor's reference answers for lab_connected.py -- NOT for students.

This folder is gitignored so it never reaches GitHub. Check it with:

    uv run weeks/03-inverse-kinematics/lab_connected.py --grade --moves weeks/03-inverse-kinematics/instructor/moves_solution.py
"""

NAME = "Reference"

LEFT_UP = {"left_shoulder_pitch": -150, "left_shoulder_roll": 0, "left_shoulder_yaw": 0, "left_elbow": 0}
RIGHT_UP = {"right_shoulder_pitch": -150, "right_shoulder_roll": 0, "right_shoulder_yaw": 0, "right_elbow": 0}
BOTH_UP = {**LEFT_UP, **RIGHT_UP}
SQUAT = {"pelvis": [0, 0, -0.20]}
CLAP = {"left_shoulder_pitch": -80, "left_shoulder_roll": -20, "left_shoulder_yaw": 0, "left_elbow": 30,
        "right_shoulder_pitch": -80, "right_shoulder_roll": 20, "right_shoulder_yaw": 0, "right_elbow": 30,
        "pelvis": [-0.04, 0, -0.03]}
BOW = {"waist_pitch": 29, "pelvis": [-0.08, 0, -0.12], "left_hip_pitch": -60, "right_hip_pitch": -60}
TURN = {"waist_yaw": 90}

MOVES = {
    1: [(1.0, LEFT_UP)],
    2: [(1.0, {"pelvis": [0, 0.06, -0.05]}), (1.5, {"pelvis": [0, -0.06, -0.05]}), (1.0, {})],
    3: [(1.5, TURN)],
    4: [(1.0, SQUAT), (1.0, {})],
    5: [(1.5, BOW), (1.5, {})],
    6: [(1.0, CLAP)],
    7: [(1.0, LEFT_UP), (1.0, BOTH_UP)],
    8: [(1.0, SQUAT), (1.0, {**SQUAT, **LEFT_UP}), (1.0, SQUAT), (1.0, {})],
    9: [(1.0, TURN), (1.0, {**TURN, **LEFT_UP}), (1.0, TURN), (1.0, {})],
    10: [(1.0, SQUAT), (1.0, {**SQUAT, **LEFT_UP}), (1.0, {**SQUAT, **BOTH_UP}), (1.5, BOTH_UP),
         (1.0, RIGHT_UP), (1.0, {})],
}
