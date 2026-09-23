"""YOUR MOVES -- the only file you edit for lab_connected.py.

    uv run weeks/03-inverse-kinematics/lab_connected.py

Ten problems, one move each. A MOVE is a list of steps, played in order under
real physics (gravity, contact, the 29 position servos):

    (seconds, pose)      blend smoothly from the previous pose to `pose`
                         over `seconds`

When the last step ends, the pose is HELD for 1.5 s. The problem passes if its
goal was reached at some moment AND the robot never fell AND it ends standing
(pelvis above 0.70 m, torso within 15 degrees of upright). There is no time
limit: a move may be as long as you need, and slower is often steadier.

A POSE is a dict. Every joint you do not name goes back to the `stand` pose.

    "left_elbow": 30            one joint, in DEGREES. Short names are fine:
                                "left_shoulder_pitch", "waist_yaw", "right_knee".
                                `--joints` prints all 29 with their ranges.
    "left_arm": [0,0,0,0,0,0,0] a whole chain in degrees, root to tip
                                (left_leg, right_leg, waist, left_arm, right_arm).
    "pelvis": [dx, dy, dz]      INVERSE KINEMATICS (this week!): move the pelvis
                                by metres (x forward, y left, z up) while both
                                feet stay where they are. The solver finds the
                                twelve leg angles. [0, 0, -0.1] is a small squat.
    "left_foot": [dx, dy, dz]   IK again: move one foot relative to the pelvis
                                target (metres). Warning: the robot then stands
                                on one foot, and that is hard.

A joint named explicitly overrides what IK chose for it, so
{"pelvis": [0, 0, -0.1], "left_hip_pitch": -30} is IK first, then your hip.

You can name a pose and reuse it: write  SQUAT = {...}  above MOVES, then use
SQUAT inside a move, or {**SQUAT, "left_elbow": 30} to add to it. Only plain
values are allowed here (numbers, lists, dicts, names defined above) --
the file is read as data, never run, so it cannot change the scorer.

HOW TO FIND THE NUMBERS: do not guess. Press E in the viewer, drag the joint
sliders on the right until the frozen robot looks right, and press ENTER --
the pose is printed as a dict you paste here. Then press the problem's number
to try it under physics and watch where it goes wrong.
"""

NAME = "Your Name"          # shown on the grading screen -- change it

# ---------------------------------------------------------------------------
# Every joint and every offset you need is already written out below. Each
# ... is a BLANK: replace it with a number. As shipped every problem stops
# with "not filled in yet" and names the lines, so the file scores 0 / 10.
# The other numbers are given to get you started; you may change them too.
# Joints are in degrees (tens); "pelvis" is [dx, dy, dz] in METRES (a few
# hundredths to tenths; x forward, y left, z up). Signs matter: find them
# with E + the sliders + ENTER in the viewer, not by guessing.
#
# The poses are named once and reused, so fixing LEFT_UP fixes it in problems
# 1, 7, 8, 9 and 10 at the same time.
# ---------------------------------------------------------------------------

LEFT_UP = {"left_shoulder_pitch": ..., "left_shoulder_roll": 0, "left_shoulder_yaw": 0, "left_elbow": 0}
RIGHT_UP = {"right_shoulder_pitch": ..., "right_shoulder_roll": 0, "right_shoulder_yaw": 0, "right_elbow": 0}
BOTH_UP = {**LEFT_UP, **RIGHT_UP}
SQUAT = {"pelvis": [0, 0, ...]}
CLAP = {"left_shoulder_pitch": -80, "left_shoulder_roll": -20, "left_shoulder_yaw": 0, "left_elbow": 30,
        "right_shoulder_pitch": -80, "right_shoulder_roll": 20, "right_shoulder_yaw": 0, "right_elbow": 30,
        "pelvis": [-0.04, 0, ...]}
BOW = {"waist_pitch": ..., "pelvis": [-0.08, 0, -0.12], "left_hip_pitch": -60, "right_hip_pitch": -60}
TURN = {"waist_yaw": ...}                                         # lean back to stay balanced

MOVES = {
    # 1  WAVE: left hand above 1.20 m.
    1: [(1.0, LEFT_UP)],

    # 2  SWAY: pelvis 5 cm to the left AND 5 cm to the right, then back.
    2: [(1.0, {"pelvis": [0, ..., 0]}),        # to the left
        (1.5, {"pelvis": [0, ..., 0]}),        # to the right
        (1.0, {})],                          # back to stand

    # 3  TWIST: torso turned 80 degrees or more.
    3: [(1.5, TURN)],

    # 4  SQUAT: pelvis below 0.60 m, then stand back up.
    4: [(1.0, SQUAT), (1.0, {})],

    # 5  BOW: torso tilted 40 degrees or more, then back upright.
    5: [(1.5, BOW), (1.5, {})],

    # 6  CLAP: hands closer than 10 cm, at least 25 cm in front of the pelvis.
    6: [(1.0, CLAP)],

    # 7  HANDS UP: both hands above 1.20 m at the same moment.
    7: [(1.0, LEFT_UP), (1.0, BOTH_UP)],

    # 8  SQUAT-WAVE: pelvis below 0.62 m with a hand above 1.10 m.
    8: [(1.0, SQUAT), (1.0, {**SQUAT, **LEFT_UP}), (1.0, SQUAT), (1.0, {})],

    # 9  TWIST-WAVE: torso turned 80 degrees or more with a hand above 1.20 m.
    9: [(1.0, TURN), (1.0, {**TURN, **LEFT_UP}), (1.0, TURN), (1.0, {})],

    # 10 BOSS: pelvis below 0.62 m with BOTH hands above 1.10 m.
    10: [(1.0, SQUAT), (1.0, {**SQUAT, **LEFT_UP}), (1.0, {**SQUAT, **BOTH_UP}),
         (1.5, BOTH_UP), (1.0, RIGHT_UP), (1.0, {})],
}
