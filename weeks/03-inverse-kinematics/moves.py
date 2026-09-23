"""YOUR MOVES -- the only file you edit for lab_connected.py.

    uv run weeks/03-inverse-kinematics/lab_connected.py

Ten problems, one move each. A MOVE is a list of steps, played in order under
real physics (gravity, contact, the 29 position servos):

    (seconds, pose)      blend smoothly from the previous pose to `pose`
                         over `seconds`

When the last step ends, the pose is HELD for 1.5 s. The problem passes if its
goal was reached at some moment AND the robot never fell AND it ends standing
(pelvis above 0.70 m, torso within 15 degrees of upright). A move may last at
most 8 seconds.

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

MOVES = {
    # 1  WAVE: left hand above 1.20 m. This one is started for you -- press 1
    #    and read the gauge: it is a near miss. Make it pass.
    1: [(1.0, {"left_shoulder_pitch": -90})],

    # 2  SWAY: pelvis 5 cm to the left AND 5 cm to the right (hint: "pelvis").
    2: [],

    # 3  TWIST: torso turned 80 degrees or more.
    3: [],

    # 4  SQUAT: pelvis below 0.60 m, then stand back up.
    4: [],

    # 5  BOW: torso tilted 40 degrees or more, then back upright.
    5: [],

    # 6  CLAP: hands closer than 10 cm, at least 25 cm in front of the pelvis.
    6: [],

    # 7  HANDS UP: both hands above 1.20 m at the same moment.
    7: [],

    # 8  SQUAT-WAVE: pelvis below 0.62 m with a hand above 1.10 m.
    8: [],

    # 9  TWIST-WAVE: torso turned 80 degrees or more with a hand above 1.20 m.
    9: [],

    # 10 BOSS: pelvis below 0.62 m with BOTH hands above 1.10 m.
    10: [],
}
