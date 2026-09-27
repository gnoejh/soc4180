"""YOUR STUDENTS -- the only file you edit for lab_imitate.py.

    uv run weeks/13-imitation/lab_imitate.py

Four problems. In each, an expert demonstrates and a student network learns
to copy it by regression. You choose what the student is SHOWN and how it is
taught; the referee trains it and then lets it drive on its own.

A student sees a list of FEATURES, any of:

    "gravity"     3 numbers   gravity in the torso frame, from the IMU
    "gyro"        3           angular velocity, from the IMU
    "joints"     12           the 12 leg angles minus the crouch
    "velocities" 12           the 12 leg joint velocities
    "previous"   12           the action it took one step ago
    "clock"       3           sin, cos of the gait phase, and "walking yet?"
    "time"        1           seconds since the start, / 10

and, for the shove problems, a dict:

    "features"   the list above
    "history"    how many past steps it sees at once (1 = only now)
    "demos"      how many demonstration episodes to learn from
    "dagger"     rounds of DAgger after that: the STUDENT drives, the expert
                 says what it should have done, the student is retrained (0 = none)

Each ... is a BLANK: replace it. As shipped every problem stops with "not
filled in yet" and names the line, so the file scores 0 / 4. The file is
read as DATA, never run.

THE QUESTION TO ASK in every problem: what did the expert's decision depend
on -- and can the student see it?
"""

NAME = "Your Name"          # shown on the grading screen -- change it

# Problem 1: copy the week-4 walker; walk 1.0 m from three starts.
# The walker is open loop: its action is a function of the TIME only.
WALKER_FEATURES = [...]

# Problem 2: copy week 12's shove policy. It sees 42 numbers: gravity, gyro,
# joints, velocities, previous.
COPY = {"features": [...], "history": 1, "demos": 40, "dagger": 0}

# Problem 3: the same, from at most 2 demonstrations, at 80 N.
FEW = {"features": ["gravity", "gyro", "joints", "velocities", "previous"], "history": 1, "demos": 2, "dagger": ...}

# Problem 4: the real robot's joint velocities are too noisy to use. No "velocities".
BLIND = {"features": ["gravity", "gyro", "joints", "previous"], "history": ..., "demos": 40, "dagger": ...}
