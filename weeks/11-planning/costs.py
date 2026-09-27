"""YOUR COSTS -- the only file you edit for lab_plan.py.

    uv run weeks/11-planning/lab_plan.py

Six problems, one planner. The planner (MPPI) tries PLANNER["samples"]
futures of PLANNER["horizon"] seconds, 25 times a second, and executes the
cheapest-looking one. You do not tell the robot what to do. You tell it what
is BAD, as a cost, and it searches.

A COST is a dict of weights, one per term. Each term is measured on every
step of every sampled future, multiplied by its weight, and summed:

    "height"       (pelvis height - "height_target") squared   target 0.72 m unless you set it
    "upright"      1 - cos(tilt): 0 upright, 1 lying down
    "over_feet"    centre of mass away from the middle of the two feet, squared (metres)
    "over_right"   centre of mass away from the RIGHT foot, squared
    "lift_left"    left foot below "lift" metres, squared; 0 once it is above
    "plant_right"  right foot off the floor, squared
    "still"        pelvis speed, squared (linear and angular)
    "fall"         1 on every step the pelvis is below 0.5 m

and two numbers that are settings, not weights: "height_target" (metres)
and "lift" (metres). A term you leave out has weight 0.

A PLANNER is four numbers:

    "samples"      futures tried per decision (1 .. 256; more is slower)
    "horizon"      seconds looked ahead (0.04 .. 1.0)
    "sigma"        how far each future's servo targets wander from the plan, radians
    "temperature"  how picky the average is: small trusts only the best futures

Each ... is a BLANK: replace it with a number. As shipped every problem stops
with "not filled in yet" and names the lines, so the file scores 0 / 6.
The file is read as DATA, never run: only plain values, and names defined
above it ({**STAND, "still": 1} is a copy of STAND with one change).

HOW TO FIND THE NUMBERS: start from the lecture's standing cost, play a
problem, watch what the robot does, and ask what the cost made it want.
Every failure has a reason you can see.
"""

NAME = "Your Name"          # shown on the grading screen -- change it

PLANNER = {"samples": 64, "horizon": 0.4, "sigma": 0.15, "temperature": 0.05}

# Problems 1, 2, 3 (and 4 uses it too): stand, then take a shove.
STAND = {"height": ..., "upright": ..., "over_feet": ..., "still": 0.01, "fall": 100}

# Problem 4: a 300 N shove with at most 16 samples. One setting matters most.
CHEAP_PLANNER = {"samples": 16, "horizon": 0.4, "sigma": ..., "temperature": 0.05}

# Problem 5: stand on the RIGHT foot with the LEFT foot 5 cm clear for 2 s.
CRANE = {"height": 10, "upright": 5, "over_right": ..., "lift_left": ..., "lift": 0.13,
         "still": 0.01, "fall": 100}

# Problem 6: squat below 0.58 m between t = 1 and 3 s, then stand back up.
# (The lab switches "height_target" back to 0.72 at t = 3 s by itself.)
SQUAT = {"height": ..., "height_target": ..., "upright": 5, "over_feet": 20, "still": 0.01, "fall": 100}
