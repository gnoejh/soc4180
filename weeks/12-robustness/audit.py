"""YOUR AUDIT -- the only file you edit for lab_robust.py.

    uv run weeks/12-robustness/lab_robust.py

Three controllers take a sideways shove: "hold" (the crouch, no feedback),
"nominal" (PPO trained in the nominal world) and "random" (PPO trained on
randomised worlds). Measure them with robust.py, then write what you found.

A LIMIT is the largest shove, in newtons and a multiple of 10, that the
controller survives in at least 7 of 10 episodes. The referee accepts your
claim L if it survives 7/10 at L and fails that at L + 20.

Each ... is a BLANK: replace it. As shipped every question stops with "not
filled in yet" and names the line, so the file scores 0 / 5. The file is
read as DATA, never run.
"""

NAME = "Your Name"          # shown on the grading screen -- change it

HOLD_LIMIT = ...            # newtons, nominal world
NOMINAL_LIMIT = ...
RANDOM_LIMIT = ...

# Torso 15 kg heavier. Which controller ("hold", "nominal", "random"), and its limit?
HEAVY = {"policy": ..., "limit": ...}

# 20 robots whose friction, torso mass and motor strength you do not know
# (friction 0.2-1.0, torso -5..+15 kg, stiffness x0.6-1.5). Ship one controller
# and promise a shove it survives on at least 16 of them (80 N or more).
SHIP = {"policy": ..., "promise": ...}
