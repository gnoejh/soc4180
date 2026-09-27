"""YOUR EYES -- the only file you edit for lab_look.py.

    uv run weeks/14-vision/lab_look.py

Five problems. Each trains a small convolutional network to say where a ball
is from the head camera's 64 x 64 picture, using a RECIPE you write:

    "pictures"   how many training pictures to render (100 .. 20000)
    "epochs"     how many passes over them (1 .. 40)
    "colour"     "red"     every training ball is red
                 "random"  every training ball is a random colour
    "light"      [low, high]: each training picture's lights are scaled by a
                 random number between low and high. [1.0, 1.0] is the room as
                 it is; [0.3, 1.0] is anything from 30% to full brightness.

More pictures and more epochs cost time: 4000 pictures x 10 epochs takes
about 20 s on a laptop. The test pictures are always NEW ones.

Each ... is a BLANK: replace it. As shipped every problem stops with "not
filled in yet" and names the line, so the file scores 0 / 5. The file is
read as DATA, never run.
"""

NAME = "Your Name"          # shown on the grading screen -- change it

# Problem 1 (and 5): find a red ball in the room as it is.
RED = {"pictures": 4000, "epochs": ..., "colour": "red", "light": [1.0, 1.0]}

# Problem 2: the ball in the test is BLUE.
BLUE = {"pictures": 4000, "epochs": 10, "colour": ..., "light": [1.0, 1.0]}

# Problem 3: the room in the test is lit at 30%.
DIM = {"pictures": 4000, "epochs": 10, "colour": "red", "light": [..., ...]}

# Problem 4: one network for red, blue AND dim.
ALL = {"pictures": ..., "epochs": ..., "colour": ..., "light": [..., ...]}

# Problem 5: waist turn per radian of estimated azimuth, every 0.1 s.
GAIN = ...
