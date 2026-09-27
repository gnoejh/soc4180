"""YOUR INSTRUCTIONS -- the only file you edit for lab_agent.py.

    uv run weeks/15-language/lab_agent.py

The robot's language skills are a keyword parser (soc4180.agent.parse) and a
network that finds a named colour in a picture. Both know exactly four colour
words -- red, green, blue, yellow -- and a handful of verbs. You supply what
they do not know:

    SYNONYMS    word -> a word the robot knows. Applied to the sentence before
                anything else, so {"scarlet": "red"} makes "the scarlet ball"
                mean "the red ball".
    PROGRAMS    phrase -> a whole plan. If the phrase appears in the sentence,
                the plan is used instead of the parser's guess. A plan is a
                list of steps, each a list: ["raise", "left"], ["lower", "right"],
                ["look", "the red ball"], ["search", "the red ball"], ["raise_both"].
    SEARCH      how the search skill turns when it cannot see what it wants:
                "step_deg" degrees per turn (it tries +1, -1, +2, -2 steps ...)
                and at most "tries" turns.
    LOOK_GAIN   how hard the waist turns per radian of estimated azimuth
                (week 14's gain, again).

Each ... is a BLANK: replace it. As shipped every problem stops with "not
filled in yet" and names the line, so the file scores 0 / 5. The file is
read as DATA, never run.

Try your own sentences:  uv run weeks/15-language/lab_agent.py --say "..."
"""

NAME = "Your Name"          # shown on the grading screen -- change it

LOOK_GAIN = ...

SYNONYMS = {"crimson": ..., "azure": ...}

PROGRAMS = {"both hands": [...]}

SEARCH = {"step_deg": ..., "tries": ...}
