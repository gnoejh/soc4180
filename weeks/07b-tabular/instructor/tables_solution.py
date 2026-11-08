"""Reference answers for lab_tables.py (instructor only; gitignored, sealed by scripts/instructor.py)."""

NAME = "Reference"

FROZEN = {"gamma": 0.99}                     # far-sighted: the long safe way round the holes
RANDOM = {"algo": "q"}                       # off-policy: learns the best table while acting at random
CLIFF_SAFE = {"algo": "sarsa"}               # on-policy: it values the moves it will really make, slips included
CLIFF_SHORT = {"algo": "q"}                  # off-policy: the greedy path along the edge

PENDULUM = {                                 # the pole's angle AND its angular velocity: a state
    "pole angle": [-0.12, -0.06, 0, 0.06, 0.12],
    "pole angular velocity": [-1, -0.5, 0, 0.5, 1],
}

BALANCE = {"bins": 21}                       # not too coarse (aliasing), not too fine (too few samples per row)
