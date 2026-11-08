"""YOUR ANSWERS -- the only file you edit for lab_tables.py.

    uv run --extra rl weeks/07b-tabular/lab_tables.py

Six problems, one learning rule. Every problem trains a table Q(s, a) with

    Q[s, a] += alpha * (target - Q[s, a])

and you decide what it is told. As shipped the file holds the reference
answers and scores 6 / 6: change one, predict what happens, then play it.
To play a problem from scratch, put ... (a BLANK) in its place: that problem
then stops with "not filled in yet" and names the line. The file is read as
DATA, never run: only plain values (numbers, strings, lists, dicts).

HOW TO FIND THE ANSWERS: play a problem with a guess, watch the arrows and
the gauge, and ask what the table was told. Every failure has a reason you
can see. The lecture's slides name each idea; none of the answers is a
number you have to tune by trial and error alone.
"""

NAME = "Your Name"          # shown on the grading screen -- change it

# Problem 1. The slippery lake (FrozenLake): the agent moves where it meant only
# one time in three. Reward 1 at the goal, 0 everywhere else, and a hole ends
# the episode. Q-learning, 5000 episodes. You choose the discount gamma
# (0 .. 1): how much a reward one step later is worth, compared with now.
FROZEN = {"gamma": 0.99}                 # far-sighted: the long safe way round the holes is worth it

# Problem 2. The same lake, but the agent ACTS AT RANDOM the whole time
# (epsilon = 1, it never uses its own table). Only the update can differ:
# "q" (target uses the best next action) or "sarsa" (target uses the action
# it actually takes next). Which one's table is still worth following?
RANDOM = {"algo": "q"}                   # off-policy: learns the best table whatever it actually did

# Problems 3 and 4. CliffWalking: 4 x 12 tiles, -1 per step, -100 and back to
# the start for stepping into the cliff along the bottom edge. Both train with
# epsilon = 0.1 (one move in ten is random) for 500 episodes.
#   3: while it is still exploring, it must rarely fall.
#   4: what it has learned (the greedy path) must be the shortest one.
# "q" or "sarsa" for each. They are not necessarily the same answer.
CLIFF_SAFE = {"algo": "sarsa"}           # on-policy: values the moves it really makes, slips included
CLIFF_SHORT = {"algo": "q"}              # off-policy: the greedy path along the edge, 13 steps

# Problem 5. MuJoCo's inverted pendulum (gymnasium InvertedPendulum-v5): a cart
# pushes with -3, 0 or +3 N to keep a pole up. The env reports four numbers:
#     "cart position" (m), "pole angle" (rad), "cart velocity" (m/s),
#     "pole angular velocity" (rad/s)
# A table needs a ROW, so you cut numbers into bins. List the cut points for
# each number you want the table to SEE; leave a number out and it is blind
# to it. n cut points make n + 1 bins; rows = product of the bin counts. The
# episode ends when the pole leans past 0.2 rad. 3000 episodes of training.
PENDULUM = {                             # angle AND angular velocity: together they are a state
    "pole angle": [-0.12, -0.06, 0, 0.06, 0.12],
    "pole angular velocity": [-1, -0.5, 0, 0.5, 1],
}

# Problem 6. Week 7's G1 balance (the LIPM: CoM x, its speed v, 7 places for the
# centre of pressure), now learned from episodes instead of solved from P.
# The table cuts both x and v into "bins" equal pieces: bins x bins rows.
# 100,000 episodes. Value iteration, which knew P, earned J = 72.12.
BALANCE = {"bins": 21}                   # not too coarse (aliasing), not too fine (too few samples per row)
