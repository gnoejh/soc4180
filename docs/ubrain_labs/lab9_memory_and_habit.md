# Lab 9 -- Memory first, habit later: episodic control and the hand-over

**Goal:** see two learners on one task: a memory that acts from one experience, and a
striatum that learns slowly but well; and a hand-over that chooses between them.

## Run

```bash
uv run examples/09_memory_and_habit.py      # about half a minute
```

## What you see

A bandit with 256 situations and two actions, one paying 0.7 and one 0.3. Three
brains play it: the `chooser` as shipped (the hand-over), the same with memory always
acting, and the `player` (striatum alone). The table gives each one's reward rate by
how many times the situation has been visited. Memory leads on early visits; the
striatum catches up around visits 11-30 and leads late. The hand-over follows the
better of the two.

## The idea

**Memory acts at once.** The episodic store keeps a vote per (situation, action),
decaying over time; after one good outcome it repeats the action. It needs no
learning curve, but its votes interfere: with many situations sharing one store, the
crosstalk grows.

**The striatum learns slowly and well.** Its values move by eta per visit (lab 1):
the datasheet gives its time constant, 1/eta = 10 visits. Long after it, the Go race
matches: it chooses each action in proportion to its value, which predicts a late
reward rate of sum p^2 / sum p = 0.58 here (the best possible is 0.70).

**The hand-over.** Memory acts while its margin clears its own crosstalk scale,
||x|| sqrt(2/d); otherwise the striatum acts. Both learn from every outcome.

## Exercises

1. **Predict, then run.** Set `S = 16`. With few situations the store has little
   crosstalk. Does memory stay ahead late? Write your answer down first.
2. Give the store more room: set the episodic store's `d` to 1024 in the wiring.
   Which rows of the table move?
3. Late, the hand-over sits a little below the striatum alone. Explain where its
   shortfall comes from.
4. All three brains score near 0.5 on the first visit. Why is 0.5 the prediction for
   every one of them?
5. **Going further:** animals rely on memory early in learning and on habit after
   long practice. Which of this lab's settings correspond to "long practice"?
