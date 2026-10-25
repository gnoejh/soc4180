# Lab 1 -- Learning from reward: the striatum

**Goal:** see a value learned from reward, and a choice rule that matches rather
than maximises.

## Run

```bash
uv run examples/01_bandit.py
```

## What you see

The world shows one of three situations; in each, three actions pay 1 with
different probabilities (the `PAY` table). The reward rate climbs over the first
thousand trials. At the end two tables: the learned values beside the payoff
probabilities, and how often each action was chosen beside a prediction.

## The idea

**Learning.** Each time action a is taken in situation s and reward r arrives, the
value moves a step toward it:

    Q[s, a] <- Q[s, a] + eta * (r - Q[s, a])

`r - Q` is the reward-prediction error -- what dopamine neurons report. With
probability p of reward, Q settles around p: the error averages to zero exactly there.

**Choosing.** Each action has a "Go" unit that races to a threshold at a speed
proportional to its value; the first to arrive is done. Action a wins with
probability `Q_a / sum_b Q_b`. So once Q is near p, choice settles at
`p_a / sum_b p_b`: the best action most often, but not always. Animals do this too
(Herrnstein's matching law).

## Exercises

1. **Predict, then run.** For situation 2 (`[0.3, 0.3, 0.9]`), write down the
   fraction of choices you expect for each action. Compare with the printout.
2. Values of rarely chosen actions come out wrong (e.g. 0.06 where p is 0.2). Why?
   How many times is that action taken? Change `TRIALS` and check your explanation.
3. Make the striatum's `eta` 0.01 and then 0.5: in `01_bandit.py`, load the profile as
   `w = profile("player")`, set `w.regions["basal_ganglia.striatum"]["eta"]`, and build the
   brain with `w`. What happens to
   the speed of learning, and to how much the values jitter? Explain both from the
   update rule.
4. Is matching a good strategy? Compute the reward rate a player would get by always
   choosing the best action, and compare it with the measured rate. When might
   matching be worth its cost? (Hint: what if the payoffs change?)
5. **Going further:** change `PAY` halfway through the run (inside the loop). How
   long until choice follows the new payoffs? Predict it from eta.
