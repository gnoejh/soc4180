# Lab 2 -- One-shot memory and its capacity: the hippocampus

**Goal:** store an episode in a single vector, see recall fail as it fills, and
predict the failure before it happens.

## Run

```bash
uv run examples/02_memory.py
```

## What you see

First a **certificate**: for episodes of 4 to 32 items, the predicted chance that
the first item comes back. Then the brain is run 200 times per load, and the
measured rate is printed beside the prediction, with its sampling error.

## The idea

An item is a code: a vector of D numbers. To store item z in slot i, the
hippocampus binds it to slot i's **role** r_i (a fixed random vector) with circular
convolution, scrambles the result by a permutation that marks the position, and
adds it to the episode vector:

    x = sum_i  P^i ( r_i (*) z_i )

To recall slot i: undo the permutation, unbind the role, and pick the known item
most similar to what comes out. The other k - 1 items are still in x, and they add
noise -- crosstalk -- that grows with k and shrinks with D. Recall fails when the
noise lets a wrong item look more similar than the right one. That chance is
computable from the codes alone: the certificate.

## Exercises

1. **Predict, then run.** Change `D` to 256. At which k do you expect recall to fall
   below 90%? (The capacity grows in proportion to D.) Run and check.
2. Change `V` (the number of known items) from 500 to 50 and to 5000. Does the
   capacity change? Why should more known items make recall harder?
3. Recall slot k-1 (the last item) instead of slot 0. Is the last item easier? Look at
   how the episode is built and explain.
4. Store the same item twice in one episode. What happens to its recall, and why?
5. **Going further:** a brain cannot hold 100 items in one episode at D = 128. Using
   the `boundary` input (it starts a new episode), how could an app split a long day
   into episodes it *can* remember? What is lost?
