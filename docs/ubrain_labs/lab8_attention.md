# Lab 8 -- Noticing: the salience gate and the modes

**Goal:** see a gate that writes only what is new, set by a false-alarm rate, and a
mode controller that switches between acting and resting.

## Run

```bash
uv run examples/08_attention.py
```

## What you see

A stream of bursts: fresh items (F), repeats of an item still in the gate's window
(r), and repeats of one seen before the window (o), with quiet ticks between bursts.
Under the stream, the gate's verdict per tick (+ novel and written, - familiar and
skipped) and the mode (E executive, d default). Then the familiar rate for each kind
of item beside its prediction, the mode counts, and how well the cortex recalls what
was replayed into it.

## The idea

**Novelty as distance.** The gate keeps the last `window` arrivals (8). An item's
novelty is 1 - its largest cosine with any of them. A recent repeat has novelty 0,
so it is always judged familiar.

**The threshold comes from a rate.** You do not set the threshold; you set alpha, the
chance that a fresh item is wrongly judged familiar when the window holds `window`
different codes. ubrain derives the threshold from alpha by the exact distribution of
a random cosine in d dimensions (`ubrain.regions.familiar_threshold`). At d = 512 and
alpha = 0.05 it is 0.8901.

**A refinement the example predicts.** When the window holds repeats, it holds fewer
than 8 distinct codes, and a fresh item has fewer chances to look familiar:
P = 1 - (1 - t)^n for n distinct codes, where t is the one-code tail. That is why the
predicted rate for fresh items is 0.044, a little under alpha.

**Modes.** A novel item is a demand: the controller switches to executive mode to
serve it, and stays there until the plans serving the burst are done (in the mode row,
a few ticks past each burst). Then it returns to default mode, and on the first idle
tick the hippocampus replays the episode into the cortex.

## Exercises

1. **Predict, then run.** Set alpha to 0.2
   (`w.regions["salience.gate"]["alpha"] = 0.2`). Before running, say which number in
   the "judged familiar" table moves, and in which direction. How many fresh items are
   lost?
2. Set the window to 32. Old repeats are items that repeat from before the window.
   What happens to them, and why?
3. Recent repeats are judged familiar every time. Is that a feature or a bug for a
   brain that must notice change?
4. The executive share of the ticks was 0.77 in the first two bursts. Predict how it
   moves if the quiet gap between bursts doubles.
5. **Going further:** the threshold depends on d. Compute
   `familiar_threshold(d, 8, 0.05)` (from `ubrain.regions`) at d = 64 and d = 2048,
   and explain the trend from the geometry of high-dimensional cosines.
