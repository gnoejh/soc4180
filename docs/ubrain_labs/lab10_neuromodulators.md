# Lab 10 -- One number each: noradrenaline and serotonin

**Goal:** see a neuromodulator as a single number written onto a region that already
exists, and predict what it does.

## Run

```bash
uv run examples/10_neuromodulators.py      # about 20 seconds
```

## What you see

Two experiments. **Noradrenaline:** a salience gate judges a stream in which a share
pi of arrivals is new; the table gives the gate's criterion and its error rate for a
criterion fixed at the wrong rate, at the right rate, and adapting. **Serotonin:** one
situation, a small reward soon or a large one late; the table gives how often the
brain waits for the large one, for low, high and adapting patience.

## The idea

**Noradrenaline sets what counts as novel.** For any threshold, the gate's two error
rates (a new item called familiar, a repeat called novel) are exact numbers
(`ubrain.regions.novelty_rates`), so the threshold that errs least at a rate pi of
novelty is a number too (`novelty_criterion`). Noradrenaline's level tracks the rate of
*judged* surprise, level <- level + eta (judged novel - level), and the criterion
follows the level. Adapting, the level settles where it equals the surprise rate its
own criterion produces: a fixed point.

**Serotonin sets patience.** A choice's eligibility decays by g = exp(-1/tau) each
tick until its outcome arrives, so a value learned for a reward r after D ticks is
r g^D. The Go race then waits for the large reward with probability

    P(late) = g^D_L / (r_S g^D_S + g^D_L).

Small tau, little patience: impulsivity, as a number. Adapting, tau follows the
measured time between decisions.

## Exercises

1. **Predict, then compute.** With r_S = 0.4, D_S = 2 and D_L = 8, compute P(late) by
   hand for tau = 2 and tau = 10. Check the table.
2. Find the tau at which the brain is indifferent (P(late) = 1/2). Is it above or
   below the adaptive tau the example reports?
3. Set the noradrenaline eta to 0.1
   (`w.regions["neuromodulator.noradrenaline"]["eta"] = 0.1`) and watch the level after
   the change. Fast adaptation counts the surprise its own criterion decides. What
   happens, and why can a fast learner talk itself out of noticing anything?
4. The criterion fixed at the wrong rate errs about twice as often as the right one.
   Which kind of error grows: new items missed, or repeats called new?
5. **Going further:** the example describes the thalamus and acetylcholine but does
   not run them. Pick one, read its region in `ubrain.regions`, and design a
   prediction you could test the same way.
