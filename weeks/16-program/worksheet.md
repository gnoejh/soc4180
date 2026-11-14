# Week 16 worksheet — Programming the whole brain

Name: ______________________   Date: __________

Write each **prediction before** you run. A wrong prediction with a reason earns full marks;
a right one without a reason does not.

## A. The world

**A1.** The default world has 4 pictures and 4 actions; block 1 earns 0.976. Predict block 1
with 8 pictures and 4 actions (each action is right for two pictures).

Prediction: ______  Reason: __________________________________________

`uv run weeks/16-program/lab_program.py --step whole --pictures 8`    Measured: ______

**A2.** With `--reverse` the right actions are shuffled at trial 1500. How many trials until a
block of 500 earns 0.9 again? Which part of the brain relearns first? (Hint: `read("source")`.)

Prediction: ______  Measured (block 4, block 5): ______ , ______

## B. The brain

**B1.** Run step ask with `--variant small`, `standard`, `large`. Which datasheet rows change?
Which do not, and why can a variant change a size but never the wiring?

| variant | items held of 128 (prefrontal.buffer) | other changed rows |
| --- | --- | --- |
| small | 43.9 | |
| standard | | |
| large | | |

**B2.** Does a larger variant earn more in step whole? Predict, then measure block 1 and block 6.

## C. Keeping

**C1.** Step keep refuses `Brain.load(path)` without the codec. Edit the lab so the loaded
brain gets a codec fitted with `seed=2` instead of `seed=1`. What happens, and why is that the
right answer?

______________________________________________________________________

**C2.** The saved file is 10.1 MB for the small variant. Predict its size after 3000 ticks
instead of 1000. Does a brain's file grow with experience?

## D. The project proposal (one page, hand in at the start of week 17)

1. **The question** — about the brain in the G1.
2. **The prediction**, written now, with a number and its reason (a law, an earlier week).
3. **The twin**: the same run with one thing changed (a fresh brain, a silenced part, another
   variant), so that a difference has a cause.
4. **The measurement**: what your script writes to a file, and how many runs.
5. **The demo**: the saved brain, loaded, in the viewer.
