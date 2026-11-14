# Week 17 worksheet — Senses and the body

Name: ______________________   Date: __________

Write each **prediction before** you run. A wrong prediction with a reason earns full marks;
a right one without a reason does not.

## A. The twin

The default tone comes 0.2 s (10 ticks) before the push; brain A falls 3 times in 20 pushes,
brain B (no tone) 14 times.

**A1.** Halve the lead: `--delay 5` (0.1 s). Predict brain A's falls, and why.

Prediction: ______  Reason: __________________________________________   Measured: ______

**A2.** Double it: `--delay 20` (0.4 s). Prediction: ______   Measured: ______

**A3.** Brain B falls more late in the run than early. Propose one cause and one run that
would test it.

______________________________________________________________________

## B. The detector

**B1.** `--step push --force 50`: how many of 16 pushes are named right? Which sides get
confused, and what in the walking gait could cause it?

Prediction: ______   Measured: ______

**B2.** Why is `push_direction` a detector the program owns and not a sense the brain has?

______________________________________________________________________

## C. The privilege

**C1.** `--step contact --vx 0.2` and `--vx 0.6`: predict the share of ticks with both feet
on the ground (0.12 at 0.4 m/s).

| speed | prediction | measured |
| --- | --- | --- |
| 0.2 | | |
| 0.6 | | |

**C2.** Name one experiment that would become unfair if foot contact were given to the brain
as a sense.

## D. The eye (`uv run --extra vision weeks/17-senses/lab_sense.py --step vision`)

**D1.** 22 of 199 glances are called new. What would make the brain call a picture new a
second time? Try it (a longer walk, `--seconds 80`, walks past the end of the gallery).

## E. Your project (hand in)

Run your project once end to end on `Brain()` with the G1's streams, and hand in its results
file. In a table, sort every signal your project uses into sense / detector / privilege.
