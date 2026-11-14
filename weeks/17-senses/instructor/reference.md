# Week 29 — reference answers (measured 2026-11-15, ubrain 0.46.0.dev2, the vendored body 4815eedf56d0)

Every number below is from a run of `lab_sense.py`.

## A. The twin

Default (delay 10 ticks = 0.2 s, 250 N, vx 0.4): A `..X.XX..............` falls **3**;
B `..X.X.X.X.XXXXXXXXXX` falls **14**.

**A1.** `--delay 5` (0.1 s): A `..XXXX.X.XX.........` falls **7**. A shorter lead leaves the
counter-move less time to act before the push lands, and it learns later (the first 11 pushes
are mixed).

**A2.** `--delay 20` (0.4 s): A `..X.XX....X.........` falls **4**: about the default. The
trace (tau 0.9 per tick of the brace) still spans 0.4 s; one extra fall at push 11.

**A3.** Open: the measured fact is that B keeps falling, more in its second half. Accept any
testable cause (the gait's phase at the push as the walk settles; the forward model's state
after resets) with a run that varies one thing.

## B. The detector

Default (100 N): **15 of 16**, one back push named right. `--force 50`: **14 of 16**. Weak
pushes are confused between front/back and the side the stance foot is on, because the gait's
own fore-aft and lateral swing is in the same accelerometer. It is the program's detector: it
needs to know WHEN the push came (the rows before and during), which the brain is never told.

## C. The privilege

| speed | both feet | left only | right only | neither |
| --- | --- | --- | --- | --- |
| 0.2 | 0.14 | 0.41 | 0.46 | 0.00 |
| 0.4 | 0.12 | 0.42 | 0.46 | 0.01 |
| 0.6 | 0.07 | 0.45 | 0.48 | 0.00 |

Faster walking, less double support (as in human gait). At 0.4 m/s a loaded foot carries 296 N
on average, both feet together 328 N (the robot's weight). C2: e.g. learning to brace or to
detect a push from contact forces a real G1 does not have.

## D. The eye

40 s at 0.5 m/s down the gallery: **199 glances, 22 new**, in bursts as a picture comes into
view. A picture is new again only if it is not in what the brain holds as recent (the salience
gate's memory); walking past the gallery's end and back, or long enough to forget, can do it.
