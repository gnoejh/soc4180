# Week 16 — reference answers (measured 2026-11-15, ubrain 0.46.0.dev2, seed 0)

Every number below is from a run of `lab_program.py` or the two short checks noted.

## A. The world

**A1.** 8 pictures, 4 actions: block 1 earns **0.832** (law 0.868), block 2 0.888, block 3
0.986, block 6 0.996. Twice the pictures means each picture is met half as often in the first
block, so less is remembered and valued by then. Accept any prediction below 0.976 with that
reason.

**A2.** With `--reverse`: block 4 (the shuffle's block) earns **0.896**, block 5 **1.000**. A
block of 500 is back above 0.9 within the shuffle's own block: about 50 of the 500 trials are
lost. Which system chooses (counted with `read("source")` over 100 ticks):

| | recalled episodes | learned values |
| --- | --- | --- |
| right after the shuffle (ticks 1500-1599) | 62 | 38 |
| at the end (ticks 2900-2999) | 83 | 17 |

So memory chooses most of the time in this small world, and more so at the end. The
hippocampus relearns from one experience of each new pairing; the striatum's values follow.

## B. The brain

**B1.** The prefrontal buffer's items recalled of a list of 128:

| variant | items |
| --- | --- |
| small | 43.9 |
| standard | 48.81 |
| large | 51.28 |

A variant sizes regions (`SIZES` in `ubrain.face`: `d_g`, `d_h`, `slots`, `s`, `gamma_h`), never
the wiring, so the rows that change are the capacities and the rows that stay are
configurations (the gate's false-alarm rate, the mode controller's arm).

**B2.** Large earns 0.970 in block 1 and 0.994 in block 6 (law 0.975, 0.990), against small's
0.976 and 0.998. A larger brain does not earn more in a four-picture world: it was never short
of capacity. Size buys capacity, and this world asks for none.

## C. Keeping

**C1.** Loading with a codec fitted with `seed=2`:
`StateError: sense.vision: the codec plugged in (gaussian, gaussian-64-2) is not the one the
state was learned with (gaussian, gaussian-64-1)`. This is the right answer: the brain's
knowledge is in the codes it learned from, and another codec gives other codes for the same
picture. A loaded brain with another eye would recognise nothing it learned.

**C2.** 1000 ticks: **10.15 MB**; 3000 ticks: **10.11 MB**. The file does not grow with
experience: a brain's state has a fixed size set by its variant. What it learns changes the
values in it, not its size. (The small difference is compression.)

## D. The proposal

Check that it has all five parts and that the prediction has a number and a reason. Projects on
the robot use the body's streams through `ubrain.bodies.g1.G1` (step g1 already does);
week 17 covers them in full.
