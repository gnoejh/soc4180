# 16 — Programming the whole brain (Part II begins; the project starts)

| | |
| --- | --- |
| **Layer** | 5 — the untrained brain as a programmer meets it: `Brain()`, port classes, the tick |
| **Runtime** | laptop, CPU |
| **Needs** | nothing beyond `uv sync`; ubrain 0.46 and the body ship in `vendor/ubrain/` |
| **Classes** | two: slides and a live demo, then a hands-on class and the project proposal |

**Part II is the untrained brain in the robot, three weeks, one project.** The brain is
ubrain's whole brain, `Brain(variant=...)`: every region on the route from the senses to the
body, wired once and the same in every copy. A program never wires it. It **attaches
streams by port class** (a sense through a frozen codec, a reward, an action, and from week
17 the body's streams) and **ticks** it with one dictionary in and one out. It can **ask** the
brain about itself (`describe`, `read`, `datasheet`, `health`) and **keep** it (`save`,
`load`).

Part II runs on one release, **ubrain 0.46**, vendored in `vendor/ubrain/` (16 program, 17
senses and the body, 18 build and measure; presentations in the final slot).

## Slides

HTML deck (deck.json and slides/); every number on a slide comes from a run of
`lab_program.py`.

```bash
uv run python weeks/16-program/build.py
```

## Before the class (about 1 minute)

```bash
uv run weeks/16-program/lab_program.py
```

It must print `ubrain 0.46.0.dev2`; step whole earns `0.976` in its first block; step keep
ends `500 more ticks: 0 choices differ`; step g1 walks 500 ticks with `falls 0`.

## The lab

`lab_program.py` runs four steps (`--step whole | ask | keep | g1`):

| step | what it shows | the measured result (seed 0) |
| --- | --- | --- |
| whole | a picture asks for an action; the brain earns, beside the striatum's race law | 0.976 in block 1, 0.998 by block 6 (chance 0.25); `--reverse`: 0.896 in the block after the shuffle, 1.000 the next |
| ask | `describe`, `read`, `datasheet`, `health` | 9 classes, 6 observables, 0 health warnings |
| keep | save, load without the codec (refused), load with it, continue | 10.1 MB; 0 of 500 choices differ |
| g1 | the same whole brain walks the G1 (`ubrain.bodies.g1.G1`) | prediction error 96.9% of no-change in second 1, 17.2% in second 9; no falls |

Options: `--pictures`, `--actions`, `--trials`, `--variant small|standard|large`,
`--reverse`, `--seconds`, `--vx`, `--viewer` (step g1 in real time), `--seed`.
Every run appends rows to `results.csv`.

## The hands-on class

The worksheet: predict, then run, for the world (more pictures, a reversal), the brain (the
three variants), keeping (a different codec at load). Then the **project proposal**: one page
with the question, the prediction written first, the twin it is compared against, and the
demo.

## For the instructor

`instructor/reference.md` holds the measured answers. `scripts/check_labs.py --only 16` runs
the four steps.
