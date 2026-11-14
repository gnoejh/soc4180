# 28 — Programming the whole brain (the project starts)

| | |
| --- | --- |
| **Layer** | 5 — the untrained brain as a programmer meets it: `Brain()`, port classes, the tick |
| **Runtime** | laptop, CPU |
| **Needs** | nothing beyond `uv sync`; ubrain 0.45 and the body ship in `vendor/ubrain/` |
| **Classes** | two: slides and a live demo, then a hands-on class and the project proposal |

Weeks 16-27 each built a **circuit**: a profile wired from chosen regions (walker, chooser,
explorer ...). This week is the **whole brain**, `Brain(variant=...)`: every region on the
route from the senses to the body, wired once and the same in every copy. A program never
wires it. It **attaches streams by port class** (a sense through a codec, a reward, an
action) and **ticks** it with one dictionary in and one out. It can **ask** the brain about
itself (`describe`, `read`, `datasheet`, `health`) and **keep** it (`save`, `load`).

This is the first week of the three-week project (28 program, 29 senses and the body, 30
build and measure; presentations in the final slot).

**ubrain 0.45 for this week.** Weeks 16-27 stay on 0.14, the release their numbers were
measured on (`vendor/ubrain/ubrain_vendor.py` loads a named release; this week's
`ubrain_local.py` names 0.45). In 0.45 the whole brain has no body classes yet: the G1 still
runs through the walker profile here. Release 0.46 (week 29) adds them.

## Slides

HTML deck in week 16's format; every number on a slide comes from a run of `lab_program.py`.

```bash
uv run python weeks/28-program/build.py
```

## Before the class (about 1 minute)

```bash
uv run weeks/28-program/lab_program.py
```

It must print `ubrain 0.45.0.dev1`; step whole earns `0.976` in its first block; step keep
ends `500 more ticks: 0 choices differ`; step g1 walks 500 ticks with `falls 0`.

## The lab

`lab_program.py` runs four steps (`--step whole | ask | keep | g1 | all`):

| step | what it shows | the measured result (seed 0) |
| --- | --- | --- |
| whole | a picture asks for an action; the brain earns, beside the striatum's race law | 0.976 in block 1, 0.998 by block 6 (chance 0.25); `--reverse`: 0.896 in the block after the shuffle, 1.000 the next |
| ask | `describe`, `read`, `datasheet`, `health` | 5 classes, 6 observables, 0 health warnings |
| keep | save, load without the codec (refused), load with it, continue | 10.1 MB; 0 of 500 choices differ |
| g1 | the G1 through the walker profile | prediction error 96.9% of no-change in second 1, 17.2% in second 9; no falls |

Options: `--pictures`, `--actions`, `--trials`, `--variant small|standard|large`,
`--reverse`, `--seconds`, `--vx`, `--viewer` (step g1 in real time), `--seed`.
Every run appends rows to `results.csv`.

## The hands-on class

The worksheet: predict, then run, for the world (more pictures, a reversal), the brain (the
three variants), keeping (a different codec at load). Then the **project proposal**: one page
with the question, the prediction written first, the twin it is compared against, and the
demo.

## For the instructor

`instructor/reference.md` (gitignored) holds the measured answers. `scripts/check_labs.py
--only 28` runs the four steps.
