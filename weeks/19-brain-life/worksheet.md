# Week 19 worksheet -- A brain's life: save, load, datasheet, health

**Goal:** treat the G1's brain as a device. Save what it learned, load it in a new
session and prove it is the same brain, read its datasheet before it runs and check it
against the robot, and watch its health while it walks.

**Slides:** `slides_students.html` in this folder (build it with
`uv run python weeks/19-brain-life/build.py`, or open it on the course site).

**Every command below runs from this folder**, `weeks/19-brain-life/`.

## Before the class (at home, about 2 minutes)

```bash
uv sync                          # from the repository root, as every week
cd weeks/19-brain-life
uv run lab_life.py sheet         # prints the walker's datasheet
```

## The tool

```bash
uv run lab_life.py --help        # the six steps
uv run lab_life.py train --help  # each step's flags
```

The brain is week 16's `walker`; the push is week 16's (a tone, then 250 N from behind
10 ticks later). Saved brains go in `brains/` (yours; git ignores it). Every run adds a
row to `results.csv`.

**Rule for every experiment: write your prediction before you run.**

## Part A -- save, load, and the same brain

| # | experiment | command | your prediction | measured |
|---|---|---|---|---|
| A1 | learn and save | `train` | the record of 12 pushes; the file's size: | |
| A2 | a new session: load it | close the terminal, open a new one, `test` | the record of 6 pushes: | |
| A3 | the control | `test --fresh` | | |
| A4 | the same brain? | `identical` | commands that agree out of 500: | |
| A5 | another laptop's brain | swap `brains/brace.npz` with another group, then `test` | does it load? the record? | |
| A6 | another body | `refuse` | | |

1. A2: what in the file makes the loaded brain brace from the first push? Use
   `observe`'s numbers in the output (trials, peak).
2. A2: the loaded brain still falls sometimes. Did it fall on the same pushes as during
   training? Why could it not be the same?
3. A4: the two brains drive two different bodies and never exchange a number. Why are
   their commands identical to the last bit? What would break that if the file did not
   save the random generators?
4. A5: the other group's brain learned on another laptop. Does it load? Do its pushes
   fall where they fell on that laptop? (Week 1: what is identical across machines and
   what is not?)
5. A6: the other body has the same channels, the same sizes. Why must the load refuse it?
   What would happen if it did not?

## Part B -- the datasheet and health

The brace's datasheet figure is its **lead**: how far back its trace of the tone reaches,
1 / (1 − τ) ticks.

| # | experiment | command | your prediction | measured |
|---|---|---|---|---|
| B1 | the datasheet | `sheet --tau 0.5`, `--tau 0.9`, `--tau 0.97` | the lead for each, before you run: | |
| B2 | against the robot | `sheet --tau T --measure 12` for each τ, then again with `--delay 25` | which τ learns which delay? | |
| B3 | health while walking | `watch` | when does health turn True? what happens at the turn? | |
| B4 | a loaded brain's health | `watch --load brains/brace.npz --seconds 12` | | |

6. B1: compute 1 / (1 − τ) by hand for the three τ before running `sheet`.
7. B2: at τ 0.9 the learned counter-move peaks near the push, not at the lead; at τ 0.97
   it peaks later still. What does the lead predict, then, and what does it not?
8. B2: τ 0.9 has a lead of 10 ticks but still learns a push at 25 ticks, with more falls.
   How much of the trace is left at tick 25 (0.9^25)? Why is the lead not a hard limit?
9. B3: the forward model's error jumps when the robot turns. Why? Why does it fall again?
10. B4: the loaded brain predicts straight walking better than a fresh one, but the turn
    worse. Explain from what it lived through.
11. What should a robot do when its forward model's health turns False? Which region from
    weeks 16-18 could act on such a signal?

## Show your results

Each group presents one experiment of its own choice (or a new one): the question, the
prediction, the command, the measured result, and an explanation. Five minutes per group.

## Going further

- Train a brace at `--tau 0.97`, save it, and load it: is its datasheet the same as a
  fresh brain's at 0.97? (The datasheet reads the configuration, not the experience.)
- Build a watcher: a second `Brain` on `monitor_manifest(brain)` whose senses are
  `monitor(brain)` every tick (ubrain's example 12 does it for working memory). Raise a
  flag when `cerebellum.forward.stride_ratio` rises above 0.6.
