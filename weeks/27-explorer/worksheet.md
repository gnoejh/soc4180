# Week 27 worksheet -- The explorer: walking, seeing, remembering

**Goal:** run the whole pipeline on the walking G1 -- glance, fovea, CLIP, recognise,
segment, remember -- break one stage at a time, and say for every missing picture which
stage lost it.

**Slides:** `slides_students.html` in this folder (build it with
`uv run python weeks/27-explorer/build.py`, or open it on the course site).

**Every command below runs from this folder**, `weeks/27-explorer/`, and always with
`uv run --extra vision`.

## Before the class (at home, about 2 minutes, after week 26's download)

```bash
uv sync --extra vision                 # from the repository root
cd weeks/27-explorer
uv run --extra vision lab_explore.py   # about 30 s
```

## The tool

```bash
uv run --extra vision lab_explore.py --help
```

It prints the certificate for the 21 pictures, walks, and reports four lines: `written`
(what entered the episode, in order), `recalled` (what each slot returns), `standing` (the
gallery's own order), and two scores: recall of what was written, and how many pictures
the memory holds in the gallery's order. `--verbose` prints every glance; `last_run.json`
keeps them all.

**Rule for every experiment: write your prediction before you run.**

## Part A -- one stage at a time

| # | experiment | command | your prediction | written / recalled / in order |
|---|---|---|---|---|
| A1 | as shipped | `lab_explore.py --verbose` | | |
| A2 | no segmentation | `lab_explore.py --steady 1` | | |
| A3 | no recognising | `lab_explore.py --no-snap` | | |
| A4 | no fovea | `lab_explore.py --no-fovea` | | |

- **Q1.** In A1, find the clock and the shoe in the `--verbose` output. Which stage lost
  each one?
- **Q2.** A2 writes as many pictures as A1 and recalls them all. Why is the order worse?
- **Q3.** A3 writes the same 12 but recalls half. The certificate says 0.997. What does
  the certificate assume that a raw glance breaks? (Look at the `familiar` column.)
- **Q4.** A4: why does looking at the whole frame lose pictures?

## Part B -- the walk

| # | experiment | command | your prediction | written / in order / last written at |
|---|---|---|---|---|
| B1 | when does it stop writing? | A1's output | | |
| B2 | hold the heading | `lab_explore.py --steer 0.5` | | |
| B3 | a stiffer hold | `lab_explore.py --steer 2` | | |
| B4 | speed | `--steer 0.5 --vx 0.7`, then `--steer 0.5 --vx 0.3 --seconds 90` | | |

- **Q5.** The robot has no position sensor. What tells you, from A1's output alone, that it
  turned? What does the heading hold use, and what can it not correct?
- **Q6.** In `lab_explore.py --viewer`, when do the panels leave the camera's view?
- **Q7.** B4: is slower better? Measure, then give one reason why it might not be.

## Part C -- memory

| # | experiment | command | your prediction | written / recalled / certificate at that k |
|---|---|---|---|---|
| C1 | a small memory | `lab_explore.py --steer 0.5 --d 64` | | |
| C2 | a medium memory | `lab_explore.py --steer 0.5 --d 128` | | |

- **Q8.** Compare recalled / written with the certificate's number at the nearest k. Which
  stages does the certificate speak for, and which not?

## To hand in

Your tables, Q1-Q8, and one paragraph: for each picture the walk as shipped did not
remember, name the stage that lost it and the change that recovers it.
