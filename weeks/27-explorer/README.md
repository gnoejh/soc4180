# 27 — The explorer: walking, seeing, remembering

| | |
| --- | --- |
| **Layer** | 3 and 5 — a frozen trained eye (CLIP) feeding an untrained memory, on layer 4's walker |
| **Runtime** | laptop, CPU |
| **Needs** | `uv sync --extra vision` and the CLIP weights (week 26); ubrain and the body ship in `vendor/ubrain/` |
| **Classes** | two: slides and a live demo, then a hands-on class |

The G1 walks straight down a gallery of 21 CC0 photographs (on stands every 1.2 m,
alternately 0.65 m left and right of its line) with a head camera. Every tenth tick the
lab takes a frame, looks at its more structured half (a **fovea**), codes it with CLIP,
**recognises** it as the nearest of the 21 pictures (a `Sense` with `snap`), and writes it
into week 20's hippocampal episode once it has been recognised on three glances in a row
(**event segmentation**). When the walk is over a cue recalls each slot. The report sets
what was written, what was recalled and the gallery's order side by side, so every error
has a home: perception and walking change what is written, memory changes what is
recalled (and the certificate predicts it). Walking "straight" open loop, the G1 turns to
the right and leaves the gallery after half a minute; a heading hold from the integrated
gyro (week 17's signal) keeps it on the line.

## Slides

HTML deck in week 16's format; the figure slide is drawn by `figures.py` from
`traces/walk_steer*.json` (`--run` re-measures it, about a minute).

```bash
uv run python weeks/27-explorer/build.py
```

## Before the class (at home, about 2 minutes, after week 26's download)

```bash
uv sync --extra vision
uv run --extra vision weeks/27-explorer/lab_explore.py
```

It must print `body f787c763e67a ...` (the gallery body's digest) and end with
`recall of what was written: 12/12 slots` and `11/21 pictures in order`.

## Lab class: on your laptop

```bash
uv run --extra vision weeks/27-explorer/lab_explore.py --help
uv run --extra vision weeks/27-explorer/lab_explore.py --viewer        # watch it, real time (58 s)
uv run --extra vision weeks/27-explorer/lab_explore.py --verbose       # every glance
uv run --extra vision weeks/27-explorer/lab_explore.py --steer 0.5     # hold the heading: 20 written
uv run --extra vision weeks/27-explorer/lab_explore.py --steady 1      # no event segmentation
uv run --extra vision weeks/27-explorer/lab_explore.py --no-snap       # store raw glances: 6 of 12
uv run --extra vision weeks/27-explorer/lab_explore.py --steer 0.5 --d 64
```

What students change: `--steer`, `--steady`, `--no-snap`, `--no-fovea`, `--d`, `--vx`,
`--seconds`, `--glance`, and the `fovea` function in the script.

What they should see (measured 2026-10-25; the walk is deterministic, so every laptop gets
the same lists):

- as shipped: 12 of 21 written, 12 of 12 recalled, 11 of 21 in the gallery's order; the
  clock taken for the laptop, shoe, horse and butterfly never recognised three glances
  running, nothing new written after 31.6 s;
- the reason: the robot turns 8° right in 58 s; its true sideways offset is −0.84 m at
  30 s and −2.77 m at 55 s (an instructor probe of the simulator; the robot has no position
  sensor); with `--steer 0.5` it stays within 0.5 m and writes 20, recalls 20 of 20, 18 of
  21 in order (`--steer 2`: 20, 17 in order);
- `--steady 1`: still 12 written and recalled, but flickers come first: 8 in order;
- `--no-snap`: the same 12 written, 6 of 12 recalled (10 of 20 with `--steer 0.5`);
- `--no-fovea`: 8 written, all recalled;
- a smaller memory fails as certified: `--steer 0.5 --d 64` recalls 8 of 15 (certificate
  0.56 at k 16), `--d 128` 14 of 19 (0.72 at k 21);
- speed: `--vx 0.7` writes 18 (17 in order), `--vx 0.3 --seconds 90` writes 16.

Every run appends a row to `results.csv` and saves every glance in `last_run.json`. The
exercises are in [worksheet.md](worksheet.md); measured reference answers are in
[instructor/reference.md](instructor/reference.md).
