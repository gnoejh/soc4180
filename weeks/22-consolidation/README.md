# 22 — From hippocampus to cortex: consolidation

| | |
| --- | --- |
| **Layer** | 5 — behaviour learned by local plasticity, not by gradient descent |
| **Runtime** | laptop, CPU |
| **Needs** | this repo's `uv sync`; ubrain ships in `vendor/ubrain/` (a local copy while ubrain is not public) |
| **Classes** | two: slides and a live demo, then a hands-on class |

Week 20's pushes and sense, now through ubrain's `rememberer` profile: the hippocampus
writes each episode of 8 sensed pushes into one vector, and at every boundary retrieves it
and replays it into the cortex (a d × d matrix, Hebb's rule under fresh keys), then clears
itself. Three stages are measured against their laws: the episode (the decode law at d),
the cortex (the same law at d²), end to end (their product). The cortex's law counts
different pairs; the robot's world has 8 kinds of push, so the robot's cortex falls far
below its law, while `--distinct` (the same schedule, every item different, no robot)
follows it. A forgetting cortex (`--gamma`) keeps its newest episodes.

## Slides

HTML deck in week 16's format; the figure slide is drawn by `figures.py` from
`traces/points.json` (`--run` re-measures it, about 2 minutes in parallel).

```bash
uv run python weeks/22-consolidation/build.py
```

## Before the class (at home, about 2 minutes)

```bash
uv sync
uv run weeks/22-consolidation/lab_consolidate.py --episodes 2
```

## Lab class: on your laptop

```bash
uv run weeks/22-consolidation/lab_consolidate.py                         # 10 episodes
uv run weeks/22-consolidation/lab_consolidate.py --episodes 20
uv run weeks/22-consolidation/lab_consolidate.py --distinct --episodes 20 # the law's world
uv run weeks/22-consolidation/lab_consolidate.py --episodes 20 --gamma 0.97
uv run weeks/22-consolidation/lab_consolidate.py --episodes 20 --d 128
```

What students change: the number of episodes and their length, the dimension, the
cortex's forgetting, and the world (`--distinct`).

What they should see (measured 2026-10-25, d 64): hippocampus 0.90-0.94 (law 0.90),
flat; robot cortex 0.95 / 0.86 / 0.62 / 0.41 at 40 / 80 / 160 / 320 pushes (law 1.00 /
1.00 / 1.00 / 0.96); `--distinct` cortex 1.00 / 1.00 / 0.94 / 0.56 (law 1.00 / 1.00 / 0.91 /
0.59); the hippocampus empty after every boundary; at γ 0.97 only the last two or three
episodes are kept; d 128 lifts the robot's cortex to 0.83 at 160 pushes, still below 1.00.

Every run appends a row to `results.csv`. The exercises are in
[worksheet.md](worksheet.md); measured reference answers are in [instructor/reference.md](instructor/reference.md).
