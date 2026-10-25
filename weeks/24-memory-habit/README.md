# 24 — Memory first, habit later

| | |
| --- | --- |
| **Layer** | 5 — behaviour learned by local plasticity, not by gradient descent |
| **Runtime** | laptop, CPU |
| **Needs** | this repo's `uv sync`; ubrain ships in `vendor/ubrain/` (a local copy while ubrain is not public) |
| **Classes** | two: slides and a live demo, then a hands-on class |

Week 18's braking task (a cue says the push comes soon or late; brake early, brake late or
keep walking; reward = still standing) with three brains: the striatum (week 18), an
episodic memory that acts from one experience, and ubrain's `chooser`, which hands each
decision to memory while memory's votes clear their own crosstalk and to the striatum
otherwise. The G1 is deterministic, so its world (240 trials: 2 cues × 3 moves × 40 stride
points) is measured once and kept in `traces/world.json`; the brains play against the
robot's own outcomes, many seeds in seconds (`--live` drives the robot itself). On the
robot memory wins early and late; on ubrain's laptop bandit (`--bandit S`) habit overtakes
once enough situations share the memory.

## Slides

HTML deck in week 16's format; the figure slide is drawn by `figures.py` from
`traces/late.json` (`--run` re-measures it, about 2 minutes in parallel).

```bash
uv run python weeks/24-memory-habit/build.py
```

## Before the class (at home, about 2 minutes)

```bash
uv sync
uv run weeks/24-memory-habit/lab_habit.py --seeds 2
```

## Lab class: on your laptop

```bash
uv run weeks/24-memory-habit/lab_habit.py                     # three brains, the robot's world
uv run weeks/24-memory-habit/lab_habit.py --d 8                # a small memory
uv run weeks/24-memory-habit/lab_habit.py --bandit 256 --trials 20000 --seeds 5
uv run weeks/24-memory-habit/lab_habit.py --live --brain memory --trials 20 --viewer
```

What students change: the memory's fade and room, the striatum's rate, the number of
situations (`--bandit`), and which brain runs live.

What they should see (measured 2026-10-25, 20 seeds of 200 trials): late reward striatum
0.66, memory 0.81, chooser 0.82 (95 % of decisions to memory); on the bandit, memory late
0.70 / 0.68 / 0.61 / 0.54 at 2 / 16 / 64 / 256 situations against the striatum's 0.58;
at d 8 the chooser hands over more (61 % to memory) and earns less (0.73) than memory alone
(0.79).

Every run appends one row per brain to `results.csv`. The exercises are in
[worksheet.md](worksheet.md); measured reference answers are in [instructor/reference.md](instructor/reference.md).
