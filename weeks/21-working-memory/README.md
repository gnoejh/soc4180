# 21 — Holding a list: working memory

| | |
| --- | --- |
| **Layer** | 5 — behaviour learned by local plasticity, not by gradient descent |
| **Runtime** | laptop, CPU |
| **Needs** | this repo's `uv sync`; ubrain ships in `vendor/ubrain/` (a local copy while ubrain is not public) |
| **Classes** | two: slides and a live demo, then a hands-on class |

Week 20's pushes and sense, a different region: ubrain's `holder` profile
(`prefrontal.buffer`) keeps the robot's list of 16 pushes in one vector by position, and
fades everything by γ before each write. So recall depends on age: the newest pushes are
sharp, the oldest drown. The count law (`ubrain.datasheet.buffer_count`) predicts how many
are held before a single push (6.83 of 16 at γ 0.75, d 64: about seven). Rehearsal in free
moments changes which pushes are held, never more of them: a full cycle changes nothing.

## Slides

HTML deck in week 16's format; the figure slide is drawn by `figures.py` from
`traces/by_age.json` (`--run` re-measures it, about 2 minutes).

```bash
uv run python weeks/21-working-memory/build.py
```

## Before the class (at home, about 2 minutes)

```bash
uv sync
uv run weeks/21-working-memory/lab_hold.py --lists 1
```

## Lab class: on your laptop

```bash
uv run weeks/21-working-memory/lab_hold.py --law                      # the prediction
uv run weeks/21-working-memory/lab_hold.py --lists 20                 # the robot
uv run weeks/21-working-memory/lab_hold.py --lists 20 --gamma 0.9
uv run weeks/21-working-memory/lab_hold.py --lists 20 --rehearse 16   # a full cycle: no change
uv run weeks/21-working-memory/lab_hold.py --lists 20 --rehearse 16 --schedule newest
uv run weeks/21-working-memory/lab_hold.py --viewer --lists 1
```

What students change: the fade (`--gamma`), the dimension (`--d`), the list length, the
rehearsal (`--rehearse`, `--schedule`).

What they should see (measured 2026-10-25, 20 lists each): held 7.30 / 10.75 / 4.80 at γ
0.75 / 0.9 / 0.5 (law 6.83 / 9.87 / 4.62); d 128: 7.95 (7.86); recall by age near 1.00 for
the newest three, near chance (0.125) for the oldest; a full rehearsal cycle leaves recall
identical, position by position; rehearsing only the newest or the oldest three makes them
certain and drops the count to about 5.

Every run appends a row to `results.csv`. The exercises are in
[worksheet.md](worksheet.md); measured reference answers are in [instructor/reference.md](instructor/reference.md).
