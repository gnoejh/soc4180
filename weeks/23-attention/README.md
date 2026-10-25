# 23 — Noticing: the salience gate and the modes

| | |
| --- | --- |
| **Layer** | 5 — behaviour learned by local plasticity, not by gradient descent |
| **Runtime** | laptop, CPU |
| **Needs** | this repo's `uv sync`; ubrain ships in `vendor/ubrain/` (a local copy while ubrain is not public) |
| **Classes** | two: slides and a live demo, then a hands-on class |

Week 20's pushes come in bursts with quiet walking between; ubrain's `attender` profile
puts a salience gate in front of the robot's memory. The gate judges each sensed push
against the last 8 that arrived (novelty = 1 − the largest cosine); a direction felt
recently is skipped, a direction not felt lately is written, and the threshold is derived
from one number, α, the chance a fresh push is wrongly judged familiar. A mode controller
keeps the brain executive while novel pushes keep coming and lets it rest when nothing
new is pending; the first idle moment replays the episode into the cortex (week 22).

## Slides

HTML deck in week 16's format; the figure slide is drawn by `figures.py` from
`traces/windows.json` (`--run` re-measures it, about a minute in parallel).

```bash
uv run python weeks/23-attention/build.py
```

## Before the class (at home, about 2 minutes)

```bash
uv sync
uv run weeks/23-attention/lab_notice.py --bursts 3
```

## Lab class: on your laptop

```bash
uv run weeks/23-attention/lab_notice.py                    # a random day
uv run weeks/23-attention/lab_notice.py --world gusty      # mostly the same two sides
uv run weeks/23-attention/lab_notice.py --alpha 0.3
uv run weeks/23-attention/lab_notice.py --window 2         # a gate with a short memory
uv run weeks/23-attention/lab_notice.py --quiet 4          # too little rest
```

What students change: the world, the gate's window and α, the rest between bursts, d.

What they should see (measured 2026-10-25): recent repeats skipped every time; 45 of 120
pushes written on a random day, 29 on a gusty one; α 0.3 loses 18 % of fresh directions;
window 2 writes 73 % and the cortex recalls 0.77, window 16 writes 16 % and it recalls 0.95;
with only 4 quiet moments half of what was written never reaches the cortex (23 of 46).

Every run appends a row to `results.csv`. The exercises are in
[worksheet.md](worksheet.md); measured reference answers are in [instructor/reference.md](instructor/reference.md).
