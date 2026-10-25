# 20 — One-shot memory: the hippocampus

| | |
| --- | --- |
| **Layer** | 5 — behaviour learned by local plasticity, not by gradient descent |
| **Runtime** | laptop, CPU |
| **Needs** | this repo's `uv sync`; ubrain ships in `vendor/ubrain/` (a local copy while ubrain is not public) |
| **Classes** | two: slides and a live demo, then a hands-on class |

The G1 walks and is pushed, now and then, in one of 8 directions in its own frame. It
senses each direction with its pelvis accelerometer (`perceive()` in the lab), and its
hippocampus (ubrain's `observer` profile, `hippocampus.episode`) stores each percept the
moment it arrives, in one vector. Afterwards every push is recalled. Three things are
scored: perception (percept = push), memory (recalled = percept), both; and memory is
set beside the **certificate**, recall predicted from the codes before anything is stored.
No camera and no CLIP: the robot's own body is the sense.

## Slides

HTML deck in week 16's format (`deck.json`, `slides/<id>.html`); the figure slide is drawn
by `figures.py` from `traces/points.json` (`--run` re-measures it, about 3 minutes).

```bash
uv run python weeks/20-hippocampus/build.py
```

## Before the class (at home, about 2 minutes)

```bash
uv sync
uv run weeks/20-hippocampus/lab_remember.py
```

It must end with the three rates beside the certificate (0.94 at the defaults).

## Lab class: on your laptop

```bash
uv run weeks/20-hippocampus/lab_remember.py --certify                  # the prediction
uv run weeks/20-hippocampus/lab_remember.py --episodes 10              # 8 pushes, D 64
uv run weeks/20-hippocampus/lab_remember.py --episodes 10 --pushes 24
uv run weeks/20-hippocampus/lab_remember.py --episodes 10 --d 32
uv run weeks/20-hippocampus/lab_remember.py --episodes 10 --force 60   # a weaker push
uv run weeks/20-hippocampus/lab_remember.py --viewer --episodes 1      # watch it
```

What students change: the episode length (`--pushes`), the workspace (`--d`), the push
(`--force`), the number of directions, and the sense itself (`perceive()` in the file).

What they should see (measured 2026-10-25, 10 episodes each):

- perception 0.90-0.97 at 100 N; 0.57 at 60 N; 1.00 at 60 N with 4 directions;
- memory against the certificate: D 64, 8 / 16 / 24 pushes: 0.94 / 0.79 / 0.67 against
  0.94 / 0.76 / 0.64; D 32: 0.71 / 0.53 / 0.46 against 0.76 / 0.56 / 0.46;
- the memory keeps misperceptions: both ≈ perception × memory (0.88 ≈ 0.90 × 0.94);
- the same 24 pushes as three episodes of 8 are remembered far better than as one.

Every run appends a row to `results.csv`. The exercises are in
[worksheet.md](worksheet.md); measured reference answers are in [instructor/reference.md](instructor/reference.md).
