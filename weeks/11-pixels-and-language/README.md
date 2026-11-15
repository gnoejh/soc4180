# 11 — Pixels and language

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/11-pixels-and-language/lab.ipynb)

**The task: "look at the green ball".** The G1 stands in front of four coloured balls
with a camera in its head. One file, [`lab.py`](lab.py), does it in four steps: words →
a colour (`colour_of`, a lookup; unknown words refused), the colour → where it is in the
picture (`find_ball`: colour distance + centroid; azimuth = atan(−u) with the 90° camera),
a P controller on the waist (week 5), and **a check with the picture** (refuse if the
ball is not there). The deck's main point is the check: a check that uses the same sense
that failed passes its own mistake.

| | |
| --- | --- |
| **Layer** | 2 (a camera) feeding 5 (reading it; words) |
| **Runtime** | CPU; on Colab pick a **GPU runtime**, needed there only to render. Laptops render with any OpenGL |
| **Wall clock** | the deck's cells ~15 s; the battery in `lab.py` ~2 s |
| **Needs** | `uv sync --extra rl` (`soc4180.vision` imports torch; nothing trains) |

The deck opens with **where this week sits on the map** of RL methods from week 8, and every
major function of `lab.py` is printed and then explained line by line (`| code | what it does | why |`).

## Lab class: on your laptop

```bash
uv run weeks/11-pixels-and-language/lab.py                              # the battery, then a replay
uv run weeks/11-pixels-and-language/lab.py --say "look at the blue ball"
uv run weeks/11-pixels-and-language/lab.py --dim 0.3 --no-viewer
```

**What you change:** `find_ball()` itself, `THRESHOLD` (0.25), `MIN_PIXELS` (4), `GAIN`
(0.8), and `--dim`. **What you see:** a table of ten tests (scene, instruction, what the
robot says, and the **true error** from the simulator, which the robot never uses), the
count carried out, then the viewer replaying the last turn. **Right looks like:** all 8
looks done, every ball within 0.3° of the middle; the hidden green ball and the word
"crimson" refused.

**Try** (predict first; measured with `lab.py`):

1. `THRESHOLD = 0.05`: refuses all 10. `THRESHOLD = 0.6`: says "done" 9 times, up to 39°
   off, and "sees" the hidden green ball, 149° away.
2. `--dim 0.3`: "done" 9 times, up to 41° off; "sees" the hidden ball. The check uses the
   same sense that failed, so it cannot catch it.
3. `GAIN = 2.5`: 4 of 8 refused (the four balls ~40° or more off): the torso swings, the
   robot starts to fall, the ball leaves the picture.
4. `MIN_PIXELS = 1`: no change here.
5. Write a check that catches 1 and 2 without the true-error column: a second,
   independent sense (the blob's size and shape, a stricter test, a small extra turn).

## Measured elsewhere, quoted in the deck

The learned versions of steps 1–2, measured in earlier versions of this week
(`torch.set_num_threads(4)`): a small CNN (146,370 parameters) reads the ball's direction
to 1.0°, but 27.8° on an unseen blue ball and 15.2° in a dim room (guessing the average:
26.0°); visual randomisation gives 4.4 / 9.4 / 4.2°. A 20k-parameter words + pixels
attention model: 2.0° on trained phrasings, 2.1° on new ones, ~31° (chance) for
"crimson", and confident heat-map peaks (up to 0.98) for a ball that is not in the picture.

## Term project (weeks 12–15)

Starting points: **P11 — Eyes on the ball**, **P12 — Words to actions**, **P4 — Reach
while standing** (point at the ball), **P14 — your own**. See
[`weeks/projects/README.md`](../projects/README.md) and
[`weeks/projects/catalogue.md`](../projects/catalogue.md). Method: predict first, a twin
with one change, `results.csv`, a `CHECK` line, grade by margins. The true-error column
is for grading your robot, never for your robot to use.

## Rebuild

```bash
quarto render weeks/11-pixels-and-language/slides.qmd
```
