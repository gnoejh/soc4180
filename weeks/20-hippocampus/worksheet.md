# Week 20 worksheet -- One-shot memory: the hippocampus

**Goal:** measure the robot's sense of a push, then its one-shot memory of a sequence of
pushes against the certificate, and tell a perception error from a memory error.

**Slides:** `slides_students.html` in this folder (build it with
`uv run python weeks/20-hippocampus/build.py`, or open it on the course site).

**Every command below runs from this folder**, `weeks/20-hippocampus/`.

## Before the class (at home, about 2 minutes)

```bash
uv sync                          # from the repository root, as every week
cd weeks/20-hippocampus
uv run lab_remember.py           # about 10 s: three episodes, then the three rates
```

## The tool

```bash
uv run lab_remember.py --help
uv run lab_remember.py --certify # the prediction, no robot
```

Each episode prints three rows: **pushed** (what the experimenter did), **perceived**
(what the robot's accelerometer made of it), **recalled** (what its hippocampus gave
back). Directions are where the robot is pushed: F, FL, L, BL, B, BR, R, FR. The rates:
**perception** = perceived equals pushed; **memory** = recalled equals perceived;
**both** = recalled equals pushed. Use `--episodes 10` for numbers you report.

**Rule for every experiment: write your prediction before you run.**

## Part A -- the sense

| # | experiment | command | your prediction | measured |
|---|---|---|---|---|
| A1 | the default push | `--episodes 10` | perception: | |
| A2 | a weaker push | `--episodes 10 --force 60` | | |
| A3 | fewer directions | `--episodes 10 --force 60 --directions 4` | | |
| A4 | your own sense | edit `perceive()` (try the torso accelerometer, `slice(3, 6)`) | | |

1. A1: which directions does the robot confuse when it errs? Neighbours, or opposites?
   Why?
2. A2: why does a weaker push make the sense worse? (Week 6: what else does the
   accelerometer read while the robot walks?)
3. A3: with 4 directions the weak push is read perfectly. Why do fewer directions help?
4. The lab pushes in the robot's own frame. The first version pushed in world
   coordinates and every reading was 45° off. Why? (Week 6: is heading observable?)

## Part B -- the memory

| # | experiment | command | your prediction | measured |
|---|---|---|---|---|
| B1 | the certificate | `--certify` | the K where memory drops below 0.9 at D 64: | |
| B2 | longer episodes | `--episodes 10 --pushes 16`, then `--pushes 24` | | |
| B3 | half the workspace | `--episodes 10 --d 32 --pushes 8`, `16`, `24` | | |
| B4 | the robot's mistakes | read the rows of B2 | when perceived ≠ pushed, what is recalled? | |
| B5 | a long day | 24 pushes as `--episodes 3 --pushes 8`, against one `--pushes 24` episode | | |
| B6 | fewer directions, long | `--episodes 10 --directions 4 --pushes 24` | | |

5. B1/B2: does the robot's memory match the certificate? How close is close, with 80 or
   240 pushes measured?
6. B3: what does halving D cost? The certificate's capacity grows in proportion to D.
7. B4: why does the memory keep the robot's misperceptions? Could it ever correct one?
8. B4: both ≈ perception × memory. What does that say about the two kinds of error?
9. B5: the same 24 pushes are remembered much better as three episodes of 8. What is lost
   by cutting the day into episodes? (The `boundary` event ends an episode.)
10. B6: why do 4 directions raise the certificate, with the same D and the same episode?

## Show your results

Each group presents one experiment of its own choice (or a new one): the question, the
prediction, the command, the measured result (`results.csv`), and an explanation. Five
minutes per group.

## Going further

- Plot your own certificate curve against your measurements: `uv run python figures.py --run`
  re-measures the deck's figure (about 3 minutes).
- ubrain's laptop example (`02_memory.py` in the ubrain kit) runs the same memory on 500
  items: the vocabulary is the other knob of capacity.
