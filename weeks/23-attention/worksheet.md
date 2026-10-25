# Week 23 worksheet -- Noticing: the salience gate and the modes

**Goal:** measure what a salience gate in front of the G1's memory skips and what it
writes, from one setting (a false-alarm rate), and what rest has to do with keeping it.

**Slides:** `slides_students.html` in this folder (build it with
`uv run python weeks/23-attention/build.py`, or open it on the course site).

**Every command below runs from this folder**, `weeks/23-attention/`.

## Before the class (at home, about 2 minutes)

```bash
uv sync                          # from the repository root, as every week
cd weeks/23-attention
uv run lab_notice.py --bursts 3  # about 5 s
```

## The tool

```bash
uv run lab_notice.py --help
```

Pushes come in bursts of 6, with quiet walking between (8 idle moments of 1 s). Each run
prints the first bursts column by column: **perceived**, the **gate** (+ written, -
skipped), the **mode** (E executive, d default). Then: how often a push was judged
familiar, split by whether its direction was felt within the gate's window, beside the
prediction; how much was written; the executive share; the replays into the cortex;
how much of what was written reached the cortex, and how much of that it recalls.

**Rule for every experiment: write your prediction before you run.**

## Part A -- the gate

| # | experiment | command | your prediction | measured |
|---|---|---|---|---|
| A1 | a random day | default | recent repeats skipped; written of 120: | |
| A2 | a gusty day | `--world gusty` | | |
| A3 | a looser gate | `--alpha 0.3` | which number moves? | |
| A4 | the gate's memory | `--window 2`, then `--window 16` | | |

1. A1: a push from a side felt within the last 8 pushes is skipped every time. Is that a
   feature or a bug for a robot that must notice change?
2. A1: the prediction for a fresh direction is 0.03 at α 0.05, the robot measures 0.00.
   Why can 8 fixed direction codes give exactly 0 while the prediction is not 0?
3. A3: at α 0.3, what share of fresh directions is lost? What did you buy for it?
4. A4: a short window writes most pushes. What happens to the cortex's recall, and why?
   What does a long window cost the robot?

## Part B -- modes and rest

| # | experiment | command | your prediction | measured |
|---|---|---|---|---|
| B1 | the modes | read the mode rows of A1 | when is the brain executive, when default? | |
| B2 | too little rest | `--quiet 4` | written pushes that reach the cortex: | |
| B3 | more rest | `--quiet 12` | | |
| B4 | a gusty day, rest | `--world gusty`, compare the executive share with A1 | | |

5. B1: why does the brain stay executive a few moments after each novel push?
6. B2: with 4 quiet moments only about half of what was written reaches the cortex. Why?
   Where are the other pushes when the run ends?
7. B3: does more rest help beyond 8 moments? Why not?
8. B4: why does a gusty day leave the brain resting more?

## Show your results

Each group presents one experiment of its own choice (or a new one): the question, the
prediction, the command, the measured result (`results.csv`), and an explanation. Five
minutes per group.

## Going further

- Redraw the deck's figure from your own runs: `uv run python figures.py --run`.
- Compute `familiar_threshold(d, 8, 0.05)` (from `ubrain.regions`) at d 32, 64, 512, and
  explain the trend from the geometry of random codes in high dimensions.
