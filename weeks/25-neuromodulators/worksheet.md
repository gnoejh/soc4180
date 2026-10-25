# Week 25 worksheet -- One number each: serotonin and noradrenaline

**Goal:** turn one knob at a time on regions you know -- serotonin on the striatum's
patience (on the G1's measured world), noradrenaline on the salience gate's criterion
(on the laptop) -- and check each against its law.

**Slides:** `slides_students.html` in this folder (build it with
`uv run python weeks/25-neuromodulators/build.py`, or open it on the course site).

**Every command below runs from this folder**, `weeks/25-neuromodulators/`.

## Before the class (at home, about 2 minutes)

```bash
uv sync                          # from the repository root, as every week
cd weeks/25-neuromodulators
uv run lab_patience.py serotonin # about a second
```

## The tool

```bash
uv run lab_patience.py serotonin --help
uv run lab_patience.py noradrenaline --help
```

**serotonin:** at each gust the robot chooses *stop* (a sure `--small` = 0.4, 2 moments
later) or *walk on* (1.0 if the G1 stays up, 8 moments later; the G1's own outcomes, from
`traces/world.json`: up at 22 of 40 stride points). It prints P(walk on), law and measured,
and the reward per moment, for each patience τ.

**noradrenaline:** a salience gate judges a stream against 64 known codes; it prints each
criterion's error, law and measured.

**Rule for every experiment: write your prediction before you run.**

## Part A -- serotonin

| # | experiment | command | your prediction | measured |
|---|---|---|---|---|
| A1 | by hand | compute P(walk) = 0.55 g^8 / (0.4 g^2 + 0.55 g^8), g = e^(−1/τ), for τ 2 and 10 | | |
| A2 | three patiences | `serotonin` | which earns more per moment? | |
| A3 | a different deal | `serotonin --small 0.2`, then `--small 0.1` | | |
| A4 | adapting | read the adaptive τ in A2 and A3 | where does τ settle? | |
| A5 | your patience | `serotonin --tau T` for a few T | the τ where the robot is indifferent (P = 0.5): | |

1. A2: why does more patience make the robot walk on more?
2. A2: the impatient robot earns most per moment when stopping pays 0.4. Why? Is it
   being "irrational"?
3. A3: when does patience pay? State serotonin's law in your own words.
4. A2: the measured P(walk on) is a little below the law at high τ. Walking on pays 0 or
   1, never 0.55. Why would a race in proportion to a noisy value choose it less often?
5. A4: the adaptive τ settles near the mean trial time. Why there?

## Part B -- noradrenaline (laptop)

| # | experiment | command | your prediction | measured |
|---|---|---|---|---|
| B1 | the arms | `noradrenaline` | which criterion errs most? | |
| B2 | adapting | read the adaptive arm and its level | | |
| B3 | too fast | `noradrenaline --eta 0.1` | | |

6. B1: the criterion fixed for 10 % new errs about twice as often once half the stream is
   new. Which error grows: new items missed, or repeats called new?
7. B2: after the change the adaptive level settles at 0.50. Why does it settle where it
   does (a fixed point)?
8. B3: with η 0.1 the level runs to 0 and the gate judges nothing novel. Explain the
   loop: what does the level count, and who decides what it counts?
9. A robot's world just changed. Which knob would you turn, and how fast?

## Show your results

Each group presents one experiment of its own choice (or a new one): the question, the
prediction, the command, the measured result (`results.csv`), and an explanation. Five
minutes per group.

## Going further

- Redraw the deck's figure: `uv run python figures.py --run` (seconds).
- Acetylcholine splits the cortex's plasticity between encoding the new and replaying
  the old (`neuromodulator.acetylcholine`, ubrain's `modulated` profile). Design a
  measurement on week 22's consolidation that would show it.
