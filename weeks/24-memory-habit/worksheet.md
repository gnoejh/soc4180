# Week 24 worksheet -- Memory first, habit later

**Goal:** run three brains on week 18's braking task -- the striatum, an episodic memory
that acts, and the chooser that hands over between them -- on the G1's measured world, and
find out when habit overtakes memory.

**Slides:** `slides_students.html` in this folder (build it with
`uv run python weeks/24-memory-habit/build.py`, or open it on the course site).

**Every command below runs from this folder**, `weeks/24-memory-habit/`.

## Before the class (at home, about 2 minutes)

```bash
uv sync                          # from the repository root, as every week
cd weeks/24-memory-habit
uv run lab_habit.py --seeds 2    # seconds: the world, then the three brains
```

## The tool

```bash
uv run lab_habit.py --help
```

The robot's world (stayed up or fell, for every cue, move and stride point) was measured
once on the G1 and is kept in `traces/world.json`; the brains play against it. Each run
prints the world, then each brain's share of trials the robot stayed up, by how many times
the cue had been seen (1, 2-3, 4-10, 11-30, 31-60, 61+), and the chooser's share of
decisions handed to memory. `--live` runs the robot itself.

**Rule for every experiment: write your prediction before you run.**

## Part A -- the robot

| # | experiment | command | your prediction | measured |
|---|---|---|---|---|
| A1 | three brains | default | late (61+): striatum, memory, chooser: | |
| A2 | early only | `--trials 40` | | |
| A3 | a forgetting memory | `--gamma 0.9`, then `--gamma 0.8` | | |
| A4 | watch memory act | `--live --brain memory --trials 20 --viewer` | | |

1. A1: the striatum ends near 0.66, memory near 0.81. Why? (Week 18: what does the race do
   with values near the payoffs? What does memory do with its votes?)
2. A1: the first time a cue is seen, every brain is near 0.59. Why that number?
3. A3: why does forgetting hurt memory so little on this task?
4. A4: after how many trials does memory stop choosing the wrong brake? Why so few?

## Part B -- situations, and the hand-over

| # | experiment | command | your prediction | measured |
|---|---|---|---|---|
| B1 | many situations | `--bandit 2`, `--bandit 16`, `--bandit 64` (`--trials 4000 --seeds 10`), `--bandit 256 --trials 20000 --seeds 5` | where habit overtakes: | |
| B2 | a small memory | `--d 16`, then `--d 8` | memory alone against the chooser: | |
| B3 | the hand-over's share | read "handed to memory" in B1 and B2 | | |
| B4 | your design | | | |

5. B1: why does memory's late reward fall as situations are added, while the striatum's
   does not?
6. B2: with a small memory the chooser does worse than memory alone. Why does it hand over
   more, and why does that cost reward here?
7. B3: the hand-over asks whether memory can tell its moves apart, not whether it is
   right. Find a setting where those come apart, and one where they agree.
8. B4: a robot that meets thousands of situations: memory, habit, or the chooser? What
   would you measure to decide?

## Show your results

Each group presents one experiment of its own choice (or a new one): the question, the
prediction, the command, the measured result (`results.csv`), and an explanation. Five
minutes per group.

## Going further

- Redraw the deck's figure: `uv run python figures.py --run` (about 2 minutes).
- Measure the robot's world again (`--remeasure`) and check that it comes back identical.
