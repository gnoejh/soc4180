# Week 22 worksheet -- From hippocampus to cortex: consolidation

**Goal:** follow the robot's pushes through two stores -- the episode, replayed at each
boundary into the cortex -- measure each stage against its law, and find out what happens
when the robot's world does not match the cortex's assumption.

**Slides:** `slides_students.html` in this folder (build it with
`uv run python weeks/22-consolidation/build.py`, or open it on the course site).

**Every command below runs from this folder**, `weeks/22-consolidation/`.

## Before the class (at home, about 2 minutes)

```bash
uv sync                          # from the repository root, as every week
cd weeks/22-consolidation
uv run lab_consolidate.py --episodes 2   # about 5 s
```

## The tool

```bash
uv run lab_consolidate.py --help
```

Episodes of 8 pushes (week 20's pushes and sense); a boundary after each. The run prints
three stages beside their laws: **hippocampus** (retrieved = perceived, before the
boundary), **cortex** (recollected = retrieved: faithful), **end to end** (recollected =
perceived); the hippocampus after each boundary; and the cortex by episode.

**Rule for every experiment: write your prediction before you run.**

## Part A -- the stages, and the law's assumption

| # | experiment | command | your prediction | measured |
|---|---|---|---|---|
| A1 | 10 episodes | default | hippocampus, cortex, end to end: | |
| A2 | more episodes | `--episodes 20`, then `--episodes 40` | | |
| A3 | the law's world | `--distinct --episodes 20`, then `--episodes 40` | | |
| A4 | after the boundary | read the "hippocampus after each boundary" line | | |

1. A1/A2: why does the hippocampus's rate stay flat as episodes are added, while the
   cortex's falls?
2. A1: end to end ≈ hippocampus × cortex. Why the product?
3. A2/A3: at 160 pushes the robot's cortex reads 0.62 where its law says 1.00, but the
   `--distinct` cortex matches its law. What is different about the robot's world? Why do
   copies of one direction hurt every read?
4. A3: in `--distinct` the hippocampus is at 0.45, not 0.94. Why? (What does it decode
   against?)
5. A4: what does the hippocampus give back after a boundary? Why is that the right
   behaviour for a store that hands its contents on?

## Part B -- forgetting, and the dimension

| # | experiment | command | your prediction | measured |
|---|---|---|---|---|
| B1 | a forgetting cortex | `--episodes 20 --gamma 0.99`, then `--gamma 0.97` | which episodes survive: | |
| B2 | a bigger cortex | `--episodes 20 --d 128`, then `--d 32` | | |

6. B1: which episodes does a forgetting cortex keep? Compare week 21's working memory:
   what is the same, what is different?
7. B1: why might a robot want a cortex that forgets?
8. B2: d 128 helps the robot's cortex (its capacity grows with d²) but does it bring it to
   its law? What would?

## Show your results

Each group presents one experiment of its own choice (or a new one): the question, the
prediction, the command, the measured result (`results.csv`), and an explanation. Five
minutes per group.

## Going further

- Redraw the deck's figure from your own runs: `uv run python figures.py --run`.
- Episodes of 4 instead of 8 (`--pushes 4 --episodes 40`): the same number of pushes in
  the cortex, a better hippocampus. Which stage limits end to end now?
