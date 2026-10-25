# Week 21 worksheet -- Holding a list: working memory

**Goal:** measure how many of its last 16 pushes the G1 holds in working memory, and which
ones, against the count law; then see what rehearsal moves and what it cannot.

**Slides:** `slides_students.html` in this folder (build it with
`uv run python weeks/21-working-memory/build.py`, or open it on the course site).

**Every command below runs from this folder**, `weeks/21-working-memory/`.

## Before the class (at home, about 2 minutes)

```bash
uv sync                          # from the repository root, as every week
cd weeks/21-working-memory
uv run lab_hold.py --lists 1     # about 5 s: one list, then recall by age beside the law
```

## The tool

```bash
uv run lab_hold.py --help
uv run lab_hold.py --law         # the prediction, no robot
```

A list is 16 pushes (week 20's pushes and sense). Each list prints **perceived** and
**recalled**, oldest first. At the end: recall by **age** (pushes since; 0 = the newest),
the law beside it, and the count **held** per list. Use `--lists 20` for numbers you report;
a single age point from 20 lists is ±0.1, the count is the robust number.

**Rule for every experiment: write your prediction before you run.**

## Part A -- the count law

| # | experiment | command | your prediction | measured |
|---|---|---|---|---|
| A1 | the law | `--law` | held of 16 at γ 0.75: | |
| A2 | the robot | `--lists 20` | | |
| A3 | a slower / faster fade | `--lists 20 --gamma 0.9`, then `--gamma 0.5` | | |
| A4 | twice the dimension | `--lists 20 --d 128` | | |
| A5 | about seven | `--law --gamma G --d D` for a few pairs | a second pair that holds about seven: | |
| A6 | a shorter list | `--lists 20 --pushes 7` | | |

1. A2: which positions are held: the first pushes of the list or the last? Why, from the
   weights γ^(age)?
2. A3: at γ 0.9 the robot holds more, but its newest push is no longer certain. Why?
3. A3/A4: which buys more items held, doubling d or moving γ from 0.75 to 0.9? Predict
   with `--law` first.
4. A5: is "about seven" a property of one setting, or of many? What does that say about
   explaining a human's span by one number?
5. A6: a list of 7 holds about 5. Why not 7?

## Part B -- rehearsal

`--rehearse R` gives the buffer R free moments before recall; each re-adds one held push
by the `--schedule` (cycle: every push in turn; newest / oldest: the 3 newest / oldest).

| # | experiment | command | your prediction | measured |
|---|---|---|---|---|
| B1 | a full cycle | `--lists 20 --rehearse 16` | held, and which ages: | |
| B2 | focus on the newest | `--lists 20 --rehearse 16 --schedule newest` | | |
| B3 | focus on the oldest | `--lists 20 --rehearse 16 --schedule oldest` | | |
| B4 | half a cycle | `--lists 20 --rehearse 8` | | |

6. B1: compare recall by age with A2, number by number. Why does a full cycle change
   nothing at all? (What happens to every weight, and what does a read compare?)
7. B2/B3: rehearsing three pushes makes them certain. What does it do to the count held,
   and why?
8. B4: half a cycle moves the strong positions. Which ones, and why those?
9. Week 20's hippocampus kept 16 pushes at about 0.76 each; working memory keeps the last
   few near 1.00 and the oldest at chance. Which should a robot use to react to the push it
   just felt, and which to learn from the walk afterwards?

## Show your results

Each group presents one experiment of its own choice (or a new one): the question, the
prediction, the command, the measured result (`results.csv`), and an explanation. Five
minutes per group.

## Going further

- Redraw the deck's figure from your own runs: `uv run python figures.py --run`.
- A random schedule (`--schedule random`): does it change the count? Which ages?
