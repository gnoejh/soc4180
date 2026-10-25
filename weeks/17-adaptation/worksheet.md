# Week 17 worksheet -- Walking in a side wind: the cerebellum adapts

**Goal:** change the G1's adaptation learner, predict what the robot will do, measure it,
and explain the result. A steady side wind (a force on the pelvis, sideways in the
robot's own heading frame) turns the walking robot; the cerebellum feels the turn (the
gyroscope's yaw rate), never the wind, and learns a counter-turn.

**Slides:** `slides_students.html` in this folder (build it with
`uv run python weeks/17-adaptation/build.py`, or open it on the course site).

**Every command below runs from this folder**, `weeks/17-adaptation/`.

## Before the class (at home, about 2 minutes)

```bash
uv sync                          # from the repository root, as every week
cd weeks/17-adaptation
uv run lab_adapt.py              # about 25 s: three blocks, no FELL line
```

If it does not end with three blocks, send its full output to your teacher before the
class. ubrain is the course's local copy in `vendor/ubrain/`; nothing else to install.

## The tools

```bash
uv run lab_adapt.py --help       # every flag and schedule, explained
uv run lab_adapt.py              # basic: 5 calm strides, 30 of wind, 30 calm
uv run lab_adapt_walk.py         # steer it yourself: W wind, L learning, X quit
```

A stride is 40 ticks, 0.8 s. Every run prints, for some strides, the robot's **turn**
(rad/s, + is to the left) and the learned **counter-turn**, then one line per block:
how many **degrees** the robot turned in it, the degrees in its first 10 strides, its
turn at the start and the end, and the counter-turn at the end. Every run adds a row to
`results.csv` in this folder; label it with `--name` (`--name "group 3: B2"`).

**Rule for every experiment: write your prediction before you run.**

## Part A -- adaptation and the after-effect

The single learner sums the error, `u <- u + eta e` (week 5's I term). The dual learner
has a fast state that learns quickly and forgets quickly, and a slow state that learns
slowly and keeps it; both forget a little every tick.

| # | experiment | command | your prediction | measured |
|---|---|---|---|---|
| A1 | no learning | `--learner off` | degrees turned in the wind: | |
| A2 | one rate, two rates | `--learner single`, then `--learner dual` | turn at the end of the wind; degrees after it stops: | |
| A3 | a weaker wind | `--wind 10` with each learner | strides until the turn is half its peak (read the stride list): | |
| A4 | the after-effect's size | compare A2's "turned after" with "turned in the wind" | which learner has the bigger after-effect, and why: | |
| A5 | the wind from the other side | `--wind -15` | | |
| A6 | too much wind | `--wind 20` with `off` and `dual` | | |

Questions:

1. A1: without learning, does the robot keep turning at the same rate for the whole
   wind block, or does the walking policy correct some of it? (Read the stride list.)
2. A2: why does the single learner's turn go to (almost) exactly zero, and the dual
   learner's not? Use week 5: which controller term leaves a steady error, and why?
3. A2: the dual learner's leftover turn at 10 N and at 15 N -- how do they compare? Why
   is that the signature of a steady error, not of slow learning?
4. A3: is a stronger wind learned more slowly? What does a linear learner predict?
5. A4: is a bigger after-effect a sign of better learning? Of what is it a sign?
6. A5: is the robot symmetric? Compare the degrees turned for +15 and -15 N.
7. A6: the counter-turn is learned at 20 N too. Why does the robot fall anyway?

## Part B -- savings, the error clamp, and spontaneous recovery

The **error clamp** (`c` after a block, `--blocks "...:0c"`) silences the teaching
signal: the counter-turn is still sent, and each state only forgets. A block of `u`
strides runs until the counter-turn crosses zero: unlearning, until behaviour is back
at baseline.

| # | experiment | command | your prediction | measured |
|---|---|---|---|---|
| B1 | the same wind twice | `--schedule savings` with `single` and `dual` | degrees in the first 10 strides, 1st and 2nd wind: | |
| B2 | a short break | `--blocks "5:0,30:15,6:0,20:15"` with `single` | | |
| B3 | spontaneous recovery | `--schedule recovery` with `single` and `dual` | counter-turn at the end of the clamp: | |
| B4 | walking slower | `--vx 0.3` with `off`, `single`, `dual` | | |
| B5 | your own schedule | `--blocks "..."` | | |

Questions:

8. B1: which learner relearns faster the second time, and by how much? What in the dual
   learner remembers the first wind?
9. B2: the second wind is learned much faster. Is that savings? Look at the counter-turn
   at the end of the break before you answer.
10. B3: after unlearning, the clamp leaves only forgetting. Why does the dual learner's
    counter-turn move back toward the adapted side, and the single learner's not?
11. B3: how many strides of counter-wind did each learner need to unlearn? Why fewer for
    the dual learner?
12. B4: at 0.3 m/s, does the robot adapt as well? Does the wind turn it more or less than
    at 0.5 m/s, per second and per metre walked?
13. Which learner would you put in a robot that is pushed by changing winds, and which in
    one that meets the same wind every day? Use your measurements.

## Show your results

Each group presents one experiment of its own choice (or a new one): the question, the
prediction, the command, the measured result (`results.csv` or `last_run.json`), and an
explanation. Five minutes per group.

## Going further

- Feel it: `uv run lab_adapt_walk.py`, press W, wait until the turn is small, press W
  again and watch the heading: the after-effect, in real time. Then L (learning off)
  before switching the wind off: what changes?
- Read the profile -- the same parameters, as the brain's anatomy:
  `uv run python -c "import ubrain_local; ubrain_local.load(); from ubrain import profile; print(profile('adapter'))"`
- Redraw the deck's figures from your own runs: `uv run python figures.py --run`.
