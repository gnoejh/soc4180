# Week 18 worksheet -- Learning from reward: the striatum

**Goal:** see what a learner taught only by a reward finds out, and what it then does.
First on a laptop bandit (seconds per run), then on the G1, which learns when to brake
against a push from whether it is still standing.

**Slides:** `slides_students.html` in this folder (build it with
`uv run python weeks/18-striatum/build.py`, or open it on the course site).

**Every command below runs from this folder**, `weeks/18-striatum/`.

## Before the class (at home, about 2 minutes)

```bash
uv sync                          # from the repository root, as every week
cd weeks/18-striatum
uv run lab_choose.py --trials 20 # about 15 s: ends with the values beside the world's payoffs
```

If it does not end with the values table, send its full output to your teacher before
the class. ubrain is the course's local copy in `vendor/ubrain/`; nothing else to install.

## The tools

```bash
uv run lab_striatum.py --help    # the laptop bandit: 3 situations x 3 actions, 6000 trials in seconds
uv run lab_choose.py --help      # the G1: 2 cues x 3 moves, about 0.5 s a trial
uv run lab_choose.py --map       # the G1's payoff table, measured with no brain
```

The striatum keeps a value Q per (situation, action), moves the chosen one toward the
outcome by `Q <- Q + eta (r - Q)`, and chooses by a race in which each action runs faster
the higher its value. **Matching** is choosing each action in proportion to what it pays;
for a row of payoffs p, it earns sum(p^2) / sum(p).

**Rule for every experiment: write your prediction before you run.**

## Part A -- warm-up: the laptop bandit (everyone, 15 minutes)

| # | experiment | command | your prediction | measured |
|---|---|---|---|---|
| A1 | the default world | `uv run lab_striatum.py` | reward rate, last third: | |
| A2 | one action clearly best | `--world easy` | best action chosen, against matching's 0.82: | |
| A3 | three close actions | `--world close` | | |
| A4 | the world changes | `--reverse 3000` | | |
| A5 | a slow learner | `--eta 0.01` | | |

1. A1: compare the choices of the last third with matching's column. Close?
2. A2: the striatum is MORE exclusive than matching here. Why? (How many outcomes does a
   rarely chosen action get, and what happens to its value?)
3. A4: after the payoffs swap halfway, does the striatum follow? How do you know from the
   values and the choices?

## Part B -- the G1 learns when to brake

A tone announces a push from behind (250 N, 0.1 s), either **soon** (0.2 s after the cue)
or **late** (0.5 s). The striatum picks **brake early**, **brake late** or **keep walking**.
Reward: 1 if the robot is still up 3 s after the push, 0 if it fell.

| # | experiment | command | your prediction | measured |
|---|---|---|---|---|
| B1 | the world, no brain | `--map` (about 40 s) | the best move after each cue; is the wrong brake better than nothing? | |
| B2 | the striatum learns | `uv run lab_choose.py` (120 trials, about a minute) | values at the end; stayed up in the last third: | |
| B3 | the same, longer | `--trials 300` | right brake chosen in the last third: | |
| B4 | a weaker push | `--map --force 230`, then `--force 230` | | |
| B5 | watch it | `--viewer --trials 20` | | |

4. B1: why is the wrong brake worse than keeping walking? (Week 16: what did the trained
   brace do while the push landed?)
5. B2: compare the values with the world's payoffs. Which value is furthest off, and how
   often was that move chosen?
6. B3: will the striatum ever pick the right brake every time? Compare with matching
   (0.52 for soon, 0.45 for late) and with "always the right brake" (stays up 0.85).
7. B4: at 230 N, what does the striatum learn, and why does its choice not matter?

## Part C -- the learning rate, and the cue

Single runs are noisy (the last third of 300 trials is 100 trials). **Pool with other
groups:** each group runs two seeds (`--seed 0`, `--seed 1`, ...), and the class averages
the rows of `results.csv`.

| # | experiment | command | your prediction | measured (class mean) |
|---|---|---|---|---|
| C1 | a slow learner | `--trials 300 --eta 0.05` | how close are the values to the world? stayed up: | |
| C2 | faster learners | `--trials 300 --eta 0.5`, then `0.8`, then `1.0` | | |
| C3 | no cue | `--trials 300 --blind` | | |

8. C1/C2: which learning rate knows the world best (values closest to the payoffs)? Which
   keeps the robot up most? Why are those different settings?
9. C2: what happens to a move's value right after one fall at eta 0.8? What does that do
   to the race?
10. C2: why is eta 1.0 worse again than 0.8?
11. C3: blind to the cue, what is the best the striatum can learn? How much is the cue
    worth? Compare week 16, where turning the tone off made the brace useless.
12. Week 16's brace learned to survive these pushes in about 5 trials. Why does the
    striatum need about a hundred, and still explore a quarter of the time?

## Show your results

Each group presents one experiment of its own choice (or a new one): the question, the
prediction, the command, the measured result (`results.csv`, `last_run.json`), and an
explanation. Five minutes per group.

## Going further

- Plot the values trial by trial from `last_run.json` (`Q_before_outcome`), or redraw
  the deck's figures from your own runs: `uv run python figures.py --run`.
- Change the moves: `WINDOW` at the top of `lab_choose.py` sets when each brake acts.
  Measure the new world with `--map` before you let the striatum loose on it.
- Read the profile: `uv run python -c "import ubrain_local; ubrain_local.load(); from ubrain import profile; print(profile('player'))"`
