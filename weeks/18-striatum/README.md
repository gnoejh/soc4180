# 18 — Learning from reward: the striatum

| | |
| --- | --- |
| **Layer** | 5 — behaviour learned by local plasticity, not by gradient descent |
| **Runtime** | laptop, CPU |
| **Needs** | this repo's `uv sync`; ubrain ships in `vendor/ubrain/` (a local copy while ubrain is not public) |
| **Classes** | two: slides and a live demo, then a hands-on class |

Weeks 16 and 17 taught the cerebellum with an error. This week's teacher is a reward: one
number, with no direction in it. ubrain's `player` profile (`basal_ganglia.striatum`)
keeps a value per (situation, action), learns `Q <- Q + eta (r - Q)`, and chooses by a
race. First on a laptop bandit (`lab_striatum.py`, from ubrain's `01_bandit.py`), then on
the G1 (`lab_choose.py`): a cue announces a push from behind, soon or late, and the
striatum picks brake early, brake late or keep walking; the reward is whether the robot
is still standing 3 s later.

## Slides

HTML deck in week 16's format: `deck.json` and `slides/<id>.html`. Assemble it with

```bash
uv run python weeks/18-striatum/build.py
```

which writes `slides.html` (speaker notes hidden; **N** shows them) and
`slides_students.html` (no notes); both are build artifacts. The two figure slides are
drawn from measured trial logs by `figures.py` (`--run` re-measures them: 21 runs in
parallel, about 3 minutes; the logs are in `traces/`).

Sections: two teachers (an error, a reward); the striatum's law and its race; the laptop
bandit; the G1 as a bandit and its payoff table; what the striatum learns (the values)
and what it chooses (matching); the learning rate's inverted U; the cue; the brace and
the striatum compared; the code; the demo and the hands-on class.

## Before the class (at home, about 2 minutes)

```bash
uv sync
uv run weeks/18-striatum/lab_choose.py --trials 20
```

It must print `body 4815eedf56d0 ...` and end with the values beside the world's payoffs.

## Lab class: on your laptop

```bash
uv run weeks/18-striatum/lab_striatum.py                     # the laptop bandit, seconds
uv run weeks/18-striatum/lab_choose.py --map                 # the G1's payoffs, no brain (~40 s)
uv run weeks/18-striatum/lab_choose.py                       # 120 trials, about a minute
uv run weeks/18-striatum/lab_choose.py --trials 300 --eta 0.8
uv run weeks/18-striatum/lab_choose.py --trials 300 --blind
uv run weeks/18-striatum/lab_choose.py --viewer --trials 20  # watch it (6 s a trial)
```

What students change: in `lab_striatum.py` the world (`--world`), the learning rate, a
mid-run reversal; in `lab_choose.py` the learning rate (`--eta`), the starting values
(`--q0`), the number of trials, the cue (`--blind`), the world (`--force`, `--brake`,
`WINDOW` at the top of the file), and `--map` to measure any world before learning in it.

What they should see (measured 2026-10-25):

- the G1's payoffs (`--map --phases 40`): the brake timed to the push keeps it up 0.85,
  the wrong brake 0.25 or 0.48 (worse than keeping walking, 0.55);
- the values approach those payoffs (eta 0.05: mean error 0.06 over 4 seeds);
- the choices sit at matching: the right brake 0.48 of the time over 300 trials (matching
  0.48), staying up 0.65 (matching 0.66; always the right brake 0.85);
- the learning rate's inverted U: staying up 0.645 / 0.647 / 0.745 / 0.788 / 0.680 at
  eta 0.05 / 0.2 / 0.5 / 0.8 / 1.0, while the values get less accurate all the way;
- blind to the cue: 0.60 (matching predicts 0.59).

Every run appends a row to `results.csv` (`--name` labels it); `lab_choose.py` saves every
trial with the values in `last_run.json`. The exercises are in [worksheet.md](worksheet.md);
measured reference answers are in [instructor/reference.md](instructor/reference.md).
