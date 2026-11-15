# 10 — Planning and imitation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/10-planning-and-imitation/lab.ipynb)

**One task, one file.** The crouched G1 is shoved sideways on the torso (0.2 s at t = 1 s);
holding the crouch falls at 70 N. Part 1 survives 150 N by **planning** (MPPI, with a
`cost()` you write). Part 2 **copies** an expert — the planner, or week 8's ankle rule —
into a linear reflex fitted by least squares from the inputs you choose, and tests the
copy alone.

| | |
| --- | --- |
| **Layer** | 4 — Control & Planning (part 1); 5 — learning without a reward (part 2) |
| **Runtime** | CPU only. Colab: a CPU runtime is enough |
| **Wall clock** | `lab.py` ~4 min; the deck's cells ~15 s |
| **Needs** | `soc4180` only (no torch) |

## The lab

```bash
uv run weeks/10-planning-and-imitation/lab.py                 # plan, copy, test, replay
uv run weeks/10-planning-and-imitation/lab.py --push 300      # a harder shove
uv run weeks/10-planning-and-imitation/lab.py --no-viewer     # the tables only
```

**What you change**, at the top of `lab.py`:

- `cost()` — what is bad about an imagined future (stay tall, upright, mass over the
  feet, do not rush, do not fall). You never say *how* to balance.
- `SAMPLES, HORIZON, SIGMA, TEMPERATURE` — how hard the planner searches (64, 0.4 s,
  0.15 rad, 0.05).
- `INPUTS` (`tilt`, `spin`, `legs`, `speeds`, `time`) and `EXPERT` (`"planner"` or
  `"ankle"`) — what the copy sees, and whom it copies.
- `--push` — the test shove, newtons.

**What you see**, measured with `lab.py` on the course machine (your laptop differs in
milliseconds, not in who falls):

- Part 1, 150 N: holding falls; MPPI stands **by stepping** — its feet move **3.04 m**,
  because nothing in `cost()` asks it to stop. **81–98 ms per decision** on 16 threads
  (about 0.4× real time). At λ = 0.05 one sample of 64 decides every decision (effective
  sample size 1.0; averaging starts near λ = 5).
- Part 2, 400 decisions recorded on shoves of ±100 and ±150 N, who stands:

  | expert | inputs | fit error | 60 N | 75 N | 100 N | 150 N |
  | --- | --- | --- | --- | --- | --- | --- |
  | planner | tilt spin legs speeds (29) | 0.188 rad | FELL | FELL | FELL | FELL |
  | planner | tilt (2) | 0.201 rad | stood | stood | FELL | FELL |
  | ankle | tilt spin legs speeds (29) | 0.003 rad | FELL | FELL | FELL | FELL |
  | ankle | tilt (2) | 0.011 rad | stood | stood | FELL | FELL |

  The experts themselves: the planner stands at all four, the ankle rule at 60 and 75 N,
  holding at 60 N only. The planner's own spread is 0.202 rad and the fit leaves
  0.188–0.201 of it: what MPPI does is mostly its random samples, which no input
  explains. The 29-input ankle copy fits to 0.003 rad and falls everywhere: it reads the
  ankle angles (the servo's *result*, correlation 0.996 with the command, against 0.985
  for tilt) and gives tilt the wrong sign. More inputs made the copy worse.

**Try** (write the prediction first):

1. Delete the "still" term from `cost()`. Does it still stand? Does it stop?
2. Add a term that keeps the feet where they started. Does the drift stop?
3. `EXPERT = "ankle"`, then `INPUTS = ["tilt"]`: reproduce the table.
4. Add `"time"` to `INPUTS`. The planner never looks at the clock — does the copy improve?
5. `--push 300`: does the planner survive? Does the copy?

> **The rule:** the expert's action must be a function of what the student sees,
> deterministic, and closed loop.

## The deck

`slides.qmd` → `slides.html` + `lab.ipynb`. Every code slide prints `lab.py` itself
(`inspect.getsource`); the setup cell downloads `lab.py` on Colab. The cells run one
MPPI shove (4 s of robot, ~9 s here), the ankle copies live (~3 s) and some numpy. The
planner rows of the copy table are quoted from `lab.py`'s docstring (recording the planner
takes minutes).

The deck opens with **where this week sits on the map** of RL methods from week 8, and every
major function of `lab.py` is printed and then explained line by line (`| code | what it does | why |`).

## Term project

Weeks 12–15 ([`../projects/README.md`](../projects/README.md),
[`../projects/catalogue.md`](../projects/catalogue.md)). From this week: **P5** push
recovery by planning (recover *and* stop), **P6** new tasks for the planner, **P7** a
planner with a wrong model, **P10** distil the expert (add DAgger), **P14** your own.
Start from `lab.py` and `soc4180.planning`. The method: a prediction first, a twin with
one change, `results.csv`, a `CHECK` line; grade by margins; never a float from a fall.
