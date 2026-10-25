# 17 — Walking in a side wind: the cerebellum adapts

| | |
| --- | --- |
| **Layer** | 5 — behaviour learned by local plasticity, not by gradient descent (on top of layer 4's frozen walking policy) |
| **Runtime** | laptop, CPU |
| **Needs** | this repo's `uv sync`; ubrain ships in `vendor/ubrain/` (a local copy while ubrain is not public) |
| **Classes** | two: slides and a live demo, then a hands-on class |

A steady side wind (a sustained force on the pelvis, sideways in the robot's own heading
frame) turns the walking G1. The robot has no heading sensor (week 6: yaw is
privileged), but it feels its turn rate. ubrain's `adapter` profile (week 16's walker
plus `cerebellum.adapt`) learns a counter-turn from that turn and adds it to the walking
command, with one learning rate (an integrator: week 5's I term) or two (a fast and a
slow state that forget: Smith, Ghazizadeh and Shadmehr 2006). The week measures what
each predicts on the robot: adaptation, the after-effect, savings and spontaneous
recovery under an error clamp, and the limit (20 N falls, learning or not).

## Slides

HTML deck in week 16's format: `deck.json` and `slides/<id>.html`. Assemble it with

```bash
uv run python weeks/17-adaptation/build.py
```

which writes `slides.html` (speaker notes hidden; **N** shows them) and
`slides_students.html` (no notes); both are build artifacts. The two figure slides are
drawn from measured traces by `figures.py` (`--run` re-measures them, about 3 minutes;
the traces are in `traces/`).

Sections: why a steady push turns the robot; the law, one rate or two, and what each
predicts; the measurements; the code (`adapter.toml`, the tick loop, the flags); the
demo and the hands-on class.

## Before the class (at home, about 2 minutes)

```bash
uv sync
uv run weeks/17-adaptation/lab_adapt.py
```

It must print `body 4815eedf56d0 ...` and end with three blocks and no `FELL` line.

## Lab class: on your laptop

```bash
uv run weeks/17-adaptation/lab_adapt.py --help             # every flag and schedule
uv run weeks/17-adaptation/lab_adapt.py --learner off      # no learning: 230 degrees in 24 s
uv run weeks/17-adaptation/lab_adapt.py                    # two rates: +70, then -25 after
uv run weeks/17-adaptation/lab_adapt.py --learner single   # one rate: +44, then -43 after
uv run weeks/17-adaptation/lab_adapt.py --schedule savings
uv run weeks/17-adaptation/lab_adapt.py --schedule recovery
uv run weeks/17-adaptation/lab_adapt_walk.py               # steer it: W wind, L learning, X quit
```

What students change: the learner (`--learner dual|single|off`), the schedule
(`--schedule basic|savings|recovery`, or their own with `--blocks "5:0,30:15,30:0c"`:
strides:newtons, `c` clamped, `u` strides = until the counter-turn is back at zero), the
wind (`--wind`), the walking speed (`--vx`). In `lab_adapt_walk.py`: `WIND_N` and
`WIND_STEP` at the top, and the keys.

What they should see (measured 2026-10-25, defaults):

- with no learning the robot turns at 0.17 rad/s, 230° in the 24 s of wind;
- with learning the turn falls (halved 3–4 strides after its peak, at 10 N and at 15 N
  alike); one rate ends at a turn of 0.001 rad/s, two rates at 0.036 (it doubles with the
  wind: a steady error, like week 5's sag);
- when the wind stops, both turn the other way: −43° (one rate), −25° (two rates);
- savings only with two rates and only a little (34.0° → 31.4° in the first 10 strides);
- after unlearning and under the clamp, two rates drift back to the adapted side
  (counter-turn −0.027), one rate does not (+0.015);
- 20 N falls in 8 (no learning) or 11 (two rates) strides: adaptation corrects heading,
  not balance.

Every run appends a row to `weeks/17-adaptation/results.csv` (`--name` labels it) and
saves the whole trace in `last_run.json`. The exercises are in
[worksheet.md](worksheet.md); measured reference answers are in [instructor/reference.md](instructor/reference.md).
