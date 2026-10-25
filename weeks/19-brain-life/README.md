# 19 — A brain's life: save, load, datasheet, health

| | |
| --- | --- |
| **Layer** | 5 — behaviour learned by local plasticity; this week, the brain as a device |
| **Runtime** | laptop, CPU |
| **Needs** | this repo's `uv sync`; ubrain 0.14 ships in `vendor/ubrain/` (a local copy while ubrain is not public) |
| **Classes** | two: slides and a live demo, then a hands-on class |

An untrained brain learns all its life, so what it holds is what it learned. ubrain 0.14
makes that a file and a set of instruments, shown here on week 16's `walker` on the G1:
`brain.save` / `Brain.load` (a brain that continues exactly, and refuses another body),
`datasheet` (figures from the regions' laws before a single tick: the brace's lead
1 / (1 − τ)), `health` (the forward model against six tenths, the walking robot's bar),
`observe` / `monitor` (a brain's interior as channels another brain could read).

## Slides

HTML deck in week 16's format: `deck.json` and `slides/<id>.html`; every number on them
was measured with `lab_life.py`. Assemble with

```bash
uv run python weeks/19-brain-life/build.py
```

which writes `slides.html` (speaker notes hidden; **N** shows them) and
`slides_students.html`; both are build artifacts.

## Before the class (at home, about 2 minutes)

```bash
uv sync
uv run weeks/19-brain-life/lab_life.py sheet
```

It must print the walker's datasheet (granule code 128 of 2048; brace lead 10.0).

## Lab class: on your laptop

```bash
uv run weeks/19-brain-life/lab_life.py train          # 12 pushes, saved to brains/brace.npz
uv run weeks/19-brain-life/lab_life.py test           # (in a new terminal) loaded: braced at once
uv run weeks/19-brain-life/lab_life.py test --fresh
uv run weeks/19-brain-life/lab_life.py identical      # 500 of 500 commands identical
uv run weeks/19-brain-life/lab_life.py refuse
uv run weeks/19-brain-life/lab_life.py sheet --tau 0.97 --measure 12
uv run weeks/19-brain-life/lab_life.py watch          # health and the monitor while it walks
```

What students change: τ and the push delay (`--tau`, `--delay`) against the datasheet,
the number of pushes, which brain file is loaded (their own, another group's), and the
watch length; in `watch --load` the brain whose health they read.

What they should see (measured 2026-10-25):

- `train`: `XXXXX.X..X..`, a 1.8 MB file whose header names wiring `walker` and body
  `mujoco/g1 (4815eedf56d0)`;
- `test`: the loaded brain `.X..X.` against a fresh brain's `XXXXX.`;
- `identical`: state equal at the save, 500 of 500 commands identical on two bodies,
  state equal at the end;
- `refuse`: StateError, "the state was learned on mujoco/g1 (4815eedf56d0), not ...";
- `sheet`: lead 2 / 10 / 33.3 ticks at τ 0.5 / 0.9 / 0.97; pushes at 10 ticks fall 12 /
  7 / 2 of 12; at 25 ticks 9 / 3 / 2 of 12;
- `watch`: health False at 2 s (0.89), True from 3 s (0.54); a turn lifts the error from
  0.06 to 0.57.

Saved brains go in `brains/` (gitignored). Every run appends a row to `results.csv`.
The exercises are in [worksheet.md](worksheet.md); measured reference answers are in
[instructor/reference.md](instructor/reference.md).
