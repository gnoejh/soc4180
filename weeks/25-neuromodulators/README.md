# 25 — One number each: serotonin and noradrenaline

| | |
| --- | --- |
| **Layer** | 5 — behaviour learned by local plasticity, not by gradient descent |
| **Runtime** | laptop, CPU |
| **Needs** | this repo's `uv sync`; ubrain ships in `vendor/ubrain/` (a local copy while ubrain is not public) |
| **Classes** | two: slides and a live demo, then a hands-on class |

A neuromodulator in ubrain is one number written onto a knob a region already has.
**Serotonin** is the striatum's patience (week 18): at a gust the G1 chooses to stop (a
sure 0.4 soon) or walk on (1.0 later, if it stays up -- the G1's own outcomes, from week
24's measured world, copied to `traces/world.json`); ubrain's law predicts how often it
walks on for each patience τ, and the adaptive τ's fixed point. **Noradrenaline** sets the
salience gate's criterion (week 23) from the recent rate of judged surprise; that part is
ubrain's example 10, ported (laptop only).

## Slides

HTML deck in week 16's format; the figure slide is drawn by `figures.py` from
`traces/patience.json` (`--run` re-measures it in seconds).

```bash
uv run python weeks/25-neuromodulators/build.py
```

## Before the class (at home, about 2 minutes)

```bash
uv sync
uv run weeks/25-neuromodulators/lab_patience.py serotonin
```

## Lab class: on your laptop

```bash
uv run weeks/25-neuromodulators/lab_patience.py serotonin              # τ 2, 10, adaptive
uv run weeks/25-neuromodulators/lab_patience.py serotonin --small 0.1  # a worse deal for stopping
uv run weeks/25-neuromodulators/lab_patience.py serotonin --tau 6
uv run weeks/25-neuromodulators/lab_patience.py noradrenaline
uv run weeks/25-neuromodulators/lab_patience.py noradrenaline --eta 0.1   # runs away
```

What they should see (measured 2026-10-25): P(walk on) 0.068 / 0.392 at τ 2 / 10 (law
0.064 / 0.430); the impatient robot earns most per moment when stopping pays 0.4 (0.093 vs
0.072) and the patient one when it pays 0.1 (0.049 vs 0.038); the adaptive τ settles at 5.85
(predicted 6.02); the noradrenaline criterion fixed at the wrong rate errs 0.051 (law 0.049),
the adaptive one 0.025 (0.026), and at η 0.1 it runs away to judging nothing novel (0.49).

Every run appends rows to `results.csv`. The exercises are in
[worksheet.md](worksheet.md); measured reference answers are in [instructor/reference.md](instructor/reference.md).
