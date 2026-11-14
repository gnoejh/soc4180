# 18 — Build and measure (project week 3)

| | |
| --- | --- |
| **Layer** | 5 — a project on the whole untrained brain in the G1, measured against a twin |
| **Runtime** | laptop, CPU |
| **Needs** | `uv sync`; ubrain 0.46 and the body ship in `vendor/ubrain/` |
| **Classes** | two: slides and the three project starts measured live, then a hands-on class with the checklist |

The week's method, five parts in order: **a question** about the brain in the G1, **a
prediction written before the run** (a number and its reason), **a twin** (the same run with
one thing changed), **a measurement** (a script that writes its numbers to a file) and **a
check** (one PASS/FAIL line decided before the run). The three labs are ubrain's
`examples/05_g1_choose.py`, `06_g1_keep.py` and `07_g1_lead.py`, run on the course's copy; each ends with its CHECK line and
appends it to `results.csv`.

| lab | the question | measured (defaults) |
| --- | --- | --- |
| `lab_choose.py` | from a picture GO LEFT / GO RIGHT, does the brain learn which way to turn? reward from the integrated gyro (a detector) | the brain that sees: wrong twice, then right every trial (1.00 in the last half); the twin shown unrelated pictures 0.35 (chance 0.5); PASS |
| `lab_keep.py` | train the brace, save, demonstrate the LOADED brain against a fresh twin | loaded 0 falls in 10 pushes, fresh 3; saved 11.2 MB; PASS |
| `lab_brace.py` | how long a tone's lead can the cerebellum use? | lead 5: 7 falls; 10: 3; 20: 4; 40: 4; no tone 14 -- the trace-law prediction (falls rise with the lead) FALSIFIED; PASS on what the data supports |

Two rules the labs found, now in the slides: **the whole brain explores only while its values
are near zero** (make a wrong choice pay nothing: a first design that paid the wrong action
half the time was never unlearned), and **a sense that never changes never engages the brain**
(week 17).

## Slides

```bash
uv run python weeks/18-build/build.py
```

## Before the class (about 5 minutes)

```bash
uv run weeks/18-build/lab_choose.py
uv run weeks/18-build/lab_keep.py
```

Both must end with `CHECK PASS`. `lab_brace.py` takes about 3 minutes (five brains).

## For the instructor

`instructor/reference.md` holds the measured answers. `scripts/check_labs.py --only 30` runs
the three labs (brace at one lead). The presentation's length per team is yours to set; the
slides leave a placeholder.
