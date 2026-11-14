# 17 — Senses and the body (project week 2)

| | |
| --- | --- |
| **Layer** | 4 and 5 — the G1's body as streams of the whole untrained brain; a frozen eye (CLIP) as an option |
| **Runtime** | laptop, CPU |
| **Needs** | `uv sync`; step vision also `--extra vision` and the CLIP weights (fetched once, about 600 MB); ubrain 0.46 and the body ship in `vendor/ubrain/` |
| **Classes** | two: slides and live runs, then a hands-on class and the project's first end-to-end run |

**ubrain 0.46 gives the whole brain the body.** Week 16 walked the G1 in one step; this
week is the body in full. `Brain()` drives it through four streams: `body.proprio` (70 numbers of
proprioception), `cue.tone` (the tone before a push), `goal` (the walk the program wants) and
`command` (the walk the brain sends: the goal plus the cerebellum's counter-move).
`ubrain.bodies.g1.G1` attaches them and converts a body state into a tick and a command into
an act request. Inside is a certified cerebellum, held bit for bit to the circuit it was
certified in (a preview: the whole does not disturb it).

The week sorts every signal into three kinds, and the project report must say which each of
its signals is: **a sense** (the brain receives it as a stream), **a detector** (the program
computes it, knowing what it did: `push_direction`), **a privilege** (the simulator has it, a
real robot does not: the feet's contact forces, the robot's position).

## Slides

HTML deck (deck.json and slides/); every number on a slide comes from a run of `lab_sense.py`.

```bash
uv run python weeks/17-senses/build.py
```

## Before the class (about 3 minutes)

```bash
uv run weeks/17-senses/lab_sense.py --step brace
```

It must print `ubrain 0.46.0.dev2` and end with `A hears the tone ..X.XX.............. falls 3`
and `B does not hear ..X.X.X.X.XXXXXXXXXX falls 14`.

## The lab

`lab_sense.py --step brace | push | contact | vision`:

| step | what it shows | measured (defaults) |
| --- | --- | --- |
| brace | the twin: one brain hears the tone, one does not; 20 pushes of 250 N | falls 3 against 14 |
| push | `push_direction`, a detector: 16 pushes of 100 N from a random side | 15 of 16 named right |
| contact | the feet's forces (privileged) over 4 s of walking at 0.4 m/s | both feet 0.12, one foot 0.88; 328 N on the feet |
| vision | the head camera through CLIP every 10th tick, 40 s down the gallery (`--extra vision`) | 199 glances, 22 called new |

Options: `--pushes`, `--force`, `--vx`, `--delay` (ticks from the tone to the push),
`--seconds`, `--viewer`, `--seed`. Every run appends rows to `results.csv`.

## For the instructor

`instructor/reference.md` holds the measured answers. `scripts/check_labs.py --only 29` runs
steps brace (short), push and contact; step vision needs the vision extra (`uv run --extra vision`).
