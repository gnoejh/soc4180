# 16 — The cerebellum in a walking robot

| | |
| --- | --- |
| **Layer** | 5 — behaviour learned by local plasticity, not by gradient descent |
| **Runtime** | laptop, CPU |
| **Needs** | this repo's `uv sync`; ubrain ships in `vendor/` |
| **Classes** | two: slides and a live demo, then a hands-on class |

From this week on the course teaches **ubrain**, an untrained brain that learns from
experience while it runs. Here its cerebellum sits on top of the G1's frozen walking
policy (the "spinal cord") and does two things: a **forward model** that predicts how
the body will move next, and a **brace** that learns a timed counter-move to a push a
tone announces (eyeblink conditioning).

This is the first week whose lab does not run on the `soc4180` package. It runs on
ubrain's compiled wheel and its G1 body server, both in `vendor/` in this folder:
`vendor/wheels/` (ubrain 0.13.0.dev1, Python 3.12, under the ubrain Binary Licence in
`vendor/wheels/LICENSE-BINARY`) and `vendor/g1_body/` (Apache-2.0; the walking policy is
Unitree's, BSD-3-Clause).

## Slides

The deck is plain HTML, not a Quarto `slides.qmd`: `deck.json` gives the order and
`slides/<id>.html` holds one slide each. Assemble it with

```bash
uv run python weeks/16-cerebellum/build.py
```

which writes `slides.html` (speaker notes hidden; **N** shows them) and
`slides_students.html` (no notes). Both are build artifacts and gitignored. Open either
file in a browser: arrow keys or click to move, **F** for full screen.

The sources were copied from `papercode/teaching/lecture_robot/cerebellum_g1` and are
the same files as the claude.ai deck; `build.py` only assembles them.

Sections:

1. What the two classes are for
2. Cerebellum principles: the circuit, the forward model, timing
3. How the cerebellum is connected to the G1
4. The code: the wiring, the tick loop, the parameters
5. The demo and the hands-on class

## Before the class (at home, about 2 minutes)

Pull the repository (or clone it), then from the repository root as for every other week:

```bash
uv sync
uv run weeks/16-cerebellum/lab_cerebellum_walk.py --headless --seconds 5
```

It must print `body 4815eedf56d0 ...` and end with `falls 0`. Nothing is installed
beyond `uv sync`: `ubrain_local.py` unpacks the wheel for your laptop (Windows, macOS
Intel or Apple silicon, Linux x86-64; Python 3.12, which `uv` already picks) and starts
the body on this repository's own Python, whose mujoco 3.12.0 / numpy 2.5.2 /
menagerie 2026.9.0 are exactly the body's pins. If it fails, send the full output to the
instructor before the class.

## Lab class: on your laptop

Two scripts, both copied from ubrain's examples and changed only to run here.
The exercises (Part A, the forward model; Part B, the brace) are in
[worksheet.md](worksheet.md). The worksheet and the slides `cd weeks/16-cerebellum` first
and drop the prefix; both forms work.

**`lab_cerebellum_walk.py`: drive it with the keys** (from ubrain's `g1_walker.py`). The
keys set the walking goal; the cerebellum's forward model learns while it walks, and
the status line shows its prediction error falling, jumping when you change the command,
and falling again.

```bash
uv run weeks/16-cerebellum/lab_cerebellum_walk.py
```

| Key | Does |
| --- | --- |
| Up / Down (W / S) | faster / slower, 0.1 m/s a press (past zero walks backward) |
| Left / Right (A / D) | turn, 0.2 rad/s a press; **C** straight again |
| Q / E | sidestep left / right |
| space | stop; **R** resets the body; **X** quits |
| P | the tone, then a 265 N push from the front 0.2 s later: press it again and again, and watch it learn to brace |
| B | bracing on / off (it keeps learning either way) |

Keys work in the viewer window **and in the terminal** you started it from. If the window
ignores keys (an IME, focus), type them in the terminal. Ctrl + right-drag in the viewer
pushes the robot by hand.

**`lab_cerebellum.py`: measured runs with parameters.**

```bash
uv run weeks/16-cerebellum/lab_cerebellum.py --help        # every parameter, explained
uv run weeks/16-cerebellum/lab_cerebellum.py --viewer      # walk; the forward model learns
uv run weeks/16-cerebellum/lab_cerebellum.py --pushes 6 --bracing off   # tone, then push
uv run weeks/16-cerebellum/lab_cerebellum.py --pushes 12                # learn to brace
```

What students change: the forward model's granule count (`--granules`), sparseness
(`--active`), learning rate (`--eta-forward`) and random wiring (`--seed`); the
brace's push force, tone-to-push delay, trace decay (`--tau`) and learning rate
(`--eta-brace`). And the conditioning experiment around it, each run against its
`--bracing off` control: no cue (`--tone off`), where the push comes from (`--from
behind|front|right|left`), an unreliable interval (`--jitter`), and extinction
(`--tone-only N --test M`). In `lab_cerebellum_walk.py` the same knobs are `PUSH_N`,
`PUSH_DELAY` and `PUSH_DIR` at the top of the file. Not `HZ`: the body always advances
20 ms per tick, so `HZ` only changes playback speed. The worksheet's Part B is built on
these; measured reference results are in `instructor/` (gitignored).

What they should see (measured on the instructor's machine, 2026-10-22, defaults):

- every second, the forward model's **prediction error** as a percentage of predicting
  "no change" (100 % = no better than not predicting), falling as it learns its body:
  89 % at 2 s, 38 % at 4 s, 11 % at 8 s;
- with pushes, one mark per push: **X** fell, **.** stayed up. `--pushes 6 --bracing off`:
  `XXXXXX`, fell 6 of 6. `--pushes 12`: `XXXXX.X..X..`, fell 7 of 12, the falls stopping
  once the brace is learned. Falls are chaotic, so a laptop's marks will differ; the
  trend, not the string, is the result.

Every run appends a row to `weeks/16-cerebellum/results.csv` (use `--name` to label
it), so a group or the whole class can compare runs. **Write the prediction before each
run.** `--plot` needs matplotlib, which `uv.lock` does not install:
`uv run --with matplotlib weeks/16-cerebellum/lab_cerebellum.py --plot`.
