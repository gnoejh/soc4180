# 16 — The cerebellum in a walking robot

| | |
| --- | --- |
| **Layer** | 5 — behaviour learned by local plasticity, not by gradient descent |
| **Runtime** | laptop, CPU |
| **Needs** | **ubrain**, a copy handed out by the instructor; *not* this repo's `uv.lock` |
| **Classes** | two: slides and a live demo, then a hands-on class |

From this week on the course teaches **ubrain**, an untrained brain that learns from
experience while it runs. Here its cerebellum sits on top of the G1's frozen walking
policy (the "spinal cord") and does two things: a **forward model** that predicts how
the body will move next, and a **brace** that learns a timed counter-move to a push a
tone announces (eyeblink conditioning).

This is the first week whose lab does not run on the `soc4180` package. Students need
a copy of ubrain, set up once at home.

**ubrain is not public yet.** Its repository (github.com/untrained-brain) opens after
ICLR 2027. Until then the instructor hands out the ubrain folder directly; do not
redistribute it.

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

## Before the class (at home, about 10 minutes)

Unpack the ubrain folder from the instructor, then:

```bash
cd ubrain
cd bodies/mujoco && uv sync && cd ../..      # the G1 body (a few minutes, once)
uv run examples/check_setup.py               # must end with READY
```

If the check does not say READY, send its full output to the instructor before the
class. The commands below run **from the ubrain folder**, not from this repository.

## Lab class: on your laptop

```bash
uv run examples/lab_cerebellum.py --help        # every parameter, explained
uv run examples/lab_cerebellum.py --viewer      # walk; the forward model learns
uv run examples/lab_cerebellum.py --viewer --pushes 6 --bracing off   # tone, then push
uv run examples/lab_cerebellum.py --viewer --pushes 12                # learn to brace
```

What students change: the forward model's granule count (`--granules`), sparseness
(`--active`), learning rate (`--eta-forward`) and random wiring (`--seed`); the
brace's push force, tone-to-push delay, trace decay (`--tau`) and learning rate
(`--eta-brace`).

What they should see:

- every second, the forward model's **prediction error** as a percentage of predicting
  "no change" (100 % = no better than not predicting), falling as it learns its body;
- with pushes, one mark per push: **X** fell, **.** stayed up — with bracing off it
  keeps falling, with bracing on the falls stop once the brace is learned.

Every run appends a row to `results.csv` (use `--name` to label it), so a group or the
whole class can compare runs. **Write the prediction before each run.** The full
exercise sheet is `docs/labs/lab4_walker.md` in the ubrain folder.
