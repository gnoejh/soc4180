# Lab 4 -- A brain in a body: the cerebellum

**Goal:** change the G1's cerebellum, predict what will happen, measure it, and explain
the result. You will work with its two parts: the **forward model**, which predicts how
the body moves, and the **brace**, which learns a timed counter-move to a push it hears
coming.

**Slides:** [lab4_slides.html](lab4_slides.html) -- the cerebellum, the G1 connection
and the code, from the first class. Open the file from your clone in a browser (arrow keys
to move, F for full screen); GitHub shows it only as source.

## Before the class (at home, about 10 minutes)

```bash
git clone https://github.com/untrained-brain/ubrain
cd ubrain
cd bodies/mujoco && uv sync && cd ../..      # the G1 body (a few minutes, once)
uv run examples/check_setup.py               # must end with READY
```

If the check does not say READY, send its full output to your teacher before the class.

## The tool

```bash
uv run examples/lab_cerebellum.py --help        # every parameter, explained
uv run examples/lab_cerebellum.py               # 30 s walk with the default cerebellum
uv run examples/lab_cerebellum.py --viewer      # the same, watching the robot
```

Each run prints, every second, the forward model's **prediction error** as a percentage
of the error of predicting "no change" (100 % = no better than not predicting; lower is
better), and at the end its average over the last 5 seconds. With pushes it also prints
one mark per push: **X** the robot fell, **.** it stayed up. Every run adds one row to
`results.csv`; use `--name` to label your runs (for example `--name "group 3: sparse"`).
`uv run --extra plot examples/lab_cerebellum.py --plot` also saves a PNG of the curve.

**Rule for every experiment: write your prediction before you run.** The most
interesting results are the ones you predicted wrong.

## Part A -- the forward model (walking, no pushes)

The body state and the walking command enter through **mossy fibres** and are expanded
into many **granule cells** (`--granules`); only the most active few may fire
(`--active`). **Purkinje cells** read the granule code out as a prediction of the next
body state, and the prediction error (the **climbing fibre** signal) corrects the read-out
weights by the delta rule at rate `--eta-forward`.

| # | experiment | change | your prediction | measured (last 5 s) |
|---|---|---|---|---|
| A1 | the default | nothing | | |
| A2 | sparser code | `--active 8`, then `--active 32` | | |
| A3 | no sparseness | `--active 2048` (all granule cells fire) | | |
| A4 | fewer / more granule cells | `--granules 256 --active 16`, `--granules 8192 --active 512` | | |
| A5 | slow learning | `--eta-forward 0.05` | | |
| A6 | fast learning | `--eta-forward 1.0`, then `1.5`, `2.0`, `3.0` | | |

Questions:
1. Why is a sparse code better than letting every granule cell fire? Why is *too*
   sparse worse again?
2. Why do more granule cells help? What would it cost a real brain?
3. At some learning rate the error explodes. Between which values? Explain with the
   delta rule: what happens when each correction overshoots?
4. Turn while walking (`--wz 0.5`). Does the error rise? Why does the model have to
   learn turning separately?

## Part B -- the brace (tone, then push)

A tone sounds and 0.2 s later the walking robot is pushed from the front (`--pushes 12`).
The brain never feels the push itself, only what it does to the body. The tone starts a
**delay line**; each moment after the tone leaves an **eligibility trace** that fades by
`--tau` per tick (50 ticks per second). When the push jolts the body, every trace still
alive is credited, and after each trial the counter-move weights move toward that credit
at rate `--eta-brace`. Next time, the cerebellum moves the walking command into the push
just before it lands.

(During push runs the forward model's error is high: falls and resets disturb its
predictions. Use Part A runs, without pushes, for the forward model.)

| # | experiment | change (all with `--pushes 12`) | your prediction (falls of 12) | measured |
|---|---|---|---|---|
| B1 | no bracing | `--bracing off` | | |
| B2 | the default | nothing | | |
| B3 | a short trace | `--tau 0.5` | | |
| B4 | a long trace | `--tau 0.97` | | |
| B5 | slow / fast learning | `--eta-brace 0.05`, `--eta-brace 0.9` | | |
| B6 | a later push | `--delay 25` (0.5 s after the tone) with the default tau, then with `--tau 0.97` | | |
| B7 | a softer / harder push | `--force 220`, `--force 265` | | |

Questions:
5. With `--tau 0.5`, how much of the tone's trace is left 10 ticks later (0.5^10)? Why can
   the brace not learn then?
6. Which trace worked best in your runs? Is the default the best? Why might a longer
   trace help here?
7. With a later push (B6), what must the trace be able to do? Did your results agree?
8. Bracing never stops every fall. Watch a fall with `--viewer`: what goes wrong?

## Show your results

Each group presents one experiment of its own choice (or a new one): the question, the
prediction, the command, the measured result (`results.csv` and, if you like, a plot),
and an explanation. Five minutes per group.

## Going further

- Record a video of your best brace: `uv run --extra video examples/g1_bracing.py --record mine`.
- Read the walker profile: `print(profile("walker"))` in Python, or its file in the folder
  `python -c "import ubrain.wiring as w; print(w.PROFILES)"` names --
  the same parameters, as the brain's anatomy.
- Try your own question: which parameter matters most for learning a turn?
