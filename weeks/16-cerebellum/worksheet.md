# Week 16 worksheet -- A brain in a body: the cerebellum

**Goal:** change the G1's cerebellum, predict what will happen, measure it, and explain
the result. You will work with its two parts: the **forward model**, which predicts how
the body moves, and the **brace**, which learns a timed counter-move to a push it hears
coming.

**Slides:** `slides_students.html` in this folder (build it with
`uv run python weeks/16-cerebellum/build.py`, or open it on the course site).

Adapted from ubrain's `docs/labs/lab4_walker.md` to run in this repository. **Every
command below runs from this folder**, `weeks/16-cerebellum/`.

## Before the class (at home, about 2 minutes)

Pull (or clone) the course repository; ubrain is already in `vendor/`. Then:

```bash
uv sync                          # from the repository root, as every week
cd weeks/16-cerebellum
uv run lab_cerebellum_walk.py    # the robot walks; arrow keys steer it; X quits
```

It must print `body 4815eedf56d0 ...` and walk, with `falls 0` on the status line. If not, send its full output
to your teacher before the class.

## The tool

```bash
uv run lab_cerebellum.py --help        # every parameter, explained
uv run lab_cerebellum.py               # 30 s walk with the default cerebellum
uv run lab_cerebellum.py --viewer      # the same, watching the robot
```

Each run prints, every second, the forward model's **prediction error** as a percentage
of the error of predicting "no change" (100 % = no better than not predicting; lower is
better), and at the end its average over the last 5 seconds. With pushes it also prints
one mark per push: **X** the robot fell, **.** it stayed up. Every run adds one row to
`results.csv` in this folder; use `--name` to label your runs (for example `--name "group 3: sparse"`).
`uv run --with matplotlib lab_cerebellum.py --plot` also saves a PNG of the curve.

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

## Part B -- the brace: classical conditioning in a robot

A tone sounds and 0.2 s later the walking robot is pushed from behind (`--pushes 12`).
This is the eyeblink experiment: the tone is the **cue**, the push the event it
predicts. The brain never feels the push itself, only what it does to the body. The
tone starts a **delay line**; each moment after the tone leaves an **eligibility trace**
that fades by `--tau` per tick (50 ticks per second). When the push jolts the body, every
trace still alive is credited, and after each trial the counter-move weights move toward
that credit at rate `--eta-brace`. Next time, the cerebellum moves the walking command
into the push just before it lands.

**Every experiment needs its control.** Run each one twice, as given and again with
`--bracing off`: only the difference between the two is the cerebellum's doing. A fall
is chaotic, so your record will not match your neighbour's mark for mark; a difference
like 12 of 12 against 3 of 12 is real, one like 5 against 7 may not be. Compare with
other groups in `results.csv` before you conclude.

(During push runs the forward model's error is high: falls and resets disturb its
predictions. Use Part A runs, without pushes, for the forward model.)

### B1-B4: does it learn, and what does it need?

| # | question | change (all with `--pushes 12`) | prediction (falls of 12) | braced | `--bracing off` |
|---|---|---|---|---|---|
| B1 | does it learn at all? | nothing | | | |
| B2 | is the cue needed? | `--tone off` (the push, unannounced) | | | |
| B3 | timing: a sooner / later push | `--delay 5`, then `--delay 25` | | | |
| B4 | an unreliable cue | `--jitter 8` (each delay 2 .. 18 ticks) | | | |

### B5-B6: what is learned, and does it last?

| # | question | change | prediction | braced | `--bracing off` |
|---|---|---|---|---|---|
| B5 | where the push comes from | `--pushes 12 --from front`, then `--from right` | | | |
| B6 | does it forget? | `--pushes 10 --tone-only 8 --test 4`, then `--pushes 10 --test 4` | falls in the 4 test pushes | | |

### B7-B9: the brace's own parameters

| # | question | change (all with `--pushes 12`) | prediction (falls of 12) | measured |
|---|---|---|---|---|
| B7 | a short / long trace | `--tau 0.5`, `--tau 0.97`; then with `--delay 25` | | |
| B8 | slow / fast learning | `--eta-brace 0.05`, `--eta-brace 0.9` | | |
| B9 | a softer / harder push | `--force 220`, `--force 265` | | |

You can also feel B2-B5 by hand: in `lab_cerebellum_walk.py`, change `PUSH_N`,
`PUSH_DELAY` or `PUSH_DIR` at the top of the file and press P again and again.

Questions:
5. B2: the push and the robot are the same with and without the tone. Why does the
   brace need the tone at all? What does it learn *from*, and what does it learn *to*?
6. With `--tau 0.5`, how much of the tone's trace is left 10 ticks later (0.5^10)? Why can
   the brace not learn then? And with `--delay 25` and the default tau, how much is left
   (0.9^25)? Did the result agree with your arithmetic?
7. B4: the cue no longer says exactly *when*. What would you expect a timing memory to do?
   What did it do?
8. B5: from one side it hardly matters whether the brace is on; from another, it does not
   help. The brace's output is a change of the walking command (vx, vy, wz). Which
   direction of push can such a counter-move answer, and which not?
9. B6: in animals, a tone that keeps sounding with nothing after it stops working
   (extinction). Does it here? What would the brace need to have, to forget?
10. B8: with a fast learning rate, look at the pattern of X and . -- what happens to the
    counter-move from one trial to the next?
11. Bracing never stops every fall. Watch a fall with `--viewer`: what goes wrong?

## Show your results

Each group presents one experiment of its own choice (or a new one): the question, the
prediction, the command, the measured result (`results.csv` and, if you like, a plot),
and an explanation. Five minutes per group.

## Going further

- Drive it yourself: `uv run lab_cerebellum_walk.py`, then turn (Left/Right) and change
  speed (Up/Down) and watch the prediction error jump and fall again. P is the tone and
  push; press it again and again and count how many it takes to learn the brace.
- Read the walker profile -- the same parameters, as the brain's anatomy:
  `uv run python -c "import ubrain_local; ubrain_local.load(); from ubrain import profile; print(profile('walker'))"`
- Try your own question: which parameter matters most for learning a turn?
