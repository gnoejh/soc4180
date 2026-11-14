# The capstone project — a guide

Part II's project: **a question about the untrained brain in the G1, answered by measurement.**
Three weeks from Monday 17 November; presentations in the final-exam slot.

Read this guide in order: what you hand in, the path through the material, what the brain can
and cannot do, then pick (or adapt) a project.

---

## 1. What you hand in

Every project has the same five parts (week 18's method):

| part | what it is | where it goes |
| --- | --- | --- |
| **the question** | one question about the brain in the G1 | the proposal, the report |
| **the prediction** | written **before** the run, with a number and its reason | the proposal (dated), the report |
| **the twin** | the same run with **one** thing changed, so a difference has a cause | your script |
| **the measurement** | a script that writes its numbers to `results.csv` | your script, the report's table |
| **the check** | one line, `CHECK PASS` or `CHECK FAIL`, decided before the final run | your script's last line |

| week | you do | you hand in |
| --- | --- | --- |
| 16 | learn the whole brain; pick a question | the one-page proposal (start of week 17) |
| 17 | learn the body; run your project once end to end | the first results file |
| 18 | build, twin, measure; the checklist (week 18's worksheet) | the twin and the check, signed in class |
| final slot | present | the script, `results.csv`, the saved brain, a short report, the talk |

**The report**, in this order: the question and the prediction as written; the setup (variant,
streams, and every signal sorted into *sense / detector / privilege*); the twin; the table from
the results file with the number of runs; the check, PASS or FAIL — **a failed prediction is
reported as failed**, with your best explanation; what you would run next.

**The presentation**: load the saved brain and run it in the viewer; show the twin; show the
table and the check line; tell what surprised you. [Instructor: the time per team.]

---

## 2. The path through the material

ubrain's examples and the lectures teach the same things; learn them together. The examples
live in the public repository, github.com/untrained-brain/ubrain (`examples/`); every one ends
with a `CHECK` line, like your project will.

| week | the lecture teaches | its lab | the ubrain examples that show the same |
| --- | --- | --- | --- |
| 16 | the whole brain: `Brain()`, port classes, the tick; asking it; keeping it | `lab_program.py` | `01_whole_brain`, `02_ask`, `03_keep` |
| 17 | the body's streams; the twin; sense, detector, privilege; the eye | `lab_sense.py` | `00_check_setup`, `04_g1_brace`, `08_g1_vision` |
| 18 | the method; three project starts; a failed prediction | `lab_choose.py`, `lab_keep.py`, `lab_brace.py` | `05_g1_choose`, `06_g1_keep`, `07_g1_lead` |
| 19 | your project | `project_template.py` (this folder) | the example your project starts from |

Everything runs on **one wheel: ubrain 0.46.0.dev3**, the same in the course, in the examples
and in your project.

---

## 3. What the brain can do — and what it cannot yet

**Measured** (every number from a run you can repeat):

| it can | measured | where |
| --- | --- | --- |
| learn which action each picture asks for | 0.976 in the first 500 trials, 0.998 later; chance 0.25 | `01_whole_brain`, week 16 |
| relearn when the rule changes | 0.896 in the block of the change, 1.000 the next | week 16 `--reverse` |
| tell you what it is and what it holds | `describe`, `read`, `datasheet`, `health` | `02_ask` |
| be saved, loaded and continue exactly | 0 of 500 choices differ; refuses a wrong codec | `03_keep` |
| walk the G1 and predict its body | prediction error 97% → 17% of "no change" in 10 s | week 16 step g1 |
| learn to brace on a warning tone | 3 falls in 20 pushes, against 14 without the tone | `04_g1_brace` |
| learn which way to turn from a picture | right every trial after the first two | `05_g1_choose` |
| keep a trained skill for the demo | loaded brain 0 falls in 10 pushes, a fresh one 3 | `06_g1_keep` |
| see through CLIP and notice what is new | 22 of 199 glances new; a covered camera 1 | `08_g1_vision` |

**A preview** (it works, held to a certified circuit, but the whole is not yet certified as a
whole): the body's streams — `body.proprio`, `cue.tone`, `goal`, `command` through
`ubrain.bodies.g1.G1`.

**Not available in this release** (do not plan on them): walking adaptation in the whole brain;
recalling a sequence of things in order; touch as a sense (foot forces are a *privilege*, for
rewards and analysis only); a movement chosen inside the brain (your program maps an action to
a walk).

**Two rules every project needs** (found while building the labs):

1. **A sense must change.** The whole brain engages, and chooses, only on what is new or
   surprising. A sense that shows one fixed thing never engages it: its action stays `None`.
2. **A wrong choice must pay nothing.** The brain explores only while its values are near zero;
   once an action has paid, an action never tried is almost never tried. A reward that pays the
   wrong action half the time is never unlearned.

---

## 4. Possible projects

Each starts from an example that runs; change one thing at a time. **"Not yet run"** means
nobody has measured it: your prediction is genuinely open. Difficulty: ★ a parameter study,
★★ a new world or measure, ★★★ an open research question.

| # | the question | start from | the twin | a check (decide yours) | ★ |
| --- | --- | --- | --- | --- | --- |
| 1 | How strong a push can a warned brain survive? (100–350 N) | `04_g1_brace` | no tone | falls(no tone) − falls(tone) ≥ 5 at some force | ★ |
| 2 | Why does a 5-tick warning fail when 40 works? (lead 5 → 7 falls, 40 → 4) | `07_g1_lead` | another lead | your explanation predicts a new lead's falls within 2 | ★★★ |
| 3 | Does the brace generalise to a push from the side or the back? | `06_g1_keep` (trained from the front) | a fresh brain | trained − fresh, in falls, for each side | ★★ (not yet run) |
| 4 | Two tones, only one followed by a push: does it brace only on the right one? | `04_g1_brace` | one tone only | bracing on the decoy below a bound you set | ★★ (not yet run) |
| 5 | How many pictures can it map to turns before it slows down? (2, 4, 8, 16) | `05_g1_choose` | unrelated pictures | trials to 90% right, against the number of pictures | ★ |
| 6 | When the meaning of the pictures flips, how fast does it relearn on the robot? | `05_g1_choose` + week 16 `--reverse` | no flip | trials lost after the flip | ★★ (not yet run on the G1) |
| 7 | Rule 2 tested: let the wrong turn pay 0.5 | `05_g1_choose` | the wrong turn pays 0 | right-rate with 0.5 below the twin's | ★★ (not yet run) |
| 8 | What makes a picture new again? Walk the gallery twice, or longer | `08_g1_vision` | one pass | new glances on the second pass | ★★ (needs `--extra vision`) |
| 9 | Does a bigger brain (`small` / `standard` / `large`) learn the robot's task faster? | any of 04, 05 | another variant | your measure per variant | ★ |
| 10 | Combine: a picture says which way to turn **and** a tone warns of a push | `04` + `05` | one task alone | both tasks' measures within a bound of their alone values | ★★★ |

Your own question is welcome: write it in the five parts and check it against section 3 first.

---

## 5. Getting started

```bash
uv sync                                         # the course's environment: ubrain 0.46.0.dev3 included
uv run weeks/19-project/project_template.py     # runs as it is and ends with CHECK PASS
```

`project_template.py` is a working project: a question, a twin, a measurement into
`results.csv`, a check. Copy it, rename it, and change the places marked `# CHANGE`.

**When something goes wrong:** the robot falls at once → is `goal` set (a walk, not zeros)?
The brain never chooses → rule 1. It keeps a bad choice → rule 2. `Brain.load` refuses → give it
the same codec you trained with. On Windows, the ubrain repository needs `git clone -c
core.longpaths=true`.
