# Project report

Copy to `weeks/projects/teams/<team-name>/report.md`. Four to six pages with figures. In
this order:

## 1. The question and the prediction

Copy both **exactly as dated in the proposal**. If the question changed, give the
original, then the new one and why.

## 2. The setup

- The robot, the scene and every change you made to them (an `MjSpec` edit, a mass, a
  floor).
- The controller or learner, and where its code came from (course week or file, or an
  outside source with its licence).
- What the controller is allowed to sense, and what it is not (truth from the simulator
  is a *privilege*: say when you use it).

## 3. The twin

What exactly changed between the two runs, and why that one change answers the question.

## 4. The results

- The table from `results.csv`, with the number of runs (seeds, episodes) behind each
  cell.
- One or two figures, each drawn from `results.csv` by a script in your folder.
- Margins, not floats from falls: "falls from 80 N in 3 of 3 seeds", not "falls at
  2.4931 s".

## 5. The check

The rule, as committed in week 14, and the line your script printed: `CHECK PASS` or
`CHECK FAIL`. **A failed prediction is reported as failed.**

## 6. Why

Your explanation of the result, and **the run that tested it** (a third condition, a
logged quantity, a sweep). An explanation with no run behind it is labelled a
hypothesis.

## 7. What we would run next

One or two experiments, each with its prediction.

## Appendix: reproduce

The exact command that regenerates `results.csv`, the time it takes, and the machine.
