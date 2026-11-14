# Week 18 worksheet — the project checklist

Team: ______________________   Date: __________

Tick each line when it is true, and show the instructor B and C in class.

## A. Run a start

- [ ] I ran the lab closest to my project and it printed `CHECK PASS`.
- [ ] I changed ONE thing, wrote my prediction first, and ran it.

  Lab: ________  Change: ______________  Prediction: ______  Measured: ______

## B. The twin (instructor signs)

- [ ] My twin differs in exactly one thing: ________________________________
- [ ] Both runs write their numbers to my results file.

  Instructor: ________

## C. The check (instructor signs)

- [ ] My CHECK line is written down before the final run:

  `CHECK PASS if ______________________________________________`

  Instructor: ________

## D. Save and demonstrate

- [ ] The trained brain is saved (`brain.save`), loaded (`Brain.load(path, codecs=...)`), and
      behaves as trained.

## E. The report

- [ ] 1. the question and the prediction, as written before the run
- [ ] 2. the setup: variant, streams, and every signal sorted into sense / detector / privilege
- [ ] 3. the twin
- [ ] 4. the table from the results file, with the number of runs
- [ ] 5. the check, PASS or FAIL; a failed prediction reported as failed
- [ ] 6. what we would run next

## F. Questions to answer by experiment (pick one)

1. Why does a 5-tick lead fail when 40 works (`lab_brace.py --leads 3 5 8 10`)?
2. Make the wrong turn pay 0.5 in `lab_choose.py`. Predict, then run: does it still learn?
3. Load the braced brain onto pushes from the back (`lab_keep.py`, force `[-250, 0]`). Predict first.
