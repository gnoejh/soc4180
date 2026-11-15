# Checklist before you present

Signed by the instructor at check-in 2 (items 1-5) and on the day (6-10).

1. [ ] The question and the prediction are in `proposal.md`, dated before the first results.
2. [ ] `project.py` runs headless (`--no-viewer`) from a clean `uv sync` and writes `results.csv`.
3. [ ] The twin differs from the experiment in exactly one thing, named in a comment at the top.
4. [ ] More than one seed or episode backs every cell that matters, and the count is printed.
5. [ ] The `CHECK` rule is committed (git log shows it) before the final runs.
6. [ ] Every number in the report and on the slides appears in `results.csv`.
7. [ ] No float from a fall is compared. Falls are judged by margins and counts.
8. [ ] A `FAIL` is reported as a `FAIL`, with the explanation tested by a run (or labelled a hypothesis).
9. [ ] The demo replays in the viewer on the presenting laptop, and a recording exists as a fallback.
10. [ ] Outside code and assets are cited with their licence. AI assistance is acknowledged.
