# Lab 5 -- Seeing and remembering: the explorer

**Goal:** put it all together -- a body, a trained vision model, attention, event
segmentation and one-shot memory -- and separate what perception gets wrong from what
memory gets wrong.

## Run

```bash
uv run --extra vision --extra video examples/g1_explorer.py --viewer
uv run --extra vision --extra video examples/g1_explorer.py --headless --seconds 40 --record walk
```

The G1 walks past 21 photographs (public-domain, CC0) standing along its path. The
second command writes `walk.mp4`: the robot's-eye view.

## What you see

Every fifth of a second the robot glances: it takes a camera frame, looks at the half
with more going on (a crude fovea), and CLIP says which known picture it looks like.
When the same picture has been recognised on 3 glances in a row, it counts as seen and
is written into the episode -- once. At the end every slot is recalled. The report
sets three lists side by side: what was written, what was recalled, and the pictures'
true order.

## The idea

Two systems, two kinds of error:

* **Perception** (CLIP + attention + "3 in a row"): can mistake one picture for
  another, or miss one. Compare *written* with the true order.
* **Memory** (the hippocampus): can return a different picture than was written.
  Compare *recalled* with *written*. The certificate predicts this one.

The explorer **recognises first, then remembers**: the episode stores the code of the
recognised picture, not the raw glance. Storing raw glances recalls far worse, because
a glance is only loosely like its picture's code and the certificate assumes stored
items are the vocabulary's own codes.

## Exercises

1. Run headless with `--verbose`. Find a glance where perception was wrong. Why might
   CLIP have confused those two pictures?
2. Change `STEADY` from 3 to 1 in the script. How many items are written now? What
   happens to recall, and why? (Look at the certificate for that many items.)
3. The robot drifts off the path after about 14 pictures. Using the walking command,
   can you keep it on the path longer? What sense would a brain need to do this
   itself?
4. Separate the errors: over 3 runs, count perception's errors and memory's errors.
   Which system limits this robot?
5. **Going further:** store raw glances instead (in the script, use a Sense without
   `snap=True`). Measure recall. Explain the drop with the certificate's assumption.
