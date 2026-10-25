# 26 — Plugging in a trained model: codecs and the certificate

| | |
| --- | --- |
| **Layer** | 3 — perception from a frozen trained model (CLIP), feeding layer 5's untrained memory |
| **Runtime** | laptop, CPU |
| **Needs** | `uv sync --extra vision` (torch, transformers); the CLIP weights, about 600 MB, once; ubrain ships in `vendor/ubrain/` |
| **Classes** | two: slides and a live demo, then a hands-on class |

The brain is untrained and has no eyes. A **codec** is a frozen trained model that turns
an item into a code; here CLIP (ViT-B/32) turns a word or a picture into 512 numbers, and
week 20's hippocampus stores and recalls them. Two things decide whether that works. CLIP's
raw codes sit in a **cone** (two unrelated words have a cosine of 0.82), and a memory that
adds codes drowns in their common part. A `Sense` fitted once on the vocabulary
**centres** the codes, and then they behave like random ones. The **certificate**
(`ubrain.certify`) says, from the codes alone and before anything is stored, how often an
item of a k-item episode will come back. The week measures it against recall, on words
and on the 21 gallery photographs the G1 walks past in week 27.

## Slides

HTML deck in week 16's format; the figure slide is drawn by `figures.py` from
`traces/capacity.json` (`--run` re-measures it, about 2 minutes).

```bash
uv run python weeks/26-codecs/build.py
```

## Before the class (at home, about 5 minutes and a 600 MB download)

```bash
uv sync --extra vision
uv run --extra vision weeks/26-codecs/lab_codecs.py words --download
```

It must end with `recalled: the old farmer saw a red fox near the river   (10 of 10)`.
**Always run with `uv run --extra vision`**: a plain `uv run` re-syncs the environment to
the default and removes torch.

## Lab class: on your laptop

```bash
uv run --extra vision weeks/26-codecs/lab_codecs.py --help
uv run --extra vision weeks/26-codecs/lab_codecs.py words               # 10 of 10
uv run --extra vision weeks/26-codecs/lab_codecs.py words --raw         # not centred: 4 of 10
uv run --extra vision weeks/26-codecs/lab_codecs.py words --d 64        # small workspace: 4 of 10
uv run --extra vision weeks/26-codecs/lab_codecs.py pictures --k 16     # 16 of 16
uv run --extra vision weeks/26-codecs/lab_codecs.py compare             # CLIP, random, bag of letters
uv run --extra vision weeks/26-codecs/lab_codecs.py capacity --d 128    # certificate against recall
uv run --extra vision weeks/26-codecs/lab_codecs.py name                # pictures named by words
```

What students change: the step, `--raw`, `--d`, `--sentence`, `--k`, `--episodes`, and the
`Letters` codec in the script (a codec is any object with `name`, `modality`, `license`,
`dim`, `digest` and `encode(items)`).

What they should see (measured 2026-10-25):

- the cone: CLIP text 0.82 raw, −0.01 centred; the 21 pictures 0.60 and −0.05;
- with centring, the certificate equals white codes' and recall follows it: the sentence
  10 of 10, 18 words 18 of 18, 16 pictures 16 of 16;
- not centred, the sentence comes back 4 of 10 (20 random episodes: 0.69 at k 8, 0.51 at
  k 16), although the workspace is the full 512; the pictures, a shallower cone, lose one
  of 16;
- a small workspace fails as certified: at d 128 the certificate says 0.909 / 0.555 at
  k 8 / 16 and 20 episodes give 0.931 / 0.522; at d 64, 0.583 / 0.276 against 0.512 / 0.244;
- the bag of letters gives 'eat' and 'tea' (and 'baker' and 'break') the same code, and its
  certificate shows it: 0.941 at k 16 against CLIP's 0.995;
- CLIP's picture codes and word codes share one space: each of the 21 pictures is nearest
  the word that names it, 21 of 21 (`name`).

Every run appends rows to `results.csv`. The exercises are in
[worksheet.md](worksheet.md); measured reference answers are in [instructor/reference.md](instructor/reference.md).
