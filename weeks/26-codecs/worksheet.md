# Week 26 worksheet -- Plugging in a trained model: codecs and the certificate

**Goal:** plug a frozen trained model (CLIP) into the untrained brain's memory, find what
makes its codes storable, and check the certificate's promise against what comes back.

**Slides:** `slides_students.html` in this folder (build it with
`uv run python weeks/26-codecs/build.py`, or open it on the course site).

**Every command below runs from this folder**, `weeks/26-codecs/`, and always with
`uv run --extra vision` (a plain `uv run` removes torch).

## Before the class (at home, about 5 minutes and a 600 MB download)

```bash
uv sync --extra vision                                  # from the repository root
cd weeks/26-codecs
uv run --extra vision lab_codecs.py words --download    # fetches CLIP once
```

## The tool

```bash
uv run --extra vision lab_codecs.py --help
```

Each step loads CLIP (about 20 s), prints the cone (mean cosine between different items'
codes, raw and centred) and the certificate (recall predicted for k-item episodes, beside
white codes), then stores and recalls.

**Rule for every experiment: write your prediction before you run.**

## Part A -- the cone

| # | experiment | command | your prediction | measured |
|---|---|---|---|---|
| A1 | a sentence | `lab_codecs.py words` | words recalled of 10: | cone raw / centred: ; recalled: |
| A2 | the same, not centred | `lab_codecs.py words --raw` | | |
| A3 | pictures, not centred | `lab_codecs.py pictures --k 16` then `--k 16 --raw` | | |
| A4 | random episodes, not centred | `lab_codecs.py capacity --raw` | | k 8: ; k 16: ; k 32: |

- **Q1.** In A1, what is the mean cosine between two unrelated words before centring?
  What does that say about the direction of every CLIP code?
- **Q2.** Why does a memory that adds k codes suffer from a common part they all share?
  (Think of what the sum looks like for k = 10.)
- **Q3.** A3: the pictures survive without centring much better than the words. Which of
  the two numbers the step prints explains it?

## Part B -- capacity and the certificate

| # | experiment | command | your prediction | measured |
|---|---|---|---|---|
| B1 | workspace 128 | `lab_codecs.py words --d 128` | | certificate k 8 / 16: ; recalled: |
| B2 | workspace 64 | `lab_codecs.py words --d 64` | | |
| B3 | certificate against recall | `lab_codecs.py capacity --d 64` (and `--d 128`) | | |
| B4 | a long sentence | `lab_codecs.py words --sentence "..."` (your own 18 words) | | |

- **Q4.** In B3, how close is the certificate to the measured recall? Where is it off by
  the most, and is that more than 20 episodes can resolve?
- **Q5.** The certificate assumes the codes are centred. Using A4 and B3, what would a
  certificate for raw codes have to know that this one does not?

## Part C -- codecs compared

| # | experiment | command | your prediction | measured |
|---|---|---|---|---|
| C1 | three codecs | `lab_codecs.py compare` | which is worst at k 16: | CLIP / random / letters: |
| C2 | naming | `lab_codecs.py name` | pictures named of 21: | |
| C3 | your own codec | edit `Letters` (e.g. letter pairs) and run `compare` | | |

- **Q6.** The bag of letters loses words at k 16. Find two words in the vocabulary with the
  same code. Why does the certificate notice them before anything is stored?
- **Q7.** Random codes remember as well as CLIP. What can CLIP do that random codes cannot?
  (Use C2.)

## To hand in

Your table, Q1-Q7, and one paragraph: what does the certificate promise, what does it
assume, and what would you check before trusting a new codec on the robot?
