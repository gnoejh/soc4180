# Week 26 worksheet: reference results (instructor only)

Measured 2026-10-25, ubrain 0.14.0.dev1 (vendor/ubrain), CLIP openai/clip-vit-base-patch32
(revision 3d74acf9a28c) under transformers 5.20.2, CPU. 168 words in `words` (the
vocabulary plus the sentence), 164 in `compare` and `capacity`; 21 gallery pictures.

## Part A

| Row | Command | Cone raw / centred | Certificate k 4 / 8 / 16 | Recalled |
| --- | --- | --- | --- | --- |
| A1 | `words` | 0.82 / −0.01 | 1.000 / 1.000 / 0.996 | 10 of 10 |
| A2 | `words --raw` | (same) | (same; assumes centring) | 4 of 10: `the old give book the red fox small a plane` |
| A3 | `pictures --k 16` | 0.60 / −0.05 | 1.000 / 1.000 / 1.000 | 16 of 16 |
| A3 | `pictures --k 16 --raw` | | | 15 of 16 (`cup` came back as `guitar`) |
| A3 | `pictures` (k 8), with or without `--raw` | | | 8 of 8 |
| A4 | `capacity --raw` (20 episodes) | | 1.000 / 1.000 / 0.995 | k 2 1.000, 4 1.000, 8 0.688, 16 0.509, 32 0.172 |

- Q1: 0.82: every code is mostly one shared direction plus a small part that is the word's
  own.
- Q2: the sum of k codes carries k times the common part and only one copy of each item's
  own part; the cue's similarity to every stored item is dominated by the shared part, so
  items are confused (crosstalk grows with k).
- Q3: the cone: 0.60 for the pictures against 0.82 for the words (and only 21 pictures
  to confuse against 168 words).

## Part B

| Row | Command | Certificate k 4 / 8 / 16 | Recalled |
| --- | --- | --- | --- |
| B1 | `words --d 128` | 0.998 / 0.901 / 0.564 | 7 of 10 (`a sad farmer saw ship red fox near the river`) |
| B2 | `words --d 64` | 0.941 / 0.579 / 0.269 | 4 of 10 |
| B4 | the 18-word sentence | 1.000 / 1.000 / 0.994 | 18 of 18 |

B3, `capacity` (certificate / measured, 20 episodes per k):

| d | k 2 | k 4 | k 8 | k 16 | k 32 |
| --- | --- | --- | --- | --- | --- |
| 512 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.995 / 1.000 | 0.881 / 0.863 |
| 128 | 1.000 / 1.000 | 0.999 / 1.000 | 0.909 / 0.931 | 0.555 / 0.522 | 0.263 / 0.272 |
| 64 | 1.000 / 1.000 | 0.946 / 0.887 | 0.583 / 0.512 | 0.276 / 0.244 | 0.115 / 0.117 |
| 32 | 0.987 / 0.975 | 0.609 / 0.637 | 0.272 / 0.250 | 0.099 / 0.131 | 0.052 / 0.061 |

- Q4: within a few hundredths almost everywhere; the largest gap is d 64, k 8 (0.583
  against 0.512). 20 episodes of 8 words are 160 recalls: a standard error of about 0.04,
  so the gap is under two standard errors.
- Q5: the common part's size (the cone) and how it adds up in the episode; this
  certificate's spectral law is computed on centred codes.

## Part C

| Codec (164 words) | Cone raw / centred | Certificate k 4 / 8 / 16 |
| --- | --- | --- |
| CLIP text | 0.82 / −0.01 | 1.000 / 1.000 / 0.995 |
| random (GaussianCodes) | 0.00 / −0.01 | 1.000 / 1.000 / 0.999 |
| bag of letters | 0.23 / −0.01 | 0.982 / 0.984 / 0.941 |

C2 (`name`): 21 of 21 pictures named by the nearest of the 21 words; random codes would name
about 1 by chance.

- Q6: `eat` and `tea`, `baker` and `break` (anagrams: the same letters, so the same sum).
  Identical codes are a spectral fact of the vocabulary, visible to the certificate
  without storing anything; whenever one of a pair is in the episode, the other can be
  recalled instead.
- Q7: CLIP's codes carry meaning, and its text and image codes share one space: a picture
  of a dog lands near the word "dog" (C2, `name`: all 21 pictures are nearest their own name,
  cosines 0.21-0.31). Random codes are only labels; they cannot recognise
  a new view of a thing (week 27's glances) or name it.
