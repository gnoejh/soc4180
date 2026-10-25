# Week 21 worksheet: reference results (instructor only)

Measured 2026-10-25, ubrain 0.14.0.dev1 (vendor/ubrain), body digest 4815eedf56d0.
`lab_hold.py --lists 20` unless stated; perception 0.93-0.94 throughout (week 20's sense).

## Part A

| Row | Setting | Held of 16 | Law | Recall by age 0..15 |
| --- | --- | --- | --- | --- |
| A2 | γ 0.75, d 64 | 7.30 | 6.83 | 1.00 1.00 0.90 0.95 0.80 0.35 0.35 0.20 0.45 0.25 0.10 0.25 0.25 0.20 0.00 0.25 |
| A3 | γ 0.9 | 10.75 | 9.87 | 0.95 1.00 0.90 0.90 1.00 0.85 0.70 0.75 0.70 0.50 0.50 0.35 0.40 0.70 0.25 0.30 |
| A3 | γ 0.5 | 4.80 | 4.62 | 1.00 1.00 0.65 0.45 0.30 0.15 0.10 0.10 ... |
| A4 | d 128 | 7.95 | 7.86 | 1.00 1.00 1.00 1.00 0.80 0.65 0.60 0.35 ... |
| A6 | 7 pushes | 5.10 | 5.17 | 1.00 1.00 0.95 0.80 0.80 0.40 0.15 |

Laptop check (no robot, 300 lists): 16 with repeats 6.96 (law 6.83), 8 distinct 5.60
(5.45); γ 0.9: 10.32 (9.87), 8 distinct 7.14 (6.85). The law is slightly conservative at
V = 8; the robot's 20 lists carry about ±0.35.

A5, the law (16 pushes): d 64 γ 0.75 6.83; d 32 γ 0.85 6.99; d 128 γ 0.7 7.03; d 256 γ 0.6
6.44; d 32 γ 0.9 7.66; d 128 γ 0.75 7.86; d 64 γ 0.9 9.87.

- Q1: the last (newest) ones: weight γ^age; the newest has weight 1 against faint old
  crosstalk, the oldest γ^15 = 0.013 against loud new crosstalk.
- Q2: with a slow fade the old items stay loud, so the newest item's crosstalk is larger.
- Q3: γ 0.75 → 0.9 (6.83 → 9.87) buys far more than d 64 → 128 (→ 7.86).
- Q4: many (d, γ) pairs hold about seven (d 32 γ 0.85, d 128 γ 0.7, ...): a span alone does
  not identify a mechanism.
- Q5: in a list of 7 the oldest two have weights 0.18 and 0.24 against the crosstalk of
  the five newer ones: about 0.4 and 0.15 right.

## Part B (16 rehearsals unless stated)

| Row | Schedule | Held | Recall by age 0..15 |
| --- | --- | --- | --- |
| B1 | cycle (full) | 7.30 | identical to A2, every number, all 20 lists |
| B2 | newest | 4.95 | 1.00 1.00 1.00 0.20 0.10 0.10 0.20 0.05 0.15 0.20 0.05 0.15 0.30 0.15 0.15 0.15 |
| B3 | oldest | 5.05 | 0.25 0.20 0.15 0.05 0.10 0.30 0.25 0.20 0.15 0.10 0.05 0.10 0.15 1.00 1.00 1.00 |
| B4 | cycle, 8 rehearsals | 7.65 | 0.25 0.40 0.20 0.05 0.45 0.15 0.20 0.05 1.00 1.00 1.00 0.85 0.70 0.65 0.45 0.25 |
| -- | random | 6.40 | 0.50 0.05 1.00 1.00 0.20 0.15 0.25 0.00 0.80 0.40 0.15 0.05 0.40 0.35 1.00 0.10 |

(The random schedule rehearsed the same positions in every list of this run -- ages 2, 3
and 14 at 1.00 in all 20 -- so its choices appear not to vary from list to list here.)

- Q6: a full cycle re-adds every held term once; each weight is multiplied by the same
  factor (the profile's rehearsal law: a full cycle restores the presentation's profile),
  and a read compares an item with its competitors, so the factor cancels.
- Q7: the rehearsed three become certain; every other weight shrinks relative to them, so
  their crosstalk rises and the count falls by about two.
- Q8: the cycle starts with the oldest (position 0); 8 rehearsals re-add positions 0-7, so
  ages 8-15 become the newest writes.
- Q9: react with working memory (the newest pushes, certain); learn with the episode (all
  of it, a little blurred) -- a robot wants both.

