# Project proposal (one page)

Copy to `weeks/projects/teams/<team-name>/proposal.md`. Due at the start of week 13.

**Team:** <name> · **Members:** <names> · **Catalogue entry:** P<n> (or P14) · **Date:** <YYYY-MM-DD>

## 1. The question (one sentence)

> Example: Does the ankle strategy keep the crouched G1 up under a larger sideways shove
> than holding the crouch, when it reads its tilt from the filtered torso IMU instead
> of the truth?

The quantity that answers it, and its unit: <e.g. the largest sideways force survived, N>

## 2. The prediction (written before the first real run)

> We predict <number + unit>, because <the reason, from a lecture, a measurement or the
> physics>.

## 3. The twin

The same run with exactly one thing changed: <what changes>. Everything else (robot,
seed, push, duration) stays the same.

## 4. The experiment

- Starts from: <script / package function>
- Sweep: <what is varied, which values>
- Repeats: <seeds or episodes; why that many>
- One run takes about <seconds>; the whole sweep takes about <minutes>. Measured in week 12: yes / no.

## 5. The check rule (draft; frozen in week 14)

> PASS if <rule>, e.g. "ours survives at least 10 N more than the twin, in every seed".

## 6. Risks and the fallback

What could make this unanswerable (compute, a training run that does not converge, a
knife-edge parameter), and the smaller question you will answer instead.
