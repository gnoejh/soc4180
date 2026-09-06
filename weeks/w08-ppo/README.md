# Week 8 — Policy Gradients and PPO

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/w08-ppo/lab.ipynb)

**The first week that trains anything.**

| | |
| --- | --- |
| **Runtime** | Local: CPU. **Colab: pick a GPU runtime** (Runtime > Change runtime type > T4). |
| **Wall clock** | **~6 min** — three training runs. Longest lab so far. |
| **Convergence risk** | Low for CartPole and InvertedPendulum; the G1 run is *designed* to fail. |
| **Needs** | `soc4180[rl]` — torch and stable-baselines3 |

## Objectives

1. State the policy gradient theorem and explain why it avoids differentiating
   the simulator.
2. Implement REINFORCE from scratch and read its learning curve.
3. Explain baselines, advantage, and why PPO clips.
4. Diagnose a failed training run as a **specification** failure rather than an
   algorithm failure.

## Measured results

| Run | Result | Time |
| --- | --- | --- |
| REINFORCE, CartPole (from scratch) | 60 → **489** (max 500) | 173 s |
| PPO, InvertedPendulum-v5 | 26 → **1000** (solved) | 99 s |
| PPO, `G1WalkEnv`, 30k steps | 774 → **95** | 69 s |

## The two results that carry the week

**The untrained policy already scores 774** — exactly the hold-the-crouch
baseline. That is the Week 7 residual-action design paying off: a network with
near-zero outputs commands the nominal crouch and simply stands. The policy
starts from competence. It also sets a cruel bar, since the easiest way to score
774 is to do nothing.

**Training made it worse: 774 → 95.** The trained policy lunges at 1.28 m/s
against a 0.5 target and falls after 46 of 500 steps.

The obvious explanation — that the reward rewards diving — is **wrong**, and the
lab checks it rather than asserting it. Evaluating the reward by hand:

| behaviour | per step | over the episode |
| --- | --- | --- |
| walk at the target | 2.500 | **1250** (the best available) |
| stand still | 1.552 | 776 |
| lunge, fall at 46 steps | 1.129 | 52 |

There is no loophole. The reward wants exactly what we want.

**The real cause is exploration noise.** PPO starts with an action standard
deviation of 1.0 on a `[-1, 1]` action space, so:

- untrained **deterministic** policy (what we evaluate): survives **500 of 500** steps
- untrained **stochastic** policy (what we train on): survives **38**

Over 90% of collected experience is a robot falling over. PPO never sees the
competent behaviour its own mean produces. Setting `log_std_init=-2.0`
(std 0.135) raises stochastic survival to ~180 steps, and the same 30k-step run
then scores **776 instead of 95**.

> A policy is a distribution. If the distribution is too wide, your data
> describes a policy you would never deploy.

**But it still is not walking** — 776 is the standing baseline, and walking is
worth 1250. Fixing this bug removed a bug; 30k steps is 0.02% of a real
locomotion run. That is Week 10's problem.

## A note on REINFORCE's batch size

Updates use a **batch of 8 episodes**. With one episode per update the variance
is so large that training frequently goes nowhere: two runs of identical code
reached 260 and 88 respectively. The batched version is reliable across seeds
(~30 → ~485 for seeds 0, 1, 2). The slides make that variance the lesson rather
than hiding it.

## Rebuild

```bash
quarto render weeks/w08-ppo/slides.qmd     # ~6 minutes; it trains
```
