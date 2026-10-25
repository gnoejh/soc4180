"""lab_striatum: the striatum learns which action pays, and chooses by a race.

    uv run weeks/18-striatum/lab_striatum.py                     # 6000 trials, the defaults
    uv run weeks/18-striatum/lab_striatum.py --eta 0.01          # a slower learner
    uv run weeks/18-striatum/lab_striatum.py --world easy        # one action clearly best
    uv run weeks/18-striatum/lab_striatum.py --reverse 3000      # the payoffs swap halfway

A "body" shows one of a few situations; in each, every action pays 1 with its own
probability (the WORLD's table). The striatum keeps a value Q per (situation, action):

    Q[s, a] <- Q[s, a] + eta * (r - Q[s, a])        the three-factor rule

(r - Q is the reward-prediction error, what dopamine neurons report), and chooses by the
GO RACE: each action's Go unit runs to threshold at a speed proportional to its value,
the first to arrive wins, so action a is chosen with probability Q_a / sum_b Q_b.

What you can change:
    --eta E          learning rate (default 0.1): how far Q moves toward each outcome
    --q0 Q           the value every action starts from (default 0.5)
    --world NAME     the payoff table: default | easy | close  (printed at the start)
    --trials N       trials in all (default 6000)
    --reverse T      at trial T, every situation's payoffs are reversed (the best action
                     becomes the worst); 0 = never (default)
    --seed S         the world's random draws (default 0); the brain draws its own
    --name LABEL     a label for this run in results.csv

Measured, and predicted before the run:
  * the learned values beside the payoff probabilities (the rule's fixed point: Q -> p);
  * the share of choices per action in the last third of the run, beside MATCHING,
    p_a / sum_b p_b (the race's prediction once Q is near p);
  * the reward rate, beside what always taking the best action would earn.

Every run appends one row to results.csv beside this file. From ubrain's
examples/01_bandit.py; ubrain itself is the course's copy (see ubrain_local.py).
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

import ubrain_local

ubrain_local.load()                 # the course's wheel onto sys.path, before `import ubrain`

from ubrain import Brain, Channel, Manifest, profile  # noqa: E402

HERE = Path(__file__).resolve().parent

WORLDS = {
    "default": [[0.8, 0.4, 0.1],     # situation 0: action 0 pays most often
                [0.2, 0.7, 0.2],     # situation 1: action 1
                [0.3, 0.3, 0.9]],    # situation 2: action 2
    "easy": [[0.9, 0.1, 0.1],
             [0.1, 0.9, 0.1],
             [0.1, 0.1, 0.9]],
    "close": [[0.6, 0.5, 0.4],
              [0.4, 0.6, 0.5],
              [0.5, 0.4, 0.6]],
}


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--q0", type=float, default=0.5)
    ap.add_argument("--world", choices=sorted(WORLDS), default="default")
    ap.add_argument("--trials", type=int, default=6000)
    ap.add_argument("--reverse", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="")
    return ap.parse_args(argv)


def run(a) -> dict:
    pay = np.array(WORLDS[a.world], dtype=float)
    S, A = pay.shape
    w = profile("player")
    w.regions["basal_ganglia.striatum"]["eta"] = a.eta
    w.regions["basal_ganglia.striatum"]["q0"] = a.q0
    body = Manifest("course/bandit", "none", (Channel("situation", "event", S),
                                              Channel("reward", "reward", 1),
                                              Channel("action", "choice", A)))
    brain = Brain(w, body, rng=np.random.default_rng(a.seed + 1))
    rng = np.random.default_rng(a.seed)

    final = pay[:, ::-1].copy() if a.reverse else pay        # the table in force at the end
    last = a.trials - a.trials // 3                           # the last third is measured
    print(f"world '{a.world}' (rows: situations, columns: actions), eta {a.eta}, q0 {a.q0}, "
          f"{a.trials} trials" + (f", reversed at trial {a.reverse}" if a.reverse else ""))
    for s in range(S):
        print("   " + "  ".join(f"{p:.2f}" for p in pay[s]))
    print("prediction: Q -> p; choice -> p_a / sum p (matching); best-action rate "
          f"{final.max(axis=1).mean():.3f}, matching rate {(final ** 2 / final.sum(1, keepdims=True)).sum(1).mean():.3f}\n")

    counts = np.zeros((S, A))
    r, earned, late = None, 0.0, 0.0
    for t in range(a.trials):
        if a.reverse and t == a.reverse:
            pay = final
        s = int(rng.integers(S))
        act = brain.tick({"situation": np.eye(S)[s], "reward": r})["action"]
        r = float(rng.random() < pay[s, act])
        earned += r
        if t >= last:
            counts[s, act] += 1
            late += r
        if (t + 1) % 1000 == 0:
            print(f"trials {t + 1:5d}: reward rate so far {earned / (t + 1):.3f}")
    brain.tick({"reward": r})                                 # the last outcome
    Q = brain.regions["basal_ganglia.striatum"].Q
    alloc = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    match = final / final.sum(axis=1, keepdims=True)

    print("\nlearned values Q                 payoff probabilities p (at the end)")
    for s in range(S):
        print("   " + "  ".join(f"{q:.2f}" for q in Q[s]) + "              " + "  ".join(f"{p:.2f}" for p in final[s]))
    print("\nchoice, last third of the run    matching p_a / sum p")
    for s in range(S):
        print("   " + "  ".join(f"{x:.2f}" for x in alloc[s]) + "              " + "  ".join(f"{x:.2f}" for x in match[s]))
    n_late = a.trials - last
    rate = late / n_late
    se = np.sqrt(rate * (1 - rate) / n_late)
    m_rate = (final ** 2 / final.sum(1, keepdims=True)).sum(1).mean()
    print(f"\nreward rate, last third: {rate:.3f} (+-{se:.3f}); matching predicts {m_rate:.3f}; "
          f"always the best would earn {final.max(axis=1).mean():.3f}")
    return {"when": datetime.now().isoformat(timespec="seconds"), "name": a.name, "world": a.world,
            "eta": a.eta, "q0": a.q0, "trials": a.trials, "reverse": a.reverse, "seed": a.seed,
            "rate_last_third": round(rate, 4), "matching_rate": round(float(m_rate), 4),
            "best_rate": round(float(final.max(axis=1).mean()), 4),
            "max_choice_gap": round(float(np.abs(alloc - match).max()), 4),
            "max_value_gap": round(float(np.abs(Q - final).max()), 4)}


def main(argv=None) -> None:
    row = run(parse(argv))
    out = HERE / "results.csv"
    new = not out.is_file()
    with out.open("a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(row))
        if new:
            wr.writeheader()
        wr.writerow(row)
    print(f"(one row added to {out.name})")


if __name__ == "__main__":
    main()
