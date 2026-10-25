"""lab_patience: one number each -- serotonin's patience on the G1, noradrenaline's criterion.

    uv run weeks/25-neuromodulators/lab_patience.py serotonin                 # three patiences
    uv run weeks/25-neuromodulators/lab_patience.py serotonin --tau 6
    uv run weeks/25-neuromodulators/lab_patience.py serotonin --small 0.2     # a smaller sure reward
    uv run weeks/25-neuromodulators/lab_patience.py noradrenaline             # the gate's criterion (laptop)

A neuromodulator in ubrain is ONE NUMBER written onto a knob a region already has.

SEROTONIN is the striatum's patience (week 18's striatum). On the G1: each brain moment
is 0.4 s of walking. A gust is announced; the robot chooses
    action 0  stop and wait it out   a sure reward (--small, 0.4), 2 moments later
    action 1  walk on through it     1.0 if it is still up 8 moments later, 0 if it fell
Walking on through a 250 N push from behind keeps the G1 up at 22 of 40 stride points
(measured: week 24's traces/world.json, copied here), so it pays 0.55 on average --
more per trial than stopping, less per second (it takes 10 moments to stopping's 4).
A choice's eligibility fades by g = exp(-1/tau) each moment until its outcome, so the
striatum's values settle at Q = r g^D and its race walks on with probability
    P(walk on) = p g^8 / (r_S g^2 + p g^8),      p = 0.55
Low tau is impulsivity, as a number. With kappa > 0, tau follows the measured time
between decisions and settles where it equals the trial time its own choices produce.
    --tau T         a fixed patience (default: runs tau 2, 10 and adaptive)
    --kappa K       adaptation rate of tau (default 0.02 for the adaptive run)
    --small R       the stop's sure reward (default 0.4)
    --trials N      trials per run (default 2400; the second half is measured)

NORADRENALINE sets the salience gate's criterion (week 23) from the recent rate of
judged surprise. Laptop only: the gate judges a stream against 64 known codes, a share
pi of arrivals new; the criterion fixed at the wrong rate, at the right rate, and
adapting (ubrain's example 10, ported).

Every run appends rows to results.csv. ubrain is the course's local copy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

import ubrain_local

ubrain_local.load()                 # the vendored wheel onto sys.path, before `import ubrain`

from ubrain import Brain, Channel, FrozenCodes, Manifest, Sense, Wiring, profile  # noqa: E402

HERE = Path(__file__).resolve().parent
WORLD = HERE / "traces" / "world.json"          # week 24's measured world: outcome[cue][move][phase]
D_S, D_L, GAP = 2, 8, 2                         # moments to the stop's reward, to the walk's, between trials
WALK_ON = (0, 2)                                # the world's (cue soon, keep walking): a push at 0.2 s, no brake


def log(rows: list[dict]) -> None:
    new = not (HERE / "results.csv").exists()
    with open(HERE / "results.csv", "a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["when", "step", "arm", "predicted", "measured", "extra"])
        if new:
            wr.writeheader()
        for r in rows:
            wr.writerow({"when": datetime.now().isoformat(timespec="seconds"), **r})


# ------------------------------------------------------------------ serotonin

def walk_outcomes() -> np.ndarray:
    world = np.array(json.loads(WORLD.read_text(encoding="utf-8"))["outcome[cue][move][phase]"])
    return world[WALK_ON[0]][WALK_ON[1]]                 # 40 stride points: 1 stayed up, 0 fell


def p_walk(tau: float, r_s: float, p: float) -> float:
    g = math.exp(-1.0 / tau)
    return p * g ** D_L / (r_s * g ** D_S + p * g ** D_L)


def tau_fixed_point(r_s: float, p: float) -> float:
    """tau* = the mean trial time its own choices produce: D_S + GAP + P(walk)(D_L - D_S)."""
    lo, hi = D_S + GAP, D_L + GAP
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if D_S + GAP + p_walk(mid, r_s, p) * (D_L - D_S) > mid else (lo, mid)
    return 0.5 * (lo + hi)


def patient_brain(tau: float, kappa: float, seed: int) -> Brain:
    w = Wiring("patient", "1.0",
               regions={"neuromodulator.serotonin": {"tau": tau, "kappa": kappa},
                        "basal_ganglia.striatum": {"eta": 0.1, "q0": 0.5}},
               inputs={"neuromodulator.serotonin.state": "situation", "basal_ganglia.striatum.state": "situation",
                       "basal_ganglia.striatum.reward": "reward",
                       "basal_ganglia.striatum.patience": "neuromodulator.serotonin.patience"},
               outputs={"action": "basal_ganglia.striatum.choice"})
    body = Manifest("course/g1-gust", "none", (Channel("situation", "event", 1), Channel("reward", "reward", 1),
                                              Channel("action", "choice", 2)))
    return Brain(w, body, rng=np.random.default_rng(seed))


def patience_run(tau: float, kappa: float, r_s: float, outcomes: np.ndarray, trials: int, seed: int):
    brain = patient_brain(tau, kappa, seed)
    rng = np.random.default_rng(seed + 100)
    clock, due, late, taus, earned, moments = 0, None, [], [], 0.0, 0

    def tick(senses):
        nonlocal clock, due
        if due is not None and due[0] == clock:
            senses = senses | {"reward": due[1]}
            due = None
        clock += 1
        return brain.tick(senses)

    for t in range(trials):
        cue = clock
        a = tick({"situation": 0})["action"]
        if a == 0:
            delay, r = D_S, r_s                                   # stop: sure, soon
        else:
            delay, r = D_L, float(outcomes[rng.integers(len(outcomes))])   # walk on: the G1's outcome
        due = (cue + delay, r)
        while clock < cue + delay + GAP:
            tick({})
        if t >= trials // 2:
            late.append(a)
            taus.append(brain.regions["neuromodulator.serotonin"].tau)
            earned += r
            moments += delay + GAP
    return np.array(late, dtype=float), np.array(taus), earned / moments


def step_serotonin(a) -> None:
    outcomes = walk_outcomes()
    p = float(outcomes.mean())
    print(f"SEROTONIN on the G1: stop = {a.small} after {D_S} moments; walk on = 1.0 after {D_L} moments if still "
          f"up (the G1 stays up at {int(outcomes.sum())} of {len(outcomes)} stride points: p = {p:.2f}); "
          f"{GAP} moments between trials")
    print(f"  per trial: stop {a.small:.2f}, walk on {p:.2f};  per moment: stop {a.small / (D_S + GAP):.3f}, "
          f"walk on {p / (D_L + GAP):.3f}")
    print("\n  patience                       P(walk on) predicted   measured        reward per moment")
    rows = []
    arms = [(a.tau, 0.0, f"tau {a.tau:g} fixed")] if a.tau else [(2.0, 0.0, "tau 2 fixed (low serotonin)"),
                                                                  (10.0, 0.0, "tau 10 fixed (high serotonin)")]
    for tau, kappa, name in arms:
        x, _, rate = patience_run(tau, kappa, a.small, outcomes, a.trials, seed=int(tau * 10))
        law = p_walk(tau, a.small, p)
        print(f"  {name:31s}  {law:.3f}                {x.mean():.3f} (+-{math.sqrt(law * (1 - law) / x.size):.3f})"
              f"   {rate:.3f}")
        rows.append({"step": "serotonin", "arm": name, "predicted": round(law, 3), "measured": round(float(x.mean()), 3),
                     "extra": f"small {a.small}; reward per moment {rate:.3f}"})
    if not a.tau:
        tau_star = tau_fixed_point(a.small, p)
        x, taus, rate = patience_run(4.0, a.kappa, a.small, outcomes, a.trials, seed=3)
        law = p_walk(tau_star, a.small, p)
        print(f"  {f'tau adaptive (kappa {a.kappa})':31s}  {law:.3f}                {x.mean():.3f}"
              f"            {rate:.3f}")
        print(f"\n  adaptive tau: predicted {tau_star:.2f} moments (the trial time its own choices produce), "
              f"measured {taus.mean():.2f} (sd {taus.std():.2f})")
        rows.append({"step": "serotonin", "arm": f"adaptive kappa {a.kappa}", "predicted": round(law, 3),
                     "measured": round(float(x.mean()), 3),
                     "extra": f"tau* {tau_star:.2f}, measured {taus.mean():.2f}; reward per moment {rate:.3f}"})
    log(rows)


# ------------------------------------------------------------------ noradrenaline (ubrain's example 10, ported)

def step_noradrenaline(a) -> None:
    from ubrain.promoted.cls.hrr import normalize
    from ubrain.promoted.control.salience import noisy_repeat
    from ubrain.regions import NE_THETAS, novelty_criterion, novelty_rates

    w0 = profile("modulated")
    ne = w0.regions["neuromodulator.noradrenaline"]
    d, M, sigma, eta = ne["d"], ne["M"], ne["sigma"], ne["eta"]
    N = a.events

    def manifest(V):
        return Manifest("course/modulated", "none", (
            Channel("routes", "vector", d), Channel("goal", "vector", d), Channel("attend", "event", 4),
            Channel("camera", "codec", 1), Channel("known", "vector", d), Channel("recollect", "event", 64),
            Channel("situation", "event", 1), Channel("reward", "reward", 1), Channel("workspace", "command", d),
            Channel("attended", "command", d), Channel("recollection", "command", d),
            Channel("recollected", "choice", V), Channel("mode", "choice", 2), Channel("action", "choice", 2)))

    def judge(w, pis, seed):
        rng = np.random.default_rng(seed)
        known = normalize(rng.standard_normal((M, d)))
        novel = rng.random(len(pis)) < pis
        codes = np.array([normalize(rng.standard_normal(d)) if nv else noisy_repeat(known[rng.integers(M)], sigma, rng)
                          for nv in novel])
        E = np.vstack([known, codes])
        sense = Sense(FrozenCodes(E), mean=np.zeros(d), d=d, vocabulary=E, items=list(range(len(E))))
        brain = Brain(w, manifest(len(E)), codecs={"camera": sense}, rng=np.random.default_rng(seed + 1))
        brain.tick({"known": known})
        gate, nreg = brain.regions["salience.gate"], brain.regions["neuromodulator.noradrenaline"]
        errs, levels = [], []
        for t in range(len(pis)):
            before = gate.novel
            brain.tick({"camera": M + t})
            errs.append((gate.novel > before) != novel[t])
            levels.append(nreg.level)
        return np.array(errs, dtype=float), np.array(levels)

    def err(pi, k):
        ff, rn = novelty_rates(d, M, sigma)
        return pi * ff[k] + (1 - pi) * rn[k]

    def fixed_point(pi, level, eta_):
        ff, rn = novelty_rates(d, M, sigma)
        for _ in range(20000):
            k = novelty_criterion(level, d, M, sigma)
            level += eta_ * (pi * (1 - ff[k]) + (1 - pi) * rn[k] - level)
        return level

    print(f"NORADRENALINE (laptop): the gate judges against {M} known codes, d {d}, repeats at noise {sigma}")
    print("  arm                           criterion   error predicted   error measured")
    rows = []
    for pi in (0.1, 0.5):
        w = profile("modulated")
        w.regions["neuromodulator.noradrenaline"].update(eta=0.0, level=pi)
        k = novelty_criterion(pi, d, M, sigma)
        e, _ = judge(w, np.full(N, pi), seed=int(pi * 100))
        rows.append((f"fixed at {pi}, stream at {pi}", NE_THETAS[k], err(pi, k), e))
    pis = np.where(np.arange(N) < N // 2, 0.1, 0.5)
    late = slice(N // 2 + 500, N)
    w = profile("modulated")
    w.regions["neuromodulator.noradrenaline"].update(eta=0.0, level=0.1)
    k = novelty_criterion(0.1, d, M, sigma)
    e, _ = judge(w, pis, seed=7)
    rows.append(("fixed at 0.1, stream at 0.5", NE_THETAS[k], err(0.5, k), e[late]))
    w = profile("modulated")
    w.regions["neuromodulator.noradrenaline"]["eta"] = a.eta
    e, levels = judge(w, pis, seed=7)
    L = fixed_point(0.5, levels[N // 2 - 1], a.eta)
    k = novelty_criterion(L, d, M, sigma)
    rows.append((f"adaptive (eta {a.eta}), at 0.5", NE_THETAS[k], err(0.5, k), e[late]))
    out = []
    for name, theta, law, e in rows:
        se = math.sqrt(law * (1 - law) / e.size)
        print(f"  {name:28s}  {theta:.3f}       {law:.3f}             {e.mean():.3f} (+-{se:.3f})")
        out.append({"step": "noradrenaline", "arm": name, "predicted": round(law, 3), "measured": round(float(e.mean()), 3),
                    "extra": f"criterion {theta:.3f}"})
    print(f"  adaptive level after the change: predicted {L:.3f} (its fixed point), measured "
          f"{levels[late].mean():.3f} (sd {levels[late].std():.3f})")
    log(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.split("\n")[6:]))
    sub = ap.add_subparsers(dest="step", required=True)
    p = sub.add_parser("serotonin")
    p.add_argument("--tau", type=float, default=0.0)
    p.add_argument("--kappa", type=float, default=0.02)
    p.add_argument("--small", type=float, default=0.4)
    p.add_argument("--trials", type=int, default=2400)
    p = sub.add_parser("noradrenaline")
    p.add_argument("--eta", type=float, default=0.02)
    p.add_argument("--events", type=int, default=4000)
    a = ap.parse_args(argv)
    (step_serotonin if a.step == "serotonin" else step_noradrenaline)(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
