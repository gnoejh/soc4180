"""Tabular reinforcement learning (week 7b): a table of values, filled from samples.

Week 7 solved an MDP whose P and R were arrays (value iteration). This module
does the same jobs WITHOUT P: the agent only acts and is told what happened,
and a table Q(s, a) is nudged toward each sample. Then it puts a table on a
continuous problem (an inverted pendulum, by cutting its state into bins) and
on week 7's G1 balance, where the exact answer is known -- and shows where
tables stop.

    value_iteration(P, nS, nA, gamma)      the model-based answer (week 7), for toy-text P
    Learner(env, algo="q"|"sarsa", ...)    Q-learning / SARSA, one episode at a time
    Binner(edges, take)                    continuous observation -> one table row
    pendulum_learner(...)                  a Learner on InvertedPendulum-v5, 3 forces
    BalanceMDP / balance_q_learning        week 7's LIPM balance, learned from samples

Every update is the same line, written out:

    Q[s, a] += alpha * (target - Q[s, a])
    target = r + gamma * max_a' Q[s', a']        Q-learning (off-policy)
    target = r + gamma * Q[s', a']               SARSA: a' is the action it WILL take

Nothing here imports mujoco directly; gymnasium's MuJoCo environments do.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["Binner", "BalanceLearner", "BalanceMDP", "KeepBest", "Learner", "balance_q_learning",
           "greedy_policy", "keep_best", "pendulum_learner", "run_greedy", "value_iteration"]


# --- 1. the model-based answer: value iteration on a toy-text P --------------------------------------

def value_iteration(P, n_states: int, n_actions: int, gamma: float, tol: float = 1e-10):
    """V*, Q* and the number of sweeps, from gymnasium's ``env.unwrapped.P``.

    ``P[s][a]`` is a list of (probability, next state, reward, done). A done
    transition has no future: its value is the reward alone.
    """
    V = np.zeros(n_states)
    for sweep in range(1, 100_000):
        Q = np.array([[sum(p * (r + (0.0 if done else gamma * V[s2])) for p, s2, r, done in P[s][a])
                       for a in range(n_actions)] for s in range(n_states)])
        V_new = Q.max(axis=1)
        if np.abs(V_new - V).max() < tol:
            return V_new, Q, sweep
        V = V_new
    raise RuntimeError("value iteration did not converge")


def greedy_policy(Q):
    """The action with the highest value in every row."""
    return np.asarray(Q).argmax(axis=1)


# --- 2. learning from samples: Q-learning and SARSA ------------------------------------------------

class Learner:
    """Tabular Q-learning or SARSA with epsilon-greedy exploration, one episode per ``episode()``.

    ``index(obs) -> row`` and ``act(a) -> what env.step takes`` adapt any
    gymnasium env: identity for toy-text, a ``Binner`` and a force table for
    the pendulum. Epsilon is multiplied by ``eps_decay`` after every episode,
    never below ``eps_min``. Episode k is reset with seed ``seed * 100000 + k``,
    so a run repeats exactly.
    """

    def __init__(self, env, n_rows: int, n_actions: int, *, algo: str = "q", alpha: float = 0.1,
                 gamma: float = 0.99, epsilon: float = 0.1, eps_decay: float = 1.0, eps_min: float = 0.0,
                 seed: int = 0, index=None, act=None, max_steps: int = 10_000):
        if algo not in ("q", "sarsa"):
            raise ValueError(f"algo must be 'q' or 'sarsa', not {algo!r}")
        self.env, self.algo = env, algo
        self.alpha, self.gamma = float(alpha), float(gamma)
        self.epsilon, self.eps_decay, self.eps_min = float(epsilon), float(eps_decay), float(eps_min)
        self.seed, self.max_steps = int(seed), int(max_steps)
        self.index = index or (lambda obs: int(obs))
        self.act = act or (lambda a: int(a))
        self.Q = np.zeros((n_rows, n_actions))
        self.rng = np.random.default_rng(seed)
        self.episodes = 0
        self.returns: list[float] = []          # what the agent earned WHILE exploring
        self.lengths: list[int] = []

    def pick(self, s: int) -> int:
        """Epsilon-greedy: a random action with probability epsilon, else the best (ties broken at random)."""
        n = self.Q.shape[1]
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(n))
        row = self.Q[s]
        return int(self.rng.choice(np.flatnonzero(row == row.max())))

    def episode(self) -> float:
        obs, _ = self.env.reset(seed=self.seed * 100_000 + self.episodes)
        s = self.index(obs)
        a = self.pick(s)
        total, steps = 0.0, 0
        while True:
            obs, r, terminated, truncated, _ = self.env.step(self.act(a))
            s2 = self.index(obs)
            total += r
            steps += 1
            if self.algo == "sarsa":
                a2 = self.pick(s2)                                       # choose first ...
                future = self.Q[s2, a2]                                  # ... and value THAT action
            else:
                future = self.Q[s2].max()                                # value the best action
            target = r + (0.0 if terminated else self.gamma * future)   # a terminal state has no future
            self.Q[s, a] += self.alpha * (target - self.Q[s, a])
            if self.algo == "q":
                a2 = self.pick(s2)
            s, a = s2, a2
            if terminated or truncated or steps >= self.max_steps:
                break
        self.episodes += 1
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        self.returns.append(total)
        self.lengths.append(steps)
        return total

    def train(self, episodes: int):
        for _ in range(episodes):
            self.episode()
        return self


def run_greedy(env, Q, n: int, seed0: int, *, index=None, act=None, max_steps: int = 10_000):
    """Play the greedy policy of Q for n episodes (seeds seed0 ...). Returns (returns, lengths, last rewards)."""
    index = index or (lambda obs: int(obs))
    act = act or (lambda a: int(a))
    rets, lens, last = [], [], []
    for k in range(n):
        obs, _ = env.reset(seed=seed0 + k)
        total, steps, r = 0.0, 0, 0.0
        while True:
            obs, r, terminated, truncated, _ = env.step(act(int(np.argmax(Q[index(obs)]))))
            total += r
            steps += 1
            if terminated or truncated or steps >= max_steps:
                break
        rets.append(total)
        lens.append(steps)
        last.append(r)
    return np.array(rets), np.array(lens), np.array(last)


# --- 3. a continuous state, cut into bins -----------------------------------------------------------

class Binner:
    """Map chosen observation entries to one table row.

    ``take`` lists which entries of the observation are used; ``edges[k]`` are
    the cut points for entry ``take[k]`` (n edges make n + 1 bins; values past
    the end land in the outer bins). Rows = product of the bin counts.
    """

    def __init__(self, edges, take):
        self.edges = [np.asarray(e, float) for e in edges]
        self.take = list(take)
        if len(self.edges) != len(self.take):
            raise ValueError("one list of edges per observation entry taken")
        self.sizes = [len(e) + 1 for e in self.edges]
        self.n = int(np.prod(self.sizes))

    def __call__(self, obs) -> int:
        row = 0
        for e, k, n in zip(self.edges, self.take, self.sizes):
            row = row * n + int(np.searchsorted(e, obs[k]))
        return row


PENDULUM_OBS = ("cart position", "pole angle", "cart velocity", "pole angular velocity")


def pendulum_learner(edges, take=(1, 3), forces=(-3.0, 0.0, 3.0), *, alpha=0.1, gamma=0.99, eps_decay=0.998,
                     eps_min=0.02, seed=0, env=None):
    """A Q-learner on gymnasium's ``InvertedPendulum-v5`` (MuJoCo), with a few fixed forces as its actions.

    The observation is (cart position, pole angle, cart velocity, pole angular
    velocity); ``take=(1, 3)`` bins only the pole. The env pays +1 per step and
    ends the episode when the pole leans past 0.2 rad, or after 1000 steps.
    """
    import gymnasium as gym

    env = env or gym.make("InvertedPendulum-v5")
    binner = Binner(edges, take)
    forces = np.asarray(forces, float)
    lo, hi = float(env.action_space.low[0]), float(env.action_space.high[0])
    if forces.min() < lo or forces.max() > hi:
        raise ValueError(f"forces must lie in [{lo}, {hi}] N")
    learner = Learner(env, binner.n, len(forces), algo="q", alpha=alpha, gamma=gamma, epsilon=1.0,
                      eps_decay=eps_decay, eps_min=eps_min, seed=seed, index=binner,
                      act=lambda a: np.array([forces[a]]))
    learner.binner, learner.forces = binner, forces
    return learner


class KeepBest:
    """Train a learner, and every ``every`` episodes play its greedy table from ``starts``; keep the best.

    A coarse table forgets: it can balance for 1000 steps at one check and
    fall at the next, because one row lumps together states that need
    different pushes. Keeping the best checked table is model selection by
    evaluation, not a change to the learning rule. ``advance(n)`` trains n
    more episodes (checking on the way), so a viewer can stay live.
    """

    def __init__(self, learner, episodes, every=250, starts=(500_000, 500_001, 500_002)):
        self.learner, self.episodes, self.every, self.starts = learner, int(episodes), int(every), tuple(starts)
        self.best, self.best_Q, self.when, self.curve = -1.0, learner.Q.copy(), 0, []
        self.done = 0
        self._check()

    def _check(self):
        L = self.learner
        score = float(np.mean([run_greedy(L.env, L.Q, 1, s, index=L.index, act=L.act)[1][0] for s in self.starts]))
        self.curve.append((L.episodes, score))
        if score > self.best:
            self.best, self.best_Q, self.when = score, L.Q.copy(), L.episodes

    @property
    def finished(self) -> bool:
        return self.done >= self.episodes

    def advance(self, n):
        for _ in range(min(n, self.episodes - self.done)):
            self.learner.episode()
            self.done += 1
            if self.done % self.every == 0:
                self._check()
        return self


def keep_best(learner, episodes, every=250, starts=(500_000, 500_001, 500_002)):
    """``KeepBest`` run to the end. Returns (best Q, best score, episode found at, [(episode, score)])."""
    kb = KeepBest(learner, episodes, every, starts).advance(episodes)
    return kb.best_Q, kb.best, kb.when, kb.curve


# --- 4. week 7's G1 balance, learned from samples ------------------------------------------------------

class BalanceMDP:
    """Week 7's sagittal balance: the LIPM with the G1's numbers, 50 Hz, 7 centre-of-pressure actions.

    The same constants as weeks/07-mdp/mdp.py, so the exact answer is known:
    value iteration's pi* earns J = 72.12 over the pushes -0.30 .. +0.60 m/s.
    """

    G, H = 9.81, 0.661
    HEEL, TOE = -0.05, 0.12
    FALL_LO, FALL_HI = HEEL - 0.05, TOE + 0.05
    DT, V_MAX = 0.02, 1.0
    PUSHES = np.round(np.arange(-0.30, 0.601, 0.01), 2)
    J_STAR = 72.12                           # value iteration (weeks/07-mdp/mdp.py)

    def __init__(self, gamma=0.99, weights=(1.0, 1.0, 0.1), n_actions=7):
        self.gamma, self.weights = gamma, weights
        self.ps = np.linspace(self.HEEL, self.TOE, n_actions)
        w = math.sqrt(self.G / self.H)
        self._c, self._s, self._w = math.cosh(w * self.DT), math.sinh(w * self.DT), w

    def step(self, x, v, p):
        """One decision of x'' = (g/h)(x - p), solved exactly."""
        return p + (x - p) * self._c + v * self._s / self._w, (x - p) * self._w * self._s + v * self._c

    def fallen(self, x, v):
        return (x < self.FALL_LO) | (x > self.FALL_HI) | (np.abs(v) > self.V_MAX)

    def reward(self, x, p):
        alive, upright, effort = self.weights
        return alive - upright * (x / 0.1) ** 2 - effort * (p / 0.1) ** 2

    def J(self, pi, seconds=10.0):
        """Mean discounted return of the policy pi(x, v) -> p from every push in PUSHES (no noise)."""
        x, v = np.zeros(len(self.PUSHES)), self.PUSHES.astype(float)
        up, ret, disc = np.ones(len(x), bool), np.zeros(len(x)), 1.0
        for _ in range(int(round(seconds / self.DT))):
            p = pi(x, v)
            ret += np.where(up, disc * self.reward(x, p), 0.0)
            x, v = self.step(x, v, p)
            up &= ~self.fallen(x, v)
            disc *= self.gamma
        return float(ret.mean())


class BalanceLearner:
    """Q-learning on the balance from episodes only: a bins x bins table over (x, v), 7 actions.

    ``batch`` episodes run side by side (numpy), each 10 s from x = 0 with a
    random push from PUSHES; ``advance(rounds)`` runs that many batches.
    ``check()`` appends (episodes so far, J of the greedy table) to ``curve``.
    """

    def __init__(self, bins=21, *, alpha=0.1, eps_decay=0.995, eps_min=0.05, seed=0, batch=64, mdp=None):
        self.mdp = mdp or BalanceMDP()
        self.bins, self.alpha, self.eps_decay, self.eps_min, self.batch = int(bins), alpha, eps_decay, eps_min, batch
        self.xs = np.linspace(self.mdp.FALL_LO, self.mdp.FALL_HI, bins + 1)[1:-1]
        self.vs = np.linspace(-self.mdp.V_MAX, self.mdp.V_MAX, bins + 1)[1:-1]
        self.Q = np.zeros((self.bins * self.bins, len(self.mdp.ps)))
        self.rng = np.random.default_rng(seed)
        self.epsilon, self.episodes, self.curve = 1.0, 0, []

    def cell(self, x, v):
        return np.searchsorted(self.xs, x) * self.bins + np.searchsorted(self.vs, v)

    def policy(self, x, v):
        """The greedy table as a policy on the real (continuous) pendulum: p for each (x, v)."""
        return self.mdp.ps[self.Q[self.cell(x, v)].argmax(axis=1)]

    def check(self):
        self.curve.append((self.episodes, round(self.mdp.J(self.policy), 2)))
        return self.curve[-1][1]

    def advance(self, rounds=1):
        m, Q, B, rng = self.mdp, self.Q, self.batch, self.rng
        for _ in range(rounds):
            x, v = np.zeros(B), rng.choice(m.PUSHES, B)
            up = np.ones(B, bool)
            for _ in range(500):
                s = self.cell(x, v)
                a = np.where(rng.random(B) < self.epsilon, rng.integers(len(m.ps), size=B), Q[s].argmax(axis=1))
                r = m.reward(x, m.ps[a])
                x, v = m.step(x, v, m.ps[a])
                fell = m.fallen(x, v)
                s2 = self.cell(x, v)
                target = r + np.where(fell, 0.0, m.gamma * Q[s2].max(axis=1))   # fallen: no future
                np.add.at(Q, (s[up], a[up]), self.alpha * (target[up] - Q[s[up], a[up]]))
                up &= ~fell
                if not up.any():
                    break
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
            self.episodes += B
        return self


def balance_q_learning(bins=21, episodes=20_000, *, seed=0, report=8, **kw):
    """``BalanceLearner`` for ``episodes``, checking J ``report`` times. Returns (learner, curve)."""
    L = BalanceLearner(bins, seed=seed, **kw)
    rounds = episodes // L.batch
    every = max(1, rounds // report)
    L.check()
    for it in range(1, rounds + 1):
        L.advance(1)
        if it % every == 0 or it == rounds:
            L.check()
    return L, L.curve
