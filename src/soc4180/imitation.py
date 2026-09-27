"""Imitation learning: demonstrations, a student network, behaviour cloning and DAgger. Week 13.

Needs torch (``uv sync --extra rl``). Not imported by ``soc4180`` itself:
``from soc4180 import imitation``.

A *student* is a small network trained by regression -- squared error --
to output what an *expert* did in the same situation. Everything in this
module is about what "the same situation" means: which numbers the student
is shown (``FEATURES``), how many past steps (``history``), and whose states
the examples come from (the expert's own -- behaviour cloning -- or the
student's -- DAgger).

    features  = ["clock", "time"]                         # what the student sees
    X, Y      = walker_demos(env, 10, features)           # expert: the week-4 walker
    student   = Student.fit(X, Y)                         # behaviour cloning
    walk(env, student, features)                          # did it work?

    X, Y      = policy_demos(env, teacher, 40, features, history=5)
    student   = dagger(env, teacher, X, Y, features, history=5, iterations=3)
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

__all__ = ["FEATURES", "Student", "clock", "dagger", "observe", "policy_demos", "shove_test", "walk", "walker_demos"]

#: What a student may be shown, name -> slice of G1WalkEnv's 42-number observation
#: (or a feature computed from the time). The expert may see more; that is the point.
FEATURES = {
    "gravity": slice(0, 3),         # gravity in the torso frame, from the IMU
    "gyro": slice(3, 6),            # angular velocity, from the IMU
    "joints": slice(6, 18),         # 12 leg angles minus the crouch
    "velocities": slice(18, 30),    # 12 leg joint velocities
    "previous": slice(30, 42),      # the previous action
    "clock": "clock",               # sin and cos of the gait phase, and "walking yet?" (3)
    "time": "time",                 # seconds since the start, / 10 (1)
}


def clock(t: float, params) -> np.ndarray:
    """The gait phase as a student can use it: (sin, cos, walking) -- zeros while the walker settles."""
    if t < params.settle_time:
        return np.zeros(3, np.float32)
    phase = 2 * math.pi * (t - params.settle_time) / (2 * params.step_time)
    return np.array([math.sin(phase), math.cos(phase), 1.0], np.float32)


def observe(obs: np.ndarray, t: float, features, params=None) -> np.ndarray:
    """The student's view of one step: the chosen slices of ``obs``, then clock/time if chosen."""
    parts = []
    for f in features:
        if f not in FEATURES:
            raise ValueError(f"no feature called {f!r}; features are {', '.join(FEATURES)}")
        s = FEATURES[f]
        if s == "clock":
            parts.append(clock(t, params))
        elif s == "time":
            parts.append(np.array([t / 10.0], np.float32))
        else:
            parts.append(obs[s])
    return np.concatenate(parts).astype(np.float32)


class Student(nn.Module):
    """Two hidden layers of 256, inputs standardised by the training data's mean and spread."""

    def __init__(self, n_in: int, n_out: int = 12, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(),
                                 nn.Linear(hidden, n_out))
        self.register_buffer("mu", torch.zeros(n_in))
        self.register_buffer("sd", torch.ones(n_in))
        self.train_mse = float("nan")

    def forward(self, x):
        return self.net((x - self.mu) / self.sd)

    @classmethod
    def fit(cls, X, Y, *, epochs: int = 30, lr: float = 1e-3, batch: int = 256, seed: int = 0) -> "Student":
        """Behaviour cloning: minimise the mean squared difference from the expert's actions."""
        torch.manual_seed(seed)
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        Y = torch.as_tensor(np.asarray(Y), dtype=torch.float32)
        s = cls(X.shape[1], Y.shape[1])
        s.mu[:] = X.mean(0)
        s.sd[:] = X.std(0) + 1e-3
        opt = torch.optim.Adam(s.parameters(), lr)
        for _ in range(epochs):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), batch):
                b = perm[i:i + batch]
                loss = ((s(X[b]) - Y[b]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            s.train_mse = float(((s(X) - Y) ** 2).mean())
        return s

    def act(self, x) -> np.ndarray:
        with torch.no_grad():
            return np.clip(self(torch.as_tensor(x, dtype=torch.float32)[None])[0].numpy(), -1.0, 1.0)


# --- the walker as an expert ------------------------------------------------------------------------

def _walker_env():
    from .envs import G1WalkEnv

    # 100 Hz and a full radian per unit: the only setting at which the week-4
    # walker survives as env actions (week 7's grid).
    return G1WalkEnv(control_hz=100.0, action_scale=1.0, episode_seconds=10.0)


def walker_demos(n: int, features, *, seed: int = 0, env=None, params=None):
    """Run the week-4 walker ``n`` times (small random starting crouches); record (student view, action)."""
    from .envs import walker_actions
    from .walking import GaitParams, WalkingController

    env = env or _walker_env()
    params = params or GaitParams(n_steps=40)
    X, Y = [], []
    for k in range(n):
        obs, _ = env.reset(seed=seed + k)
        walker = WalkingController(env.model, params)
        while True:
            t = env.data.time
            a = walker_actions(env, walker, t)
            X.append(observe(obs, t, features, params)); Y.append(a)
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    return np.array(X), np.array(Y)


def walk(student, features, *, seeds=(100, 101, 102), env=None, params=None, record=False, push=0.0):
    """Let the student drive for 10 s from each seed. Returns [(decisions survived, metres), ...] (and frames).

    ``push`` shoves the torso sideways with that many newtons for 0.2 s at t = 3 s.
    """
    import mujoco

    from .walking import GaitParams

    env = env or _walker_env()
    params = params or GaitParams(n_steps=40)
    torso = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    out, frames = [], []
    for k, s in enumerate(seeds):
        obs, _ = env.reset(seed=s)
        n = 0
        while True:
            env.data.xfrc_applied[torso, 1] = push if 3.0 <= env.data.time < 3.2 else 0.0
            obs, _, term, trunc, _ = env.step(student.act(observe(obs, env.data.time, features, params)))
            n += 1
            if record and k == 0:
                frames.append(env.data.qpos.copy())
            if term or trunc:
                break
        out.append((n, float(env.data.qpos[0])))
    return (out, frames) if record else out


# --- a policy as an expert (weeks 12-13) -----------------------------------------------------------------

def _student_view(history, features, h):
    views = [observe(o, 0.0, features) for o in history[-h:]]
    return np.concatenate(views)


def _run(env, teacher, student, features, history, *, seed, beta=0.0, rng=None, world=None, push=None,
         record=True, frames=None):
    """One G1PushEnv episode. Acts with the teacher, the student, or a beta-mix; labels every step with the teacher."""
    from .envs import RANDOM_WORLD

    env.world = {k: (v, v) for k, v in world.items()} if world is not None else dict(RANDOM_WORLD)
    obs, _ = env.reset(seed=seed)
    if push is not None:
        env.push_force = np.array([0.0, float(push)])                 # sideways, exactly `push` newtons
    past = [obs] * history
    X, Y = [], []
    while True:
        view = _student_view(past, features, history) if (student is not None or record) else None
        a_teacher = teacher.predict(obs, deterministic=True)[0] if teacher is not None else None
        if record:
            X.append(view); Y.append(a_teacher)
        use_teacher = student is None or (rng is not None and rng.random() < beta)
        a = a_teacher if use_teacher else student.act(view)
        obs, _, term, trunc, _ = env.step(a)
        past = past[1:] + [obs]
        if frames is not None:
            frames.append(env.data.qpos.copy())
        if term or trunc:
            return X, Y, not term


def policy_demos(env, teacher, n: int, features, *, history: int = 1, seed: int = 0):
    """Behaviour-cloning data: the teacher drives ``n`` randomised, shoved episodes; every step is an example."""
    X, Y = [], []
    for k in range(n):
        x, y, _ = _run(env, teacher, None, features, history, seed=seed + k)
        X += x; Y += y
    return np.array(X), np.array(Y)


def dagger(env, teacher, X, Y, features, *, history: int = 1, iterations: int = 3, episodes: int = 20,
           epochs: int = 30, seed: int = 1000, log=None):
    """DAgger (Ross et al., 2011): let the STUDENT drive, ask the teacher what it should have done, retrain.

    Iteration 1 mixes: each step is driven by the teacher with probability
    1/2, so the first student's mistakes do not end every episode at once.
    After that the student drives alone. The dataset only grows.
    """
    student = Student.fit(X, Y, epochs=epochs)
    rng = np.random.default_rng(seed)
    X, Y = list(X), list(Y)
    for it in range(1, iterations + 1):
        for k in range(episodes):
            x, y, _ = _run(env, teacher, student, features, history, seed=seed * it + k,
                           beta=0.5 if it == 1 else 0.0, rng=rng)
            X += x; Y += y
        student = Student.fit(X, Y, epochs=epochs)
        if log is not None:
            log(it, len(X), student)
    return student


def shove_test(env, policy, features=None, *, history: int = 1, push: float = 100.0, episodes: int = 20,
               world=None, seed: int = 5000, frames=None) -> int:
    """Episodes (of ``episodes``) survived under a sideways shove. ``policy`` is a Student or an SB3 model."""
    from .envs import NOMINAL_WORLD

    world = world or {k: v[0] for k, v in NOMINAL_WORLD.items()}
    ok = 0
    for k in range(episodes):
        if isinstance(policy, Student):
            _, _, alive = _run(env, None, policy, features, history, seed=seed + k, world=world, push=push,
                               record=False, frames=frames if k == 0 else None)
        else:
            _, _, alive = _run(env, policy, None, features or [], 1, seed=seed + k, world=world, push=push,
                               record=False, frames=frames if k == 0 else None)
        ok += alive
    return ok
