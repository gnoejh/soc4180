"""An MDP written down in full and solved: the G1's forward balance by value iteration.

    uv run weeks/07-mdp/mdp.py
    uv run weeks/07-mdp/mdp.py --push 0.4
    uv run weeks/07-mdp/mdp.py --policy ankle --push 0.4
    uv run weeks/07-mdp/mdp.py --gamma 0.5 --no-viewer
    uv run weeks/07-mdp/mdp.py --alive 0 --no-viewer

G1WalkEnv (lab_env.py, env_run.py) is a *simulator*: it hands you one sample
s' and r when you give it s and a. It never shows you P or R as a whole, so
nothing can be solved from it directly -- weeks 8-10 learn from samples
instead. This script takes the opposite route on a problem small enough to
write down: every piece of the tuple (S, A, P, R, gamma) is an array you can
print, and the Bellman OPTIMALITY equation

    V*(s) = max_a [ R(s,a) + gamma * sum_s' P(s'|s,a) V*(s') ]

is iterated until it holds. The policy is then READ OFF V*:
pi*(s) = argmax_a [ ... ]. That is where R decides the policy.

The problem is the G1 standing still, seen from the side: week 4's linear
inverted pendulum (LIPM) with the G1's numbers, measured at the settled
crouch of G1WalkEnv -- mass 33.3 kg, centre of mass 0.661 m above the floor,
contact spheres from 0.05 m behind the ankle to 0.12 m in front of it.

    state   s = (x, v)   CoM ahead of the ankle (m), and its speed (m/s)
    action  a = p        where the centre of pressure goes under the foot (m),
                         7 values from the heel (-0.05) to the toe (+0.12);
                         moving it is what the ankle torque does (tau = m g p)
    dynamics             x'' = (g / h) (x - p), held for one decision (0.02 s,
                         the env's 50 Hz), solved exactly
    fallen               x < -0.10 or x > 0.17 (5 cm past heel or toe):
                         an absorbing state worth 0, the deck's "terminated"
    reward   R = alive - upright (x / 0.1)^2 - effort (p / 0.1)^2
             the env's alive / upright / effort terms, in this model's units

Five numbered steps; read them with the Bellman slides open. The script
prints the tuple, solves it, maps pi*, compares four policies, and replays
one push in the viewer (blue CoM, orange centre of pressure, purple capture
point). About 13 s headless; value iteration itself is 3 s.

    --gamma 0.99   --alive 1 --upright 1 --effort 0.1   reward weights
    --grid 81      states per axis (81 x 81 = 6561)
    --policy optimal|ankle|damped|hold   which policy the replay uses
    --ankle-gain 2 0      p = kx x + kv v: lab_env.py's idea, the lean only
    --damped-gain 2 0.5   the same rule with a velocity term
    --push 0.3     the replay starts at x = 0 with this speed (m/s; + forward)
    --no-viewer    --speed 1

Measured with this script (defaults unless named):

 1. Value iteration converges in 1376 sweeps (3 s). The largest change per
    sweep shrinks by exactly gamma (ratio 0.9900; 0.9500 at --gamma 0.95,
    0.9000 at 0.9): the Bellman update is a gamma-contraction, which is why
    it converges at all.
 2. Nobody told it about week 4. pi* keeps the robot up for 10 s from 32.3 %
    of the grid, and the capture point x + v/omega lies on the foot on 32.3 %:
    the two sets agree on 100.0 % of states. Pushes survived: -0.19 .. +0.46
    m/s against the capture point's -0.193 .. +0.462 (+0.46 stands, +0.47
    falls). pi* is almost bang-bang: heel or toe, rarely in between.
 3. The grid is an approximation and says so: J(pi*) = 72.12 measured by
    rollouts against 71.97 from the Bellman equation on the grid (--grid 41:
    71.95 against 71.01; --grid 161: 72.17 against 72.30, VI 13 s).
 4. Hand-written rules, judged by the same R:
        optimal   J 72.12   saved 32.3 %   pushes -0.19 .. +0.46
        damped    J 70.80   saved 32.3 %   pushes -0.19 .. +0.46  (p = 2x + 0.5v)
        ankle     J 34.41   saved  0.2 %   pushes -0.03 .. +0.04  (p = 2x)
        hold      J 12.66   saved  0.0 %   stands only at exact rest
    The lean-only rule is an undamped spring in this model: --policy ankle
    --push 0.1 swings and falls at 3.62 s. A good hand design (damped) saves
    the same states and earns slightly less: R is what tells them apart.
 5. gamma decides how far ahead pi* looks (horizon dt/(1-gamma); the pendulum's
    time constant is 0.26 s):
        gamma 0.99  2.0 s   saved 32.3 %   pushes -0.19 .. +0.46
        gamma 0.9   0.2 s   saved 31.8 %
        gamma 0.8   0.1 s   saved 20.6 %   pushes -0.17 .. +0.31
        gamma 0.7   0.07 s  saved  0.7 %   falls from every push
        gamma 0.5   0.04 s  saved  0.0 %
    At 0.7 pi* still scores best ON ITS OWN R (J 3.17 against damped's 3.00,
    which stands): optimal means optimal for R and gamma, not for what you meant.
 6. R decides what is worth saving. --alive 0 (penalties only): 26.1 % saved,
    pushes up to +0.40 -- falling ends the penalties, so pi* lets some
    saveable states go. --effort 3 (an expensive ankle): 28.7 %, up to +0.39.
    --upright 0 changes nothing that falls (32.3 %).
 7. pi* is optimal for the grid's MDP, not exactly for the pendulum: at
    --upright 0 damped earns 77.59 against pi*'s 77.56.

Exercises: find the smallest gamma that still saves 30 % of the grid; find an
--effort at which pi* stops saving the +0.30 push; set --grid 41 and 161 and
watch item 3's gap; make --ankle-gain's rule save as much as pi*.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

# -- the G1's numbers (measured at G1WalkEnv's settled crouch) -----------------------------
G = 9.81
H = 0.661                    # CoM height above the floor (m)
MASS = 33.34                 # kg
HEEL, TOE = -0.05, 0.12      # contact spheres, relative to the ankle (m)
FALL_LO, FALL_HI = HEEL - 0.05, TOE + 0.05
DT = 0.02                    # one decision: G1WalkEnv's 50 Hz
W = math.sqrt(G / H)         # the pendulum's rate, 1/s (week 4's omega)
V_MAX = 1.0                  # the velocity grid's edge (m/s); faster is past saving


# == 1. the dynamics: one decision of the pendulum ==========================================

def step(x, v, p):
    """Exact LIPM solution over DT with the centre of pressure held at p.

    x'' = W^2 (x - p) has solution x - p = A e^{Wt} + B e^{-Wt}; written with
    cosh and sinh it is the two lines below. Works on arrays too.
    """
    c, s = math.cosh(W * DT), math.sinh(W * DT)
    return p + (x - p) * c + v * s / W, (x - p) * W * s + v * c


def fallen(x, v):
    return (x < FALL_LO) | (x > FALL_HI) | (np.abs(v) > V_MAX)


def reward(x, p, alive, upright, effort):
    """R(s, a): alive bonus, minus a lean cost, minus an ankle-torque cost (tau = m g p)."""
    return alive - upright * (x / 0.1) ** 2 - effort * (p / 0.1) ** 2


# == 2. the MDP as arrays ==================================================================

class MDP:
    """S, A, P, R as arrays. State n (the last index) is FALLEN: absorbing, reward 0."""

    def __init__(self, n, alive, upright, effort, n_actions=7):
        self.xs = np.linspace(FALL_LO, FALL_HI, n)
        self.vs = np.linspace(-V_MAX, V_MAX, n)
        self.ps = np.linspace(HEEL, TOE, n_actions)
        X, V = np.meshgrid(self.xs, self.vs, indexing="ij")
        self.X, self.V = X.ravel(), V.ravel()            # state i = (X[i], V[i])
        self.n_states = X.size + 1                       # + FALLEN
        self.FALLEN = X.size
        nS, nA = X.size, len(self.ps)

        # R(s, a): one number per state and action
        self.weights = (alive, upright, effort)
        self.R = np.zeros((self.n_states, nA))
        for a, p in enumerate(self.ps):
            self.R[:nS, a] = reward(self.X, p, *self.weights)

        # P(s'|s, a): the pendulum lands between grid points, so its probability is
        # split over the four corners around it (bilinear weights, summing to 1).
        # A landing past the edge goes to FALLEN with probability 1. Stored sparse:
        # for each (s, a) four next-state indices and four probabilities.
        self.nxt = np.full((self.n_states, nA, 4), self.FALLEN, dtype=np.int64)
        self.prob = np.zeros((self.n_states, nA, 4))
        self.prob[:, :, 0] = 1.0                         # default: certainly FALLEN
        for a, p in enumerate(self.ps):
            x2, v2 = step(self.X, self.V, p)
            ok = ~fallen(x2, v2)
            idx, wts = self.corners(x2[ok], v2[ok])
            self.nxt[:nS][ok, a] = idx
            self.prob[:nS][ok, a] = wts

    def corners(self, x, v):
        """The four grid states around (x, v), and the bilinear weight of each."""
        n = len(self.xs)
        fx = (x - self.xs[0]) / (self.xs[1] - self.xs[0])
        fv = (v - self.vs[0]) / (self.vs[1] - self.vs[0])
        i = np.clip(np.floor(fx).astype(int), 0, n - 2); tx = fx - i
        j = np.clip(np.floor(fv).astype(int), 0, n - 2); tv = fv - j
        idx = np.stack([i * n + j, (i + 1) * n + j, i * n + j + 1, (i + 1) * n + j + 1], -1)
        wts = np.stack([(1 - tx) * (1 - tv), tx * (1 - tv), (1 - tx) * tv, tx * tv], -1)
        return idx, wts

    def expected_next(self, V):
        """sum_s' P(s'|s,a) V(s'), for every s and a at once: shape (states, actions)."""
        return (self.prob * V[self.nxt]).sum(-1)

    def value_at(self, V, x, v):
        """V at a point between grid states, by the same bilinear weights."""
        x, v = np.atleast_1d(x), np.atleast_1d(v)
        out = np.zeros(len(x))
        ok = ~fallen(x, v)
        idx, wts = self.corners(x[ok], v[ok])
        out[ok] = (wts * V[idx]).sum(-1)
        return out


# == 3. value iteration (Bellman optimality) and policy evaluation (Bellman expectation) =====

def value_iteration(mdp, gamma, tol=1e-6, log=None):
    V = np.zeros(mdp.n_states)
    for sweep in range(1, 100000):
        Q = mdp.R + gamma * mdp.expected_next(V)         # Q(s,a) = R + gamma E[V(s')]
        V_new = Q.max(axis=1)                            # V(s) = max_a Q(s,a)
        V_new[mdp.FALLEN] = 0.0                          # a fallen robot's future is worth 0
        err = np.abs(V_new - V).max()
        if log is not None:
            log.append(err)
        V = V_new
        if err < tol:
            return V, Q.argmax(axis=1), sweep
    raise RuntimeError("value iteration did not converge")


def evaluate(mdp, actions, gamma, tol=1e-6):
    """V^pi for a fixed table of actions: the Bellman equation with no max."""
    V = np.zeros(mdp.n_states)
    rows = np.arange(mdp.n_states)
    R = mdp.R[rows, actions]
    while True:
        V_new = R + gamma * mdp.expected_next(V)[rows, actions]
        V_new[mdp.FALLEN] = 0.0
        if np.abs(V_new - V).max() < tol:
            return V_new
        V = V_new


# == 4. policies that act on the real (continuous) pendulum ===================================
# Each takes arrays x, v and returns an array of centres of pressure p.

def greedy(mdp, V, gamma):
    """Act greedily on V: try every action one step ahead, keep the best. This IS pi*."""
    def pi(x, v):
        q = np.stack([reward(x, p, *mdp.weights) + gamma * mdp.value_at(V, *step(x, v, p))
                      for p in mdp.ps])
        return mdp.ps[q.argmax(axis=0)]
    return pi


def ankle(kx, kv):
    """A hand-designed rule: p = kx x + kv v, kept on the foot. kv = 0 is lab_env.py's
    ankle strategy (it reads only the tilt); kv > 0 adds damping."""
    return lambda x, v: np.clip(kx * x + kv * v, HEEL, TOE)


def hold(x, v):
    return np.zeros_like(x)


def table_of(mdp, pi):
    """A policy as a table over the grid: the nearest action at each state."""
    p = pi(mdp.X, mdp.V)
    acts = np.abs(p[:, None] - mdp.ps[None, :]).argmin(axis=1)
    return np.r_[acts, 0]


def rollout(pi, x, v, gamma, weights, seconds=10.0):
    """Run every start (x, v) in the arrays at once. Returns whether each stayed up, the
    discounted return each earned (the quantity V estimates), and the first one's path."""
    x, v = np.array(x, float), np.array(v, float)
    up = ~fallen(x, v)
    ret = np.zeros_like(x)
    disc = 1.0
    path = [(x[0], v[0], 0.0)]
    for _ in range(int(round(seconds / DT))):
        p = pi(x, v)
        ret += np.where(up, disc * reward(x, p, *weights), 0.0)
        x, v = step(x, v, p)
        up &= ~fallen(x, v)
        disc *= gamma
        if up[0]:
            path.append((x[0], v[0], p[0]))
    return up, ret, path


# == 5. printing ================================================================================

def ascii_map(mdp, table, fmt, rows=11, cols=13):
    """A coarse picture of a per-state table: x down the side, v along the top."""
    n = len(mdp.xs)
    ix = np.linspace(0, n - 1, rows).round().astype(int)
    iv = np.linspace(0, n - 1, cols).round().astype(int)
    print("   x \\ v " + "".join(f"{mdp.vs[j]:+6.2f}" for j in iv))
    for i in ix[::-1]:
        print(f"  {mdp.xs[i]:+6.3f} " + "".join(fmt(table[i * n + j]) for j in iv))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--alive", type=float, default=1.0)
    ap.add_argument("--upright", type=float, default=1.0)
    ap.add_argument("--effort", type=float, default=0.1)
    ap.add_argument("--grid", type=int, default=81)
    ap.add_argument("--policy", choices=["optimal", "ankle", "damped", "hold"], default="optimal")
    ap.add_argument("--ankle-gain", type=float, nargs=2, default=[2.0, 0.0], metavar=("KX", "KV"))
    ap.add_argument("--damped-gain", type=float, nargs=2, default=[2.0, 0.5], metavar=("KX", "KV"))
    ap.add_argument("--push", type=float, default=0.3)
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    # -- 1-2. the tuple, as arrays
    t0 = time.time()
    mdp = MDP(args.grid, args.alive, args.upright, args.effort)
    nS, nA = mdp.n_states, len(mdp.ps)
    print(f"S: {nS - 1} grid states (x {args.grid} from {FALL_LO:+.2f} to {FALL_HI:+.2f} m, "
          f"v {args.grid} from {-V_MAX:+.1f} to {V_MAX:+.1f} m/s) + 1 FALLEN")
    print(f"A: {nA} centre-of-pressure positions " + " ".join(f"{p:+.3f}" for p in mdp.ps) + " m")
    print(f"P: a table {nS} x {nA} x {nS} = {nS * nA * nS:,} numbers; each row has at most 4 "
          f"non-zero entries, and every row sums to 1 (largest error "
          f"{np.abs(mdp.prob.sum(-1) - 1).max():.1e})")
    print(f"R: {nS} x {nA} numbers, from {mdp.R[:-1].min():+.2f} to {mdp.R[:-1].max():+.2f}; "
          f"R = {args.alive:g} alive - {args.upright:g} (x/0.1)^2 - {args.effort:g} (p/0.1)^2")
    print(f"gamma = {args.gamma:g}: horizon 1/(1-gamma) = {1 / (1 - args.gamma):.0f} decisions "
          f"= {DT / (1 - args.gamma):.2f} s;  pendulum time constant 1/omega = {1 / W:.3f} s")

    # -- 3. solve: iterate the Bellman optimality equation until it holds
    log = []
    V, acts, sweeps = value_iteration(mdp, args.gamma, log=log)
    print(f"\nvalue iteration: {sweeps} sweeps, {time.time() - t0:.1f} s; max |V_new - V| per sweep: "
          + ", ".join(f"{log[k]:.3g}" for k in (0, 1, 9, 99, 999) if k < len(log))
          + f"; ratio of the last two {log[-1] / log[-2]:.4f} (gamma {args.gamma:g})")
    pis = {"optimal": greedy(mdp, V, args.gamma), "ankle": ankle(*args.ankle_gain),
           "damped": ankle(*args.damped_gain), "hold": hold}
    weights = mdp.weights
    up, ret, _ = rollout(pis["optimal"], mdp.X, mdp.V, args.gamma, weights)
    xi = mdp.X + mdp.V / W                                           # week 4's capture point
    capt = (xi >= HEEL) & (xi <= TOE)
    print(f"pi* keeps the robot up for 10 s from {up.mean():.1%} of the grid; the capture point "
          f"inside the foot (HEEL <= x + v/omega <= TOE) holds on {capt.mean():.1%}; "
          f"they agree on {(up == capt).mean():.1%} of states")
    err = np.abs(V[:-1] - ret)[up] if up.any() else np.zeros(1)
    print(f"V* on the grid against the return pi* actually earns (10 s rollouts, saved states): "
          f"mean |difference| {err.mean():.2f}, largest {err.max():.2f}; at rest V* = "
          f"{mdp.value_at(V, 0.0, 0.0)[0]:.2f} (standing still for ever earns "
          f"{args.alive / (1 - args.gamma):.2f})")

    print("\npi*(s): where value iteration puts the centre of pressure, cm ahead of the ankle "
          "('.' = falls within 10 s whatever it does)")
    table = np.where(np.r_[up, False], mdp.ps[acts] * 100, np.nan)
    ascii_map(mdp, table, lambda p: "     ." if np.isnan(p) else f"{p:+6.1f}")

    # -- 4. compare policies: V^pi by the Bellman expectation equation, and pushes for real
    pushes = np.round(np.arange(-0.30, 0.601, 0.01), 2)
    print(f"\nstarts rho0: at rest, pushed at {pushes[0]:+.2f} .. {pushes[-1]:+.2f} m/s (every 0.01). "
          f"J = the mean discounted return actually earned (rollouts); grid = the mean of V^pi "
          f"from the Bellman equation on the grid")
    print(f"{'policy':>8} {'rule':>22} {'J(pi)':>7} {'grid':>7} {'saved':>7}   pushes survived for 10 s "
          f"(m/s; the capture point allows {-W * -HEEL:+.3f} .. {W * TOE:+.3f})")
    rules = {"optimal": "argmax_a Q*(s,a)", "ankle": "p = %g x + %g v" % tuple(args.ankle_gain),
             "damped": "p = %g x + %g v" % tuple(args.damped_gain), "hold": "p = 0"}
    for name, pi in pis.items():
        Vp = V if name == "optimal" else evaluate(mdp, table_of(mdp, pi), args.gamma)
        stood, J, _ = rollout(pi, np.zeros_like(pushes), pushes, args.gamma, weights)
        ok = pushes[stood]
        saved = rollout(pi, mdp.X, mdp.V, args.gamma, weights)[0].mean()
        rng = f"{ok.min():+.2f} .. {ok.max():+.2f}" if len(ok) else "none"
        grid = mdp.value_at(Vp, np.zeros_like(pushes), pushes).mean()
        print(f"{name:>8} {rules[name]:>22} {J.mean():7.2f} {grid:7.2f} {saved:7.1%}   {rng}")

    # -- 5. one push, printed and replayed
    up1, ret1, path = rollout(pis[args.policy], [0.0], [args.push], args.gamma, weights)
    print(f"\n{args.policy}, push {args.push:+.2f} m/s: "
          + ("stands" if up1[0] else f"FALLS at {DT * len(path):.2f} s")
          + f"; discounted return {ret1[0]:.2f}")
    for k in range(0, min(len(path), 51), 5):
        x, v, p = path[k]
        print(f"   t {k * DT:4.2f} s  x {x:+.3f}  v {v:+.3f}  p {p:+.3f}  capture point {x + v / W:+.3f}")

    if args.no_viewer:
        return 0
    return replay(path, args.speed)


# == replay: the pendulum drawn over a foot, in MuJoCo's passive viewer ============================

SCENE = f"""
<mujoco>
  <visual><global offwidth="800" offheight="600"/></visual>
  <worldbody>
    <light pos="0 -1 2"/>
    <geom type="plane" size="1 1 0.01" rgba="0.8 0.8 0.8 1"/>
    <geom type="box" pos="{(HEEL + TOE) / 2} 0 0.01" size="{(TOE - HEEL) / 2} 0.05 0.01" rgba="0.3 0.3 0.3 1"/>
    <site name="ankle" pos="0 0 0.035" size="0.01" rgba="1 1 1 1"/>
  </worldbody>
</mujoco>
"""


def replay(traj, speed):
    import mujoco
    import soc4180
    if soc4180.is_colab():
        return 0
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    print("\nreplaying: blue ball = CoM, orange = centre of pressure, purple = capture point "
          "(close the window to stop)")

    def draw(viewer, x, p):
        scn = viewer.user_scn
        scn.ngeom = 0
        for pos, rgba, r in (((x, 0, H), (0.23, 0.49, 0.85, 1), 0.04),
                             ((p, 0, 0.025), (0.91, 0.45, 0.23, 1), 0.015)):
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([r, 0, 0]), np.array(pos, float), np.eye(3).ravel(),
                                np.array(rgba, float))
            scn.ngeom += 1
        mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, 0.006,
                             np.array([p, 0, 0.025]), np.array([x, 0, H]))
        scn.geoms[scn.ngeom].rgba[:] = (0.5, 0.5, 0.5, 1)
        scn.ngeom += 1

    with soc4180.launch_viewer(model, data, passive=True) as viewer:
        while viewer.is_running() and time.time() < deadline:
            for x, v, p in traj:
                if not viewer.is_running() or time.time() > deadline:
                    break
                with viewer.lock():
                    draw(viewer, x, p)
                    xi = x + v / W
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                        mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.015, 0, 0]),
                                        np.array([xi, 0, 0.005]), np.eye(3).ravel(),
                                        np.array([0.56, 0.27, 0.68, 1.0]))
                    viewer.user_scn.ngeom += 1
                viewer.sync()
                time.sleep(DT / speed)
            time.sleep(0.8)
    return 0


if __name__ == "__main__":
    sys.exit(main())
