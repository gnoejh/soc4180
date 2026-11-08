"""Tables learned from samples, measured: four worlds, one update rule, printed then replayed.

    uv run --extra rl weeks/07b-tabular/qlearn.py --task frozen
    uv run --extra rl weeks/07b-tabular/qlearn.py --task cliff
    uv run --extra rl weeks/07b-tabular/qlearn.py --task pendulum --see angle rate
    uv run --extra rl weeks/07b-tabular/qlearn.py --task balance --bins 11 21 41
    uv run --extra rl weeks/07b-tabular/qlearn.py --task frozen --gamma 0.5 --no-viewer

Week 7 SOLVED an MDP: P and R were arrays and value iteration iterated
Bellman's equation on them. Here the agent never sees P. It acts, is told
(s', r), and nudges one entry of a table:

    Q[s, a] += alpha * (target - Q[s, a])
    target   = r + gamma * max_a' Q[s', a']        Q-learning
    target   = r + gamma * Q[s', a']               SARSA (a' = the action it will take)

    --task frozen     FrozenLake 4 x 4, slippery. Value iteration (it may read P)
                      against Q-learning (it may not); the two tables side by side.
    --task cliff      CliffWalking 4 x 12. SARSA against Q-learning: what each
                      earns while exploring, and the path each one learned.
    --task pendulum   gymnasium's MuJoCo InvertedPendulum-v5, the state cut into
                      bins. The best checked table is kept, tested, and played
                      live -- ctrl + right-drag the pole to push it.
    --task balance    week 7's G1 balance (LIPM), learned from episodes, against
                      value iteration's J = 72.12; one row per table size.

    --episodes N  --alpha  --gamma  --epsilon  --seed   (defaults per task, printed)
    --see angle rate cart cartvel   what the pendulum's table sees (6 bins each)
    --bins 21 ...                   balance table sizes (bins x bins rows)
    --no-viewer   --speed 1

Measured with this script (defaults unless named):

 1. frozen: value iteration 571 sweeps, V*(start) 0.542, pi* reaches the
    goal 0.737. Q-learning (seed 0): success 0.093 / 0.158 / 0.270 / 0.646 /
    0.737 after 100 / 500 / 1000 / 2000 / 5000 episodes, and at 5000 the same
    action as pi* on every tile that can reach the goal (max |V - V*| 0.079).
    --gamma 0.5: pi* itself only 0.465; Q-learning 0.070.
 2. cliff: SARSA earns -25.1 per episode while exploring and falls in 11 of
    the last 200; its greedy path is 17 steps, along the top. Q-learning
    earns -49.5 and falls in 51; its greedy path is 13 steps, along the edge.
 3. pendulum --see angle rate (36 rows): checks 30, 99, ..., 1000 at 1750,
    then 170, 125, 163, 1000; the kept and the final table both balance 1000
    steps from 10 new starts (35 s). --see angle (6 rows): kept table 7-31
    steps. --see angle rate cart cartvel (1296 rows): 14-57.
 4. balance --bins 5 11 21 41 81: second-half J 19.1 / 51.1 / 65.5 / 66.1 /
    53.3 (hold 12.66, value iteration 72.12); about a minute per size.

Five numbered sections; read them with the lecture open.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

import soc4180
from soc4180 import tabular

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lab_tables import CLIFF_MOVES, CLIFF_ROWS, FROZEN_MOVES, FROZEN_ROWS, grid_scene  # noqa: E402

ARROWS = {"frozen": "<v>^", "cliff": "^>v<"}            # action number -> arrow character


def table_map(rows, Q, chars, terminal):
    """The greedy action of every tile, drawn with the world's own letters where nothing is learned."""
    out = []
    cols = len(rows[0])
    for r, line in enumerate(rows):
        cells = []
        for c, ch in enumerate(line):
            s = r * cols + c
            cells.append(ch if s in terminal or not Q[s].any() or ch in "HC" else chars[int(np.argmax(Q[s]))])
        out.append("    " + " ".join(cells))
    return "\n".join(out)


# == 1. FrozenLake: value iteration (P known) against Q-learning (samples only) =========================

def frozen(args):
    env = gym.make("FrozenLake-v1", is_slippery=True)
    P = env.unwrapped.P
    print("1. The slippery lake. P[s][a] lists (probability, next state, reward, done); for tile 0, 'right':")
    for p, s2, r, d in P[0][2]:
        print(f"     p = {p:.3f}  -> tile {s2:2d}  reward {r}  done {d}")
    V, Qstar, sweeps = tabular.value_iteration(P, 16, 4, args.gamma)
    star = tabular.run_greedy(env, Qstar, 2000, 10**6)[0].mean()
    print(f"\n   value iteration, gamma {args.gamma}: {sweeps} sweeps; V*(start) = {V[0]:.3f}; "
          f"pi* reaches the goal {star:.3f} of 2000 tries")
    print(table_map(FROZEN_ROWS, Qstar, ARROWS["frozen"], {5, 7, 11, 12, 15}))

    print(f"\n   Q-learning: alpha {args.alpha}, gamma {args.gamma}, epsilon 1 -> x0.9995 per episode (floor 0.05), "
          f"seed {args.seed}. No P: only env.step.")
    L = tabular.Learner(env, 16, 4, algo="q", alpha=args.alpha, gamma=args.gamma, epsilon=1.0, eps_decay=0.9995,
                        eps_min=0.05, seed=args.seed)
    print("   episodes   success (greedy, 2000 tries)   max |V - V*| on tiles that can reach the goal   same action as pi*")
    safe = V > 0
    for upto in sorted({100, 500, 1000, 2000, 5000, args.episodes}):
        if upto > args.episodes:
            continue
        L.train(upto - L.episodes)
        succ = tabular.run_greedy(env, L.Q, 2000, 10**6)[0].mean()
        gap = np.abs(L.Q.max(1) - V)[safe].max()
        same = (L.Q.argmax(1) == Qstar.argmax(1))[safe].mean()
        print(f"   {L.episodes:8d}   {succ:.3f}{'':27s}{gap:.3f}{'':45s}{same:.0%}")
    print(table_map(FROZEN_ROWS, L.Q, ARROWS["frozen"], {5, 7, 11, 12, 15}))
    return FROZEN_ROWS, FROZEN_MOVES, [("Q-learning", L)]


# == 2. CliffWalking: SARSA against Q-learning ===========================================================

def cliff(args):
    print(f"2. The cliff. -1 per step, -100 and back to the start for the cliff. epsilon {args.epsilon}, "
          f"alpha {args.alpha}, gamma {args.gamma}, {args.episodes} episodes, seed {args.seed}.")
    out = []
    for algo in ("sarsa", "q"):
        L = tabular.Learner(gym.make("CliffWalking-v1"), 48, 4, algo=algo, alpha=args.alpha, gamma=args.gamma,
                            epsilon=args.epsilon, seed=args.seed, max_steps=1000).train(args.episodes)
        r = np.array(L.returns)
        falls = int((r[-200:] <= -100).sum())
        ret, steps, _ = tabular.run_greedy(L.env, L.Q, 1, 0, max_steps=100)
        name = "SARSA" if algo == "sarsa" else "Q-learning"
        path = f"{int(steps[0])} steps, return {ret[0]:.0f}" if ret[0] > -100 else "never arrives"
        print(f"\n   {name:10s}  while exploring: mean return of the last 200 episodes {r[-200:].mean():6.1f}, "
              f"episodes with a fall {falls}\n{'':15s}greedy path: {path}")
        print(table_map(CLIFF_ROWS, L.Q, ARROWS["cliff"], {47}))
        out.append((name, L))
    return CLIFF_ROWS, CLIFF_MOVES, out


# == 3. The inverted pendulum: a continuous state, cut into bins ===========================================

SEE = {"rate": (3, np.linspace(-1.5, 1.5, 7)[1:-1]), "angle": (1, np.linspace(-0.2, 0.2, 7)[1:-1]),
       "cart": (0, np.linspace(-0.6, 0.6, 7)[1:-1]), "cartvel": (2, np.linspace(-1.0, 1.0, 7)[1:-1])}


def pendulum(args):
    see = args.see
    take, edges = [SEE[k][0] for k in see], [SEE[k][1] for k in see]
    L = tabular.pendulum_learner(edges, take, alpha=args.alpha, gamma=args.gamma, seed=args.seed)
    print(f"3. InvertedPendulum-v5 (MuJoCo, 25 decisions/s, forces -3 / 0 / +3 N, +1 per step, ends past 0.2 rad).")
    print(f"   the table sees: {', '.join(tabular.PENDULUM_OBS[k] for k in take)} -> {L.binner.n} rows x 3 actions")
    t0 = time.time()
    kb = tabular.KeepBest(L, args.episodes)
    print("   episode   greedy check (mean of 3 starts, max 1000)   mean length of the last 250 training episodes")
    while not kb.finished:
        kb.advance(kb.every)
        print(f"   {kb.done:7d}   {kb.curve[-1][1]:6.0f}{'':38s}{np.mean(L.lengths[-250:]):6.1f}")
    _, kept, _ = tabular.run_greedy(L.env, kb.best_Q, 10, 999_000, index=L.index, act=L.act)
    _, final, _ = tabular.run_greedy(L.env, L.Q, 10, 999_000, index=L.index, act=L.act)
    print(f"\n   {time.time() - t0:.0f} s. Kept table (from episode {kb.when}), 10 new starts: {' '.join(map(str, kept))}")
    print(f"   The FINAL table, same starts:                     {' '.join(map(str, final))}")
    if "angle" in see and "rate" in see and len(see) == 2:
        Qb = kb.best_Q.reshape(len(edges[0]) + 1, len(edges[1]) + 1, 3)
        print("\n   The kept table's push (- left, 0 none, + right), angle down the side, angular velocity across:")
        print("        " + "".join(f"{'<' + format(e, '+.1f'):>7s}" for e in edges[1]) + f"{'>' + format(edges[1][-1], '+.1f'):>7s}")
        lab = [f"<{e:+.2f}" for e in edges[0]] + [f">{edges[0][-1]:+.2f}"]
        for i in range(Qb.shape[0]):
            print(f"   {lab[i]:>6s}" + "".join(f"{'-0+'[int(np.argmax(Qb[i, j]))] if Qb[i, j].any() else '.':>7s}"
                                             for j in range(Qb.shape[1])))
    return L, kb.best_Q


# == 4. Week 7's G1 balance, from samples ===================================================================

def balance(args):
    m = tabular.BalanceMDP()
    hold = m.J(lambda x, v: np.zeros_like(x))
    print(f"4. Week 7's balance (LIPM, 50 Hz, 7 centre-of-pressure actions). From episodes only, "
          f"{args.episodes:,} each, seed {args.seed}.")
    print(f"   hold (p = 0): J = {hold:.2f}   value iteration's pi* (it knew P): J = {m.J_STAR}")
    print("   bins  rows    J at 8 checks along the way                         second half   seconds")
    last = None
    for b in args.bins:
        t0 = time.time()
        L = tabular.BalanceLearner(b, seed=args.seed)
        rounds = args.episodes // L.batch
        for k in range(1, rounds + 1):
            L.advance(1)
            if k % (rounds // 8) == 0 and len(L.curve) < 8:
                L.check()
        js = [j for _, j in L.curve]
        print(f"   {b:4d}  {b * b:5d}   {' '.join(f'{j:5.1f}' for j in js)}   {np.mean(js[-4:]):11.1f}   "
              f"{time.time() - t0:7.0f}")
        last = L
    return last


# == 5. replay ==============================================================================================

def replay_grid(rows, moves, learners, speed):
    import mujoco
    from soc4180.game import arrow
    model = mujoco.MjModel.from_xml_string(grid_scene(rows))
    data = mujoco.MjData(model)
    cols = len(rows[0])
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    print("\nreplaying each table's greedy policy (arrows = best action); close the window to stop")
    with soc4180.launch_viewer(model, data, passive=True) as viewer:
        viewer.cam.lookat[:] = ((cols - 1) * 0.25, -(len(rows) - 1) * 0.25, 0)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.25 * cols * 0.5 + 1.5, 90, -89
        while viewer.is_running() and time.time() < deadline:
            for name, L in learners:
                s, _ = L.env.reset(seed=int(time.time()) % 1000)
                for _ in range(60):
                    r, c = divmod(int(s), cols)
                    data.mocap_pos[0] = (c * 0.5, -r * 0.5, 0.12)
                    mujoco.mj_forward(model, data)
                    with viewer.lock():
                        viewer.user_scn.ngeom = 0
                        for t in range(L.Q.shape[0]):
                            if L.Q[t].any():
                                rr, cc = divmod(t, cols)
                                dx, dy = moves[int(np.argmax(L.Q[t]))]
                                arrow(viewer.user_scn, (cc * 0.5 - 0.12 * dx, -rr * 0.5 - 0.12 * dy, 0.03),
                                      (0.24 * dx, 0.24 * dy, 0), (0.1, 0.1, 0.1, 1), radius=0.03)
                    viewer.sync()
                    time.sleep(0.3 / speed)
                    if not viewer.is_running() or time.time() > deadline:
                        return
                    s, _, te, tr, _ = L.env.step(int(np.argmax(L.Q[s])))
                    if te or tr:
                        break
                time.sleep(0.6)


def replay_pendulum(L, Q, speed):
    env = L.env.unwrapped
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    print("\nplaying the kept table live: ctrl + right-drag the pole to push it; close the window to stop")
    obs, _ = L.env.reset(seed=7)
    with soc4180.launch_viewer(env.model, env.data, passive=True) as viewer:
        viewer.cam.lookat[:] = (0, 0, 0.3)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.6, 90, -10
        while viewer.is_running() and time.time() < deadline:
            obs, _, te, tr, _ = L.env.step(L.act(int(np.argmax(Q[L.index(obs)]))))
            if te or tr:
                print("  fell (or 1000 steps): reset")
                obs, _ = L.env.reset()
            viewer.sync()
            time.sleep(env.dt / speed)


def replay_balance(L, speed):
    import mujoco
    from lab_tables import BALANCE_SCENE
    from soc4180.game import sphere
    m = L.mdp
    model = mujoco.MjModel.from_xml_string(BALANCE_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    print(f"\nreplaying the {L.bins} x {L.bins} table catching pushes -0.2, +0.3, +0.45 m/s "
          "(blue = CoM, orange = centre of pressure); close the window to stop")
    with soc4180.launch_viewer(model, data, passive=True) as viewer:
        viewer.cam.lookat[:] = (0.05, 0, 0.35)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.6, 90, -5
        while viewer.is_running() and time.time() < deadline:
            for push in (-0.2, 0.3, 0.45):
                x, v = np.zeros(1), np.array([push])
                for _ in range(150):
                    p = L.policy(x, v)
                    x, v = m.step(x, v, p)
                    with viewer.lock():
                        viewer.user_scn.ngeom = 0
                        sphere(viewer.user_scn, (x[0], 0, m.H), (0.23, 0.49, 0.85, 1), 0.04)
                        sphere(viewer.user_scn, (p[0], 0, 0.025), (0.91, 0.45, 0.23, 1), 0.015)
                    viewer.sync()
                    time.sleep(m.DT / speed)
                    if m.fallen(x, v)[0] or not viewer.is_running() or time.time() > deadline:
                        break
                time.sleep(0.5)


DEFAULTS = {"frozen": dict(episodes=5000, alpha=0.05, gamma=0.99, epsilon=1.0),
            "cliff": dict(episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1),
            "pendulum": dict(episodes=3000, alpha=0.1, gamma=0.99, epsilon=1.0),
            "balance": dict(episodes=100_000, alpha=0.1, gamma=0.99, epsilon=1.0)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", choices=list(DEFAULTS), default="frozen")
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--gamma", type=float)
    ap.add_argument("--epsilon", type=float, help="cliff only (the others decay from 1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--see", nargs="+", choices=list(SEE), default=["angle", "rate"])
    ap.add_argument("--bins", type=int, nargs="+", default=[11, 21, 41])
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args(argv)
    for k, v in DEFAULTS[args.task].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    show = not args.no_viewer and not soc4180.is_colab()
    if args.task in ("frozen", "cliff"):
        rows, moves, learners = (frozen if args.task == "frozen" else cliff)(args)
        if show:
            replay_grid(rows, moves, learners, args.speed)
    elif args.task == "pendulum":
        L, Q = pendulum(args)
        if show:
            replay_pendulum(L, Q, args.speed)
    else:
        L = balance(args)
        if show:
            replay_balance(L, args.speed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
