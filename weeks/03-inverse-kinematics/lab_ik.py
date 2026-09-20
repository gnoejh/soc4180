"""Week 3 lab, on your laptop: move a target, and make the leg reach it.

    uv run weeks/03-inverse-kinematics/lab_ik.py

A window opens with the G1 frozen in the week 4 crouch. Physics is never
stepped: this is inverse kinematics, geometry only, so the robot holds poses it
could never balance in and the pelvis never moves.

    arrows                 move the target forward/back (left/right) and up/down
    , and .                move the target sideways (y)
    [ and ]                damping lambda, ten times smaller / larger
    S                      swap the seed: straight leg (the `stand` singularity) <-> crouch
    R                      put the target back on the foot
    SPACE                  next entry of TARGETS
    ENTER                  print target, foot, residual and the six joint angles
    left-drag / right-drag / wheel   orbit / pan / zoom
    Keys go to the MuJoCo window, not the terminal -- click the window first.
    If no key does anything in the window, press the same key in this terminal
    instead -- single keys, no enter, arrows included. `q` stops it.

Three spheres. The RED one is the target you move. The GREEN one is the left
foot site, where the leg actually is. The big TRANSLUCENT one is the reach of
the leg -- thigh plus shin, measured from the model -- centred on the hip. When
the red sphere leaves it, no solver in the world can put the green one on it.

Until you write `dls_step` below, the green sphere never moves: the solver is
yours to write, not the package's. The script does everything around it --
measures the error, asks MuJoCo for the Jacobian, clamps to the joint limits,
and iterates a few times per frame so you watch it converge.

What to change, in order, and show the instructor:

1. Write `dls_step`: damped least squares, `dq = J^T (J J^T + lambda^2 I)^-1 e`.
   Move the target a few centimetres. The green sphere should follow it and the
   printed residual should fall below 1e-4 within a frame or two.
2. Press `[` until lambda is 1e-6, then `S` for the straight-leg seed. Press the
   down arrow. Explain what the printed |dq| does, and why the foot still does
   not come down. Press `S` again: same target, same solver, and it works.
3. Push the target outward until the red sphere leaves the translucent one.
   The residual stops shrinking. Press ENTER a few times and show that it is
   the same number every time: this is geometry, not a solver running out of
   iterations. Compare the printed hip distance to the printed reach.
4. Press `]` until lambda is 1.0 and move the target. The foot now creeps. Say
   what lambda is trading away, and find the largest lambda that still tracks
   a 5 cm move within a second.
5. Add a pose to TARGETS with the foot 10 cm forward and 5 cm up, and one you
   believe is unreachable. Press SPACE to cycle through them, and be right.
"""

from __future__ import annotations

import os
import time

import mujoco
import numpy as np

import soc4180
from soc4180 import kinematics as kin

# Offsets from the foot's starting position (metres, world frame: x forward,
# y left, z up). SPACE cycles through them. Add your own.
TARGETS = [
    ("start: on the foot",        [0.00, 0.00, 0.00]),
    ("5 cm up",                   [0.00, 0.00, 0.05]),
    ("10 cm forward",             [0.10, 0.00, 0.00]),
    ("forward and up",            [0.12, 0.00, 0.08]),
    # add your own below
]

SIDE = "left"
STEP = 0.01                  # metres per key press
ITERATIONS_PER_FRAME = 3


def dls_step(J: np.ndarray, err: np.ndarray, lam: float) -> np.ndarray | None:
    """One damped-least-squares step. Return dq (6,), or None until written.

    `J` is the 6 x 6 site Jacobian for the six leg joints (three position rows,
    three rotation rows) and `err` is the 6-vector [position error; rotation
    error] that would carry the foot onto the target. Solve

        dq = J^T (J J^T + lam^2 I)^-1 err

    with numpy: `J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), err)`.
    """
    return None


# --- nothing below needs editing ------------------------------------------

def leg_lengths(model, data):
    """Thigh and shin, measured between joint anchors (not |body_pos|)."""
    def anchor(body):
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        return data.xpos[b] + data.xmat[b].reshape(3, 3) @ model.jnt_pos[model.body_jntadr[b]]
    hip, knee, ankle = (anchor(f"{SIDE}_{n}_link") for n in ("hip_pitch", "knee", "ankle_pitch"))
    return hip, np.linalg.norm(knee - hip), np.linalg.norm(ankle - knee)


def sphere(geom, pos, rgba, radius):
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0, 0], dtype=float),
                        np.asarray(pos, dtype=float),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=float))


def main() -> int:
    if soc4180.is_colab():
        print("This lab needs a desktop window; run it on your laptop.")
        return 1

    model = soc4180.load_g1()
    controller = soc4180.WalkingController(model)
    data = controller.initial_data()                 # the crouch, pelvis at 0.74 m
    crouch = data.qpos.copy()
    straight = crouch.copy()
    for s in ("left", "right"):
        straight[kin.leg_qpos_indices(model, s)] = 0.0   # every leg joint zero: `stand`

    site = kin.foot_site_id(model, SIDE)
    qidx = kin.leg_qpos_indices(model, SIDE)
    didx = kin.leg_dof_indices(model, SIDE)
    slider_qpos = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{SIDE}_{j}")
              for j in kin.LEG_JOINTS]
    lo, hi = model.jnt_range[joints, 0], model.jnt_range[joints, 1]
    hip0, l1, l2 = leg_lengths(model, data)
    reach = l1 + l2

    foot0 = data.site_xpos[site].copy()
    flat = data.site_xmat[site].reshape(3, 3).copy()   # keep the foot level
    state = {"target": foot0.copy(), "lam": 1e-2, "seed": "crouch", "i": 0,
             "print": False, "moved": True}

    print(f"thigh {l1:.4f} m + shin {l2:.4f} m = reach {reach:.4f} m from the hip")
    print(f"foot site starts at {foot0.round(4)}")

    def on_key(keycode):
        t = state["target"]
        if keycode == 265:                                   # up arrow
            t[2] += STEP
        elif keycode == 264:                                 # down arrow
            t[2] -= STEP
        elif keycode == 262:                                 # right arrow: forward
            t[0] += STEP
        elif keycode == 263:                                 # left arrow: back
            t[0] -= STEP
        elif keycode == 44:                                  # ','  to the robot's right
            t[1] -= STEP
        elif keycode == 46:                                  # '.'  to the robot's left
            t[1] += STEP
        elif keycode == 91:                                  # '['
            state["lam"] = max(state["lam"] / 10, 1e-9)
        elif keycode == 93:                                  # ']'
            state["lam"] = min(state["lam"] * 10, 1e3)
        elif keycode == 83:                                  # 'S'
            state["seed"] = "straight" if state["seed"] == "crouch" else "crouch"
            data.qpos[:] = straight if state["seed"] == "straight" else crouch
            mujoco.mj_forward(model, data)
        elif keycode == 82:                                  # 'R'
            t[:] = data.site_xpos[site]
        elif keycode == 32:                                  # SPACE
            state["i"] = (state["i"] + 1) % len(TARGETS)
            name, off = TARGETS[state["i"]]
            t[:] = foot0 + np.asarray(off, float)
            print(f"\n[{name}]")
        elif keycode in (257, 335):                          # ENTER
            state["print"] = True
        else:
            return
        state["moved"] = True

    def solve_once():
        """One iteration of *your* solver, on the live pose."""
        err = kin.pose_error(data, site, state["target"], flat)
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, site)
        J = np.vstack([jp, jr])[:, didx]
        dq = dls_step(J, err, state["lam"])
        if dq is None:
            return err, None
        dq = np.asarray(dq, float).reshape(6)
        data.qpos[qidx] = np.clip(data.qpos[qidx] + dq, lo, hi)
        mujoco.mj_forward(model, data)
        return err, float(np.linalg.norm(dq))

    def report(err, dq_norm):
        foot = data.site_xpos[site]
        dist = np.linalg.norm(state["target"] - hip0)
        dq_text = "(dls_step not written)" if dq_norm is None else f"{dq_norm:.2e} rad"
        print(f"  target {state['target'].round(4)}  foot {foot.round(4)}")
        print(f"  residual {np.linalg.norm(err[:3]):.2e} m   |dq| {dq_text}"
              f"   lambda {state['lam']:.0e}   seed {state['seed']}")
        outside = "   <- OUTSIDE: unreachable" if dist > reach else ""
        print(f"  hip->target {dist:.4f} m of reach {reach:.4f} m{outside}")
        print(f"  angles {np.round(data.qpos[qidx], 3).tolist()}")

    print("\n\n".join(__doc__.split("\n\n")[2:4]))
    soc4180.terminal_keys(on_key)
    print("  [terminal] keys dead in the window? press them here instead; 'q' stops.",
          flush=True)
    # SOC4180_AUTOCLOSE=<seconds> closes the window by itself: the smoke test.
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    with soc4180.launch_viewer(model, data, passive=True, key_callback=on_key) as viewer:
        while viewer.is_running() and time.time() < deadline:
            err, dq_norm = None, None
            for _ in range(ITERATIONS_PER_FRAME):
                err, dq_norm = solve_once()
            data.ctrl[:] = data.qpos[slider_qpos]          # sliders show the solution
            if state["print"] or state["moved"]:
                state["print"] = state["moved"] = False
                report(err, dq_norm)
            with viewer.lock():
                scn = viewer.user_scn
                sphere(scn.geoms[0], hip0, (0.3, 0.5, 1.0, 0.12), reach)
                sphere(scn.geoms[1], state["target"], (0.9, 0.1, 0.1, 0.9), 0.02)
                sphere(scn.geoms[2], data.site_xpos[site], (0.1, 0.9, 0.2, 0.6), 0.03)
                scn.ngeom = 3
            viewer.sync()
            time.sleep(0.02)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
