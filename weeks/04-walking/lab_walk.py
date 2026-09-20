"""Week 4 lab, on your laptop: the analytic walker, live, and your own LIPM.

    uv run weeks/04-walking/lab_walk.py

A window opens with the G1 in the crouch. Physics runs in real time and the
week 4 controller drives it: one linear ODE, a footstep plan, and week 3's IK.

    1 2 3 ...              switch to that entry of GAITS and restart
    R                      restart the current gait
    SPACE                  pause / resume
    ENTER                  print distance, pelvis height, ZMP range so far
    G                      toggle Moon gravity (-1.62) without touching the controller
    F                      toggle ice: floor friction 0.2 instead of 1.0
    double-click a body, then ctrl-drag    push the robot (the viewer's own feature)
    left-drag / right-drag / wheel   orbit / pan / zoom
    Keys go to the MuJoCo window, not the terminal -- click the window first.
    If no key does anything in the window, press the same key in this terminal
    instead -- single keys, no enter. `q` stops it.

Drawn on the floor:

    grey boxes      the footstep plan -- where each foot is supposed to land
    white dots      the controller's commanded centre-of-mass path for the
                    current step, from the package's LIPM
    blue dots       the same path from YOUR `predict_com` below. Right means the
                    blue dots sit inside the white ones. Until you write it
                    there are no blue dots.
    red sphere      the pelvis, projected onto the floor
    yellow sphere   the measured ZMP -- the centre of pressure under the feet

Watch the yellow sphere. The theory says it can never leave the foot it is
standing on. Watch how far it actually swings sideways.

What to change, in order, and show the instructor:

1. Write `predict_com`: the LIPM closed form
       x(t) = p + (x0 - p) cosh(wt) + (v0 / w) sinh(wt).
   The blue dots must land inside the white ones on every step. Get the sign
   of the sinh term wrong and they run away backwards -- try it, on purpose.
2. Press 2 (no double support). Count the steps before it falls, and say which
   way it went and why that way.
3. Press 3 (rushed). Compare the step time to 1/omega, printed at start-up.
   Bring the step time down in GAITS until it fails, and note the value.
4. Press G. Predict the failure from omega = sqrt(g / z_c) BEFORE you look.
5. Push the robot (ctrl-drag) gently while it walks. It never reacts. Say what
   information a controller would need to react, and where it would get it
   (that is week 6).
6. Add a gait of your own to GAITS that walks further than the default in the
   same time. Prove it with ENTER.
"""

from __future__ import annotations

import math
import os
import time

import mujoco
import numpy as np

import soc4180
from soc4180 import GaitParams

# name, then the gait. Keys 1-9 select. Add your own.
GAITS = [
    ("default (tuned by sweep)",      GaitParams()),
    ("no double support",             GaitParams(double_support=0.0)),
    ("rushed: step_time 0.45 s",      GaitParams(step_time=0.45)),
    ("overstride: step_length 0.22",  GaitParams(step_length=0.22)),
    ("high steps: step_height 0.16",  GaitParams(step_height=0.16)),
    # add your own below
]


def predict_com(x0: float, v0: float, zmp: float, t: float, omega: float) -> float | None:
    """Where the LIPM puts the centre of mass after `t` seconds, or None until written.

    Released at position `x0` with velocity `v0`, with the zero moment point
    held at `zmp`, and omega = sqrt(g / z_c). The equation of motion is
    x'' = omega^2 (x - zmp) and its solution is a cosh/sinh pair. Use
    `math.cosh` and `math.sinh`.
    """
    return None


# --- nothing below needs editing ------------------------------------------

def sphere(geom, pos, rgba, radius):
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0, 0], dtype=float),
                        np.asarray(pos, dtype=float),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=float))


def box(geom, pos, half, rgba):
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX,
                        np.asarray(half, dtype=float), np.asarray(pos, dtype=float),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=float))


class Walk:
    """One run of one gait, in place: the viewer keeps watching the same MjData."""

    def __init__(self, model, data, name, params):
        self.name = name
        self.data = data
        self.controller = soc4180.WalkingController(model, params)
        fresh = self.controller.initial_data()
        mujoco.mj_resetData(model, data)
        data.qpos[:] = fresh.qpos
        data.ctrl[:] = self.controller.control(0.0)
        mujoco.mj_forward(model, data)
        self.zmp_y = []
        self.fell = False
        print(f"\n[{name}]  step_time {params.step_time} s, step_length {params.step_length} m,"
              f" double support {params.double_support:.0%}")
        print(f"  omega = {self.controller.lipm.omega:.2f} rad/s, 1/omega = "
              f"{1 / self.controller.lipm.omega:.3f} s, {len(self.controller.plan)} steps")

    def step(self, model):
        d = self.data
        if d.time >= self.controller.total_time:
            return
        d.ctrl[:] = self.controller.control(d.time)
        mujoco.mj_step(model, d)
        _, zy = self.controller.zmp(d)
        if d.time > self.controller.params.settle_time and not math.isnan(zy):
            self.zmp_y.append(zy)
        if d.qpos[2] < 0.5 and not self.fell:
            self.fell = True
            print(f"  FELL at t = {d.time:.2f} s after {d.qpos[0]:+.2f} m")

    def report(self):
        d = self.data
        span = (f"[{min(self.zmp_y):+.3f}, {max(self.zmp_y):+.3f}] m" if self.zmp_y else "n/a")
        print(f"  t = {d.time:5.2f} s  travelled {d.qpos[0]:+.3f} m  pelvis z {d.qpos[2]:.3f} m"
              f"  ZMP y {span}  feet at +-{self.controller.params.stance_width:.3f}"
              f"  {'FELL' if self.fell else 'upright'}")

    def com_dots(self):
        """Commanded (white) and predicted (blue) centre-of-mass path for this step."""
        c, p = self.controller, self.controller.params
        walk_t = self.data.time - p.settle_time
        if walk_t < 0:
            return [], []
        k = min(int(walk_t / p.step_time), len(c.plan) - 1)
        x0, vx, y0, vy, zx, zy = c._segments[k]       # this step's boundary values
        w = c.lipm.omega
        ref, mine = [], []
        for tau in np.linspace(0, p.step_time, 9):
            ref.append((c.lipm.evolve(x0, vx, zx, tau)[0], c.lipm.evolve(y0, vy, zy, tau)[0]))
            px, py = predict_com(x0, vx, zx, tau, w), predict_com(y0, vy, zy, tau, w)
            if px is not None and py is not None:
                mine.append((px, py))
        return ref, mine


def main() -> int:
    if soc4180.is_colab():
        print("This lab needs a desktop window; run it on your laptop.")
        return 1

    model = soc4180.load_g1()
    data = mujoco.MjData(model)
    g_earth = model.opt.gravity.copy()
    friction = model.geom_friction[0, 0]
    state = {"i": 0, "walk": Walk(model, data, *GAITS[0]), "paused": False,
             "moon": False, "ice": False, "print": False}

    def on_key(keycode):
        if 49 <= keycode <= 57 and keycode - 49 < len(GAITS):        # '1'..'9'
            state["i"] = keycode - 49
            state["walk"] = Walk(model, data, *GAITS[state["i"]])
        elif keycode == 82:                                          # 'R'
            state["walk"] = Walk(model, data, *GAITS[state["i"]])
        elif keycode == 32:                                          # SPACE
            state["paused"] = not state["paused"]
        elif keycode == 71:                                          # 'G'
            state["moon"] = not state["moon"]
            model.opt.gravity[:] = [0, 0, -1.62] if state["moon"] else g_earth
            print(f"  gravity now {model.opt.gravity[2]:+.2f} (controller unchanged)")
        elif keycode == 70:                                          # 'F'
            state["ice"] = not state["ice"]
            model.geom_friction[0, 0] = 0.2 if state["ice"] else friction
            print(f"  floor friction now {model.geom_friction[0, 0]:.1f}")
        elif keycode in (257, 335):                                  # ENTER
            state["print"] = True

    def draw(viewer, walk):
        d, c = walk.data, walk.controller
        with viewer.lock():
            scn = viewer.user_scn
            n = 0
            for _, _, _, swing_to in c.plan:
                box(scn.geoms[n], [*swing_to, 0.002], [0.085, 0.03, 0.002], (0.5, 0.5, 0.5, 0.35))
                n += 1
            ref, mine = walk.com_dots()
            for x, y in ref:
                sphere(scn.geoms[n], [x, y, 0.01], (1, 1, 1, 0.9), 0.012)
                n += 1
            for x, y in mine:
                sphere(scn.geoms[n], [x, y, 0.01], (0.2, 0.4, 1.0, 0.9), 0.008)
                n += 1
            sphere(scn.geoms[n], [d.qpos[0], d.qpos[1], 0.01], (0.9, 0.1, 0.1, 0.9), 0.02)
            n += 1
            zx, zy = c.zmp(d)
            if not math.isnan(zx):
                sphere(scn.geoms[n], [zx, zy, 0.01], (1.0, 0.85, 0.1, 0.9), 0.02)
                n += 1
            scn.ngeom = n

    print("\n\n".join(__doc__.split("\n\n")[2:6]))
    soc4180.terminal_keys(on_key)
    print("  [terminal] keys dead in the window? press them here instead; 'q' stops.",
          flush=True)
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    with soc4180.launch_viewer(model, data, passive=True, key_callback=on_key) as viewer:
        while viewer.is_running() and time.time() < deadline:
            wall = time.time()
            walk = state["walk"]
            if not state["paused"]:
                walk.step(model)
            if state["print"]:
                state["print"] = False
                walk.report()
            draw(viewer, walk)
            viewer.sync()
            lag = model.opt.timestep - (time.time() - wall)
            if lag > 0:
                time.sleep(lag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
