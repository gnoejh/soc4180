"""Week 0 lab, on your laptop: the five layers, switched on one at a time.

    uv run weeks/00-robot-stack/lab_stack.py

A window opens with the G1 standing. Physics runs in real time. Each key
below leaves a different set of layers switched on, and the robot shows you
which ones were doing the work.

    1     Layers 1+2 only: physics and the model, every servo dead (a rag doll)
    0     ctrl = 0: Layer 4 running, commanding every joint to angle zero
    2     Layer 4, simplest form: the servos hold the `stand` keyframe
    3     Layer 4, in full: the week 4 walking controller
    4     Layer 4, yours: `my_controller` below (holds `stand` until written)
    G     change Layer 1: Moon gravity on/off, nothing else touched
    R     reset to `stand`          SPACE   pause
    ENTER print the loop: physics steps per second achieved, torso height,
          number of contacts, and the sum of the contact forces against the
          robot's weight
    double-click a body, then ctrl-drag    push the robot
    left-drag / right-drag / wheel   orbit / pan / zoom
    Keys go to the MuJoCo window, not the terminal -- click the window first.
    If no key does anything in the window, press the same key in this terminal
    instead -- single keys, no enter. `q` stops it.

Drawn on the floor: one arrow per contact, as long as the force is large.
That is Layer 1 made visible -- the floor pushing back.

What to change, in order, and show the instructor:

1. Press 1, then R, then 2. Same physics, same model, same starting pose.
   Say in one sentence what the difference is, and which layer it lives in.
2. Press R, then 0. The robot does NOT fall. Explain why "zero command" is a
   command, and what you would have to do to get "no controller" (press 1).
3. Press G, then R, then 1. Which layers had to change for the Moon? Then
   press 3 with G still on and say what the walker does not know.
4. Write `my_controller`: hold `stand` and wave the left arm (the shoulder
   pitch actuator, a sinusoid). Press 4. The rest of the robot must keep
   standing while the arm moves -- which layer is doing that for you?
5. Press ENTER while standing: the contact forces sum to the robot's weight.
   Press ENTER during the walk (3). Say why the sum is no longer constant.
6. Push the standing robot (2) and the walking one (3) with ctrl-drag.
   Neither reacts to the push. Name the layer that is missing.
"""

from __future__ import annotations

import math
import os
import time

import mujoco
import numpy as np

import soc4180


def my_controller(model: mujoco.MjModel, data: mujoco.MjData, t: float) -> np.ndarray | None:
    """Your Layer 4: return the 29 servo targets (radians), or None to hold `stand`.

    Start from the keyframe's own targets, `model.key_ctrl[0].copy()`, change
    a few, and return the array. Actuator names and indices:
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    The left shoulder pitch is actuator 15; try
        ctrl[15] = -1.2 + 0.6 * math.sin(2.0 * t)
    """
    return None


# --- nothing below needs editing ------------------------------------------

def arrow(geom, origin, direction, length, rgba, radius=0.006):
    quat = np.zeros(4); mat = np.zeros(9)
    mujoco.mju_quatZ2Vec(quat, np.asarray(direction, float))
    mujoco.mju_quat2Mat(mat, quat)
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW,
                        np.array([radius, radius, length], dtype=float),
                        np.asarray(origin, dtype=float), mat, np.asarray(rgba, dtype=float))


def contact_forces(model, data):
    """(position, world-frame force) for every active contact."""
    out, wrench = [], np.zeros(6)
    for i in range(data.ncon):
        mujoco.mj_contactForce(model, data, i, wrench)
        c = data.contact[i]
        out.append((c.pos.copy(), c.frame.reshape(3, 3).T @ wrench[:3]))
    return out


def main() -> int:
    if soc4180.is_colab():
        print("This lab needs a desktop window; run it on your laptop.")
        return 1

    model = soc4180.load_g1()
    data = soc4180.keyframe_data(model, "stand")
    g_earth = model.opt.gravity.copy()
    weight = float(model.body_mass.sum() * 9.81)
    hold = soc4180.hold(model, "stand")
    limp_flag = int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    MODES = {49: "limp", 48: "zero", 50: "hold", 51: "walk", 52: "mine"}
    LAYERS = {"limp": "layers 1+2: physics and model, no controller",
              "zero": "layer 4: ctrl = 0 -- every joint commanded to angle zero",
              "hold": "layer 4: servos hold `stand`",
              "walk": "layer 4: the week 4 walker (LIPM + IK)",
              "mine": "layer 4: my_controller"}
    state = {"mode": "hold", "walker": None, "moon": False, "paused": False,
             "print": False, "reset": False, "steps": 0, "t0": time.time()}

    def reset():
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.time = 0.0
        state["walker"] = None
        state["steps"], state["t0"] = 0, time.time()
        print(f"\n[{LAYERS[state['mode']]}]  gravity {model.opt.gravity[2]:+.2f}")

    def on_key(keycode):
        if keycode in MODES:
            state["mode"] = MODES[keycode]; state["reset"] = True
        elif keycode == 71:                                  # 'G'
            state["moon"] = not state["moon"]
            model.opt.gravity[:] = [0, 0, -1.62] if state["moon"] else g_earth
            print(f"  layer 1 changed: gravity {model.opt.gravity[2]:+.2f}. Nothing else was touched.")
        elif keycode == 82:
            state["reset"] = True
        elif keycode == 32:
            state["paused"] = not state["paused"]
        elif keycode in (257, 335):
            state["print"] = True

    def step():
        mode = state["mode"]
        model.opt.disableflags = (model.opt.disableflags | limp_flag) if mode == "limp" \
            else (model.opt.disableflags & ~limp_flag)
        if mode == "zero":
            data.ctrl[:] = 0.0
        elif mode == "hold":
            hold(model, data)
        elif mode == "walk":
            if state["walker"] is None:
                state["walker"] = soc4180.WalkingController(model)
                data.qpos[:] = state["walker"].initial_data().qpos
                mujoco.mj_forward(model, data)
            w = state["walker"]
            data.ctrl[:] = w.control(min(data.time, w.total_time - 1e-3))
        elif mode == "mine":
            c = my_controller(model, data, data.time)
            data.ctrl[:] = model.key_ctrl[0] if c is None else np.asarray(c, float)
        mujoco.mj_step(model, data)
        state["steps"] += 1

    def report():
        forces = contact_forces(model, data)
        total = sum(f[2] for _, f in forces)
        rate = state["steps"] / max(time.time() - state["t0"], 1e-6)
        print(f"  t = {data.time:5.2f} s   {rate:6.0f} physics steps/s achieved (real time needs 500)"
              f"   torso z = {data.qpos[2]:.3f} m")
        print(f"  contacts {data.ncon:2d}   sum of vertical contact forces {total:6.1f} N"
              f"   weight m g = {weight:.1f} N")

    print("\n\n".join(__doc__.split("\n\n")[2:4]))
    soc4180.terminal_keys(on_key)
    print("  [terminal] keys dead in the window? press them here instead; 'q' stops.",
          flush=True)
    reset()
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    with soc4180.launch_viewer(model, data, passive=True, key_callback=on_key) as viewer:
        while viewer.is_running() and time.time() < deadline:
            wall = time.time()
            if state["reset"]:
                state["reset"] = False; reset()
            if not state["paused"]:
                step()
            if state["print"]:
                state["print"] = False; report()
            with viewer.lock():
                scn = viewer.user_scn; n = 0
                for pos, f in contact_forces(model, data)[:40]:
                    mag = float(np.linalg.norm(f))
                    if mag > 1.0:
                        arrow(scn.geoms[n], pos, f / mag, min(mag / weight, 1.0) * 0.6, (0.9, 0.3, 0.1, 0.8))
                        n += 1
                scn.ngeom = n
            viewer.sync()
            lag = model.opt.timestep - (time.time() - wall)
            if lag > 0:
                time.sleep(lag)
    model.opt.disableflags &= ~limp_flag
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
