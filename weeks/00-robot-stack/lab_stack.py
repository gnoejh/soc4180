"""Week 0 lab, on your laptop: the five layers, switched on one at a time.

    uv run weeks/00-robot-stack/lab_stack.py

A window opens with the G1 standing. Physics runs in real time. Each key
below leaves a different set of layers switched on, and the robot shows you
which ones were doing the work.

    1     Layers 1+2 only: physics and the model, every servo dead (a rag doll)
    0     ctrl = 0: Layer 4 running, commanding every joint to angle zero
    2     Layer 4, simplest form: the servos hold the `stand` keyframe
    3     Layer 4, in full: the week 4 walking controller
    4     Layer 4, yours: `my_controller` below (as shipped: hold `stand`, wave the left arm)
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

Everything here is complete and explained. `my_controller` in section 1 is a
working Layer 4 that you change; sections 2-4 are the loop around it, with the
MuJoCo call each step needs named where it happens. `stack.py` in the same
folder is the non-interactive version: same layers as flags, the numbers
printed, then a replay.

Experiments, in order, and what to show the instructor (numbers measured
with stack.py, the same physics):

1. Press 1, then R, then 2. Same physics, same model, same starting pose. A
   rag doll ends with its torso at 0.13 m; the held pose stays at 0.792 m. Say
   in one sentence what the difference is, and which layer it lives in.
2. Press R, then 0. The robot does NOT fall: torso 0.791 m, straight-legged.
   Explain why "zero command" is a command, and what you would have to do to
   get "no controller" (press 1).
3. Press G, then R, then 1. Which layers had to change for the Moon? ENTER:
   the contact forces now sum to 54 N, the Moon weight. Then press 3 with G
   still on and say what the walker does not know.
4. Press 4: `my_controller` waves the left arm while the rest keeps standing.
   Change the actuator index, the amplitude, the frequency; make the right arm
   wave too (actuator 22). Which layer keeps the legs standing for you?
5. Press ENTER while standing: the contact forces sum to 327.1 N, the
   robot's weight, on 8 contacts (four spheres under each foot). Press ENTER
   during the walk (3): 280-340 N on 4-5 contacts. Say why the sum is no
   longer constant.
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


# --- 1. Layer 4, yours -----------------------------------------------------------

def my_controller(model: mujoco.MjModel, data: mujoco.MjData, t: float) -> np.ndarray:
    """Return the 29 servo targets (radians) for time t. This is a Layer 4.

    Start from the keyframe's own targets, `model.key_ctrl[0]` -- the `stand`
    pose as ctrl -- change a few entries, and return the array. The servos
    (position actuators with kp = 500) do the rest: every joint you leave at
    its `stand` value is held there, which is why the robot keeps standing
    while one arm moves. Actuator names and indices:

        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)

    15 is the left shoulder pitch, 22 the right. As shipped: hold `stand` and
    swing the left arm as a sinusoid, amplitude 0.6 rad at 2 rad/s.
    """
    ctrl = model.key_ctrl[0].copy()               # every servo at its `stand` target ...
    ctrl[15] = -1.2 + 0.6 * math.sin(2.0 * t)     # ... except one, which we move
    return ctrl


# --- 2. two helpers: drawing a force, and reading the contacts --------------------

def arrow(geom, origin, direction, length, rgba, radius=0.006):
    """Fill one viewer geom with an arrow from `origin` along `direction`."""
    quat = np.zeros(4); mat = np.zeros(9)
    mujoco.mju_quatZ2Vec(quat, np.asarray(direction, float))   # rotation taking +z to `direction`
    mujoco.mju_quat2Mat(mat, quat)
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW,
                        np.array([radius, radius, length], dtype=float),
                        np.asarray(origin, dtype=float), mat, np.asarray(rgba, dtype=float))


def contact_forces(model, data):
    """(position, world-frame force) for every active contact.

    `data.ncon` contacts exist after a step. `mj_contactForce` returns the
    force in the contact's own frame (normal first); `contact.frame` rotates
    it into the world so the arrows point the right way.
    """
    out, wrench = [], np.zeros(6)
    for i in range(data.ncon):
        mujoco.mj_contactForce(model, data, i, wrench)
        c = data.contact[i]
        out.append((c.pos.copy(), c.frame.reshape(3, 3).T @ wrench[:3]))
    return out


# --- 3. the loop ---------------------------------------------------------------------

def main() -> int:
    if soc4180.is_colab():
        print("This lab needs a desktop window; run it on your laptop.")
        return 1

    # Layers 1 and 2: the world (gravity, timestep) and the robot, compiled from
    # MJCF into MjModel; MjData is the state, set to the `stand` keyframe.
    model = soc4180.load_g1()
    data = soc4180.keyframe_data(model, "stand")
    g_earth = model.opt.gravity.copy()
    weight = float(model.body_mass.sum() * 9.81)
    hold = soc4180.hold(model, "stand")                       # the simplest Layer 4
    limp_flag = int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)    # the bit that kills every servo
    MODES = {49: "limp", 48: "zero", 50: "hold", 51: "walk", 52: "mine"}
    LAYERS = {"limp": "layers 1+2: physics and model, no controller",
              "zero": "layer 4: ctrl = 0 -- every joint commanded to angle zero",
              "hold": "layer 4: servos hold `stand`",
              "walk": "layer 4: the week 4 walker (LIPM + IK)",
              "mine": "layer 4: my_controller"}
    state = {"mode": "hold", "walker": None, "moon": False, "paused": False,
             "print": False, "reset": False, "steps": 0, "t0": time.time()}

    def reset():
        mujoco.mj_resetDataKeyframe(model, data, 0)          # back to `stand`, at rest
        data.time = 0.0
        state["walker"] = None
        state["steps"], state["t0"] = 0, time.time()
        print(f"\n[{LAYERS[state['mode']]}]  gravity {model.opt.gravity[2]:+.2f}")

    def on_key(keycode):
        """Called with a GLFW keycode from the window, or from the terminal thread."""
        if keycode in MODES:
            state["mode"] = MODES[keycode]; state["reset"] = True
        elif keycode == 71:                                  # 'G'
            state["moon"] = not state["moon"]
            model.opt.gravity[:] = [0, 0, -1.62] if state["moon"] else g_earth
            print(f"  layer 1 changed: gravity {model.opt.gravity[2]:+.2f}. Nothing else was touched.")
        elif keycode == 82:                                  # 'R'
            state["reset"] = True
        elif keycode == 32:                                  # SPACE
            state["paused"] = not state["paused"]
        elif keycode in (257, 335):                          # ENTER
            state["print"] = True

    def step():
        """One control decision, then one physics step: the whole stack, once."""
        mode = state["mode"]
        # limp = actuation disabled: ctrl is ignored and the joints go free
        model.opt.disableflags = (model.opt.disableflags | limp_flag) if mode == "limp" \
            else (model.opt.disableflags & ~limp_flag)
        if mode == "zero":
            data.ctrl[:] = 0.0                               # a command, not an absence of one
        elif mode == "hold":
            hold(model, data)                                # ctrl = the keyframe's targets
        elif mode == "walk":
            if state["walker"] is None:
                state["walker"] = soc4180.WalkingController(model)
                data.qpos[:] = state["walker"].initial_data().qpos   # it starts from a crouch
                mujoco.mj_forward(model, data)
            w = state["walker"]
            data.ctrl[:] = w.control(min(data.time, w.total_time - 1e-3))
        elif mode == "mine":
            data.ctrl[:] = np.asarray(my_controller(model, data, data.time), float)
        mujoco.mj_step(model, data)                          # layers 1+2: one timestep (0.002 s)
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
    soc4180.terminal_keys(on_key)                            # the same callback, from the terminal
    print("  [terminal] keys dead in the window? press them here instead; 'q' stops.",
          flush=True)
    reset()
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)

    # -- 4. real time: one physics step per loop, then sleep off the remainder --------
    with soc4180.launch_viewer(model, data, passive=True, key_callback=on_key) as viewer:
        while viewer.is_running() and time.time() < deadline:
            wall = time.time()
            if state["reset"]:
                state["reset"] = False; reset()
            if not state["paused"]:
                step()
            if state["print"]:
                state["print"] = False; report()
            with viewer.lock():                              # the contact arrows, redrawn every step
                scn = viewer.user_scn; n = 0
                for pos, f in contact_forces(model, data)[:40]:
                    mag = float(np.linalg.norm(f))
                    if mag > 1.0:
                        arrow(scn.geoms[n], pos, f / mag, min(mag / weight, 1.0) * 0.6, (0.9, 0.3, 0.1, 0.8))
                        n += 1
                scn.ngeom = n
            viewer.sync()
            lag = model.opt.timestep - (time.time() - wall)  # 2 ms per step is real time
            if lag > 0:
                time.sleep(lag)
    model.opt.disableflags &= ~limp_flag
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
