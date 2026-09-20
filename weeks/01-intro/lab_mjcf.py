"""Week 1 lab, on your laptop: write a robot, compile it, drive it, break it.

    uv run weeks/01-intro/lab_mjcf.py

The text in MJCF below is the whole robot. The script compiles it, prints
what the compiler made of it (nq, nv, nu, nbody), opens the viewer with
physics running, and keeps the servos on TARGET. Edit the XML, run again.

    A     actuation on / off (off = the servos are dead; the leg swings)
    0     ctrl = 0: command every joint to zero (still a command!)
    T     ctrl = TARGET again
    G     Moon gravity on / off
    R     reset to TARGET, at rest
    ENTER print the joint angles, their droop from TARGET, the contact count,
          and how many times the integrator has warned about instability
    double-click a body, then ctrl-drag    push it
    left-drag / right-drag / wheel   orbit / pan / zoom
    Keys go to the MuJoCo window, not the terminal -- click the window first.
    If no key does anything in the window, press the same key in this terminal
    instead -- single keys, no enter. `q` stops it.

Everything here is complete and explained: section 1 is the robot as text,
section 2 compiles it and prints the arrays the compiler made, section 3 is
the real-time loop. `mjcf_run.py` in this folder is the non-interactive
version: the same XML (or any file), kp / timestep / servos as flags, a
printed table of the droop over time, then a replay.

What to change, in order, and show the instructor (each one is a change to
the XML, then a fresh run; numbers measured with mjcf_run.py, same physics):

1. Run it as shipped. Press A: the servos die and the leg becomes a double
   pendulum. Press A again and it recovers TARGET. ENTER: the hip droops
   -0.080 rad below TARGET, the knee -0.007. Say why a servo can never quite
   get there (the torque that holds the leg up IS kp times the droop).
2. Add `<freejoint/>` to `upper_leg`. The compiler refuses -- read the error
   ("more than 6 dofs in body"). Fix it with a parent body that carries the
   freejoint. Predict nq and nv before you run (7 + 2 and 6 + 2), then check
   the printout. What can the leg do now?
3. Sweep kp: 200, 100, 50, 20, 5 (edit both actuators, or run
   `mjcf_run.py --kp 50`). Hip droop: -0.080, -0.151, -0.269, -0.501, -0.858.
   Does it ever reach zero? What law does the first three follow?
4. Change `timestep="0.002"` to `0.02`. Press A (servos off): nothing warns,
   the swing is just different. Press A again (servos on): ENTER shows one
   BADQACC warning and qpos back at exactly zero. One of those is a silent
   approximation and one is a silent failure -- say which is which.
5. Add a third body -- a foot -- below `lower_leg` with its own hinge. Confirm
   nq went up by one, and that the foot moves when the knee does.
6. Move the two capsules' rgba into a `<default>` class. The picture must not
   change. This is the mechanism that hides kp=500 in the G1.
"""

from __future__ import annotations

import os
import time

import mujoco
import numpy as np

import soc4180

# --- 1. the robot, as text ------------------------------------------------------
# <option> is a diff against MuJoCo's defaults (timestep 0.002 is in fact the
# default; it is written out so you can change it). Bodies nest to make the
# tree; a joint lives in the body it moves, and the body's `pos` is relative
# to its parent. <actuator> attaches a position servo to a joint: force =
# kp (ctrl - q) - kv qdot, with kv chosen by dampratio="1" (critical damping).
MJCF = """
<mujoco model="two_link_leg">
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="5 5 0.05" rgba="0.3 0.4 0.5 1"/>

    <body name="upper_leg" pos="0 0 1">
      <joint name="hip" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05" fromto="0 0 0  0 0 -0.4" rgba="0.7 0.7 0.7 1"/>

      <body name="lower_leg" pos="0 0 -0.4">
        <joint name="knee" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.04" fromto="0 0 0  0 0 -0.4" rgba="0.2 0.2 0.2 1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="hip"  joint="hip"  kp="200" dampratio="1"/>
    <position name="knee" joint="knee" kp="200" dampratio="1"/>
  </actuator>
</mujoco>
"""

# One target per actuator, in the order the <actuator> block lists them.
# A bent pose, on purpose: gravity fights it, so the servos have to work.
# (Zero is also gravity's rest position, so a servo commanded to zero
# "succeeds" at any gain and proves nothing.)
TARGET = [1.2, -0.8]


def main() -> int:
    if soc4180.is_colab():
        print("This lab needs a desktop window; run it on your laptop.")
        return 1

    # --- 2. compile: text in, arrays out ----------------------------------------
    # MjModel is the robot's constants; the compiler refuses inconsistent XML
    # with a precise message. MjData is the state: qpos (nq numbers), qvel (nv),
    # ctrl (nu), and everything MuJoCo derives from them.
    try:
        model = mujoco.MjModel.from_xml_string(MJCF)
    except ValueError as err:
        print("the compiler refused your XML:\n   ", str(err).strip())
        return 1
    data = mujoco.MjData(model)
    target = np.asarray(TARGET, float)
    if len(target) != model.nu:
        print(f"TARGET has {len(target)} numbers but the model has nu = {model.nu} actuators")
        return 1
    print(f"compiled: nq={model.nq}  nv={model.nv}  nu={model.nu}  nbody={model.nbody}  "
          f"ngeom={model.ngeom}  timestep={model.opt.timestep}")
    for j in range(model.njnt):
        kind = mujoco.mjtJoint(model.jnt_type[j]).name.replace("mjJNT_", "").lower()
        print(f"   joint {j}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or '(unnamed)':12s}"
              f" {kind:6s} qpos[{model.jnt_qposadr[j]}]  qvel[{model.jnt_dofadr[j]}]")
    for a in range(model.nu):
        print(f"   actuator {a}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)}"
              f"  kp = {model.actuator_gainprm[a, 0]:.0f}")

    act_q = model.jnt_qposadr[model.actuator_trnid[:, 0]]   # each actuator's joint -> its qpos slot
    limp_flag = int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)  # the bit that switches all servos off
    g_earth = model.opt.gravity.copy()
    state = {"limp": False, "moon": False, "print": False, "reset": True}

    def reset():
        mujoco.mj_resetData(model, data)                     # zero state, zero time
        data.qpos[act_q] = target                            # start at the target, at rest ...
        data.ctrl[:] = target                                # ... and command it
        mujoco.mj_forward(model, data)
        print(f"\n[reset]  servos {'OFF' if state['limp'] else 'on'}, gravity {model.opt.gravity[2]:+.2f}")

    def on_key(keycode):
        """Called with a GLFW keycode from the window, or from the terminal thread."""
        if keycode == 65:                                    # 'A'
            state["limp"] = not state["limp"]
            print(f"  actuation {'OFF: the servos are dead' if state['limp'] else 'on'}")
        elif keycode == 48:                                  # '0'
            data.ctrl[:] = 0.0; print("  ctrl = 0: every joint commanded to angle zero")
        elif keycode == 84:                                  # 'T'
            data.ctrl[:] = target; print(f"  ctrl = TARGET {target.tolist()}")
        elif keycode == 71:                                  # 'G'
            state["moon"] = not state["moon"]
            model.opt.gravity[:] = [0, 0, -1.62] if state["moon"] else g_earth
            print(f"  gravity {model.opt.gravity[2]:+.2f}")
        elif keycode == 82:                                  # 'R'
            state["reset"] = True
        elif keycode in (257, 335):                          # ENTER
            state["print"] = True

    def report():
        q = data.qpos[act_q]
        # BADQACC: the integrator saw an impossible acceleration and RESET the
        # state to zero. qpos = [0, 0] after it is not a result, it is a reboot.
        warn = int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
        print(f"  t = {data.time:5.2f} s  qpos = {np.round(data.qpos, 4).tolist()}")
        print(f"  actuated joints {np.round(q, 4).tolist()}   droop from ctrl "
              f"{np.round(q - data.ctrl, 4).tolist()} rad")
        print(f"  contacts {data.ncon}   integrator instability warnings (BADQACC): {warn}"
              + ("   <- the simulation reset itself; qpos is not a result" if warn else ""))

    print("\n\n".join(__doc__.split("\n\n")[2:3]))
    soc4180.terminal_keys(on_key)                            # the same callback, from the terminal
    print("  [terminal] keys dead in the window? press them here instead; 'q' stops.",
          flush=True)
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)

    # --- 3. the real-time loop: one mj_step per timestep, then sleep off the rest --------
    with soc4180.launch_viewer(model, data, passive=True, key_callback=on_key) as viewer:
        while viewer.is_running() and time.time() < deadline:
            wall = time.time()
            if state["reset"]:
                state["reset"] = False; reset()
            model.opt.disableflags = (model.opt.disableflags | limp_flag) if state["limp"] \
                else (model.opt.disableflags & ~limp_flag)
            mujoco.mj_step(model, data)                      # servo torques, gravity, integrate
            if state["print"]:
                state["print"] = False; report()
            viewer.sync()
            lag = model.opt.timestep - (time.time() - wall)
            if lag > 0:
                time.sleep(lag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
