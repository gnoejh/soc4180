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

What to change, in order, and show the instructor (each one is a change to
the XML, then a fresh run):

1. Run it as shipped. Press A: the servos die and the leg becomes a double
   pendulum. Press A again and it recovers TARGET. ENTER: the hip droops a
   few hundredths of a radian below TARGET. Say why a servo can never quite
   get there.
2. Add `<freejoint/>` to `upper_leg`. The compiler refuses -- read the error.
   Fix it with a parent body that carries the freejoint. Predict nq and nv
   before you run, then check the printout. What can the leg do now?
3. Sweep kp: 200, 100, 50, 20, 5 (edit both actuators). ENTER each time and
   write down the hip droop. Does it ever reach zero?
4. Change `timestep="0.002"` to `0.02`. Press A (servos off): nothing warns,
   the swing is just different. Press A again (servos on): ENTER shows the
   integrator warning and qpos back at zero. One of those is a silent
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
TARGET = [1.2, -0.8]


# --- nothing below needs editing ------------------------------------------

def main() -> int:
    if soc4180.is_colab():
        print("This lab needs a desktop window; run it on your laptop.")
        return 1

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

    act_q = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    limp_flag = int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    g_earth = model.opt.gravity.copy()
    state = {"limp": False, "moon": False, "print": False, "reset": True}

    def reset():
        mujoco.mj_resetData(model, data)
        data.qpos[act_q] = target
        data.ctrl[:] = target
        mujoco.mj_forward(model, data)
        print(f"\n[reset]  servos {'OFF' if state['limp'] else 'on'}, gravity {model.opt.gravity[2]:+.2f}")

    def on_key(keycode):
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
        elif keycode == 82:
            state["reset"] = True
        elif keycode in (257, 335):
            state["print"] = True

    def report():
        q = data.qpos[act_q]
        warn = int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
        print(f"  t = {data.time:5.2f} s  qpos = {np.round(data.qpos, 4).tolist()}")
        print(f"  actuated joints {np.round(q, 4).tolist()}   droop from ctrl "
              f"{np.round(q - data.ctrl, 4).tolist()} rad")
        print(f"  contacts {data.ncon}   integrator instability warnings (BADQACC): {warn}"
              + ("   <- the simulation reset itself; qpos is not a result" if warn else ""))

    print("\n\n".join(__doc__.split("\n\n")[2:3]))
    soc4180.terminal_keys(on_key)
    print("  [terminal] keys dead in the window? press them here instead; 'q' stops.",
          flush=True)
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    with soc4180.launch_viewer(model, data, passive=True, key_callback=on_key) as viewer:
        while viewer.is_running() and time.time() < deadline:
            wall = time.time()
            if state["reset"]:
                state["reset"] = False; reset()
            model.opt.disableflags = (model.opt.disableflags | limp_flag) if state["limp"] \
                else (model.opt.disableflags & ~limp_flag)
            mujoco.mj_step(model, data)
            if state["print"]:
                state["print"] = False; report()
            viewer.sync()
            lag = model.opt.timestep - (time.time() - wall)
            if lag > 0:
                time.sleep(lag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
