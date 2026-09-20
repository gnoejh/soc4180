"""Compile an MJCF, print what the compiler made of it, simulate it, then watch it.

    uv run weeks/01-intro/mjcf_run.py
    uv run weeks/01-intro/mjcf_run.py --kp 20 --seconds 3
    uv run weeks/01-intro/mjcf_run.py --timestep 0.02 --no-viewer
    uv run weeks/01-intro/mjcf_run.py --servos off --target 1.2 -0.8
    uv run weeks/01-intro/mjcf_run.py --xml my_robot.xml --target 0.5 0.5 0

The robot is the two-link leg from the week 1 slides, embedded below, or any
MJCF file you pass with --xml. The script compiles it, prints every joint's
qpos/qvel slot and every actuator's gain, holds the servos on --target for
--seconds, printing the joint angles and their droop from the target every
--report seconds, counts the integrator's instability warnings, and then
opens the simulator and replays the motion.

    --xml FILE          an MJCF file instead of the embedded leg
    --target 1.2 -0.8   one servo target per actuator, radians (default: the bent pose)
    --kp 200            overwrite every position actuator's gain (damping rescaled to match)
    --timestep 0.002    overwrite the integrator timestep
    --servos on|off     off = actuation disabled: the leg is a double pendulum
    --gravity 9.81      1.62 for the Moon
    --seconds 3  --report 0.5  --no-viewer  --speed 1

Six numbered steps; read them with the week 1 slides on MJCF and the
simulation loop open.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

import soc4180

# The whole robot. Every tag is discussed on the week 1 slides: <option> is a
# diff against MuJoCo's defaults, bodies nest to make the tree, a joint lives
# in the body it moves, and <actuator> attaches a position servo to a joint.
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--xml", type=Path)
    ap.add_argument("--target", nargs="+", type=float)
    ap.add_argument("--kp", type=float)
    ap.add_argument("--timestep", type=float)
    ap.add_argument("--servos", choices=("on", "off"), default="on")
    ap.add_argument("--gravity", type=float, default=9.81)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--report", type=float, default=0.5)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(argv)

    # -- 1. compile: text in, MjModel out --------------------------------------------
    # The compiler turns the XML tree into flat arrays (nq numbers of position,
    # nv of velocity, nu of control) and refuses anything inconsistent. Its
    # error messages are precise; read them.
    xml = args.xml.read_text(encoding="utf-8") if args.xml else MJCF
    try:
        model = mujoco.MjModel.from_xml_string(xml)
    except ValueError as err:
        print("the compiler refused the XML:\n   ", str(err).strip())
        return 1
    if args.timestep:
        model.opt.timestep = args.timestep                     # <option timestep=...>, after the fact
    model.opt.gravity[:] = [0, 0, -args.gravity]
    if args.kp is not None:
        # A <position> actuator is force = kp * (ctrl - q) - kv * qdot, stored as
        # gainprm[0] = kp, biasprm[1] = -kp, biasprm[2] = -kv. dampratio="1" made
        # kv = 2 sqrt(kp m), so keep the ratio: scale kv by sqrt(kp_new / kp_old).
        for a in range(model.nu):
            old = model.actuator_gainprm[a, 0]
            model.actuator_gainprm[a, 0] = args.kp
            model.actuator_biasprm[a, 1] = -args.kp
            model.actuator_biasprm[a, 2] *= np.sqrt(args.kp / old) if old > 0 else 1.0
    data = mujoco.MjData(model)

    # -- 2. what the compiler made of it ----------------------------------------------
    print(f"compiled: nq={model.nq}  nv={model.nv}  nu={model.nu}  nbody={model.nbody}  "
          f"ngeom={model.ngeom}  timestep={model.opt.timestep}  gravity {args.gravity:g}")
    for j in range(model.njnt):
        kind = mujoco.mjtJoint(model.jnt_type[j]).name.replace("mjJNT_", "").lower()
        print(f"   joint {j}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or '(unnamed)':12s}"
              f" {kind:6s} qpos[{model.jnt_qposadr[j]}]  qvel[{model.jnt_dofadr[j]}]")
    for a in range(model.nu):
        print(f"   actuator {a}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)}"
              f"  kp = {model.actuator_gainprm[a, 0]:.0f}  kv = {-model.actuator_biasprm[a, 2]:.1f}")

    # -- 3. the target, and the starting state ------------------------------------------
    target = np.asarray(args.target if args.target is not None else [1.2, -0.8][:model.nu], float)
    if len(target) != model.nu:
        print(f"--target has {len(target)} numbers but the model has nu = {model.nu} actuators")
        return 1
    act_q = model.jnt_qposadr[model.actuator_trnid[:, 0]]      # each actuator's joint -> its qpos slot
    data.qpos[act_q] = target                                  # start AT the target, at rest ...
    data.ctrl[:] = target                                      # ... and command it
    if args.servos == "off":
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    mujoco.mj_forward(model, data)
    print(f"\nservos {args.servos}, target {target.tolist()}")

    # -- 4. the loop: mj_step, and a row every --report seconds ---------------------------
    steps = int(round(args.seconds / model.opt.timestep))
    every = max(1, int(round(args.report / model.opt.timestep)))
    frame_every = max(1, int(round(0.02 / model.opt.timestep)))
    frames = []
    print(f"{'t (s)':>6}  {'actuated joint angles (rad)':>32}  {'droop from target (rad)':>28}  BADQACC")
    for k in range(steps + 1):
        if k % frame_every == 0:
            frames.append(data.qpos.copy())
        if k % every == 0:
            q = data.qpos[act_q]
            warn = int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
            print(f"{data.time:6.2f}  {str(np.round(q, 4).tolist()):>32}  "
                  f"{str(np.round(q - target, 4).tolist()):>28}  {warn}")
        if k == steps:
            break
        mujoco.mj_step(model, data)

    # -- 5. summary: droop is the servo's steady-state error, BADQACC is a reset --------------
    q = data.qpos[act_q]
    warn = int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
    print(f"\nafter {args.seconds:g} s: angles {np.round(q, 4).tolist()}, droop "
          f"{np.round(q - target, 4).tolist()} rad, contacts {data.ncon}, "
          f"integrator warnings {warn}"
          + ("   <- the simulation RESET itself; the angles are not a result" if warn else ""))

    model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    if args.no_viewer or soc4180.is_colab():
        return 0

    # -- 6. replay in the simulator -------------------------------------------------------
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    print(f"\nreplaying {len(frames)} poses at {args.speed:g}x (close the window to stop)")
    with soc4180.launch_viewer(model, data, passive=True) as viewer:
        while viewer.is_running() and time.time() < deadline:
            for qpos in frames:
                if not viewer.is_running() or time.time() > deadline:
                    break
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.02 / args.speed)
            time.sleep(0.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
