"""Open the interactive MuJoCo viewer on a course robot.

Desktop only — Colab has no window to draw into.

    uv run scripts/view.py                  # the G1, standing
    uv run scripts/view.py --walk           # run the week 4 walker, live
    uv run scripts/view.py --limp           # motors off, watch it collapse
    uv run scripts/view.py --robot robotis_op3
    uv run scripts/view.py --list           # every humanoid available
    uv run scripts/view.py --static --keyframe stand         # orbit, no physics
    uv run scripts/view.py --pose=-0.35,0,0,0.70,-0.35,0     # place the left leg

``--pose`` takes the six leg angles in radians, in the week 2 order
(hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll), places them on
top of the ``stand`` keyframe, and freezes physics — so you can look at a pose
the robot could never balance in. While frozen, the viewer's Control sliders
(F3) move joints directly instead of setting servo targets. It prints where the foot site landed, which is
the number your own forward kinematics should reproduce. Write it as
``--pose=...`` with the equals sign: a value starting with a minus sign is
otherwise read as an option.

Mouse: left-drag orbits, right-drag pans, wheel zooms. Double-click a body to
select it, then ctrl-drag to push the robot around — the quickest way to see
whether a controller can take a disturbance.
"""

from __future__ import annotations

import argparse
import time

import mujoco

import soc4180


def posed_data(model, pose: str, side: str, keyframe: str) -> mujoco.MjData:
    """Place one leg at six angles on top of a keyframe, and report the foot."""
    from soc4180 import kinematics as kin

    angles = [float(v) for v in pose.split(",")]
    if len(angles) != 6:
        raise SystemExit(f"--pose wants six numbers, got {len(angles)}")
    data = soc4180.keyframe_data(model, keyframe)
    data.qpos[kin.leg_qpos_indices(model, side)] = angles
    mujoco.mj_forward(model, data)

    site = kin.foot_site_id(model, side)
    foot = data.site_xpos[site]
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    in_pelvis = data.xmat[pelvis].reshape(3, 3).T @ (foot - data.xpos[pelvis])
    names = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    print(f"{side} leg:", ", ".join(f"{n}={a:+.3f}" for n, a in zip(names, angles)))
    print(f"{side}_foot site, world frame : {foot.round(4)}")
    print(f"{side}_foot site, pelvis frame: {in_pelvis.round(4)}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", default="unitree_g1", help="Menagerie robot name")
    parser.add_argument("--entry", default="scene",
                        help="model entry point (scene, scene_mjx, g1, ...)")
    parser.add_argument("--keyframe", default=None,
                        help="named keyframe to start from, e.g. stand")
    parser.add_argument("--walk", action="store_true",
                        help="run the week 4 analytic walker (G1 only)")
    parser.add_argument("--limp", action="store_true",
                        help="disable actuation, so the robot collapses")
    parser.add_argument("--list", action="store_true",
                        help="list the humanoids and exit")
    parser.add_argument("--static", action="store_true",
                        help="never step physics; just draw the pose (week 2)")
    parser.add_argument("--pose", default=None, metavar="A,B,C,D,E,F",
                        help="six leg angles in radians: hip_pitch,hip_roll,"
                             "hip_yaw,knee,ankle_pitch,ankle_roll. Implies --static")
    parser.add_argument("--side", default="left", choices=("left", "right"),
                        help="which leg --pose applies to")
    args = parser.parse_args()

    if args.list:
        for name in soc4180.humanoids():
            print(" ", name)
        print("\nbipeds:", ", ".join(soc4180.by_category("biped")))
        return 0

    if soc4180.is_colab():
        print("The interactive viewer needs a desktop window; Colab has none.")
        return 1

    model = soc4180.load_robot(args.robot, args.entry)
    print(f"{args.robot}/{args.entry}: nq={model.nq} nv={model.nv} nu={model.nu}")

    controller = None
    if args.walk:
        if args.robot != "unitree_g1":
            parser.error("--walk is specific to the G1's analytic gait")
        controller = soc4180.WalkingController(model)
        data = controller.initial_data()
        print("running the week 4 walker; close the window to stop")
    elif args.pose is not None:
        data = posed_data(model, args.pose, args.side, args.keyframe or "stand")
        args.static = True
    elif args.keyframe:
        data = soc4180.keyframe_data(model, args.keyframe)
    else:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

    if args.static and controller is not None:
        parser.error("--static and --walk contradict each other")
    if args.static:
        print("static: physics is never stepped; this is kinematics, not balance")
        print("        the Control sliders (F3) set joint angles directly")
        # every course actuator drives one hinge; map its slider onto that hinge
        slider_qpos = model.jnt_qposadr[model.actuator_trnid[:, 0]]
        data.ctrl[:] = data.qpos[slider_qpos]

    # `actuation_disabled` is a context manager, so hold it open for the session
    import contextlib

    with contextlib.ExitStack() as stack:
        if args.limp:
            stack.enter_context(soc4180.actuation_disabled(model))
            print("actuation disabled: nothing is holding this robot up")

        viewer = stack.enter_context(
            soc4180.launch_viewer(model, data, passive=True)
        )
        step_dt = model.opt.timestep
        while viewer.is_running():
            wall = time.time()
            if args.static:
                data.qpos[slider_qpos] = data.ctrl  # sliders are the angles
                mujoco.mj_forward(model, data)     # place, never simulate
            else:
                if controller is not None:
                    data.ctrl[:] = controller.control(data.time)
                mujoco.mj_step(model, data)
            viewer.sync()
            # keep roughly real time rather than sprinting through the sim
            lag = step_dt - (time.time() - wall)
            if lag > 0:
                time.sleep(lag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
