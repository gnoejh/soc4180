"""Open the interactive MuJoCo viewer on a course robot.

Desktop only — Colab has no window to draw into.

    uv run scripts/view.py                  # the G1, standing
    uv run scripts/view.py --walk           # run the week 4 walker, live
    uv run scripts/view.py --limp           # motors off, watch it collapse
    uv run scripts/view.py --robot robotis_op3
    uv run scripts/view.py --list           # every humanoid available

Mouse: left-drag orbits, right-drag pans, wheel zooms. Double-click a body to
select it, then ctrl-drag to push the robot around — the quickest way to see
whether a controller can take a disturbance.
"""

from __future__ import annotations

import argparse
import time

import mujoco

import soc4180


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
    elif args.keyframe:
        data = soc4180.keyframe_data(model, args.keyframe)
    else:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

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
