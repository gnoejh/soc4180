"""Week 2b lab, on your laptop: drive every part of the robot, by name.

    uv run weeks/02b-robot-as-code/lab_body.py

Week 2's lab posed one leg and checked your forward kinematics against MuJoCo.
This one is about the *other twenty-three joints*: which numbers are which body
part, and what each one actually moves.

A window opens with the G1 frozen in the first pose of POSES. Physics is never
stepped, so the robot holds poses it could never balance in.

    1 2 3 4 5     highlight a chain: left leg, right leg, waist, left arm, right arm
    0             highlight nothing
    SPACE / right next pose          left arrow   previous pose
    M             mirror the current pose left <-> right
    ENTER         print the pose as a set_pose(...) call you can paste
    R             back to `stand`
    Control sliders (F3)             drag any of the 29 joints directly
    left-drag / right-drag / wheel   orbit / pan / zoom
    Keys go to the MuJoCo window, not the terminal -- click the window first.
    On a Korean keyboard, letters do nothing while the IME is in Hangul mode;
    press the Han/Eng key to switch it back to English.

    If no key does anything in the window, press the same key in this terminal
    instead -- single keys, no enter, arrows included. That route needs no
    window focus, so it works when GLFW never sees the key. `q` stops it.

The highlighted chain is drawn as a string of yellow spheres, one per joint,
from the root of the limb to its tip. The white sphere is the landmark at the
end of that chain -- the foot site, or the wrist body -- and the printout tells
you how far it moved since the last pose.

The Control sliders normally set actuator *targets*, which only reach the joints
through physics. Nothing here is simulated, so this script wires each slider
straight onto its joint angle instead: drag `left_elbow_joint` and the arm bends.

What to change, in order, and show the instructor:

1. Press 1-5 and name each chain out loud *before* the highlight appears.
2. Drag one slider per chain. For each, say which landmarks moved before you
   look. The heat map in the deck is the answer key.
3. Add three poses of your own to POSES, by joint name. No slot indices.
4. Build a pose using only the left arm, press M, and explain which signs
   flipped and which did not.
5. Find a pose where the left hand is directly above the right foot. Press ENTER
   and paste the printed call into your notebook.
"""

from __future__ import annotations

import os
import time

import mujoco
import numpy as np

import soc4180
from soc4180 import bodies as B

# Poses are dictionaries of joint name -> angle in radians, exactly as you would
# pass them to soc4180.set_pose. A chain name takes one number per joint.
POSES: list[tuple[str, dict]] = [
    ("stand: the keyframe", {}),
    ("wave (left arm + waist)", dict(left_shoulder_pitch=-1.7, left_shoulder_roll=0.5,
                                     left_elbow=1.4, waist_yaw=0.25)),
    ("bow (waist pitch)", dict(waist_pitch=0.5, left_shoulder_pitch=0.5,
                               right_shoulder_pitch=0.5)),
    ("squat (both legs, as chains)", dict(left_leg=[-0.6, 0, 0, 1.2, -0.6, 0],
                                          right_leg=[-0.6, 0, 0, 1.2, -0.6, 0])),
    ("twist (waist yaw only)", dict(waist_yaw=1.2)),
    # add your own below
]

# The landmark at the end of each chain: what you watch when that chain moves.
TIPS = {
    "left_leg": ("site", "left_foot"),
    "right_leg": ("site", "right_foot"),
    "waist": ("site", "imu_in_torso"),
    "left_arm": ("body", "left_wrist_yaw_link"),
    "right_arm": ("body", "right_wrist_yaw_link"),
}

CHAIN_KEYS = {49: "left_leg", 50: "right_leg", 51: "waist",
              52: "left_arm", 53: "right_arm"}          # keys '1'..'5'


def landmark(model, data, chain: str) -> np.ndarray:
    """World position of the thing at the end of a chain."""
    kind, name = TIPS[chain]
    if kind == "site":
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        return data.site_xpos[sid].copy()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[bid].copy()


def as_call(model, data) -> str:
    """The current pose, written as the set_pose call that would reproduce it."""
    rest = model.key_qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")]
    moved = []
    for chain in B.CHAINS:
        for jname in B.CHAINS[chain]:
            slot = B.joint_index(model, jname)
            if abs(data.qpos[slot] - rest[slot]) > 1e-4:
                moved.append(f"{jname.replace('_joint', '')}={data.qpos[slot]:.2f}")
    if not moved:
        return "soc4180.set_pose(model, data)          # this is `stand`"
    return "soc4180.set_pose(model, data, " + ", ".join(moved) + ")"


def report(model, data, name: str, chain: str | None, previous) -> None:
    print(f"\n[{name}]")
    for c in B.CHAINS:
        here = landmark(model, data, c)
        shift = ("" if previous is None else
                 f"   {1000 * np.linalg.norm(here - previous[c]):8.1f} mm since the last pose")
        mark = ">" if c == chain else " "
        print(f"  {mark} {c:10s} {np.round(here, 4)}{shift}")
    print("  " + as_call(model, data))


def sphere(geom, pos, rgba, radius):
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0, 0], dtype=float),
                        np.asarray(pos, dtype=float),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=float))


def main() -> int:
    if soc4180.is_colab():
        print("This lab needs a desktop window; run it on your laptop.")
        print("The notebook covers the same ground with rendered figures.")
        return 1

    model = soc4180.load_g1()
    data = soc4180.keyframe_data(model, "stand")

    # every G1 actuator drives exactly one hinge; this is that hinge's qpos slot
    slider_qpos = model.jnt_qposadr[model.actuator_trnid[:, 0]]

    state = {"i": 0, "chain": None, "pose": True, "print": False, "mirror": False}

    def on_key(keycode):
        if keycode in (32, 262):                 # SPACE, right arrow
            state["i"] = (state["i"] + 1) % len(POSES)
            state["pose"] = True
        elif keycode == 263:                     # left arrow
            state["i"] = (state["i"] - 1) % len(POSES)
            state["pose"] = True
        elif keycode in CHAIN_KEYS:              # '1'..'5'
            state["chain"] = CHAIN_KEYS[keycode]
            state["print"] = True
        elif keycode == 48:                      # '0'
            state["chain"] = None
        elif keycode == 77:                      # 'M'
            state["mirror"] = True
        elif keycode == 82:                      # 'R'
            state["i"] = 0
            state["pose"] = True
        elif keycode in (257, 335):              # ENTER
            state["print"] = True

    def markers(viewer, data):
        """Yellow spheres along the highlighted chain, white at its tip."""
        with viewer.lock():
            scn = viewer.user_scn
            scn.ngeom = 0
            chain = state["chain"]
            if chain is None:
                return
            for k, bid in enumerate(B.chain_bodies(model, chain)):
                sphere(scn.geoms[k], data.xpos[bid], (1.0, 0.85, 0.1, 0.85), 0.032)
            n = len(B.CHAINS[chain])
            sphere(scn.geoms[n], landmark(model, data, chain), (1, 1, 1, 0.9), 0.045)
            scn.ngeom = n + 1

    print("\n\n".join(__doc__.split("\n\n")[3:5]))      # the window + key help
    previous = None
    # Second route for every command: type it in this terminal and press enter.
    # The viewer's own keys go through GLFW, which an IME or a stray keyboard
    # focus can swallow; typing needs no window focus at all.
    soc4180.terminal_keys(on_key)
    print("  [terminal] keys dead in the window? press them here instead "
          "-- single keys, no enter, arrows included; 'q' stops.", flush=True)
    # SOC4180_AUTOCLOSE=<seconds> closes the window by itself: the smoke test.
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    with soc4180.launch_viewer(model, data, passive=True, key_callback=on_key) as viewer:
        while viewer.is_running() and time.time() < deadline:
            if state["pose"] or state["mirror"]:
                if state["mirror"]:
                    state["mirror"] = False
                    name, angles = POSES[state["i"]]
                    flat = {}
                    for part, value in angles.items():
                        if part in B.CHAINS:
                            flat.update(dict(zip(
                                (j.replace("_joint", "") for j in B.CHAINS[part]),
                                np.asarray(value, float))))
                        else:
                            flat[part] = float(value)
                    name, angles = f"{name}  MIRRORED", soc4180.mirror(flat)
                else:
                    name, angles = POSES[state["i"]]
                state["pose"] = False
                soc4180.set_pose(model, data, **angles)
                data.ctrl[:] = data.qpos[slider_qpos]      # sliders show the pose
                report(model, data, name, state["chain"], previous)
                previous = {c: landmark(model, data, c) for c in B.CHAINS}
                markers(viewer, data)
            else:
                data.qpos[slider_qpos] = data.ctrl          # sliders ARE the angles
                mujoco.mj_forward(model, data)
                if state["print"]:
                    state["print"] = False
                    report(model, data, "sliders", state["chain"], previous)
                    previous = {c: landmark(model, data, c) for c in B.CHAINS}
                markers(viewer, data)
            # MuJoCo's viewer also treats digit keys 0-5 as 'toggle geom group'. The
            # robot's meshes are group 2, so a '2' pressed for this lab would hide it.
            viewer.opt.geomgroup[:3] = 1
            viewer.sync()
            time.sleep(0.02)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
