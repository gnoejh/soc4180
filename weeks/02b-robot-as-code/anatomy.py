"""The robot as data: its five chains, the qpos map, what each joint moves -- printed, then shown.

    uv run weeks/02b-robot-as-code/anatomy.py
    uv run weeks/02b-robot-as-code/anatomy.py --nudge left_arm
    uv run weeks/02b-robot-as-code/anatomy.py --nudge waist --no-viewer
    uv run weeks/02b-robot-as-code/anatomy.py --pose left_shoulder_pitch=-1.7 left_elbow=1.4 waist_yaw=0.25
    uv run weeks/02b-robot-as-code/anatomy.py --pose left_shoulder_roll=0.8 left_elbow=1.2 --mirror

With no flags: the five chains, and for every one of the 29 joints its qpos
slot, its qvel slot (one less -- the floating base spends 7 numbers on
position and 6 on velocity) and its range. --nudge CHAIN moves each joint of
that chain by 0.1 rad from `stand`, one at a time, and prints how far the five
landmarks (two feet, the torso IMU, two wrists) moved: one row of the deck's
heat map, measured live. --pose places named joints (a chain name takes one
number per joint) and --mirror reflects the pose left <-> right. The simulator
then shows the pose, the nudged chain as yellow spheres and its landmark in
white.

Five numbered steps; read them with the week 2b slides open.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mujoco
import numpy as np

import soc4180
from soc4180 import bodies as B

# What to watch at the end of each chain. There are no hand sites on the G1
# (nsite = 4: two feet, two IMUs), so a hand is the wrist body's origin.
LANDMARKS = {
    "left foot": ("site", "left_foot"), "right foot": ("site", "right_foot"),
    "torso IMU": ("site", "imu_in_torso"),
    "left wrist": ("body", "left_wrist_yaw_link"), "right wrist": ("body", "right_wrist_yaw_link"),
}


def landmark_positions(model, data):
    out = {}
    for name, (kind, obj) in LANDMARKS.items():
        if kind == "site":
            out[name] = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, obj)].copy()
        else:
            out[name] = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj)].copy()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nudge", choices=tuple(B.CHAINS), help="move each joint of this chain by --amount")
    ap.add_argument("--amount", type=float, default=0.1)
    ap.add_argument("--pose", nargs="*", metavar="JOINT=RAD", help="named joints, or CHAIN=a,b,c,...")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--hold", type=float, default=6.0)
    args = ap.parse_args(argv)

    # -- 1. the robot, and its state at `stand` --------------------------------------------
    model = soc4180.load_g1()
    data = soc4180.keyframe_data(model, "stand")
    mujoco.mj_forward(model, data)
    print(f"nq = {model.nq} (7 for the floating pelvis + 29 joints), nv = {model.nv} (6 + 29), "
          f"nu = {model.nu} servos, nsite = {model.nsite}")

    # -- 2. the five chains: every joint's slots and range -------------------------------------
    # B.CHAINS is the layout written down: joint names per chain, root to tip,
    # exactly as they sit in qpos. joint_index/dof_index look the slots up.
    print("\nchain        joint                         qpos  qvel   range (rad)")
    for chain, joints in B.CHAINS.items():
        for j in joints:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            lo, hi = model.jnt_range[jid]
            print(f"{chain:12} {j:29} {B.joint_index(model, j):4d} {B.dof_index(model, j):5d}   [{lo:+.2f}, {hi:+.2f}]")
    print("qpos index - qvel index = 1 for every joint: the same number addresses the neighbour in the other array.")

    highlight = None
    # -- 3. nudge one chain, one joint at a time, and watch the landmarks ------------------------
    if args.nudge:
        base = landmark_positions(model, data)
        print(f"\n{args.nudge}: each joint +{args.amount} rad from `stand`; landmark motion in mm")
        print(f"{'joint':29} " + " ".join(f"{n:>12}" for n in LANDMARKS))
        for j in B.CHAINS[args.nudge]:
            short = j.replace("_joint", "")
            B.set_pose(model, data, **{short: args.amount})    # reset to `stand`, move one joint, mj_forward
            moved = landmark_positions(model, data)
            print(f"{j:29} " + " ".join(f"{1000 * np.linalg.norm(moved[n] - base[n]):12.1f}" for n in LANDMARKS))
        print("blank-looking cells are the tree: a leg joint moves one foot and nothing else; a waist "
              "joint moves both hands and neither foot. A landmark on the joint's own axis moves 0.0.")
        B.set_pose(model, data, **{B.CHAINS[args.nudge][0].replace("_joint", ""): args.amount})
        highlight = args.nudge

    # -- 4. a named pose, optionally mirrored ----------------------------------------------------
    if args.pose:
        pose = {}
        for item in args.pose:
            name, _, value = item.partition("=")
            pose[name] = [float(v) for v in value.split(",")] if name in B.CHAINS else float(value)
        if args.mirror:
            pose = B.mirror(pose)
            print(f"\nmirrored: {pose}   (roll and yaw flip sign, pitch does not)")
        before = landmark_positions(model, data)
        B.set_pose(model, data, **pose)
        after = landmark_positions(model, data)
        print(f"\npose {pose}")
        for n in LANDMARKS:
            print(f"   {n:12} {after[n].round(3)}   moved {1000 * np.linalg.norm(after[n] - before[n]):6.1f} mm")
        highlight = B.chain_of(model, next(iter(pose))) if pose else None

    if args.no_viewer or soc4180.is_colab():
        return 0

    # -- 5. show it: the chain as yellow spheres, its landmark in white ---------------------------------
    def sphere(geom, p, rgba, r):
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0], float),
                            np.asarray(p, float), np.eye(3).flatten(), np.asarray(rgba, float))

    deadline = min(time.time() + args.hold, time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12))
    with soc4180.launch_viewer(model, data, passive=True) as viewer:
        with viewer.lock():
            scn = viewer.user_scn; n = 0
            if highlight:
                for bid in B.chain_bodies(model, highlight):
                    sphere(scn.geoms[n], data.xpos[bid], (1.0, 0.85, 0.1, 0.8), 0.025); n += 1
                tip = {"left_leg": "left foot", "right_leg": "right foot", "waist": "torso IMU",
                       "left_arm": "left wrist", "right_arm": "right wrist"}[highlight]
                sphere(scn.geoms[n], landmark_positions(model, data)[tip], (1, 1, 1, 0.9), 0.03); n += 1
            scn.ngeom = n
        while viewer.is_running() and time.time() < deadline:
            viewer.sync(); time.sleep(0.05)
    return 0


if __name__ == "__main__":
    sys.exit(main())
