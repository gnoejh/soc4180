"""Week 6 lab, on your laptop: what the robot can know about which way is down.

    uv run weeks/06-sensing/lab_imu.py

A window opens with the G1 standing in the crouch. Three arrows hang from the
torso IMU, each an estimate of "down" made from the IMU alone and drawn back in
the world, so RIGHT MEANS VERTICAL. A thin white arrow is the truth.

    orange   accelerometer only: gravity is wherever the specific force points
    cyan     gyroscope only: integrate the turning rate from a known start
    green    YOUR complementary filter, `my_filter` below (no arrow until written)

    W                      walk / stand (restarts)
    N                      inject a gyro bias and accelerometer noise (toggle)
    [ and ]                your filter's alpha: trust the gyro less / more
    R                      restart, all estimates re-aligned with the truth
    ENTER                  mean and final error of each estimate over the last 3 s
    double-click a body, then ctrl-drag    push the robot: the best experiment here
    left-drag / right-drag / wheel   orbit / pan / zoom
    Keys go to the MuJoCo window, not the terminal -- click the window first.
    If no key does anything in the window, press the same key in this terminal
    instead -- single keys, no enter. `q` stops it.

What to change, in order, and show the instructor:

1. Push the standing robot sideways with ctrl-drag and let it settle. The
   orange arrow swings wildly *during* the push and is right afterwards. Say
   why (what does an accelerometer measure?).
2. Press N. The cyan arrow starts to lean, slowly, and never comes back. Press
   ENTER twice a few seconds apart: its error only grows. Name the effect.
3. Write `my_filter`: predict with the gyro, correct toward the accelerometer,
       angle = alpha * (angle + omega * dt) + (1 - alpha) * angle_from_accel
   for roll and pitch. The green arrow should sit inside the white one while
   standing, stay there under a push, and not drift with N on.
4. Press W. The robot walks. Compare the three arrows and the ENTER numbers.
   Now sweep alpha with [ and ]: find the alpha that is worst, and say what
   it is trading.
5. The pelvis IMU has been ignored. Change SITE to "pelvis". Does anything
   improve? Explain either way.
"""

from __future__ import annotations

import math
import os
import time

import mujoco
import numpy as np

import soc4180
from soc4180.walking import GaitParams, WalkingController

SITE = "torso"          # or "pelvis": the G1 has one IMU at each
ALPHA0 = 0.995
GYRO_BIAS = 0.02        # rad/s on the roll axis when N is on
ACCEL_NOISE = 0.5       # m/s^2 standard deviation when N is on


def my_filter(roll: float, pitch: float, gyro, accel, dt: float, alpha: float):
    """One complementary-filter update. Return (roll, pitch), or None until written.

    `roll`, `pitch` are your previous estimates. `gyro` is the angular velocity
    (rad/s, body frame) and `accel` the specific force (m/s^2, body frame).
    `soc4180.tilt_from_accel(accel)` gives the accelerometer's own (roll, pitch).
    Predict by integrating gyro[0] into roll and gyro[1] into pitch, then pull
    each a fraction (1 - alpha) of the way toward the accelerometer's answer.
    """
    return None


# --- nothing below needs editing ------------------------------------------

def down_in_world(R, roll, pitch):
    """A (roll, pitch) estimate as the unit 'down' vector it implies, in the world."""
    up_body = np.array([-math.sin(pitch), math.cos(pitch) * math.sin(roll),
                        math.cos(pitch) * math.cos(roll)])
    return R @ (-up_body)


def true_tilt(R):
    g = R.T @ np.array([0.0, 0.0, -1.0])            # gravity in the sensor frame
    return math.atan2(-g[1], -g[2]), math.atan2(g[0], math.hypot(g[1], g[2]))


def arrow(geom, origin, direction, rgba, length=0.4, radius=0.008):
    quat = np.zeros(4); mat = np.zeros(9)
    mujoco.mju_quatZ2Vec(quat, np.asarray(direction, float))
    mujoco.mju_quat2Mat(mat, quat)
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW,
                        np.array([radius, radius, length], dtype=float),
                        np.asarray(origin, dtype=float), mat, np.asarray(rgba, dtype=float))


def main() -> int:
    if soc4180.is_colab():
        print("This lab needs a desktop window; run it on your laptop.")
        return 1

    model = soc4180.load_g1()
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"imu_in_{SITE}")
    dt = model.opt.timestep
    rng = np.random.default_rng(0)

    state = {"walk": False, "noise": False, "alpha": ALPHA0, "print": False,
             "controller": None, "est": {}, "err": {k: [] for k in ("accel", "gyro", "mine")}}

    def restart():
        c = WalkingController(model, GaitParams(n_steps=12))
        fresh = c.initial_data()
        mujoco.mj_resetData(model, data)
        data.qpos[:] = fresh.qpos
        mujoco.mj_forward(model, data)
        state["controller"] = c
        R = data.site_xmat[site_id].reshape(3, 3)
        r, p = true_tilt(R)
        state["est"] = {"gyro": [r, p], "mine": [r, p]}
        for v in state["err"].values():
            v.clear()
        print(f"\n[{'walking' if state['walk'] else 'standing'}]  noise {'ON' if state['noise'] else 'off'}"
              f"  alpha {state['alpha']:.4f}  IMU imu_in_{SITE}")

    def on_key(keycode):
        if keycode == 87:                                    # 'W'
            state["walk"] = not state["walk"]; restart()
        elif keycode == 78:                                  # 'N'
            state["noise"] = not state["noise"]
            print(f"  noise {'ON: gyro bias %.2f rad/s, accel sigma %.1f' % (GYRO_BIAS, ACCEL_NOISE) if state['noise'] else 'off'}")
        elif keycode == 91:                                  # '['
            state["alpha"] = 1 - min((1 - state["alpha"]) * 2, 0.5)
            print(f"  alpha {state['alpha']:.4f}")
        elif keycode == 93:                                  # ']'
            state["alpha"] = 1 - (1 - state["alpha"]) / 2
            print(f"  alpha {state['alpha']:.4f}")
        elif keycode == 82:                                  # 'R'
            restart()
        elif keycode in (257, 335):
            state["print"] = True

    def step():
        c = state["controller"]
        if state["walk"] and data.time < c.total_time:
            data.ctrl[:] = c.control(data.time)
        elif not state["walk"]:
            data.ctrl[:] = c.control(0.0)                    # hold the crouch
        mujoco.mj_step(model, data)
        gyro, accel = soc4180.read_imu(model, data, SITE)
        if state["noise"]:
            gyro = gyro + np.array([GYRO_BIAS, 0.0, 0.0])
            accel = accel + rng.normal(0.0, ACCEL_NOISE, 3)
        R = data.site_xmat[site_id].reshape(3, 3)
        truth = true_tilt(R)
        est = state["est"]
        acc = soc4180.tilt_from_accel(accel)
        est["gyro"][0] += gyro[0] * dt; est["gyro"][1] += gyro[1] * dt
        mine = my_filter(est["mine"][0], est["mine"][1], gyro, accel, dt, state["alpha"])
        if mine is not None:
            est["mine"] = [float(mine[0]), float(mine[1])]
        for k, v in (("accel", acc), ("gyro", est["gyro"]), ("mine", est["mine"] if mine is not None else None)):
            if v is not None:
                e = math.degrees(math.hypot(v[0] - truth[0], v[1] - truth[1]))
                state["err"][k].append(e)
                del state["err"][k][:-1500]                  # last 3 s
        return R, acc, mine is not None

    def report():
        print(f"  t = {data.time:5.2f} s   error over the last 3 s (deg):")
        for k, label in (("accel", "accelerometer only"), ("gyro", "gyroscope only"), ("mine", "your filter")):
            e = state["err"][k]
            if not e:
                print(f"    {label:20s} (my_filter not written)")
            else:
                print(f"    {label:20s} mean {np.mean(e):6.2f}   final {e[-1]:6.2f}")

    restart()
    print("\n\n".join(__doc__.split("\n\n")[2:4]))
    soc4180.terminal_keys(on_key)
    print("  [terminal] keys dead in the window? press them here instead; 'q' stops.",
          flush=True)
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    with soc4180.launch_viewer(model, data, passive=True, key_callback=on_key) as viewer:
        while viewer.is_running() and time.time() < deadline:
            wall = time.time()
            R, acc, have_mine = step()
            if state["print"]:
                state["print"] = False
                report()
            origin = data.site_xpos[site_id]
            with viewer.lock():
                scn = viewer.user_scn
                arrow(scn.geoms[0], origin, [0, 0, -1], (1, 1, 1, 0.9), length=0.45, radius=0.004)
                arrow(scn.geoms[1], origin, down_in_world(R, *acc), (1.0, 0.55, 0.1, 0.8))
                arrow(scn.geoms[2], origin, down_in_world(R, *state["est"]["gyro"]), (0.1, 0.8, 0.9, 0.8))
                n = 3
                if have_mine:
                    arrow(scn.geoms[3], origin, down_in_world(R, *state["est"]["mine"]), (0.1, 0.9, 0.2, 0.9))
                    n = 4
                scn.ngeom = n
            viewer.sync()
            lag = dt - (time.time() - wall)
            if lag > 0:
                time.sleep(lag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
