"""Three ways to know which way is down, from the IMU alone -- measured, then shown.

    uv run weeks/06-sensing/imu.py
    uv run weeks/06-sensing/imu.py --walk --noise
    uv run weeks/06-sensing/imu.py --walk --noise --alpha 0.9985
    uv run weeks/06-sensing/imu.py --walk --noise --sweep --no-viewer
    uv run weeks/06-sensing/imu.py --site pelvis --walk

The G1 stands in the crouch or walks the week 4 gait. Every physics step, the
IMU is read (gyroscope + accelerometer, body frame) and three estimates of the
torso's tilt are updated: the accelerometer alone, the gyroscope alone, and a
complementary filter written by hand. Each is scored against the truth
(`site_xmat`, which a real robot never has). The script prints the three
errors every second and their mean and final values, --sweep repeats the run
for a range of alphas, and then the simulator replays the motion with four
arrows hanging from the IMU: white truth, ORANGE accelerometer, CYAN gyro,
GREEN the filter. Right means vertical.

    --walk           walk instead of standing      --seconds 6
    --noise          gyro bias 0.02 rad/s on roll, accelerometer noise sigma 0.5 m/s^2
    --alpha 0.995    the filter's gyro trust       --sweep   try 12 alphas, print a table
    --site torso|pelvis                            --no-viewer   --speed 1

Six numbered steps; read them with the week 6 slides open.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import mujoco
import numpy as np

import soc4180
from soc4180.walking import GaitParams, WalkingController

GYRO_BIAS = 0.02        # rad/s, on the roll axis, when --noise
ACCEL_NOISE = 0.5       # m/s^2 standard deviation, when --noise


# ---------------------------------------------------------------------------
# The three estimators. Each turns IMU readings into (roll, pitch) in radians.
# ---------------------------------------------------------------------------

def tilt_from_accel(accel):
    """Accelerometer alone: gravity is wherever the specific force points.

    At rest the accelerometer reads +g upward in the body frame, so the
    direction of its reading IS "up". Any other acceleration -- a push, a
    step -- is added to it and misread as tilt. No drift, but no patience.
    """
    ax, ay, az = accel
    return math.atan2(ay, az), math.atan2(-ax, math.hypot(ay, az))


def integrate_gyro(roll, pitch, gyro, dt):
    """Gyroscope alone: add up the turning rate from a known start.

    Perfect for fast motion, but a constant bias b adds b*t of error for
    ever: the estimate walks away and nothing brings it back.
    """
    return roll + gyro[0] * dt, pitch + gyro[1] * dt


def complementary(roll, pitch, gyro, accel, dt, alpha):
    """Predict with the gyro, correct toward the accelerometer.

        angle = alpha * (angle + omega dt) + (1 - alpha) * angle_from_accel

    alpha near 1 trusts the gyro (follows fast motion, inherits its drift);
    lower trusts gravity (no drift, but every acceleration leaks in). It is a
    low-pass on the accelerometer and a high-pass on the gyro that sum to one,
    with time constant tau = -dt / ln(alpha): 0.4 s at alpha = 0.995 and 500 Hz.
    """
    acc_roll, acc_pitch = tilt_from_accel(accel)
    return (alpha * (roll + gyro[0] * dt) + (1 - alpha) * acc_roll,
            alpha * (pitch + gyro[1] * dt) + (1 - alpha) * acc_pitch)


def true_tilt(R):
    """(roll, pitch) of the sensor from its world rotation -- the answer key."""
    g = R.T @ np.array([0.0, 0.0, -1.0])
    return math.atan2(-g[1], -g[2]), math.atan2(g[0], math.hypot(g[1], g[2]))


def down_in_world(R, roll, pitch):
    """The 'down' a (roll, pitch) estimate implies, drawn back in the world."""
    up_body = np.array([-math.sin(pitch), math.cos(pitch) * math.sin(roll),
                        math.cos(pitch) * math.cos(roll)])
    return R @ (-up_body)


def simulate(model, site_id, site, walk, noise, alpha, seconds, rows=True, record=False):
    """Run once; return (errors per estimator in degrees, frames) ."""
    c = WalkingController(model, GaitParams(n_steps=12))
    data = c.initial_data()
    dt = model.opt.timestep
    rng = np.random.default_rng(0)
    R = data.site_xmat[site_id].reshape(3, 3)
    r0, p0 = true_tilt(R)
    est = {"gyro": (r0, p0), "filter": (r0, p0)}          # start aligned with the truth
    err = {"accel": [], "gyro": [], "filter": []}
    frames = []
    if rows:
        print(f"\n{'t (s)':>6} {'accel err':>10} {'gyro err':>9} {'filter err':>11}   (degrees from the truth)")
    for k in range(int(seconds / dt)):
        data.ctrl[:] = c.control(min(data.time, c.total_time - 1e-3) if walk else 0.0)
        mujoco.mj_step(model, data)
        gyro, accel = soc4180.read_imu(model, data, site)      # sensordata, in the body frame
        if noise:
            gyro = gyro + np.array([GYRO_BIAS, 0.0, 0.0])
            accel = accel + rng.normal(0.0, ACCEL_NOISE, 3)
        R = data.site_xmat[site_id].reshape(3, 3)
        truth = true_tilt(R)
        acc = tilt_from_accel(accel)
        est["gyro"] = integrate_gyro(*est["gyro"], gyro, dt)
        est["filter"] = complementary(*est["filter"], gyro, accel, dt, alpha)
        for name, v in (("accel", acc), ("gyro", est["gyro"]), ("filter", est["filter"])):
            err[name].append(math.degrees(math.hypot(v[0] - truth[0], v[1] - truth[1])))
        if record and k % 10 == 0:
            frames.append((data.qpos.copy(), acc, est["gyro"], est["filter"]))
        if rows and (k + 1) % int(1.0 / dt) == 0:
            print(f"{data.time:6.2f} {err['accel'][-1]:10.2f} {err['gyro'][-1]:9.2f} {err['filter'][-1]:11.2f}")
    return err, frames


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--walk", action="store_true")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.995)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--site", choices=("torso", "pelvis"), default="torso")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(argv)

    # -- 1. the robot and its IMU ------------------------------------------------------
    # nsensor = 4 is two IMUs: a gyro + accelerometer pair at imu_in_torso and
    # another at imu_in_pelvis. read_imu picks the pair out of data.sensordata.
    model = soc4180.load_g1()
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"imu_in_{args.site}")
    print(f"IMU imu_in_{args.site}: {'walking' if args.walk else 'standing'} for {args.seconds:g} s, "
          f"noise {'ON (gyro bias %.2f rad/s, accel sigma %.1f)' % (GYRO_BIAS, ACCEL_NOISE) if args.noise else 'off'}, "
          f"alpha {args.alpha} (tau = {-model.opt.timestep / math.log(args.alpha):.2f} s)")

    # -- 2. and 3. the three estimators, run against the truth --------------------------------
    err, frames = simulate(model, site_id, args.site, args.walk, args.noise, args.alpha,
                           args.seconds, rows=True, record=True)

    # -- 4. summary ---------------------------------------------------------------------------
    print("\nerror over the run (degrees):        mean    final")
    for name, label in (("accel", "accelerometer only"), ("gyro", "gyroscope only"), ("filter", "complementary filter")):
        print(f"   {label:22s} {np.mean(err[name]):8.2f} {err[name][-1]:8.2f}")

    # -- 5. the alpha sweep: what the filter trades ----------------------------------------------
    if args.sweep:
        print(f"\nsweep, same run, filter error in degrees:   {'alpha':>7} {'tau (s)':>8} {'mean':>7} {'final':>7}")
        best_mean, best_final = (None, 1e9), (None, 1e9)
        for alpha in (0.9, 0.95, 0.98, 0.99, 0.995, 0.997, 0.998, 0.9985, 0.999, 0.9995, 0.9999, 1.0):
            e, _ = simulate(model, site_id, args.site, args.walk, args.noise, alpha, args.seconds, rows=False)
            m, f = np.mean(e["filter"]), e["filter"][-1]
            tau = -model.opt.timestep / math.log(alpha) if alpha < 1 else float("inf")
            print(f"{'':44}{alpha:7.4f} {tau:8.2f} {m:7.2f} {f:7.2f}")
            if m < best_mean[1]: best_mean = (alpha, m)
            if f < best_final[1]: best_final = (alpha, f)
        print(f"   smallest mean error at alpha = {best_mean[0]}, smallest final error at alpha = {best_final[0]}; "
              f"alpha = 1 is the bare gyro, alpha = 0.9 nearly the bare accelerometer")

    if args.no_viewer or soc4180.is_colab():
        return 0

    # -- 6. replay with the four arrows -----------------------------------------------------------
    def arrow(geom, origin, direction, rgba, length=0.4, radius=0.008):
        quat = np.zeros(4); mat = np.zeros(9)
        mujoco.mju_quatZ2Vec(quat, np.asarray(direction, float)); mujoco.mju_quat2Mat(mat, quat)
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.array([radius, radius, length], float),
                            np.asarray(origin, float), mat, np.asarray(rgba, float))

    data = mujoco.MjData(model)
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    print(f"\nreplaying {len(frames)} poses at {args.speed:g}x: white truth, orange accel, cyan gyro, green filter")
    with soc4180.launch_viewer(model, data, passive=True) as viewer:
        while viewer.is_running() and time.time() < deadline:
            for q, acc, gy, fi in frames:
                if not viewer.is_running() or time.time() > deadline:
                    break
                data.qpos[:] = q; mujoco.mj_forward(model, data)
                R = data.site_xmat[site_id].reshape(3, 3); o = data.site_xpos[site_id]
                with viewer.lock():
                    scn = viewer.user_scn
                    arrow(scn.geoms[0], o, [0, 0, -1], (1, 1, 1, 0.9), length=0.45, radius=0.004)
                    arrow(scn.geoms[1], o, down_in_world(R, *acc), (1.0, 0.55, 0.1, 0.8))
                    arrow(scn.geoms[2], o, down_in_world(R, *gy), (0.1, 0.8, 0.9, 0.8))
                    arrow(scn.geoms[3], o, down_in_world(R, *fi), (0.1, 0.9, 0.2, 0.9))
                    scn.ngeom = 4
                viewer.sync(); time.sleep(0.02 / args.speed)
            time.sleep(0.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
