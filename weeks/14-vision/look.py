"""Learn to see a ball: render, train a CNN, test it on a shifted world, then turn toward the ball -- printed, then replayed.

    uv run weeks/14-vision/look.py                           # needs torch: uv sync --extra rl
    uv run weeks/14-vision/look.py --randomise colour
    uv run weeks/14-vision/look.py --train 500 --epochs 10
    uv run weeks/14-vision/look.py --servo-az 40 --gain 1.5
    uv run weeks/14-vision/look.py --no-viewer

A camera is added to the G1's head, a red ball floats somewhere in front of
it, and a small convolutional network learns to say where: two angles,
azimuth (left +) and elevation (up +), from 64 x 64 pixels. The labels come
free -- the simulator knows where it put the ball. Then the network is tested
on pictures it was not trained for (a blue ball, a dim room), and finally it
closes a loop: every 0.1 s the head camera takes a picture, the network
estimates the ball's azimuth, and the waist turns by that much times --gain.

    --train 4000 --test 500 --epochs 10 --res 64 --seed 0
    --randomise none|colour|light|both   what varies in the TRAINING pictures
    --servo-az 35   where the ball is for the closed loop, degrees left
    --gain 0.8      waist turn per radian of estimated azimuth, per decision
    --no-viewer     --speed 1

Six numbered steps; read them with the week 14 slides open.
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--randomise", choices=("none", "colour", "light", "both"), default="none")
    ap.add_argument("--servo-az", type=float, default=35.0)
    ap.add_argument("--gain", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(argv)

    try:
        import torch
        from soc4180 import vision
    except ImportError:
        print("This script needs torch: run `uv sync --extra rl`, then try again.")
        return 1
    torch.set_num_threads(4)          # fixed: the thread count changes the float sums, and so the digits
    torch.manual_seed(args.seed)

    # -- 1. a camera in the head, a ball in the world -----------------------------------------
    model = vision.camera_model(("red",))
    print(f"camera '{vision.CAMERA}' on torso_link, {vision.FOVY:.0f} deg field of view; "
          f"{model.nmocap} mocap ball; pictures {args.res} x {args.res}")

    # -- 2. data: the simulator labels every picture for free -------------------------------------
    rand_colour = (lambda rng: rng.uniform(0, 1, 3)) if args.randomise in ("colour", "both") else None
    rand_light = (lambda rng: rng.uniform(0.3, 1.0)) if args.randomise in ("light", "both") else None
    t0 = time.time()
    X, Y = vision.render_dataset(model, args.train, res=args.res, seed=args.seed, colour=rand_colour, light=rand_light)
    tests = {
        "red ball (as trained)": vision.render_dataset(model, args.test, res=args.res, seed=1000),
        "blue ball": vision.render_dataset(model, args.test, res=args.res, seed=1001, colour=lambda rng: (0.1, 0.2, 0.9)),
        "red ball, dim room (x0.3)": vision.render_dataset(model, args.test, res=args.res, seed=1002, light=lambda rng: 0.3),
    }
    print(f"rendered {args.train} training + {3 * args.test} test pictures in {time.time() - t0:.1f} s "
          f"(training pictures randomise: {args.randomise})")

    # -- 3. train: pixels in, two angles out, squared error --------------------------------------------
    net = vision.BallNet(args.res)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xt, Yt = vision.as_tensor(X), torch.as_tensor(Y[:, 0])
    print(f"\nBallNet: {n_params:,} parameters. Training {args.epochs} epochs, batches of 64:")
    t0 = time.time()
    for epoch in range(args.epochs):
        perm = torch.randperm(len(Xt)); total = 0.0
        for i in range(0, len(Xt), 64):
            b = perm[i:i + 64]
            loss = ((net(Xt[b]) - Yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b)
        rms = math.degrees(math.sqrt(total / len(Xt)))
        print(f"   epoch {epoch + 1:2d}  training error {rms:5.1f} deg (rms)")
    print(f"   {time.time() - t0:.0f} s on the CPU")

    # -- 4. test: the same world, and two worlds it never saw ----------------------------------------------
    def error_deg(images, labels):
        with torch.no_grad():
            p = net(vision.as_tensor(images)).numpy()
        return np.degrees(np.linalg.norm(p - labels[:, 0], axis=1))

    baseline = np.degrees(np.linalg.norm(tests["red ball (as trained)"][1][:, 0] - Y[:, 0].mean(0), axis=1)).mean()
    print(f"\n{'test set':28s} {'mean error':>11} {'within 5 deg':>13}")
    for name, (ti, tl) in tests.items():
        e = error_deg(ti, tl)
        print(f"{name:28s} {e.mean():9.1f}°  {100 * np.mean(e < 5):11.0f}%")
    print(f"{'(always guess the average)':28s} {baseline:9.1f}°")

    # -- 5. close the loop: see, estimate, turn the waist ----------------------------------------------------
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    vision.place_ball(model, data, 0, math.radians(args.servo_az), 0.0, 1.5)
    ball = data.mocap_pos[0].copy()
    mujoco.mj_forward(model, data)                  # moves the ball's body to its new mocap_pos
    waist = soc4180.joint_index(model, "waist_yaw")
    act = [int(model.jnt_qposadr[model.actuator_trnid[a, 0]]) for a in range(model.nu)].index(waist)
    data.ctrl[:] = model.key_qpos[0][[int(model.jnt_qposadr[model.actuator_trnid[a, 0]]) for a in range(model.nu)]]
    from soc4180.render import _new_renderer
    frames = []
    print(f"\nclosed loop: ball {args.servo_az:.0f} deg to the left, 1.5 m away; every 0.1 s: picture -> network -> "
          f"waist += {args.gain} x estimate")
    print(f"{'t (s)':>6} {'waist yaw':>10} {'estimated az':>13} {'true az':>8}")
    with _new_renderer(model, args.res, args.res) as r:
        while data.time < 3.0:
            data.mocap_pos[0] = ball
            r.update_scene(data, camera=vision.CAMERA)
            img = r.render().transpose(2, 0, 1)[None]
            with torch.no_grad():
                est = float(net(vision.as_tensor(img))[0, 0])
            true = float(vision.ball_direction(model, data, 0)[0])
            data.ctrl[act] = float(np.clip(data.ctrl[act] + args.gain * est, *model.actuator_ctrlrange[act]))
            if round(data.time * 10) % 2 == 0 and data.time < 1.05 or round(data.time * 10) % 10 == 0:
                print(f"{data.time:6.2f} {math.degrees(data.qpos[waist]):9.1f}° {math.degrees(est):12.1f}° {math.degrees(true):7.1f}°")
            for _ in range(50):
                mujoco.mj_step(model, data)
                if _ % 10 == 0:
                    frames.append(data.qpos.copy())
    true = float(vision.ball_direction(model, data, 0)[0])
    print(f"\nafter 3 s the ball is {math.degrees(true):+.1f}° off the camera's axis "
          f"(it started at {args.servo_az:+.0f}°); pelvis {data.qpos[2]:.2f} m")

    if args.no_viewer or soc4180.is_colab():
        return 0

    # -- 6. replay ----------------------------------------------------------------------------------------------
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    print(f"\nreplaying the closed loop at {args.speed:g}x (close the window to stop)")
    with soc4180.launch_viewer(model, data, passive=True) as viewer:
        while viewer.is_running() and time.time() < deadline:
            for q in frames:
                if not viewer.is_running() or time.time() > deadline:
                    break
                data.qpos[:] = q; data.mocap_pos[0] = ball
                mujoco.mj_forward(model, data); viewer.sync()
                time.sleep(10 * model.opt.timestep / args.speed)
            time.sleep(0.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
