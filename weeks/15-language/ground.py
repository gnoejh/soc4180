"""Words to pixels to motion: train a tiny vision-language-action model, test what it understood, then obey it -- printed, then replayed.

    uv run weeks/15-language/ground.py                         # needs torch: uv sync --extra rl
    uv run weeks/15-language/ground.py --say "look at the green ball"
    uv run weeks/15-language/ground.py --say "look at the crimson ball"
    uv run weeks/15-language/ground.py --no-viewer

Four coloured balls float in front of the G1. An instruction names one of
them. The network reads the 64 x 64 head-camera picture as a 16 x 16 grid of
*keys*, turns the sentence into one *query* (a bag of words through a linear
layer), and scores every cell by key . query / sqrt(C): the attention of a
transformer, with a single query. The softmax of those scores is a heat map;
its centre of mass is WHERE. The loss is cross-entropy against the cell the
named ball is really in -- the simulator's label, free.

It is then tested on phrasings it never saw, on a colour word it never saw,
and on no colour word at all, and finally it drives the robot: the waist turns
until the named ball is in the middle of the picture.

    --train 3000 --epochs 10 --seed 0
    --say "look at the blue ball"      the instruction obeyed in step 5
    --save FILE     write the trained network (checkpoints/heatnet.pt is this, default settings)
    --no-viewer     --speed 1

Six numbered steps; read them with the week 15 slides open.
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

NAMES = ("red", "green", "blue", "yellow")
TEMPLATES = ["point at the {c} ball", "look at the {c} one", "{c}", "find the {c} ball",
             "where is the {c} ball", "turn toward the {c} sphere", "show me {c}"]
NEW_PHRASINGS = ["can you face the {c} ball please", "the {c} one, over there", "go {c}"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--say", default="look at the blue ball")
    ap.add_argument("--save", default=None, help="write the trained network here (lab_agent.py ships checkpoints/heatnet.pt)")
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
    rng = np.random.default_rng(args.seed)

    # -- 1. the world: four balls, one camera ---------------------------------------------------
    model = vision.camera_model(NAMES, radius=0.15)
    t0 = time.time()
    X, Y = vision.render_dataset(model, args.train, seed=args.seed, dist=(1.0, 2.5))
    Xtest, Ytest = vision.render_dataset(model, 600, seed=1000 + args.seed, dist=(1.0, 2.5))
    print(f"{len(NAMES)} balls ({', '.join(NAMES)}); rendered {args.train} + 600 pictures in {time.time() - t0:.1f} s")

    # -- 2. the language: a vocabulary and a bag of words -------------------------------------------
    vocab = vision.Vocab([t.format(c=c) for t in TEMPLATES for c in NAMES])
    print(f"vocabulary: {len(vocab)} words from {len(TEMPLATES)} templates -- {', '.join(list(vocab.index)[1:])}")

    def batch(images, labels, templates, r):
        which = r.integers(len(NAMES), size=len(images))
        said = [templates[r.integers(len(templates))].format(c=NAMES[w]) for w in which]
        bags = torch.stack([vocab.bag(s) for s in said])
        target = labels[np.arange(len(images)), which]               # (az, el) of the NAMED ball
        return vision.as_tensor(images), bags, target, said

    # -- 3. train: cross-entropy on WHERE ---------------------------------------------------------------
    net = vision.HeatNet(len(vocab))
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    print(f"\nHeatNet: {sum(p.numel() for p in net.parameters()):,} parameters, "
          f"{args.epochs} epochs:")
    t0 = time.time()
    for epoch in range(args.epochs):
        img, bags, target, _ = batch(X, Y, TEMPLATES, rng)
        cells = torch.as_tensor(vision.cell_of(vision.uv_of(target[:, 0], target[:, 1]), net.grid))
        perm = torch.randperm(len(img)); total = 0.0
        for i in range(0, len(img), 64):
            b = perm[i:i + 64]
            logits = net(img[b], bags[b])
            loss = torch.nn.functional.cross_entropy(logits, cells[b])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b)
        with torch.no_grad():
            a = torch.softmax(net(img[:256], bags[:256]), -1)
            sharp = float(a.max(1).values.mean())
        print(f"   epoch {epoch + 1:2d}  loss {total / len(img):5.2f}   heat-map peak {sharp:.2f} "
              f"(uniform would be {1 / net.grid ** 2:.3f})")
    print(f"   {time.time() - t0:.0f} s on the CPU")

    # -- 4. what did it understand? ---------------------------------------------------------------------
    def test(templates, label):
        img, bags, target, said = batch(Xtest, Ytest, templates, np.random.default_rng(5))
        with torch.no_grad():
            ang = vision.angles_of_uv(vision.heat_to_uv(net(img, bags), net.grid))
        e = np.degrees(np.linalg.norm(ang - target, axis=1))
        print(f"{label:34s} {e.mean():6.1f}°  {100 * np.mean(e < 5):6.0f}%   e.g. \"{said[0]}\"")

    print(f"\n{'instructions':34s} {'error':>7} {'<5 deg':>7}")
    test(TEMPLATES, "phrasings it was trained on")
    test(NEW_PHRASINGS, "new phrasings, known colours")
    test(["point at the crimson ball"], "a colour word it never saw")
    test(["point at the ball"], "no colour word at all")

    # -- 4b. ask for a ball that is not in the picture -------------------------------------------------------
    # Half of 200 pictures have the named ball moved behind the robot. The heat
    # map's peak is the network's "confidence"; if it cannot tell the two
    # halves apart, its own output cannot be used to check itself.
    from soc4180.render import _new_renderer
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, 0)
    mujoco.mj_forward(model, d)
    r2, peaks = np.random.default_rng(3), {"in the picture": [], "behind the robot": []}
    with _new_renderer(model, 64, 64) as rend:
        for k in range(200):
            for i in range(len(NAMES)):
                vision.place_ball(model, d, i, r2.uniform(-0.7, 0.7), r2.uniform(-0.5, 0.3), r2.uniform(1.0, 2.5))
            w = r2.integers(len(NAMES))
            if k % 2:
                vision.place_ball(model, d, w, math.pi, 0.0, 1.5)
            mujoco.mj_forward(model, d)
            rend.update_scene(d, camera=vision.CAMERA)
            img = vision.as_tensor(rend.render().transpose(2, 0, 1)[None])
            with torch.no_grad():
                p = float(torch.softmax(net(img, vocab.bag(f"look at the {NAMES[w]} ball")[None]), -1).max())
            peaks["behind the robot" if k % 2 else "in the picture"].append(p)
    for where, ps in peaks.items():
        print(f"named ball {where:18s} heat-map peak: mean {np.mean(ps):.2f}, lowest {np.min(ps):.2f}, highest {np.max(ps):.2f}")
    if args.save:
        torch.save({"state": net.state_dict(), "vocab": vocab.index}, args.save)
        print(f"saved the network to {args.save}")

    # -- 5. obey: the waist turns until the named ball is centred --------------------------------------------
    print(f"\nsay: \"{args.say}\"   (words it knows: {vocab.known(args.say) or 'none'})")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    r = np.random.default_rng(7)
    for i, az in enumerate(r.permutation(np.radians([-35, -12, 12, 35]))):
        vision.place_ball(model, data, i, az, r.uniform(-0.3, 0.2), 1.6)
    balls = data.mocap_pos.copy()
    mujoco.mj_forward(model, data)
    act_qpos = [int(model.jnt_qposadr[model.actuator_trnid[a, 0]]) for a in range(model.nu)]
    waist = soc4180.joint_index(model, "waist_yaw")
    act = act_qpos.index(waist)
    data.ctrl[:] = model.key_qpos[0][act_qpos]
    bag = vocab.bag(args.say)[None]
    frames = []
    with _new_renderer(model, 64, 64) as rend:
        while data.time < 2.0:
            rend.update_scene(data, camera=vision.CAMERA)
            img = vision.as_tensor(rend.render().transpose(2, 0, 1)[None])
            with torch.no_grad():
                az = float(vision.angles_of_uv(vision.heat_to_uv(net(img, bag), net.grid))[0, 0])
            data.ctrl[act] = float(np.clip(data.ctrl[act] + 0.8 * az, *model.actuator_ctrlrange[act]))
            for k in range(50):
                data.mocap_pos[:] = balls
                mujoco.mj_step(model, data)
                if k % 10 == 0:
                    frames.append(data.qpos.copy())
    off = {n: math.degrees(vision.ball_direction(model, data, i)[0]) for i, n in enumerate(NAMES)}
    centred = min(off, key=lambda n: abs(off[n]))
    print(f"after 2 s the waist is at {math.degrees(data.qpos[waist]):+.1f}°; the ball nearest the centre of the "
          f"picture is {centred.upper()} ({off[centred]:+.1f}°)")
    print("   every ball: " + ", ".join(f"{n} {v:+.1f}°" for n, v in off.items()))

    if args.no_viewer or soc4180.is_colab():
        return 0

    # -- 6. replay --------------------------------------------------------------------------------------------
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    print(f"\nreplaying at {args.speed:g}x (close the window to stop)")
    with soc4180.launch_viewer(model, data, passive=True) as viewer:
        while viewer.is_running() and time.time() < deadline:
            for q in frames:
                if not viewer.is_running() or time.time() > deadline:
                    break
                data.qpos[:] = q; data.mocap_pos[:] = balls
                mujoco.mj_forward(model, data); viewer.sync()
                time.sleep(10 * model.opt.timestep / args.speed)
            time.sleep(0.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
