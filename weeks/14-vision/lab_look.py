"""Week 14 lab game, on your laptop: five problems about eyes, a score out of 5.

    uv run weeks/14-vision/lab_look.py          # torch: uv sync --extra rl

YOU EDIT ONLY `eyes.py`, next to this file. Each problem trains a small
convolutional network (soc4180.vision.BallNet, the lecture's) on pictures
from the G1's head camera, with a recipe you write: how many pictures, how
many passes over them, and what VARIES in them -- the ball's colour, the
brightness of the room. The referee renders the pictures, trains the network
(the window freezes while it does: 10-40 s), and then tests it on pictures it
has never seen, live, with what the camera sees in the corner:

    white +   where the ball really is          yellow +   where the network says it is

    1 ... 5        play problem 1 ... 5
    SPACE          the current problem again
    G              GRADE: all five; the score stays on screen
    ENTER          print the live gauge in this terminal
    Keys go to the MuJoCo window -- click it first. If no key works there,
    press the same keys in this terminal (single keys, no enter). `q` quits.

    uv run weeks/14-vision/lab_look.py --grade --no-viewer
    uv run weeks/14-vision/lab_look.py --grade --problem 2 3

Things measured with this file's own code (and look.py) before it was written,
4000 pictures and 10 epochs unless stated:

- Red balls only: 1.0 deg on red, 27.8 on a blue ball (always guessing the
  middle: 26.0), 15.2 in a room lit at 30%.
- Colour randomised: 4.1 red, 5.0 blue, 27.2 dim. Light randomised over 30-100%:
  1.3 red, 28.7 blue, 1.6 dim. Both: 4.4, 9.4, 4.2. Randomisation fixes what
  it randomises and nothing else, and costs accuracy on the easy case.
- Turning the waist toward a ball 35 deg to the left, a correction every 0.1 s:
  gain 0.3 is still 4 deg short at 0.6 s; 0.8 overshoots to 41 deg and settles
  by 0.5 s; 1.5 swings to 60 and back; 2.2 throws the robot on the floor.

Everything in this file is complete and explained. You are not meant to edit
it: it is the referee, and the grading screen shows a 4-letter code computed
from it so the instructor can see it is unchanged.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import numpy as np

import soc4180
from soc4180.game import AnswersError, Attempt, Problem, grade_headless, need, number, run_viewer

HERE = Path(__file__).resolve().parent
RES = 64
_CACHE: dict = {}          # recipe -> trained network, so G does not train the same recipe twice


# --- 1. a recipe -> a trained network ----------------------------------------------------------------

def read_recipe(spec, where):
    if not isinstance(spec, dict):
        raise AnswersError(f"{where}: must be a dict like {{\"pictures\": 4000, ...}}")
    pictures = int(number(spec, "pictures", where, 100, 20000))
    epochs = int(number(spec, "epochs", where, 1, 40))
    colour = spec.get("colour")
    if colour not in ("red", "random"):
        raise AnswersError(f"{where}: \"colour\" must be \"red\" or \"random\", not {colour!r}")
    light = spec.get("light")
    if (not isinstance(light, list) or len(light) != 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in light)
            or not 0.05 <= light[0] <= light[1] <= 1.5):
        raise AnswersError(f"{where}: \"light\" must be [low, high] with 0.05 <= low <= high <= 1.5, not {light!r}")
    return pictures, epochs, colour, (float(light[0]), float(light[1]))


def trained(model, recipe):
    """Render the training pictures the recipe asks for and fit BallNet to them (cached)."""
    import torch

    from soc4180 import vision

    if recipe in _CACHE:
        return _CACHE[recipe]
    pictures, epochs, colour, (lo, hi) = recipe
    t0 = time.time()
    X, Y = vision.render_dataset(model, pictures, res=RES, seed=0,
                                 colour=(lambda r: r.uniform(0, 1, 3)) if colour == "random" else None,
                                 light=(lambda r: r.uniform(lo, hi)) if (lo, hi) != (1.0, 1.0) else None)
    torch.manual_seed(0)
    net = vision.BallNet(RES)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xt, Yt = vision.as_tensor(X), torch.as_tensor(Y[:, 0])
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            b = perm[i:i + 64]
            loss = ((net(Xt[b]) - Yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    print(f"   trained on {pictures} pictures x {epochs} epochs ({colour} ball, light {lo:g}..{hi:g}) "
          f"in {time.time() - t0:.0f} s")
    _CACHE[recipe] = net
    return net


class Eyes:
    """Built once: the camera model, a renderer for what the robot sees, and the three test sets."""

    def __init__(self):
        from soc4180 import vision

        self.vision = vision
        self.model = vision.camera_model(("red",))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.gid = self.model.body_geomadr[self.model.body_mocapid.tolist().index(0)]
        # the SAME test pictures for everyone: seeds 1000, 1001, 1002, as in look.py and the slides
        self.tests = {
            "red": (dict(seed=1000), None, None),
            "blue": (dict(seed=1001), (0.1, 0.2, 0.9), None),
            "dim": (dict(seed=1002), None, 0.3),
        }


EYES = None


def eyes():
    global EYES
    if EYES is None:
        EYES = Eyes()
    return EYES


# --- 2. the test: show 500 new pictures, one by one -------------------------------------------------

class SeeAttempt(Attempt):
    """Train on the recipe, then test on the named test sets; pass if every mean error is under its limit."""

    def __init__(self, recipe, limits: dict):
        super().__init__()
        e = eyes()
        self.e, self.recipe, self.limits = e, recipe, limits
        self.model, self.data = e.model, e.data
        self.dt = 0.02                            # 50 pictures a second
        self.net, self.queue, self.errors, self.current = None, [], {k: [] for k in limits}, None
        self.gauge = "rendering the training pictures and training the network..."
        self._renderer = None

    def step(self):
        import torch

        v = self.e.vision
        if self.net is None:
            self.net = trained(self.model, self.recipe)
            rng = np.random.default_rng(1000)
            for name in self.limits:                              # 500 pictures per test set
                r = np.random.default_rng(self.e.tests[name][0]["seed"])
                for _ in range(500):
                    self.queue.append((name, r.uniform(-0.7, 0.7), r.uniform(-0.6, 0.4), r.uniform(0.8, 2.5)))
            return True
        if not self.queue:
            return False
        name, az, el, dist = self.queue.pop(0)
        _, colour, light = self.e.tests[name]
        m, d = self.model, self.data
        diffuse0, rgba0 = m.light_diffuse.copy(), m.geom_rgba[self.e.gid].copy()
        if colour is not None:
            m.geom_rgba[self.e.gid, :3] = colour
        if light is not None:
            m.light_diffuse[:] = diffuse0 * light
        v.place_ball(m, d, 0, az, el, dist)
        mujoco.mj_forward(m, d)
        if self._renderer is None:
            from soc4180.render import _new_renderer
            self._renderer = _new_renderer(m, RES, RES)
        self._renderer.update_scene(d, camera=v.CAMERA)
        img = self._renderer.render().copy()
        m.light_diffuse[:], m.geom_rgba[self.e.gid] = diffuse0, rgba0
        with torch.no_grad():
            est = self.net(v.as_tensor(img.transpose(2, 0, 1)[None])).numpy()[0]
        err = math.degrees(math.hypot(est[0] - az, est[1] - el))
        self.errors[name].append(err)
        self.current = (img, (az, el), est)
        self.gauge = "   ".join(f"{k}: {np.mean(v):4.1f} deg over {len(v)} (need < {self.limits[k]:g})"
                               for k, v in self.errors.items() if v)
        self.met = all(len(v) == 500 and np.mean(v) < self.limits[k] for k, v in self.errors.items())
        return True

    def images(self):
        if self.current is None:
            return None
        img, (az, el), est = self.current
        big = np.kron(img, np.ones((4, 4, 1), np.uint8))           # 256 x 256
        for (a, e_), colour in (((az, el), (255, 255, 255)), ((est[0], est[1]), (255, 230, 0))):
            u, v_ = self.e.vision.uv_of(a, e_)
            x, y = int((u + 1) * 128), int((v_ + 1) * 128)
            for dx in range(-8, 9):
                for w in (-1, 0, 1):
                    for px, py in ((x + dx, y + w), (x + w, y + dx)):
                        if 0 <= px < 256 and 0 <= py < 256:
                            big[py, px] = colour
        return [(mujoco.MjrRect(10, 60, 256, 256), np.flipud(big))]

    def verdict(self):
        bad = [f"{k} {np.mean(v):.1f} deg (need < {self.limits[k]:g})" for k, v in self.errors.items()
               if not v or np.mean(v) >= self.limits[k]]
        if bad:
            return False, "; ".join(bad)
        return True, ", ".join(f"{k} {np.mean(v):.1f} deg" for k, v in self.errors.items())


# --- 3. the turn: see, estimate, turn the waist, every 0.1 s ---------------------------------------------

class TurnAttempt(Attempt):
    """The RED recipe's network drives the waist toward a ball 35 degrees to the left."""

    def __init__(self, recipe, gain):
        super().__init__()
        e = eyes()
        self.e, self.gain = e, gain
        self.model = e.model
        self.data = mujoco.MjData(self.model)
        self.recipe = recipe
        self.dt = 0.1
        self.net, self._renderer, self.current = None, None, None
        self.log = []
        self.gauge = "training the RED network (cached if problem 1 already did)..."

    def step(self):
        import torch

        v, m, d = self.e.vision, self.model, self.data
        if self.net is None:
            self.net = trained(m, self.recipe)
            mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
            v.place_ball(m, d, 0, math.radians(35.0), 0.0, 1.5)
            self.ball = d.mocap_pos[0].copy(); mujoco.mj_forward(m, d)
            self.act_q = [int(m.jnt_qposadr[m.actuator_trnid[a, 0]]) for a in range(m.nu)]
            self.waist = soc4180.joint_index(m, "waist_yaw")
            self.act = self.act_q.index(self.waist)
            d.ctrl[:] = m.key_qpos[0][self.act_q]
            from soc4180.render import _new_renderer
            self._renderer = _new_renderer(m, RES, RES)
            return True
        if d.time >= 2.0 - 1e-9:
            return False
        self._renderer.update_scene(d, camera=v.CAMERA)
        img = self._renderer.render().copy()
        with torch.no_grad():
            est = float(self.net(v.as_tensor(img.transpose(2, 0, 1)[None]))[0, 0])
        d.ctrl[self.act] = float(np.clip(d.ctrl[self.act] + self.gain * est, *m.actuator_ctrlrange[self.act]))
        for _ in range(50):
            d.mocap_pos[0] = self.ball
            mujoco.mj_step(m, d)
        yaw = math.degrees(d.qpos[self.waist])
        self.log.append((d.time, yaw, d.qpos[2]))
        self.current = (img, tuple(v.ball_direction(m, d, 0)), (est, 0.0))
        self.gauge = f"t {d.time:3.1f} s   waist {yaw:+6.1f} deg   ball at +35   network says {math.degrees(est):+5.1f}"
        return True

    images = SeeAttempt.images

    def verdict(self):
        log = np.array(self.log)
        if log[:, 2].min() < 0.5:
            return False, "the robot fell over"
        peak = log[:, 1].max()
        at = log[np.argmin(np.abs(log[:, 0] - 0.6)), 1]
        if peak > 45.0:
            return False, f"overshot to {peak:.0f} deg (limit 45)"
        if abs(at - 35.0) > 3.0:
            return False, f"at 0.6 s the waist is at {at:.1f} deg, {abs(at - 35):.1f} from the ball (need within 3)"
        return True, f"within {abs(at - 35):.1f} deg of the ball at 0.6 s, peak {peak:.0f}"


# --- 4. the problems ---------------------------------------------------------------------------------

PROBLEMS = {
    1: Problem("RED", "Find the red ball: mean error under 2 deg on 500 new pictures.  (RED)",
               lambda a: SeeAttempt(read_recipe(need(a, "RED"), "RED"), {"red": 2.0})),
    2: Problem("BLUE", "The same network idea, but the ball is BLUE: under 8 deg.  (BLUE)",
               lambda a: SeeAttempt(read_recipe(need(a, "BLUE"), "BLUE"), {"blue": 8.0})),
    3: Problem("DIM", "The room is lit at 30%: under 3 deg on the red ball.  (DIM)",
               lambda a: SeeAttempt(read_recipe(need(a, "DIM"), "DIM"), {"dim": 3.0})),
    4: Problem("ALL", "ONE network for all three: red < 6, blue < 12, dim < 6.  (ALL)",
               lambda a: SeeAttempt(read_recipe(need(a, "ALL"), "ALL"), {"red": 6.0, "blue": 12.0, "dim": 6.0})),
    5: Problem("TURN", "RED's network turns the waist: within 3 deg of the ball by 0.6 s, never past 45.  (RED, GAIN)",
               lambda a: TurnAttempt(read_recipe(need(a, "RED"), "RED"), number({"GAIN": need(a, "GAIN")}, "GAIN", "GAIN", 0.0, 5.0))),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eyes", type=Path, default=HERE / "eyes.py", help="the answers file")
    ap.add_argument("--grade", action="store_true")
    ap.add_argument("--problem", type=int, nargs="*")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(argv)
    try:
        import torch
    except ImportError:
        print("This lab needs torch: run `uv sync --extra rl`, then try again.")
        return 1
    torch.set_num_threads(4)                  # fixed: the thread count changes the float sums, and so the digits
    if args.grade or args.no_viewer:
        grade_headless(PROBLEMS, args.eyes, __file__, args.problem)
        return 0
    if soc4180.is_colab():
        print("This lab needs a desktop window; run it on your laptop.")
        return 1
    print(__doc__.split("\n\n")[1])
    e = eyes()

    def camera(viewer):
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.0, 200.0, -15.0
        viewer.cam.lookat[:] = (0.6, 0.0, 0.9)
    return run_viewer(PROBLEMS, args.eyes, __file__, idle_model=e.model, idle_data=e.data, camera=camera,
                      title="week 14: pixels")


if __name__ == "__main__":
    raise SystemExit(main())
