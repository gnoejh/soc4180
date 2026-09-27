"""A small embodied agent: instructions -> a plan of skills -> physics -> a check that each one worked. Week 15.

Needs torch (``uv sync --extra rl``) and a GL backend. Not imported by
``soc4180`` itself: ``from soc4180 import agent``.

The pieces, each small enough to read:

- ``Body``                 the G1 in the four-ball scene (weeks 14-15), standing on its servos
- ``Eyes``                 week 15's HeatNet: words + a head-camera picture -> where
- skills                   ``look``, ``raise_arm``, ``lower_arm``, ``search`` -- each runs the
                           physics and returns a ``Result`` saying whether it worked,
                           judged by a sensor, not by the skill's own belief
- ``parse``                an instruction -> a list of skill calls, by keywords
- ``run``                  execute a plan, stop at the first skill that fails

Every skill *verifies*: ``look`` checks the colour of the pixels in the
middle of the picture, ``raise_arm`` the height of the hand, and every skill
that the pelvis is still up. A plan is only as good as the checks between its
steps -- that is the week's point.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._gl import GL_BACKEND  # noqa: F401  (sets MUJOCO_GL before mujoco loads)

import mujoco
import torch

from . import vision

__all__ = ["Body", "Eyes", "Result", "parse", "run", "SKILLS", "COLOUR_WORDS"]

NAMES = ("red", "green", "blue", "yellow")
COLOUR_WORDS = set(NAMES)


@dataclass
class Result:
    ok: bool
    message: str


class Body:
    """The G1 standing on its servos in the four-ball scene, with the pieces the skills need."""

    def __init__(self, balls=None):
        self.model = vision.camera_model(NAMES, radius=0.15)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.act_q = [int(m.jnt_qposadr[m.actuator_trnid[a, 0]]) for a in range(m.nu)]
        self.stand_ctrl = m.key_qpos[0][self.act_q].copy()
        from .bodies import joint_index

        def act(name):
            return self.act_q.index(joint_index(m, name))
        self.act = act
        self.hand = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wrist_yaw_link") for s in ("left", "right")}
        self.frames = []
        self.reset(balls)

    def reset(self, balls=None):
        """Stand, and put the balls at ``balls`` = {name: (azimuth, elevation, distance)} (radians, metres)."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        balls = balls or {"red": (0.6, -0.2, 1.6), "green": (0.2, 0.1, 1.8), "blue": (-0.2, -0.1, 1.5),
                          "yellow": (-0.6, 0.15, 2.0)}
        for i, n in enumerate(NAMES):
            vision.place_ball(self.model, self.data, i, *balls[n])
        self.balls = self.data.mocap_pos.copy()
        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[:] = self.stand_ctrl
        self.frames = []

    def step(self, seconds: float):
        """Run the physics with the current ``ctrl``, keeping the balls where they are."""
        d = self.data
        for _ in range(int(round(seconds / self.model.opt.timestep))):
            d.mocap_pos[:] = self.balls
            mujoco.mj_step(self.model, d)
            if round(d.time / self.model.opt.timestep) % 17 == 0:
                self.frames.append(d.qpos.copy())

    def ramp(self, targets: dict, seconds: float = 1.5):
        """Move named joints' servo targets smoothly to ``targets`` (radians) over ``seconds``, then hold 0.5 s."""
        start = {k: float(self.data.ctrl[self.act(k)]) for k in targets}
        n = max(int(seconds / 0.05), 1)
        for i in range(1, n + 1):
            s = 0.5 - 0.5 * math.cos(math.pi * i / n)
            for k, v in targets.items():
                self.data.ctrl[self.act(k)] = start[k] + s * (v - start[k])
            self.step(0.05)
        self.step(0.5)

    @property
    def standing(self) -> bool:
        x, y = self.data.qpos[4], self.data.qpos[5]
        return self.data.qpos[2] > 0.6 and 1.0 - 2.0 * (x * x + y * y) > math.cos(math.radians(20))


class Eyes:
    """HeatNet and its vocabulary, loaded from a checkpoint; plus the pixel-colour check."""

    def __init__(self, body: Body, checkpoint):
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.vocab = vision.Vocab([])
        self.vocab.index = dict(blob["vocab"])
        self.net = vision.HeatNet(len(self.vocab))
        self.net.load_state_dict(blob["state"])
        self.net.eval()
        self.body = body
        from .render import _new_renderer

        self.renderer = _new_renderer(body.model, 64, 64)
        self.last = None

    def picture(self) -> np.ndarray:
        self.renderer.update_scene(self.body.data, camera=vision.CAMERA)
        return self.renderer.render().copy()

    def where(self, words: str, image=None):
        """(azimuth, elevation) of whatever the words pick out, from the current picture."""
        image = self.picture() if image is None else image
        with torch.no_grad():
            logits = self.net(vision.as_tensor(image.transpose(2, 0, 1)[None]), self.vocab.bag(words)[None])
        uv = vision.heat_to_uv(logits, self.net.grid)[0]
        self.last = (image, uv)
        return vision.angles_of_uv(uv)

    @staticmethod
    def colour_at(image, uv) -> str:
        """The nearest ball colour (or "none") of the 5 x 5 pixels at image point ``uv``."""
        h, w = image.shape[:2]
        x, y = int((uv[0] + 1) / 2 * w), int((uv[1] + 1) / 2 * h)
        patch = image[max(0, y - 2):y + 3, max(0, x - 2):x + 3].reshape(-1, 3).astype(float) / 255.0
        best, dist = "none", 0.25                                   # anything further than this is floor or sky
        for name in NAMES:
            ref = np.array(vision.COLOURS[name])
            dmin = float(np.min(np.linalg.norm(patch - ref, axis=1)))
            if dmin < dist:
                best, dist = name, dmin
        return best


# --- the skills -------------------------------------------------------------------------------------------

def look(body: Body, eyes: Eyes, words: str, *, gain: float = 0.8, seconds: float = 2.0) -> Result:
    """Turn the waist until what the words pick out is in the middle of the picture; then check its colour."""
    colour = next((w for w in re.findall(r"[a-z]+", words.lower()) if w in COLOUR_WORDS), None)
    if colour is None:
        return Result(False, f"no colour I know in \"{words}\" (I know {', '.join(NAMES)})")
    w = body.act("waist_yaw")
    lo, hi = body.model.actuator_ctrlrange[w]
    t_end = body.data.time + seconds
    while body.data.time < t_end:
        az, _ = eyes.where(words)
        body.data.ctrl[w] = float(np.clip(body.data.ctrl[w] + gain * az, lo, hi))
        body.step(0.1)
    # The check uses a DIFFERENT sense than the one that steered: the plain colour
    # of the pixels where the network points, and how far that is from the middle.
    az, _ = eyes.where(words)
    image, uv = eyes.last
    seen = eyes.colour_at(image, uv)
    if not body.standing:
        return Result(False, "fell over while turning")
    if seen != colour:
        return Result(False, f"looked for {colour}, but the pixels where the network points are {seen}")
    if abs(math.degrees(az)) > 5.0:
        return Result(False, f"{colour} is still {math.degrees(az):+.0f} deg from the middle of the picture")
    waist = math.degrees(body.data.qpos[body.act_q[w]])
    return Result(True, f"{colour} is in the middle of the picture (waist {waist:+.0f} deg)")


def raise_arm(body: Body, eyes, side: str) -> Result:
    """Shoulder pitch to -150 degrees, elbow straight: a hand overhead (week 3's WAVE)."""
    body.ramp({f"{side}_shoulder_pitch": math.radians(-150), f"{side}_elbow": 0.0})
    z = float(body.data.xpos[body.hand[side]][2])
    if not body.standing:
        return Result(False, f"fell over raising the {side} arm")
    if z < 1.2:
        return Result(False, f"{side} hand only reached {z:.2f} m")
    return Result(True, f"{side} hand at {z:.2f} m")


def raise_both(body: Body, eyes) -> Result:
    """Both shoulders at once -- the move week 3 found throws the robot on its face."""
    body.ramp({"left_shoulder_pitch": math.radians(-150), "left_elbow": 0.0,
               "right_shoulder_pitch": math.radians(-150), "right_elbow": 0.0})
    if not body.standing:
        return Result(False, "fell over raising both arms at once")
    lo = min(float(body.data.xpos[body.hand[s]][2]) for s in ("left", "right"))
    return Result(lo > 1.2, f"lower hand at {lo:.2f} m")


def lower_arm(body: Body, eyes, side: str) -> Result:
    k = body.model.key_qpos[0]
    from .bodies import joint_index

    body.ramp({f"{side}_shoulder_pitch": float(k[joint_index(body.model, f"{side}_shoulder_pitch")]),
               f"{side}_elbow": float(k[joint_index(body.model, f"{side}_elbow")])})
    return Result(body.standing, "arm down" if body.standing else "fell over lowering the arm")


def search(body: Body, eyes: Eyes, words: str, *, step_deg: float = 60.0, tries: int = 4) -> Result:
    """Look; if the check fails, turn the waist by ``step_deg`` (alternating sides, widening) and look again."""
    w = body.act("waist_yaw")
    lo, hi = body.model.actuator_ctrlrange[w]
    r = look(body, eyes, words)
    k = 0
    while not r.ok and k < tries and body.standing:
        k += 1
        turn = math.radians(step_deg) * ((k + 1) // 2) * (1 if k % 2 else -1)
        body.ramp({"waist_yaw": float(np.clip(turn, lo, hi))}, seconds=1.0)
        r = look(body, eyes, words)
    return Result(r.ok, r.message + (f" (after {k} turns)" if k else ""))


SKILLS = {"look": look, "raise": raise_arm, "raise_both": raise_both, "lower": lower_arm, "search": search}


# --- language -> a plan ------------------------------------------------------------------------------------

def parse(instruction: str, synonyms: dict | None = None, programs: dict | None = None):
    """Keywords to skill calls. ``synonyms`` rewrites words first; ``programs`` maps a phrase to a whole plan.

    Clauses are split on "and", "then" and commas; each clause becomes one
    skill call by the first verb it contains. It knows nothing else -- which
    is exactly how far keyword rules go.
    """
    text = " " + instruction.lower() + " "
    for word, meaning in (synonyms or {}).items():
        text = re.sub(rf"\b{re.escape(word.lower())}\b", meaning.lower(), text)
    for phrase, plan in (programs or {}).items():
        if phrase.lower() in text:
            return [tuple(step) for step in plan]
    plan = []
    for clause in re.split(r"\band\b|\bthen\b|,", text):
        words = re.findall(r"[a-z]+", clause)
        if not words:
            continue
        side = "right" if "right" in words else "left"
        if {"look", "face", "turn", "point"} & set(words):
            plan.append(("look", clause.strip()))
        elif {"find", "search"} & set(words):
            plan.append(("search", clause.strip()))
        elif {"raise", "wave", "lift"} & set(words):
            plan.append(("raise_both",) if "both" in words else ("raise", side))
        elif {"lower", "drop"} & set(words):
            plan.append(("lower", side))
        else:
            plan.append(("unknown", clause.strip()))
    return plan


def run(body: Body, eyes: Eyes, plan, *, log=print, **settings):
    """Execute the plan step by step; stop at the first step whose check fails. Returns the list of Results."""
    results = []
    for step in plan:
        name, args = step[0], step[1:]
        if name not in SKILLS:
            results.append(Result(False, f"no skill for \"{' '.join(map(str, args)) or name}\""))
        else:
            fn = SKILLS[name]
            kw = {k: v for k, v in settings.get(name, {}).items()}
            results.append(fn(body, eyes, *args, **kw))
        log(f"   {name}{tuple(args) if args else ''}: {'ok' if results[-1].ok else 'FAILED'} -- {results[-1].message}")
        if not results[-1].ok:
            break
    return results
