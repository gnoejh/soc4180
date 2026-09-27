"""A camera on the G1, coloured balls to look at, and the small networks that read the pictures.

Weeks 14 and 15. Needs torch (``uv sync --extra rl``) and a GL backend to
render. Not imported by ``soc4180`` itself: ``from soc4180 import vision``.

- ``camera_model(balls)``        the G1 scene + a camera in the head + mocap balls
- ``place_ball`` / ``ball_direction``   put a ball at (azimuth, elevation, distance)
                                  from the camera, and read the true direction back
- ``render_dataset``             pictures and labels, rendered from the head camera
- ``BallNet``                    pixels -> (azimuth, elevation) of the one ball (week 14)
- ``Vocab``, ``HeatNet``         words + pixels -> a heat map of WHERE the named ball is (week 15)

Angles are radians, azimuth positive to the robot's LEFT, elevation positive
UP, both measured from the camera's optical axis. The camera looks along the
torso's +x (forward) with a 90-degree field of view, so +-45 degrees is the
edge of the picture.
"""

from __future__ import annotations

import math
import re

import numpy as np

from ._gl import GL_BACKEND  # noqa: F401  (sets MUJOCO_GL before mujoco loads)

import mujoco
import torch
from torch import nn

__all__ = ["COLOURS", "BallNet", "HeatNet", "Vocab", "ball_direction", "camera_model",
           "cell_of", "place_ball", "render_dataset", "uv_of", "angles_of_uv", "heat_to_uv"]

#: The balls of weeks 14-15, name -> RGB.
COLOURS = {"red": (0.9, 0.1, 0.1), "green": (0.1, 0.8, 0.1), "blue": (0.1, 0.2, 0.9), "yellow": (0.9, 0.85, 0.1)}

CAMERA = "head"
FOVY = 90.0


def camera_model(balls=("red",), radius: float = 0.1) -> mujoco.MjModel:
    """The G1 scene plus a forward camera in the head and one mocap ball per name in ``balls``.

    A *mocap* body is moved by writing ``data.mocap_pos``, not by physics: it
    has no joint, feels no gravity, and here has no collision either
    (``contype = conaffinity = 0``), so a ball can float anywhere. In the
    interactive viewer you can drag one with ctrl + right-drag after
    double-clicking it.

    The camera's ``xyaxes`` give its x axis (image right) as the torso's -y and
    its y axis (image up) as +z, so it looks along +x: forward. MuJoCo cameras
    look down their own -z.
    """
    from .models import robot_path

    spec = mujoco.MjSpec.from_file(str(robot_path()))
    spec.body("torso_link").add_camera(name=CAMERA, pos=[0.08, 0, 0.45], xyaxes=[0, -1, 0, 0, 0, 1], fovy=FOVY)
    for name in balls:
        rgb = COLOURS.get(name, (0.8, 0.8, 0.8))
        b = spec.worldbody.add_body(name=f"ball_{name}", mocap=True, pos=[1.2, 0, 0.8])
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[radius, 0, 0], rgba=[*rgb, 1.0],
                   contype=0, conaffinity=0)
    return spec.compile()


def _camera(model, data):
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA)
    return data.cam_xpos[cid], data.cam_xmat[cid].reshape(3, 3)


def place_ball(model, data, index: int, azimuth: float, elevation: float, distance: float):
    """Put mocap ball ``index`` at this direction and distance from the head camera (needs ``mj_forward`` first)."""
    pos, R = _camera(model, data)
    fwd, left, up = (math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth),
                     math.sin(elevation))
    data.mocap_pos[index] = pos + R @ (distance * np.array([-left, up, -fwd]))   # camera frame: x right, y up, -z forward


def ball_direction(model, data, index: int) -> np.ndarray:
    """True (azimuth, elevation) of mocap ball ``index`` as seen from the head camera."""
    pos, R = _camera(model, data)
    x, y, z = R.T @ (data.mocap_pos[index] - pos)
    return np.array([math.atan2(-x, -z), math.atan2(y, math.hypot(x, z))])


def uv_of(az, el):
    """(azimuth, elevation) -> normalised image coordinates in [-1, 1] (u right, v down)."""
    az, el = np.asarray(az), np.asarray(el)
    f = np.cos(el) * np.cos(az)
    return np.stack([-np.cos(el) * np.sin(az) / f, -np.sin(el) / f], -1)


def angles_of_uv(uv):
    """Inverse of ``uv_of``."""
    uv = np.asarray(uv, float)
    v = np.stack([np.ones(uv.shape[:-1]), -uv[..., 0], -uv[..., 1]], -1)
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    return np.stack([np.arctan2(v[..., 1], v[..., 0]), np.arcsin(v[..., 2])], -1)


def render_dataset(model, n: int, *, res: int = 64, seed: int = 0, colour=None, light=None,
                   az=(-0.7, 0.7), el=(-0.6, 0.4), dist=(0.8, 2.5), min_gap=0.15, data=None):
    """Render ``n`` head-camera pictures with every ball at a random direction.

    Returns ``images`` (n, 3, res, res) uint8 and ``labels`` (n, nballs, 2):
    each ball's (azimuth, elevation). ``colour(rng) -> rgb`` recolours ball 0
    per picture (colour randomisation); ``light(rng) -> float`` scales the
    lights. Balls are kept ``min_gap`` radians apart in azimuth so none hides
    another.
    """
    from .render import _new_renderer

    rng = np.random.default_rng(seed)
    data = data or mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    nb = model.nmocap
    first_geom = model.body_geomadr[model.body_mocapid.tolist().index(0)] if nb else 0
    diffuse0 = model.light_diffuse.copy()
    rgba0 = model.geom_rgba[first_geom].copy()
    images = np.zeros((n, 3, res, res), np.uint8)
    labels = np.zeros((n, nb, 2), np.float32)
    try:
        with _new_renderer(model, res, res) as r:
            for k in range(n):
                azs = rng.uniform(*az, nb)
                while nb > 1 and np.min(np.diff(np.sort(azs))) < min_gap:
                    azs = rng.uniform(*az, nb)
                for i in range(nb):
                    e = rng.uniform(*el)
                    place_ball(model, data, i, azs[i], e, rng.uniform(*dist))
                    labels[k, i] = (azs[i], e)
                if colour is not None:
                    model.geom_rgba[first_geom, :3] = colour(rng)
                if light is not None:
                    model.light_diffuse[:] = diffuse0 * light(rng)
                mujoco.mj_forward(model, data)
                r.update_scene(data, camera=CAMERA)
                images[k] = r.render().transpose(2, 0, 1)
    finally:
        model.light_diffuse[:] = diffuse0
        model.geom_rgba[first_geom] = rgba0
    return images, labels


def as_tensor(images) -> torch.Tensor:
    return torch.as_tensor(np.asarray(images), dtype=torch.float32) / 255.0


class BallNet(nn.Module):
    """Pixels -> (azimuth, elevation). Three strided convolutions and two linear layers.

    64x64 -> 32 -> 16 -> 8 spatial positions, 16 -> 32 -> 32 channels; the
    last 32 x 8 x 8 numbers are flattened and a small MLP regresses the two
    angles. About 140k parameters.
    """

    def __init__(self, res: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * (res // 8) ** 2, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        return self.net(x)


def _tokens(s: str):
    return re.findall(r"[a-z]+", s.lower())


class Vocab:
    """Words -> a bag-of-words vector. Unknown words all map to one slot, ``<unk>``."""

    def __init__(self, sentences):
        self.index = {"<unk>": 0}
        for s in sentences:
            for t in _tokens(s):
                self.index.setdefault(t, len(self.index))

    def __len__(self):
        return len(self.index)

    def bag(self, sentence: str) -> torch.Tensor:
        v = torch.zeros(len(self.index))
        for t in _tokens(sentence):
            v[self.index.get(t, 0)] += 1.0
        return v

    def known(self, sentence: str):
        return [t for t in _tokens(sentence) if t in self.index]


class HeatNet(nn.Module):
    """Words + pixels -> a score for every cell of a 16 x 16 grid over the picture: WHERE.

    The picture becomes a C-number *key* per cell (convolutions), the sentence
    becomes one C-number *query* (a linear layer on the bag of words), and the
    score of a cell is their dot product divided by sqrt(C) -- the attention
    of a transformer, with one query. Softmax over cells is the heat map;
    its centre of mass is the answer.
    """

    def __init__(self, nvocab: int, res: int = 64, channels: int = 32):
        super().__init__()
        self.grid = res // 4
        self.keys = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(),
                                  nn.Conv2d(32, channels, 3, 2, 1))
        self.query = nn.Linear(nvocab, channels)

    def forward(self, images, bags):
        k = self.keys(images).flatten(2)                           # B x C x cells
        q = self.query(bags)                                        # B x C
        s = torch.einsum("bcn,bc->bn", k, q)
        return s / k.shape[1] ** 0.5                                # B x cells: logits


def cell_of(uv, grid: int = 16):
    """Index of the grid cell holding image point ``uv``."""
    ij = np.clip(((np.asarray(uv) + 1) / 2 * grid).astype(int), 0, grid - 1)
    return ij[..., 1] * grid + ij[..., 0]


def heat_to_uv(logits, grid: int = 16) -> np.ndarray:
    """Centre of mass of the softmax heat map, in image coordinates [-1, 1]."""
    a = torch.softmax(torch.as_tensor(logits), -1).detach().numpy().reshape(-1, grid, grid)
    g = (np.arange(grid) + 0.5) / grid * 2 - 1
    return np.stack([(a.sum(1) * g).sum(1), (a.sum(2) * g).sum(1)], -1)
