"""A menagerie robot in MuJoCo, behind the tick clock. Model file.

One brain event = one control tick at CONTROL_HZ (50 Hz). Physics runs at
PHYSICS_DT (0.002 s), so a tick is SUBSTEPS (10) physics steps -- the course's
numbers, recorded in robot_mujoco/CLAUDE.md. The model comes from the pinned
`mujoco_menagerie` package through the robot's `RobotSpec` (`robots/`): which
model, which scene entry, which keyframe is home, where the IMU site and the
head camera sit. Nothing in this module names a robot.

Deterministic: CPU MuJoCo is bit-reproducible for identical inputs, and this
module draws no random numbers unless `push()` is called with the sim's own
seeded generator. Two sims with the same seed and the same ctrl sequence return
the same proprioception to the bit -- tests/test_sim.py checks it.
"""

from __future__ import annotations

import threading

import numpy as np
import mujoco
import mujoco_menagerie as mm

import robots
from body.robot import RobotSpec

__all__ = [
    "SIMULATOR", "CONTROL_HZ", "PHYSICS_DT", "SUBSTEPS", "MujocoSim", "load_model",
    "HEAD_CAMERA",
]

SIMULATOR = "mujoco"
PHYSICS_DT = 0.002
CONTROL_HZ = 50
SUBSTEPS = int(round(1.0 / (CONTROL_HZ * PHYSICS_DT)))  # 10
assert abs(SUBSTEPS * PHYSICS_DT * CONTROL_HZ - 1.0) < 1e-9

HEAD_CAMERA = "head"
# Camera frame in the body frame: MuJoCo cameras look along their -z with +y up.
# We want -z_cam = +x_body (forward) and +y_cam = +z_body (up):
#   x_cam = (0,-1,0)  y_cam = (0,0,1)  z_cam = (-1,0,0)   (columns of R)
_HEAD_R = np.array([[0.0, 0.0, -1.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0]])


def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a proper rotation matrix (MuJoCo order)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


#: The gallery's layout: panel i stands GALLERY_START + i * GALLERY_STEP metres
#: ahead, alternately left and right of the walking line, turned GALLERY_YAW toward
#: the approaching walker. Chosen by measurement (2026-10-17): with these, CLIP
#: names all 21 CC0 photos from the head camera's half-frame on the panel's side
#: at 1.4 m and 1.0 m before each panel; panels square to the line, or smaller,
#: further out and unlit, scored 5-67% on whole frames.
GALLERY_START, GALLERY_STEP, GALLERY_SIDE = 2.0, 1.2, 0.65
GALLERY_HALF, GALLERY_Z, GALLERY_YAW = 0.45, 1.05, 0.6
GALLERY_LIGHT = (0.5, 0.8)          # headlight ambient, diffuse: the gallery is lit


def gallery_layout(n: int) -> list[tuple[float, float, float]]:
    """(x, y, yaw) of each of n panels: what the app reads to know where each
    picture stands."""
    out = []
    for i in range(n):
        side = 1.0 if i % 2 == 0 else -1.0
        out.append((GALLERY_START + i * GALLERY_STEP, side * GALLERY_SIDE, side * GALLERY_YAW))
    return out


def _add_gallery(mjspec, images) -> None:
    """Picture panels along the walking line (the `gallery` scene). Each is a thin
    box textured with one image, VISUAL ONLY: no collision (contype and
    conaffinity 0) and on the world body (no mass), so the walking physics are the
    default scene's exactly; only what the head camera sees differs. The headlight
    is raised (rendering only) so the pictures read."""
    mjspec.visual.headlight.ambient = [GALLERY_LIGHT[0]] * 3
    mjspec.visual.headlight.diffuse = [GALLERY_LIGHT[1]] * 3
    for i, ((x, y, yaw), path) in enumerate(zip(gallery_layout(len(images)), images)):
        tex = mjspec.add_texture()
        tex.name, tex.type, tex.file = f"gallery_tex{i}", mujoco.mjtTexture.mjTEXTURE_2D, str(path)
        mat = mjspec.add_material()
        mat.name = f"gallery_mat{i}"
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        g = mjspec.worldbody.add_geom()
        g.name, g.type = f"gallery_panel{i}", mujoco.mjtGeom.mjGEOM_BOX
        g.size = [0.01, GALLERY_HALF, GALLERY_HALF]
        g.pos = [x, y, GALLERY_Z]
        g.quat = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        g.material = mat.name
        g.contype, g.conaffinity = 0, 0


def load_model(spec: RobotSpec, entry: str | None = None,
               head_camera: bool = False, gallery=None) -> mujoco.MjModel:
    """The compiled model. With `head_camera`, the menagerie XML is loaded as an
    MjSpec and a forward-looking camera named HEAD_CAMERA is added to the
    spec's head_camera_body before compiling (the course's trick, re-done here).
    With `gallery` (a list of image files), picture panels stand along the walking
    line (`_add_gallery`); without it the model is the default scene's."""
    entry = entry or spec.entry
    rob = mm.get(spec.menagerie)
    known = {e.name for e in rob.entry_points}
    if entry not in known:
        raise ValueError(f"{entry!r} is not an entry of {spec.menagerie!r}; have {sorted(known)}")
    if not head_camera and not gallery:
        model = rob.model(entry)
    else:
        rob.model(entry)  # warms the asset cache so xml() resolves
        mjspec = mujoco.MjSpec.from_file(str(rob.xml(entry)))
        if head_camera:
            body = mjspec.body(spec.head_camera_body)
            cam = body.add_camera()
            cam.name = HEAD_CAMERA
            cam.pos = spec.head_camera_pos
            cam.quat = _quat_from_matrix(_HEAD_R)
            cam.fovy = spec.head_camera_fovy
        if gallery:
            _add_gallery(mjspec, list(gallery))
        model = mjspec.compile()
    model.opt.timestep = PHYSICS_DT
    return model


class MujocoSim:
    """The body's physics for one robot. `reset(seed)` puts it in the spec's
    home keyframe; `step(ctrl)` advances one control tick."""

    def __init__(self, robot: str | RobotSpec = "g1", entry: str | None = None,
                 seed: int = 0, head_camera: bool = False, gallery=None):
        self.spec = robots.get(robot)
        self.robot, self.entry = self.spec.name, entry or self.spec.entry
        self.gallery = [str(g) for g in (gallery or [])]
        self.model = load_model(self.spec, self.entry, head_camera=head_camera, gallery=self.gallery)
        self.data = mujoco.MjData(self.model)
        self.key_home = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, self.spec.keyframe)
        if self.key_home < 0:
            raise RuntimeError(f"the model has no {self.spec.keyframe!r} keyframe")
        self.nu = self.model.nu
        if self.nu != self.spec.n_joints:
            raise RuntimeError(f"{self.spec.name}: model has {self.nu} actuators, spec says {self.spec.n_joints}")
        # actuator -> joint qpos/qvel addresses (position actuators on hinges)
        jnt = self.model.actuator_trnid[:, 0]
        self.act_qposadr = self.model.jnt_qposadr[jnt].copy()
        self.act_dofadr = self.model.jnt_dofadr[jnt].copy()
        self.imu_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.spec.imu_site)
        if self.imu_site < 0:
            raise RuntimeError(f"model lacks site {self.spec.imu_site!r}")
        self.has_head_camera = head_camera
        self._renderer = None
        self.reset(seed)

    # -- state -------------------------------------------------------------
    def reset(self, seed: int = 0) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_home)
        self.data.ctrl[:] = self.home_ctrl
        mujoco.mj_forward(self.model, self.data)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.tick = 0

    @property
    def home_ctrl(self) -> np.ndarray:
        """Joint targets that hold the home keyframe: the keyframe's own ctrl if
        it carries one, else its qpos at each actuated joint."""
        kc = self.model.key_ctrl[self.key_home]
        if np.any(kc != 0):
            return kc.copy()
        return self.model.key_qpos[self.key_home][self.act_qposadr].copy()

    def step(self, ctrl: np.ndarray) -> None:
        ctrl = np.asarray(ctrl, dtype=float)
        if ctrl.shape != (self.nu,):
            raise ValueError(f"ctrl must have shape ({self.nu},), got {ctrl.shape}")
        self.data.ctrl[:] = ctrl
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        self.tick += 1

    def push(self, force_xy: tuple[float, float] | None = None, magnitude: float = 50.0) -> np.ndarray:
        """An external push on the IMU body for this tick only (a world event,
        seeded by the sim's own generator when no direction is given). Returns
        the force applied. The brain never sees it -- it is privileged state."""
        if force_xy is None:
            ang = self.rng.uniform(0, 2 * np.pi)
            force_xy = (magnitude * np.cos(ang), magnitude * np.sin(ang))
        bid = self.model.site_bodyid[self.imu_site]
        self.data.xfrc_applied[bid, :3] = (force_xy[0], force_xy[1], 0.0)
        return np.array(force_xy)

    def apply_field(self, force_xy_heading: tuple[float, float]) -> np.ndarray:
        """A sustained force for this tick on the IMU body, given in the frame of
        that body's heading (x forward, y left, horizontal), ADDED to any push
        already applied this tick. Returns the world-frame force. Privileged, like
        `push`: the heading is the world's knowledge, never the brain's."""
        R = self.data.site_xmat[self.imu_site].reshape(3, 3)
        hx, hy = float(R[0, 0]), float(R[1, 0])        # the body x-axis, projected
        n = float(np.hypot(hx, hy)) or 1.0
        c, s = hx / n, hy / n
        fx, fy = float(force_xy_heading[0]), float(force_xy_heading[1])
        fw = np.array([c * fx - s * fy, s * fx + c * fy])
        bid = self.model.site_bodyid[self.imu_site]
        self.data.xfrc_applied[bid, :2] += fw
        return fw

    def clear_push(self) -> None:
        self.data.xfrc_applied[:] = 0.0

    # -- derived -----------------------------------------------------------
    @property
    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[self.act_qposadr].copy()

    @property
    def joint_vel(self) -> np.ndarray:
        return self.data.qvel[self.act_dofadr].copy()

    @property
    def base_height(self) -> float:
        return float(self.data.qpos[2])

    @property
    def torso_up(self) -> float:
        """z-component of the IMU site's z-axis: 1 upright, 0 horizontal."""
        return float(self.data.site_xmat[self.imu_site][8])

    def upright(self, min_up: float = 0.5, min_height: float = 0.5) -> bool:
        return self.torso_up > min_up and self.base_height > min_height

    # -- camera ------------------------------------------------------------
    def render(self, width: int = 224, height: int = 224) -> np.ndarray:
        """RGB uint8 (height, width, 3) from the head camera. Raises if the model
        was built without one or no OpenGL context is available."""
        if not self.has_head_camera:
            raise RuntimeError("built without head_camera=True")
        if self._renderer is not None and (self._renderer.height, self._renderer.width) != (height, width):
            self.close_renderer()
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._renderer_thread = threading.get_ident()
        self._renderer.update_scene(self.data, camera=HEAD_CAMERA)
        return np.asarray(self._renderer.render()).copy()

    def render_chase(self, width: int = 640, height: int = 360, *, body: str = "torso_link",
                     distance: float = 3.6, azimuth: float = 15.0, elevation: float = -18.0) -> np.ndarray:
        """RGB uint8 (height, width, 3) from a camera that follows the robot from behind
        and to one side -- for people watching (videos, demos), never for the brain:
        no reply carries it. Its own renderer, so the head camera's is undisturbed."""
        r = getattr(self, "_chase_renderer", None)
        if r is not None and (r.height, r.width) != (height, width):
            r.close()
            r = None
        if r is None:
            r = mujoco.Renderer(self.model, height=height, width=width)
            self._chase_renderer, self._renderer_thread = r, threading.get_ident()
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
            cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
            self._chase_cam = cam
        r.update_scene(self.data, camera=self._chase_cam)
        return np.asarray(r.render()).copy()

    def close_renderer(self) -> None:
        """Release the OpenGL context. Must run on the thread that created it:
        a GLFW context destroyed from another thread aborts the process, so a
        foreign-thread call is refused (and reported) rather than attempted.
        The bridge closes the renderer at the end of each client session, in
        its own serving thread, so nothing is left for shutdown."""
        if self._renderer is None and getattr(self, "_chase_renderer", None) is None:
            return
        if threading.get_ident() != self._renderer_thread:
            raise RuntimeError("close_renderer() must be called on the thread that rendered")
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if getattr(self, "_chase_renderer", None) is not None:
            self._chase_renderer.close()
            self._chase_renderer = None

    def close(self) -> None:
        self.close_renderer()
