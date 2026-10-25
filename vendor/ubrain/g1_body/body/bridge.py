"""The tick server: the body's side of the process boundary.

    python -m body.bridge --port 5555 [--robot g1] [--seed 0] [--head-camera] [--viewer]

With `--viewer` the MuJoCo passive viewer shows the body as it is stepped. The
viewer's own ctrl+right-drag pushes the robot: that is the experimenter's hand,
applied to the world and never reported to the brain.

One client at a time (a brain is one brain); requests are served in order, one
tick per `act`. Every `state` reply carries the manifest digest so the brain can
fold it into its cache key and refuse a body it did not register against.
"""

from __future__ import annotations

import argparse
import base64
import socket
import threading
from pathlib import Path

import numpy as np

import robots
from body import protocol
from body.manifest import manifest, manifest_digest
from body.primitives import Controller
from body.robot import RobotSpec
from body.sensors import contact_dim, contact_forces, proprio_dim, proprio_layout, proprioception
from body.sim import CONTROL_HZ, SIMULATOR, MujocoSim, gallery_layout

__all__ = ["BodyServer", "main"]

CHASE_W, CHASE_H = 640, 360          # the follow camera's frame size


class BodyServer:
    """Owns one MujocoSim and serves it over TCP. `port=0` picks a free port."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, robot: str | RobotSpec = "g1",
                 entry: str | None = None, seed: int = 0, head_camera: bool = False,
                 viewer: bool = False, gallery=None, record_chase: str | None = None,
                 chase_every: int = 2):
        self.spec = robots.get(robot)
        # the follow camera's recording (for videos; never in a reply): raw RGB frames
        # appended to <record_chase>.rgb, their ticks and size in <record_chase>.json
        self._chase_path, self._chase_every = record_chase, max(1, int(chase_every))
        self._chase_ticks: list[int] = []
        self._chase_file = None
        self.sim = MujocoSim(self.spec, entry, seed=seed, head_camera=head_camera, gallery=gallery)
        self.ctl = Controller(self.sim)
        self.manifest = manifest(self.spec, self.sim.entry, gallery=self.sim.gallery)
        self.digest = manifest_digest(self.manifest)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(1)
        self.host, self.port = self._sock.getsockname()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.viewer = None
        # the experimenter's world (protocol 0.3): a pending tone, the tone heard on
        # the last tick, and scheduled pushes as (first tick, end tick, force)
        self._tone_next, self._heard = False, False
        self._pushes: list[tuple[int, int, tuple[float, float]]] = []
        self._field = (0.0, 0.0)          # protocol 0.4: the sustained heading-frame force
        if viewer:
            import mujoco.viewer   # only when asked: tests and headless runs never open a window
            self.viewer = mujoco.viewer.launch_passive(self.sim.model, self.sim.data,
                                                       show_left_ui=False, show_right_ui=False,
                                                       key_callback=self._on_key)

    # -- messages ----------------------------------------------------------
    def hello(self) -> dict:
        return {
            "type": "hello",
            "protocol": protocol.PROTOCOL_VERSION,
            "simulator": SIMULATOR,
            "robot": self.spec.name,
            "control_hz": CONTROL_HZ,
            "manifest": self.manifest,
            "manifest_digest": self.digest,
            "primitives": [{"index": p.index, "name": p.name, "n_params": p.n_params}
                           for p in self.spec.primitives],
            "proprio_layout": [[n, d] for n, d in proprio_layout(self.spec)],
            "proprio_dim": proprio_dim(self.spec),
            "contact_dim": contact_dim(self.spec),
            "has_head_camera": self.sim.has_head_camera,
            # the gallery scene: each picture and where it stands (empty by default)
            "gallery": [{"index": i, "file": Path(f).name, "x": x, "y": y}
                        for i, (f, (x, y, _)) in enumerate(zip(self.sim.gallery,
                                                               gallery_layout(len(self.sim.gallery))))],
        }

    def state(self, want_contact: bool = False, want_image: bool = False,
              image_size: list[int] | None = None) -> dict:
        msg = {
            "type": "state",
            "tick": self.sim.tick,
            "proprio": proprioception(self.sim).tolist(),
            "upright": bool(self.sim.upright()),
            "contact": contact_forces(self.sim).tolist() if want_contact else None,
            "image": None,
            "tone": bool(self._heard),
            "manifest_digest": self.digest,
        }
        if want_image:
            h, w = (image_size or [224, 224])
            img = self.sim.render(width=w, height=h)
            msg["image"] = {"shape": list(img.shape), "dtype": "uint8",
                            "b64": base64.b64encode(np.ascontiguousarray(img).tobytes()).decode()}
        return msg

    def handle(self, msg: dict) -> tuple[dict, bool]:
        """Reply for one request; the bool says whether the connection ends."""
        try:
            protocol.validate_request(msg, len(self.spec.primitives), self.spec.param_counts)
            t = msg["type"]
            if t == "reset":
                with self._locked():
                    self.sim.reset(int(msg.get("seed", 0)))
                self._tone_next, self._heard, self._pushes = False, False, []
                self._field = (0.0, 0.0)
                self._show()
                return self.state(), False
            if t == "world":
                now = self.sim.tick
                if msg.get("tone", False):
                    self._tone_next = True
                p = msg.get("push")
                if p is not None:
                    start = now + int(p["delay"])
                    self._pushes.append((start, start + int(p["ticks"]), tuple(float(x) for x in p["force"])))
                if msg.get("field") is not None:
                    self._field = tuple(float(x) for x in msg["field"])
                return {"type": "ok", "tick": now}, False
            if t == "act":
                self._show()          # the viewer's drag force lands in xfrc_applied here
                with self._locked():
                    ctrl = self.ctl.ctrl(msg["primitive"], msg.get("params", []), self.sim.tick)
                    tick = self.sim.tick
                    for a, b, f in self._pushes:          # the experimenter's scheduled pushes
                        if a <= tick < b:
                            self.sim.push(f)
                    self._pushes = [p for p in self._pushes if p[1] > tick + 1]
                    if self._field != (0.0, 0.0):              # protocol 0.4
                        self.sim.apply_field(self._field)
                    self.sim.step(ctrl)
                    self.sim.clear_push()
                    self._heard, self._tone_next = self._tone_next, False
                if self._chase_path and self.sim.tick % self._chase_every == 0:
                    self._record_chase()
                return self.state(msg.get("want_contact", False), msg.get("want_image", False),
                                  msg.get("image_size")), False
            return {"type": "bye"}, True
        except Exception as e:  # a malformed request must not kill the body
            return protocol.error(f"{type(e).__name__}: {e}"), False

    def _record_chase(self) -> None:
        frame = self.sim.render_chase(CHASE_W, CHASE_H)
        if self._chase_file is None:
            self._chase_file = open(self._chase_path + ".rgb", "wb")
        self._chase_file.write(np.ascontiguousarray(frame).tobytes())
        self._chase_ticks.append(int(self.sim.tick))

    def _finish_chase(self) -> None:
        if self._chase_file is None:
            return
        import json
        self._chase_file.close()
        self._chase_file = None
        with open(self._chase_path + ".json", "w", encoding="utf-8") as f:
            json.dump({"width": CHASE_W, "height": CHASE_H, "ticks": self._chase_ticks}, f)

    # -- viewer ------------------------------------------------------------
    @staticmethod
    def _on_key(keycode: int) -> None:
        """A key pressed in the viewer window, forwarded on stdout as `KEY <glfw code>`
        for whoever launched the body (brain_live's runner reads it). It is the
        person's channel, not the brain's: nothing here enters a state reply."""
        print(f"KEY {int(keycode)}", flush=True)

    def _locked(self):
        import contextlib
        return self.viewer.lock() if self.viewer is not None else contextlib.nullcontext()

    def _show(self) -> None:
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    # -- transport ---------------------------------------------------------
    def _serve_client(self, conn: socket.socket) -> None:
        try:
            with conn, conn.makefile("rb") as rd:
                conn.sendall(protocol.encode(self.hello()))
                for line in rd:
                    if not line.strip():
                        continue
                    try:
                        msg = protocol.decode(line)
                    except Exception as e:
                        conn.sendall(protocol.encode(protocol.error(f"bad message: {e}")))
                        continue
                    reply, done = self.handle(msg)
                    conn.sendall(protocol.encode(reply))
                    if done:
                        break
        finally:
            # The OpenGL context belongs to this thread; release it here, never
            # from shutdown() on another thread (that aborts the process).
            self._finish_chase()
            self.sim.close_renderer()

    def serve_forever(self) -> None:
        self._sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                self._serve_client(conn)
            except (ConnectionError, OSError):
                pass  # the brain went away; wait for the next one

    def start(self) -> "BodyServer":
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()
        return self

    def shutdown(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.viewer is not None:
            self.viewer.close()
        self.sim.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="serve a MuJoCo body over TCP")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--robot", default="g1", choices=robots.ROBOTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--entry", default=None)
    ap.add_argument("--head-camera", action="store_true")
    ap.add_argument("--viewer", action="store_true", help="show the body in MuJoCo's passive viewer")
    ap.add_argument("--record-chase", default=None, metavar="PATH",
                    help="record a camera following the robot (for videos): PATH.rgb + PATH.json")
    ap.add_argument("--chase-every", type=int, default=2, help="record every N ticks (2: 25 fps)")
    ap.add_argument("--gallery", default=None,
                    help="a folder of .png pictures to stand along the walking line (the gallery scene)")
    a = ap.parse_args(argv)
    gallery = sorted(str(f) for f in Path(a.gallery).glob("*.png")) if a.gallery else None
    srv = BodyServer(a.host, a.port, robot=a.robot, entry=a.entry, seed=a.seed, head_camera=a.head_camera,
                     viewer=a.viewer, gallery=gallery, record_chase=a.record_chase,
                     chase_every=a.chase_every)
    print(f"body {srv.digest} ({SIMULATOR}/{srv.spec.name}) serving on {srv.host}:{srv.port} "
          f"(mujoco {srv.manifest['mujoco']}, menagerie {srv.manifest['menagerie_commit'][:8]})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
