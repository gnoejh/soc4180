"""The tick protocol between brain and body. PURE STDLIB: the engine's adapter
mirrors this file and must not need MuJoCo or numpy to parse a message.

Transport: one TCP connection, newline-delimited JSON, one request -> one reply.

  server -> client on connect
    {"type": "hello", "protocol": "0.2", "simulator": "mujoco", "robot": "g1",
     "control_hz": 50, "manifest": {...}, "manifest_digest": "...",
     "primitives": [{"index": 0, "name": "stand", "n_params": 0}, ...],
     "proprio_layout": [["imu_torso_gyro", 3], ...], "proprio_dim": 70,
     "contact_dim": 12, "has_head_camera": bool}
    (0.2, 2026-10-02: `simulator` and `robot` added, so a brain can tell which
    body answered before it reads the manifest; the layout and dims are the
    robot's, not constants of the protocol)

  client -> server
    {"type": "reset", "seed": int}
    {"type": "act", "primitive": int, "params": [float...],
     "want_contact": bool, "want_image": bool,
     "image_size": [height, width]}          (image_size optional)
    {"type": "close"}
    {"type": "world", "tone": bool,                      (0.3: the experimenter's hand)
     "push": {"force": [fx, fy], "delay": int, "ticks": int} | null,
     "field": [fx, fy] | null}                           (0.4, optional)

  server -> client
    {"type": "state", "tick": int, "proprio": [70 floats], "upright": bool,
     "contact": [12 floats] | null,           (privileged; only if asked)
     "image": {"shape": [h, w, 3], "dtype": "uint8", "b64": "..."} | null,
     "manifest_digest": "..."}
    {"type": "bye"}
    {"type": "ok", "tick": int}                          (the reply to `world`)
    {"type": "error", "message": "..."}

  (0.3, 2026-10-13, Phase XI.2) `world` changes the world, not the body's
  command. A tone sounds during the next tick and is HEARD: the `state` reply after
  that tick carries "tone": true (every state reply carries the field). A push of
  `force` newtons on the pelvis acts over `ticks` ticks, starting `delay` ticks
  after the tone's tick. It is never reported: it is privileged, like the viewer's
  drag. The brain's client has no method that sends `world`; the experimenter's
  harness does (embody/experimenter.py, brain_live's runner).

  (0.4, 2026-10-16, Phase XI.3) `field` sets a SUSTAINED force on the pelvis, in
  newtons, in the frame of the pelvis heading (x forward, y left), applied every
  tick from the next `act` on until another `world` request changes it; [0, 0]
  switches it off, and a request without the key leaves it as it is. It is a
  force field that turns with the walker (a crosswind on one side, a load the
  body carries), so the perturbation stays the same whatever the heading. Like a
  push it is never reported. `reset` switches it off.

There is deliberately NO request that returns privileged state (masses,
friction, applied forces, the world's poses): body/sensors.PRIVILEGED names it
and tests/test_bridge.py asserts none of those words appears in any reply.
"""

from __future__ import annotations

import json

PROTOCOL_VERSION = "0.4"

REQUEST_TYPES = ("reset", "act", "close", "world")
REPLY_TYPES = ("hello", "state", "bye", "ok", "error")


def encode(msg: dict) -> bytes:
    return (json.dumps(msg, separators=(",", ":")) + "\n").encode()


def decode(line: bytes | str) -> dict:
    if isinstance(line, bytes):
        line = line.decode()
    msg = json.loads(line)
    if not isinstance(msg, dict) or "type" not in msg:
        raise ValueError("message must be a JSON object with a 'type'")
    return msg


def validate_request(msg: dict, n_primitives: int, param_counts: list[int]) -> None:
    """Raise ValueError on a malformed request; return None if well-formed."""
    t = msg.get("type")
    if t not in REQUEST_TYPES:
        raise ValueError(f"unknown request type {t!r}")
    if t == "reset":
        if not isinstance(msg.get("seed", 0), int):
            raise ValueError("reset.seed must be an integer")
    elif t == "act":
        p = msg.get("primitive")
        if not isinstance(p, int) or not 0 <= p < n_primitives:
            raise ValueError(f"act.primitive must be an int in [0, {n_primitives})")
        params = msg.get("params", [])
        if not isinstance(params, list) or len(params) != param_counts[p]:
            raise ValueError(f"act.params must be a list of {param_counts[p]} numbers")
        for k in ("want_contact", "want_image"):
            if not isinstance(msg.get(k, False), bool):
                raise ValueError(f"act.{k} must be a bool")
        size = msg.get("image_size")
        if size is not None and not (isinstance(size, list) and len(size) == 2
                                     and all(isinstance(s, int) and s > 0 for s in size)):
            raise ValueError("act.image_size must be [height, width] of positive ints")
    elif t == "world":
        if not isinstance(msg.get("tone", False), bool):
            raise ValueError("world.tone must be a bool")
        push = msg.get("push")
        if push is not None:
            f = push.get("force") if isinstance(push, dict) else None
            if not (isinstance(f, list) and len(f) == 2 and all(isinstance(x, (int, float)) for x in f)):
                raise ValueError("world.push.force must be [fx, fy] in newtons")
            for k in ("delay", "ticks"):
                v = push.get(k)
                if not isinstance(v, int) or v < (1 if k == "ticks" else 0):
                    raise ValueError(f"world.push.{k} must be an int (ticks >= 1, delay >= 0)")
        field = msg.get("field")
        if field is not None and not (isinstance(field, list) and len(field) == 2
                                      and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                              for x in field)):
            raise ValueError("world.field must be [fx, fy] in newtons (pelvis heading frame) or null")


def error(message: str) -> dict:
    return {"type": "error", "message": str(message)}
