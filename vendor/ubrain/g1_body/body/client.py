"""The reference client -- PURE STDLIB, no MuJoCo, no numpy -- so the engine's
`embody/` adapter can mirror it byte for byte (with a digest, the quarry rule)
without pulling the body's dependencies into the brain's environment.

    from body.client import BodyClient
    with BodyClient("127.0.0.1", 5555) as body:
        body.hello["manifest_digest"]          # fold into the cache key
        s = body.reset(seed=3)                 # -> state dict
        s = body.act(0)                        # stand one tick
        s = body.act(2, [0.2, 100], want_contact=True)
"""

from __future__ import annotations

import base64
import socket

from . import protocol      # relative: the engine's embody/ copies this file byte for byte

__all__ = ["BodyClient", "decode_image"]


def decode_image(image: dict) -> tuple[list[int], bytes]:
    """(shape, raw uint8 bytes) -- reshape with numpy on the caller's side."""
    return list(image["shape"]), base64.b64decode(image["b64"])


class BodyClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout: float = 30.0):
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._rd = self._sock.makefile("rb")
        self.hello = self._recv()
        if self.hello.get("type") != "hello":
            raise RuntimeError(f"expected hello, got {self.hello!r}")
        if self.hello.get("protocol") != protocol.PROTOCOL_VERSION:
            raise RuntimeError(f"protocol {self.hello.get('protocol')} != {protocol.PROTOCOL_VERSION}")
        self.digest: str = self.hello["manifest_digest"]
        self.primitives = {p["name"]: p for p in self.hello["primitives"]}

    def _send(self, msg: dict) -> dict:
        self._sock.sendall(protocol.encode(msg))
        return self._recv()

    def _recv(self) -> dict:
        line = self._rd.readline()
        if not line:
            raise ConnectionError("body closed the connection")
        return protocol.decode(line)

    def _checked(self, reply: dict) -> dict:
        if reply.get("type") == "error":
            raise RuntimeError(reply.get("message"))
        if reply.get("manifest_digest", self.digest) != self.digest:
            raise RuntimeError("the body changed identity mid-session")
        return reply

    def reset(self, seed: int = 0) -> dict:
        return self._checked(self._send({"type": "reset", "seed": int(seed)}))

    def act(self, primitive: int | str, params: list[float] | None = None, *,
            want_contact: bool = False, want_image: bool = False,
            image_size: tuple[int, int] | None = None) -> dict:
        if isinstance(primitive, str):
            primitive = self.primitives[primitive]["index"]
        msg = {"type": "act", "primitive": int(primitive), "params": list(params or []),
               "want_contact": bool(want_contact), "want_image": bool(want_image)}
        if image_size is not None:
            msg["image_size"] = [int(image_size[0]), int(image_size[1])]
        return self._checked(self._send(msg))

    def close(self) -> None:
        try:
            self._send({"type": "close"})
        except (ConnectionError, OSError):
            pass
        finally:
            self._rd.close()
            self._sock.close()

    def __enter__(self) -> "BodyClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
