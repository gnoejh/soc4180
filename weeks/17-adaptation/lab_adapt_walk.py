"""lab_adapt_walk: walk the G1 into a side wind with the keys, and feel it adapt.

    uv run weeks/17-adaptation/lab_adapt_walk.py                  # viewer + keyboard
    uv run weeks/17-adaptation/lab_adapt_walk.py --learner single # one learning rate
    uv run weeks/17-adaptation/lab_adapt_walk.py --headless --seconds 30 --wind-at 5

The G1 walks under its frozen walking policy; the brain is ubrain's `adapter` profile
(the week-16 walker plus cerebellum.adapt). Press W and a steady side wind starts
pushing the pelvis sideways in the robot's own heading frame: the walking robot turns.
The cerebellum feels the TURN (the torso gyroscope's yaw rate), never the wind, and
learns a counter-turn that it adds to the walking command. Watch the status line:
"turn" falls back toward zero while "counter" grows. Press W again: the wind stops,
the counter-turn is still there, and the robot turns the OTHER way (the after-effect)
until it unlearns.

THE SYSTEM: the same two processes as week 16 (see lab_cerebellum_walk.py's diagram);
only the brain's anatomy is different:

  proprio (70) --> cerebellum.adapt: the turn e = -(gyro z) teaches the counter-turn u
  clamp (L) -----> (when clamped, e is silenced: u is still sent, and only forgets)
                       single: u <- u + eta e          dual: x_f, x_s, u = x_f + x_s
  goal (3) + cerebellum.brace + (0, 0, u) --> walk (vx, vy, wz) --> the walking policy

Keys work in the viewer window and in this console (the console is the safe route):
    W              the side wind on / off (from the robot's right, pushing it left)
    [ / ]          a weaker / stronger wind (5 N a press; 15 N walks, 20 N falls)
    L              learning on / off (off = the error clamp: the counter-turn is kept
                   and only forgets)
    Up / Down      faster / slower (0.1 m/s a press)
    Left / Right   turn left / right (0.2 rad/s a press);  C  straight again
    space          stop        R   reset the body (the brain keeps what it learned)
    X              quit
Measured with lab_adapt.py (15 N, 0.5 m/s, 30 strides of wind): no learning turns the
robot 230 degrees, a full circle in under 30 s; the dual learner 70 degrees, then -25
when the wind stops; the single learner 44 degrees, then -43.
"""

from __future__ import annotations

import argparse
import math
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

import ubrain_local

ubrain_local.load()                 # the vendored wheel onto sys.path, before `import ubrain`

from soc4180 import terminal_keys  # noqa: E402
from ubrain import Brain, Channel, Wiring, act_03, manifest_from_hello_03, profile, senses_from_state_03  # noqa: E402
from ubrain.bodies import BodyClient, free_port, kill_tree  # noqa: E402

HZ = 50
DT = 1.0 / HZ
STRIDE = 40             # ticks per stride (0.8 s): the status line averages one stride
GYRO_Z = 2              # the torso gyroscope's yaw rate in the 70 body numbers
WIND_N = 15.0           # the wind W starts with, newtons (left positive: from the robot's right)
WIND_STEP = 5.0         # [ and ] change it by this much

# GLFW key codes the viewer forwards, as key names
GLFW = {263: "left", 262: "right", 265: "fast", 264: "slow", 32: " "}


@dataclass
class Command:
    """The walk the person at the keyboard asks for: the brain's goal."""
    vx: float = 0.5     # m/s forward
    vy: float = 0.0     # m/s left
    wz: float = 0.0     # rad/s, counter-clockwise


def key_name(code: int) -> str | None:
    return GLFW.get(code) or (chr(code).lower() if 32 < code < 127 else None)


def viewer_keys(proc: subprocess.Popen, out: "queue.Queue[str]") -> threading.Thread:
    """Drain the body's stdout (so its pipe never fills) and turn the viewer's
    `KEY <code>` lines into key names; anything else the body says is shown."""
    def pump():
        for line in proc.stdout:
            if line.startswith("KEY "):
                name = key_name(int(line.split()[1]))
                if name:
                    out.put(name)
            elif line.strip():
                print(f"\n[body] {line.rstrip()}", flush=True)
    t = threading.Thread(target=pump, daemon=True)
    t.start()
    return t


def build_brain(learner: str, manifest) -> Brain:
    w = profile("adapter")
    regions = {k: dict(v) for k, v in w.regions.items()}
    if learner == "single":
        regions["cerebellum.adapt"] = {"arm": "single"}
    return Brain(Wiring(w.name, w.contract, regions=regions, inputs=dict(w.inputs),
                        outputs=dict(w.outputs)), manifest)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="walk the G1 into a side wind; its cerebellum adapts")
    ap.add_argument("--learner", choices=("dual", "single"), default="dual")
    ap.add_argument("--headless", action="store_true", help="no viewer, no keyboard")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = until quit)")
    ap.add_argument("--wind-at", type=float, default=0.0,
                    help="turn the wind on by itself after this many seconds (0 = never)")
    a = ap.parse_args(argv)

    port = free_port()
    proc, _ = ubrain_local.start_body(port, viewer=not a.headless)
    keys: "queue.Queue[str]" = queue.Queue()
    viewer_keys(proc, keys)
    if not a.headless:
        terminal_keys(lambda code: keys.put(key_name(code)) if key_name(code) else None, quit_tokens=())
    try:
        body = BodyClient("127.0.0.1", port)
        manifest = manifest_from_hello_03(body.hello).with_channels(Channel("clamp", "flag", 1))
        brain = build_brain(a.learner, manifest)
        c = Command()
        wind_on, wind, learning = False, WIND_N, True
        heading = 0.0                           # degrees turned since the start (integrated gyro)
        turns, counters = [], []
        falls, fallen_since = 0, None

        def send_wind():
            body._send({"type": "world", "field": [0.0, wind if wind_on else 0.0]})

        state = body.reset(0)
        if not a.headless:
            print(__doc__.split("Keys work")[1].split("Measured with")[0].strip(), flush=True)
        print(f"learner {a.learner}; press W for the wind", flush=True)
        t0 = next_t = last_print = time.perf_counter()
        while True:
            now = time.perf_counter()
            if a.wind_at and not wind_on and now - t0 >= a.wind_at:
                keys.put("w")
                a.wind_at = 0.0
            while not keys.empty():
                k = keys.get_nowait()
                if k == "fast": c.vx = round(min(c.vx + 0.1, 0.8), 2)
                elif k == "slow": c.vx = round(max(c.vx - 0.1, -0.4), 2)
                elif k == "left": c.wz = round(min(c.wz + 0.2, 1.0), 2)
                elif k == "right": c.wz = round(max(c.wz - 0.2, -1.0), 2)
                elif k == "c": c.wz = 0.0
                elif k == " ": c.vx = c.vy = c.wz = 0.0
                elif k == "w":
                    wind_on = not wind_on
                    send_wind()
                elif k in ("[", "]"):
                    wind = wind + (WIND_STEP if k == "]" else -WIND_STEP)
                    send_wind()
                elif k == "l": learning = not learning
                elif k == "r":
                    state, fallen_since, heading = body.reset(0), None, 0.0
                    brain.body_reset()
                    send_wind()                 # a reset clears the body's wind; put it back
                elif k in ("x", "\x1b"):
                    raise KeyboardInterrupt

            out = brain.tick(senses_from_state_03(state) | {"goal": (c.vx, c.vy, c.wz), "clamp": not learning})
            state = body.act(*act_03(manifest, out))
            turn = state["proprio"][GYRO_Z]
            heading += math.degrees(turn * DT)
            turns, counters = (turns + [turn])[-STRIDE:], (counters + [out["walk"][2] - c.wz])[-STRIDE:]

            if not state["upright"]:
                if fallen_since is None:
                    falls, fallen_since = falls + 1, now
                    print(f"\n  fell (wind {wind if wind_on else 0:+.0f} N): resetting in 2 s", flush=True)
                elif now - fallen_since > 2.0:
                    state, fallen_since, heading = body.reset(0), None, 0.0
                    brain.body_reset()
                    send_wind()

            if now - last_print > 0.5:
                fields = [f"{now - t0:6.1f}s",
                          f"wind {wind if wind_on else 0:+3.0f}N",
                          f"turn {sum(turns) / len(turns):+.3f}",
                          f"counter {sum(counters) / len(counters):+.3f}",
                          f"heading {heading:+6.0f}deg",
                          f"learn {'on ' if learning else 'off'}",
                          f"vx{c.vx:+.1f}", "up" if state["upright"] else "DOWN", f"falls {falls}"]
                width = shutil.get_terminal_size((80, 24)).columns - 1
                line = " ".join(fields)
                while len(line) > width and len(fields) > 1:
                    fields.pop()
                    line = " ".join(fields)
                print("\r" + line.ljust(width), end="", flush=True)
                last_print = now
            if a.seconds and now - t0 >= a.seconds:
                break
            next_t += DT
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()    # do not try to catch up
        print(f"\nran {a.seconds:.0f} s; heading {heading:+.0f} deg; falls {falls}", flush=True)
        body.close()
    except KeyboardInterrupt:
        print("\nquit", flush=True)
    finally:
        kill_tree(proc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
