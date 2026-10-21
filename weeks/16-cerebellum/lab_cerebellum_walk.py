"""lab_cerebellum_walk: drive the G1 with the keys; its cerebellum learns as it walks.

    uv run weeks/16-cerebellum/lab_cerebellum_walk.py              # viewer + keyboard
    uv run weeks/16-cerebellum/lab_cerebellum_walk.py --headless --seconds 10 --vx 0.5

Copied from ubrain's examples/g1_walker.py. ubrain itself is in
weeks/16-cerebellum/vendor/ (see ubrain_local.py); nothing else beyond
`uv sync`.

It starts the MuJoCo body (vendor/g1_body, on this repository's Python) as a child process
and drives the brain, `ubrain.Brain` with the `walker` profile, against it at 50 Hz
in wall-clock time. It
is a runner, not a second brain. The brain's cerebellum predicts every tick
what its command will do to the body and learns from the error; the status line
shows that error as a share of predicting no change (watch it fall as the robot
walks, jump when you change the command, and fall again).

THE SYSTEM: two processes, one tick every 20 ms

  terminal keys ---(terminal_keys thread)----+
                                             v
                                           queue --> apply_key --> Command(vx, vy, wz)
                                             ^                            |  the goal
  viewer keys ---(the body prints "KEY 265"; |                            |
                  viewer_keys thread)--------+                            v
  +-- THIS SCRIPT: the runner, a 50 Hz loop in main() ---------------------------+
  |                                                                              |
  |   senses = senses_from_state_03(state) + goal --> brain.tick --> act_03      |
  |     ^                                                              |         |
  +-----|--------------------------------------------------------------|---------+
        |  state: proprio (70), upright, tone, tick                    |  action (32)
        |            TCP (body.act), every tick                        v  walk_offset
  +-----|-- THE BODY: a child process, vendor/g1_body ---------------------------+
  |   MuJoCo G1 at 500 Hz: 10 physics steps per tick                             |
  |   frozen walking policy, the "spinal cord": (vx, vy, wz) -> 29 joint targets |
  |   the experimenter's hand: P = tone, then push; ctrl + right-drag (viewer)   |
  +------------------------------------------------------------------------------+

INSIDE brain.tick (the walker profile; `print(profile("walker"))` shows the wiring)

  proprio (70) --+--> cerebellum.forward: predicts the next proprio and learns from
  goal (3) ------+    its error (granule cells, the k most active fire, read out by
                      Purkinje cells, delta rule). Its error is "pred" on the status
                      line. It only predicts: nothing it computes moves the robot.

  tone ----------+--> cerebellum.brace: a fading trace of the tone, credited when the
  proprio (70) --+    push jolts the body --> counter-move (3) --+   (B: on / off)
                                                                 v
  goal (3) -------------------------------------------------->  (+) --> walk (vx, vy, wz)
                                                   offsets = 29 zeros --> joint offsets

Keys work in the viewer window and in this console (the console is the safe route:
a viewer window does not take keys on every laptop). Each press adds one step:
    Up / Down      faster / slower (0.1 m/s a press; Down past zero walks backward)
    Left / Right   turn left / right (0.2 rad/s a press)
    C              go straight (turning to zero)
    Q / E          sidestep left / right (0.1 m/s a press)
    space          stop        R   reset the body        X   quit
    P              the tone, then a push from behind 0.2 s later (250 N):
                   press it again and again, and watch the robot learn to brace
    B              bracing on / off (it keeps learning either way), to compare
    (W/S/A/D work as Up/Down/Left/Right too)
In the viewer, ctrl + right-drag pushes the robot (the experimenter's hand: the
world changes, and the brain only feels the consequences).
"""

from __future__ import annotations

import argparse
import queue
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass

import ubrain_local

ubrain_local.load()                 # vendor/'s wheel onto sys.path, before `import ubrain`

from soc4180 import terminal_keys  # noqa: E402
from ubrain.bodies import BodyClient, free_port, kill_tree  # noqa: E402
from ubrain import Brain, act_03, manifest_from_hello_03, profile, senses_from_state_03  # noqa: E402



@dataclass
class Command:
    """What the person at the keyboard wants the body to do: walking velocity in the
    body's frame -- the brain's goal."""
    vx: float = 0.0     # m/s forward
    vy: float = 0.0     # m/s left
    wz: float = 0.0     # rad/s, counter-clockwise


HZ = 50                 # 50 Hz/brain 500Hz/body
DT = 1.0 / HZ
PUSH_N = 250.0          # the push, newtons: below ~220 it never falls; at 265 the brace
                        # no longer saves it (measured, 12 trials)
PUSH_DELAY = 10         # ticks from the tone to the push (0.2 s)
PUSH_DIR = (1.0, 0.0)   # world x, y: +x is from behind while it faces its start direction
                        # (-x from the front, +y from its right, -y from its left)


def port_in_use(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_body(port: int, viewer: bool) -> subprocess.Popen:
    """The body server (ubrain_local.start_body); returns once it serves."""
    proc, _ = ubrain_local.start_body(port, viewer=viewer)
    return proc


# GLFW key codes the viewer forwards, as the runner's key names
GLFW = {263: "left", 262: "right", 265: "fast", 264: "slow", 32: " "}


def connect(proc: subprocess.Popen, port: int, wait: float = 10.0) -> BodyClient:
    """Connect to the body, retrying while it comes up; if the body process has
    died, say so (its own output is printed by the pump as [body] lines)."""
    t0 = time.perf_counter()
    while True:
        try:
            return BodyClient("127.0.0.1", port)
        except OSError as e:
            if proc.poll() is not None:
                raise RuntimeError(f"the body exited with code {proc.returncode} "
                                   f"(its messages are above, marked [body])") from e
            if time.perf_counter() - t0 > wait:
                raise RuntimeError(f"could not connect to the body on port {port} within {wait:.0f} s: "
                                   f"is another program using the port? try --port 5556") from e
            time.sleep(0.2)


def key_name(code: int) -> str | None:
    return GLFW.get(code) or (chr(code).lower() if 32 < code < 127 else None)


def viewer_keys(proc: subprocess.Popen, out: "queue.Queue[str]") -> threading.Thread:
    """Drain the body's stdout (so its pipe never fills) and turn the viewer's
    `KEY <code>` lines into key names."""
    def pump():
        for line in proc.stdout:
            if line.startswith("KEY "):
                code = int(line.split()[1])
                name = key_name(code)
                if name:
                    out.put(name)
            elif line.strip():
                # anything else the body says (a warning, a traceback) is shown, never swallowed
                print(f"\n[body] {line.rstrip()}", flush=True)
    t = threading.Thread(target=pump, daemon=True)
    t.start()
    return t


def apply_key(brain: Brain, k: str) -> str | None:
    c = brain.command
    k = {"w": "fast", "s": "slow", "a": "left", "d": "right"}.get(k, k)
    if k == "fast": c.vx = round(min(c.vx + 0.1, 0.8), 2)
    elif k == "slow": c.vx = round(max(c.vx - 0.1, -0.4), 2)
    elif k == "left": c.wz = round(min(c.wz + 0.2, 1.0), 2)
    elif k == "right": c.wz = round(max(c.wz - 0.2, -1.0), 2)
    elif k == "c": c.wz = 0.0
    elif k == "q": c.vy = min(c.vy + 0.1, 0.4)
    elif k == "e": c.vy = max(c.vy - 0.1, -0.4)
    elif k == " ": c.vx = c.vy = c.wz = 0.0
    elif k == "r": return "reset"
    elif k == "p": return "push"
    elif k == "b": brain.bracing = not brain.bracing
    elif k in ("x", "\x1b"): return "quit"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="the brain in the robot, live")
    ap.add_argument("--port", type=int, default=0, help="0 (default): any free port")
    ap.add_argument("--headless", action="store_true", help="no viewer, no keyboard")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = until quit)")
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    a = ap.parse_args(argv)

    if a.port == 0:
        a.port = free_port()       # the default: a fresh port every run, so an old body can never answer
    elif port_in_use(a.port):
        new = free_port()
        print(f"port {a.port} is already taken (an old body still running? an old body may still hold it); "
              f"using {new}", flush=True)
        a.port = new
    body_proc = start_body(a.port, viewer=not a.headless) # Body server process
    from_viewer: "queue.Queue[str]" = queue.Queue()
    viewer_keys(body_proc, from_viewer)
    if not a.headless:
        # the console, as a second route into the same queue (Windows, macOS, Linux);
        # no quit word: Q is sidestep here, X quits
        terminal_keys(lambda code: from_viewer.put(key_name(code)) if key_name(code) else None,
                      quit_tokens=())
    falls = overruns = 0
    pushes = saved = 0
    watch_until = None          # the tick by which a push must have been survived
    try:
        with connect(body_proc, a.port) as body: # Connect to the body server
            print(f"body {body.digest}: {body.hello['simulator']}/{body.hello['robot']}, "
                  f"{len(body.primitives)} primitives", flush=True)
            if not a.headless:
                print(__doc__.split("Keys work")[1].split("In the viewer")[0].strip(), flush=True)
            manifest = manifest_from_hello_03(body.hello)       # the body, as contract 1.0
            brain = Brain(profile("walker"), manifest)
            brain.command = Command()                           # the goal: the keyboard's walk
            brain.command.vx, brain.command.wz = a.vx, a.wz
            state = body.reset(0)
            t0 = next_t = time.perf_counter()
            last_print, work_max, fallen_since = t0, 0.0, None
            while True:
                pressed = []
                while not from_viewer.empty():
                    pressed.append(from_viewer.get_nowait())
                for k in pressed:
                    what = apply_key(brain, k)
                    if what == "quit":
                        raise KeyboardInterrupt
                    if what == "reset":
                        state = body.reset(0)
                        brain.body_reset()
                        fallen_since = None
                    if what == "push" and watch_until is None and state["upright"]:
                        # the experimenter's hand (protocol 0.3): the brain hears the tone,
                        # never the push
                        force = [PUSH_N * PUSH_DIR[0], PUSH_N * PUSH_DIR[1]]   # world frame
                        body._send({"type": "world", "tone": True,
                                    "push": {"force": force, "delay": PUSH_DELAY, "ticks": 5}})
                        pushes += 1
                        watch_until = state["tick"] + PUSH_DELAY + 150
                w0 = time.perf_counter()
                c = brain.command
                # One tick, sense -> think -> act (shapes measured on a real tick):
                #   state  (from the body): proprio = 70 numbers (two IMUs: gyro + accelerometer,
                #          3 axes each = 12; 29 joint angles; 29 joint velocities), plus upright,
                #          tone and tick
                #   senses (to the brain):  proprio (70), upright, tone, and goal (3) = the keys' command
                #   brain.tick, as the walker profile wires it:
                #     inputs   cerebellum.forward <- proprio, goal
                #              cerebellum.brace   <- proprio, tone
                #     outputs  walk    = goal + cerebellum.brace   (3: vx, vy, wz)
                #              offsets = zero                      (29 joint offsets)
                #   act_03 packs that as primitive 4, walk_offset (3 + 29 = 32 numbers);
                #   body.act sends it, the body steps 20 ms (10 physics steps), and replies
                #   with the next state.
                senses = senses_from_state_03(state) | {"goal": (c.vx, c.vy, c.wz)}
                state = body.act(*act_03(manifest, brain.tick(senses)))
                work = time.perf_counter() - w0
                work_max = max(work_max, work)
                now = time.perf_counter()
                if watch_until is not None and state["upright"] and state["tick"] >= watch_until:
                    saved, watch_until = saved + 1, None
                if not state["upright"]:
                    watch_until = None
                    if fallen_since is None:
                        falls += 1
                        fallen_since = now
                        print(f"\n  fell at tick {state['tick']}: resetting in 2 s", flush=True)
                    elif now - fallen_since > 2.0:
                        state, fallen_since = body.reset(0), None
                        brain.body_reset()
                if now - last_print > 0.5:
                    c = brain.command
                    fw = brain.forward
                    r = fw.stride_ratio if fw is not None else None
                    pred = " watch" if r is None else f"{100 * r:5.1f}%"
                    # fitted to the console's width: a line that wraps cannot be overwritten
                    # by the carriage return, and its tail lands on a row nobody sees
                    fields = [f"{now - t0:6.1f}s vx{c.vx:+.1f} wz{c.wz:+.1f}",
                              "up  " if state["upright"] else "DOWN",
                              f"pred{pred}",
                              f"brace {'on ' if brain.bracing else 'off'}",
                              f"saved {saved}/{pushes}",
                              f"falls {falls}"]
                    width = shutil.get_terminal_size((80, 24)).columns - 1
                    line = " ".join(fields)
                    while len(line) > width and len(fields) > 1:
                        fields.pop()
                        line = " ".join(fields)
                    print("\r" + line[:width].ljust(width), end="", flush=True)
                    last_print, work_max = now, 0.0
                if a.seconds and now - t0 >= a.seconds:
                    break
                next_t += DT
                sleep = next_t - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    overruns += 1
                    next_t = time.perf_counter()        # do not try to catch up
            print(f"\nran {state['tick']} ticks in {time.perf_counter() - t0:.1f} s wall; "
                  f"overruns {overruns}; falls {falls}", flush=True)
    except KeyboardInterrupt:
        print("\nquit", flush=True)
    finally:
        kill_tree(body_proc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
