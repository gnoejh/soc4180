"""Lab: make the G1 act -- body, arms, hands and legs, one numbered action at a time.

HOW TO USE IT, step by step
---------------------------
1. Start it (a window opens with the G1 standing; the terminal shows the menu):

       uv run weeks/projects/templates/actions_demo.py

2. Click on the TERMINAL (not the 3-D window), type a number and press Enter:

       9      the G1 waves            (it finishes the action, then waits for the next number)
       d      the G1 plays DANCE      (the list in section 2 below)
       r      start again             (after a fall, or any time)
       l      show the menu again
       q      quit                    (or close the window)

   After every action the terminal prints one line: stood or FELL, the largest tilt
   of the pelvis, and the action's own number (palm error, ball distance, ...).
   In the window: drag with the left mouse to turn the view, right mouse to move it,
   the wheel to zoom.

3. Change a parameter: edit the line for that number in ACTIONS (section 1), save,
   and start the script again. Example: make 2 (squat) deeper with depth=0.18.

4. Make a dance: edit DANCE (section 2) -- a list of (snapshot, seconds). Snapshot
   names are listed when you type `s`; join two with '+' ("arms_up+squat"). Before
   playing, the script checks the dance and says what is wrong with it.

5. Change the balance layer (section 3) and see what it was doing.

Without a window (the checker, or a laptop with no display):

       uv run weeks/projects/templates/actions_demo.py --no-viewer --action 9
       uv run weeks/projects/templates/actions_demo.py --no-viewer --all
       uv run weeks/projects/templates/actions_demo.py --no-viewer --dance --tempo 1.2

TRY THIS (one change at a time; write down what you expect first). Each result was
measured with this file:

    a. 2 squat, depth 0.12 -> 0.18: stands (pelvis 0.54 m). 0.21: FALLS.
    b. 19 kick, strength 0.5 / 0.75 / 1.0 / 1.25 / 1.5: the ball rolls
       0.55 / 1.43 / 2.27 / 1.37 / 1.06 m; strength 1.75 FALLS. Harder is not farther: why?
    c. 15 step, dx 0.10 -> 0.22: stands. 0.25: FALLS.
    d. 11 reach to (1.0, -0.2, 1.0): stands, but the palm ends 0.94 m short.
       Out of reach is not an error -- read the number.
    e. Section 3, BALANCE = False, then play DANCE_TEST ("arms_forward+bow" held):
       FALLS. With balance on: stands. What did the pelvis do?
    f. Section 3, GRAVITY_FF = False, then 14 (one leg): FALLS. 2 (squat) still stands.
    g. Section 3, SHIFT_SECONDS 1.5 -> 1.0: 14 still stands. 0.8: FALLS.
    h. Section 3, ANKLE_GAIN = ANKLE_DAMP = COM_GAIN = 0, then 16 (walk): FALLS at 4.6 s,
       as the first step ends.
    i. d with --tempo 1.35: stands. 1.4: FALLS. (That is project P15.)
    j. In DANCE, change ("on_left+arms_out", 1.5) to ("on_left+right_out", 1.5): the check
       warns, and the dance FALLS at tempo 1.0 (the left arm now swings out while the right
       knee lifts). It stands at 1.2: slower is not always safer.

The actions are methods of `soc4180.actions.G1`; its docstring explains each idea
(the robot's frame, joint space / task space / balance, snapshots). Use them in your
own project code exactly as section 1 does: `g1 = G1(); g1.wave("left")`.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time

import soc4180
from soc4180.actions import G1, SNAPSHOTS

# =============================================================================
# 1. THE ACTIONS: number -> (method of G1, its parameters). Change the parameters.
#    Positions are metres in the robot's frame: x forward, y left, z up, from
#    the floor between the feet. Angles are radians (0.5 rad = 29 degrees).
# =============================================================================
ACTIONS = {
    # body
    1: ("home", dict()),                                        # crouch, arms down, hands open
    2: ("squat", dict(depth=0.12, seconds=1.5)),                # pelvis down `depth` m and back up
    3: ("stand_tall", dict(height=0.76)),                       # straighter legs (0.79 = straight)
    4: ("lean", dict(pitch=0.25, roll=0.0)),                    # tilt the pelvis forward, come back
    5: ("sway", dict(amount=0.04, count=2)),                    # weight left and right
    6: ("bow", dict(angle=0.45)),                               # bend at the waist (max 0.52)
    7: ("twist", dict(angle=0.8)),                              # turn the upper body (+ = left)
    # arms and hands
    8: ("raise_arms", dict(which="both", angle=2.9)),           # 1.5 = forward, 2.9 = straight up
    9: ("wave", dict(arm="right", count=3, speed=1.0)),         # speed = waves per second
    10: ("hands", dict(close=1.0, which="both")),               # 0 open .. 1 fist
    11: ("reach", dict(hand="right", point=(0.35, -0.20, 0.95))),  # palm to a point
    12: ("point_at", dict(hand="right", point=(1.5, -0.5, 0.8))),  # arm aimed at a far point
    13: ("clap", dict(count=3, speed=1.5)),
    # legs (slow on purpose: the weight moves over one foot first)
    14: ("one_leg", dict(stand_on="left", lift=0.08, hold=2.0)),
    15: ("step", dict(foot="left", dx=0.10, dy=0.0)),           # one step; feet stay apart after
    16: ("walk", dict(steps=2, step_length=0.10)),              # pairs of slow steps
    17: ("side_step", dict(distance=0.08)),                     # + = to the left
    18: ("turn", dict(angle=0.8)),                              # + = to the left
    19: ("kick", dict(foot="right", strength=1.0)),             # a ball appears in front of the foot
}

# =============================================================================
# 2. THE DANCE: (snapshot, seconds). Type `s` in the menu for every snapshot name.
#    Rules the check looks for: a foot lifts only after the weight is on the other
#    foot ("on_left" before "right_knee_up"), and on one foot move one thing at a time
#    (arms OR a foot). Measured: stands at every tempo from 0.5 to 1.35, falls from 1.4.
# =============================================================================
DANCE = [
    ("arms_out", 1.0),
    ("arms_up+sway_left", 1.0),
    ("arms_up+sway_right", 1.0),
    ("arms_bent+half_squat", 1.0),
    ("arms_out+twist_left", 1.0),
    ("arms_out+twist_right", 1.0),
    ("on_left+arms_out", 1.5),
    ("right_knee_up+arms_out", 1.0),
    ("on_left+arms_out", 1.0),
    ("arms_out", 1.5),
    ("on_right+arms_out", 1.5),
    ("left_knee_up+arms_out", 1.0),
    ("on_right+arms_out", 1.0),
    ("arms_out", 1.5),
    ("right_wave_a+bow", 1.0),
    ("right_wave_b+bow", 0.5),
    ("right_wave_a", 0.5),
    ("home", 1.0),
]
DANCE_TEST = [("arms_forward+bow", 1.5), ("arms_forward+bow", 1.0), ("home", 1.5)]   # try e.

# =============================================================================
# 3. THE BALANCE LAYER (see soc4180/actions.py). Defaults are the measured ones.
# =============================================================================
BALANCE = True          # keep the centre of mass over the feet
GRAVITY_FF = True       # servo targets lead by their sag under load
SHIFT_SECONDS = 1.5     # time to move the weight onto one foot
ANKLE_GAIN = 2.0        # ankle reflex: rad per metre of centre-of-mass error
ANKLE_DAMP = 0.3        # ankle reflex: rad per m/s
COM_GAIN = 0.5          # correct the centre-of-mass target by the measured error


# =============================================================================
# 4. The lab machinery: you do not need to change anything below.
# =============================================================================
def make_robot(on_tick=None) -> G1:
    g1 = G1(on_tick=on_tick)
    for name in ("GRAVITY_FF", "SHIFT_SECONDS", "ANKLE_GAIN", "ANKLE_DAMP", "COM_GAIN"):
        setattr(g1, name, globals()[name])
    if not BALANCE:
        g1.set_balance(False)
    return g1


def menu() -> str:
    lines = ["", "Type a number and press Enter:"]
    for n, (name, kw) in ACTIONS.items():
        args = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        lines.append(f"  {n:2d}  {name}({args})")
    lines.append("   d  play DANCE        t  play DANCE_TEST       s  list the snapshots")
    lines.append("   r  start again       l  this menu             q  quit")
    return "\n".join(lines)


def run_action(g1: G1, n: int):
    name, kw = ACTIONS[n]
    return getattr(g1, name)(**kw)


def run_dance(g1: G1, dance, tempo: float):
    problems = g1.check(dance)
    for p in problems:
        print("  check:", p)
    if problems:
        print("  (playing it anyway, so you can see what happens)")
    return g1.play(dance, tempo=tempo)


def headless(args) -> int:
    g1 = make_robot()
    if args.all:
        for n in ACTIONS:
            g1.reset()
            print(f"{n:2d} {run_action(g1, n)}", flush=True)
    if args.action is not None:
        print(run_action(g1, args.action))
    if args.dance:
        print(run_dance(g1, DANCE, args.tempo))
    return 0


def interactive(args) -> int:
    import mujoco  # noqa: F401  (soc4180 has chosen the GL backend already)

    commands: queue.Queue[str] = queue.Queue()

    def read_terminal():
        while True:
            try:
                line = input()
            except (EOFError, OSError):
                commands.put("q")
                return
            commands.put(line.strip().lower())

    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    clock = {"wall": time.time(), "sim": 0.0}

    def tick(g):                                  # every control step: draw, keep real time
        if not viewer.is_running() or time.time() > deadline:
            raise KeyboardInterrupt
        viewer.sync()
        ahead = (g.data.time - clock["sim"]) - (time.time() - clock["wall"])
        if ahead > 0:
            time.sleep(ahead)

    g1 = make_robot()
    with soc4180.launch_viewer(g1.model, g1.data, passive=True) as viewer:
        g1.on_tick = tick
        print(menu(), flush=True)
        threading.Thread(target=read_terminal, daemon=True).start()
        try:
            while viewer.is_running() and time.time() < deadline:
                clock.update(wall=time.time(), sim=g1.data.time)
                try:
                    cmd = commands.get_nowait()
                except queue.Empty:
                    g1.idle(0.05)
                    continue
                if cmd in ("q", "quit", "exit"):
                    break
                if cmd == "l" or cmd == "":
                    print(menu())
                elif cmd == "s":
                    print("Snapshots:", ", ".join(SNAPSHOTS))
                elif cmd == "r":
                    g1.reset()
                    print("started again")
                elif cmd in ("d", "t"):
                    print(run_dance(g1, DANCE if cmd == "d" else DANCE_TEST, args.tempo), flush=True)
                elif cmd.isdigit() and int(cmd) in ACTIONS:
                    if g1.fallen:
                        print("The G1 has fallen: type r to start again.")
                        continue
                    print(f"{cmd}: {ACTIONS[int(cmd)][0]} ...", flush=True)
                    print("   ", run_action(g1, int(cmd)), flush=True)
                    if g1.fallen:
                        print("    FELL -- type r to start again.")
                else:
                    print(f"'{cmd}' is not on the menu; type l to see it.")
        except KeyboardInterrupt:
            pass
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Make the G1 act: numbered actions and a dance.")
    ap.add_argument("--no-viewer", action="store_true", help="run without a window")
    ap.add_argument("--action", type=int, choices=sorted(ACTIONS), help="run one action (headless)")
    ap.add_argument("--all", action="store_true", help="run every action once (headless)")
    ap.add_argument("--dance", action="store_true", help="play DANCE (headless)")
    ap.add_argument("--tempo", type=float, default=1.0, help="dance speed: 2 = twice as fast")
    args = ap.parse_args(argv)
    if args.no_viewer or soc4180.is_colab() or args.all or args.action is not None or args.dance:
        return headless(args)
    return interactive(args)


if __name__ == "__main__":
    sys.exit(main())
