"""Week 3 lab game, on your laptop: ten problems, connected motions, a score out of 10.

    uv run weeks/03-inverse-kinematics/lab_connected.py

YOU EDIT ONLY `moves.py`, next to this file. Each problem asks the G1 to do
something -- wave, squat, bow, clap, both hands up -- and you answer with a
MOVE: a short list of poses the robot blends through, played under REAL
PHYSICS. The format is explained at the top of moves.py. It is re-read every
time you press a key, so save the file and press the key again; no restart.

    1 ... 9, 0     play problem 1 ... 9, 10 under physics, with a live gauge
    SPACE          play the current problem again
    E              edit mode: the robot freezes at `stand`; drag the joint
                   sliders (right panel, "Control") to pose it by hand
    K              (edit mode) show the next pose of the current problem's
                   move, frozen, with the sliders set to it -- then tweak
    ENTER          (edit mode) print the pose as a dict to paste into moves.py
    G              GRADE: all ten problems, one after another, then
                   the final score stays on screen for the instructor
    left-drag / right-drag / wheel   orbit / pan / zoom
    Keys go to the MuJoCo window -- click it first. If no key works there,
    press the same keys in this terminal (single keys, no enter). `q` quits.

    uv run weeks/03-inverse-kinematics/lab_connected.py --joints        # the 29 joint names
    uv run weeks/03-inverse-kinematics/lab_connected.py --grade --no-viewer   # grade headless

Why this is a week 3 lab. "pelvis": [0, 0, -0.2] in a pose is inverse
kinematics: you say WHERE the body should be, and the damped-least-squares leg
solver (soc4180.kinematics.ik_legs -- the lecture's solver) finds the twelve
leg angles that keep both feet where they are. And why it is more than IK:
every pose below is one IK or the sliders solve perfectly -- frozen. Under
gravity, half of them fall over. Kinematics places a pose; it does not balance
a robot. That gap is weeks 4 and 5, and you will meet it here first.

Things measured with this file's own code before it was written (--grade):

- Standing at `stand`: pelvis 0.79 m, hands hanging at 0.72 m.
- moves.py ships every move written out with its key numbers left as `...`,
  a blank; it grades 0 / 10, each problem naming the lines still blank.
  Problem 1's left arm with pitch, roll, yaw and elbow all 0 puts the
  hand at 0.88 m. "left_shoulder_pitch": -90 alone, elbow still bent as in
  `stand`, gives 1.12 m: a near miss. -70 gives 0.99.
- A squat by IK to "pelvis": [0, 0, -0.20] bottoms out at 0.57 m and stands
  back up. [0, 0, -0.25] asks for 0.54 m and the robot sits down for good.
- Raising BOTH arms overhead at once (shoulder pitch -150) throws the robot
  on its face: the arms swing forward through horizontal and carry the centre
  of mass past the toes. Raising one arm, then the other, reaches 1.36 m with
  both hands and never tips more than 6 degrees. Lowering both at once from
  there leaves it slumped at pelvis 0.65 m, torso 39 degrees over: "did not
  end standing". Connected motions: the ORDER is the answer.
- The boss is the same lesson twice: squat, left arm up, right arm up, and
  STAND UP WITH THE ARMS STILL UP, then lower them one at a time. Standing up
  after lowering them in the squat throws the robot on its back.
- Arms forward for a clap with the pelvis where it is: falls forward. Move
  the pelvis back 4 cm ("pelvis": [-0.04, 0, -0.03]) and the same arms clap
  with the hands 6 cm apart. A counterweight -- the body leaning against its
  own arms.
- Bending at the hips (hip pitch -30 plus waist pitch 30) topples forward.
  Pushing the pelvis BACK and down while bending reaches 49 degrees of bow
  and stands up again.
- Standing on one foot with open-loop poses: not found. Seventy-two
  combinations of weight shift, lift height and counter-roll all fell. It
  needs feedback (week 7). That is why no problem here asks for it.

Everything in this file is complete and explained. You are not meant to edit
it: it is the referee, and the grading screen shows a 4-letter code computed
from it so the instructor can see it is unchanged.
"""

from __future__ import annotations

# --- 0. imports ---------------------------------------------------------------------
import argparse
import ast          # moves.py is read as DATA (parsed, never executed)
import datetime
import hashlib      # the scorer code on the grading screen
import os
import time
from pathlib import Path

import mujoco
import numpy as np

import soc4180
from soc4180 import bodies, kinematics as kin

HERE = Path(__file__).resolve().parent


# --- 1. the rules -----------------------------------------------------------------------

HOLD = 1.5          # seconds the last pose is held after the move ends
                    # (a move may be as long as you like: slow is often what balances)
FALL_Z = 0.45       # pelvis below this at any moment: fallen
STAND_Z = 0.70      # at the end the pelvis must be above this ...
STAND_TILT = 15.0   # ... and the torso within this many degrees of upright


# --- 2. the ten problems -----------------------------------------------------------------
#
# Each problem is a title, one line of instructions, an optional goal height
# drawn as a translucent disc, and an `update(m, st)` that is called on EVERY
# physics step with the measurements `m` (section 4). It records the best value
# so far in `st`, sets st["met"] = True the moment the goal is reached, and
# returns the gauge line shown live on screen.

def _best(st, value, better=max):
    st["best"] = value if "best" not in st else better(st["best"], value)
    return st["best"]


def wave(m, st):
    z = m["lh"][2]
    b = _best(st, z)
    st["met"] |= z > 1.20
    return f"left hand {z:.2f} m  (best {b:.2f})   need > 1.20"


def sway(m, st):
    y = m["pelvis_y"]
    st["hi"] = max(st.get("hi", 0.0), y)
    st["lo"] = min(st.get("lo", 0.0), y)
    st["met"] |= st["hi"] >= 0.05 and st["lo"] <= -0.05
    return f"pelvis y {y:+.3f} m  (reached {st['lo']:+.3f} .. {st['hi']:+.3f})   need -0.05 and +0.05"


def twist(m, st):
    b = _best(st, abs(m["yaw"]))
    st["met"] |= abs(m["yaw"]) >= 80
    return f"torso turned {abs(m['yaw']):.0f} deg  (best {b:.0f})   need >= 80"


def squat(m, st):
    b = _best(st, m["pelvis"], min)
    st["met"] |= m["pelvis"] < 0.60
    return f"pelvis {m['pelvis']:.3f} m  (lowest {b:.3f})   need < 0.60"


def bow(m, st):
    b = _best(st, m["tilt"])
    st["met"] |= m["tilt"] >= 40
    return f"torso tilt {m['tilt']:.0f} deg  (best {b:.0f})   need >= 40"


def clap(m, st):
    gap = float(np.linalg.norm(m["lh"] - m["rh"]))
    front = min(m["lh"][0], m["rh"][0]) - m["pelvis_x"]
    counted = gap if front >= 0.25 else 9.99       # a clap behind the chest does not count
    b = _best(st, counted, min)
    st["met"] |= counted < 0.10
    return f"hands {gap:.3f} m apart, {front:+.2f} m in front  (best {b:.3f})   need < 0.10, front >= 0.25"


def hands_up(m, st):
    low = min(m["lh"][2], m["rh"][2])               # the LOWER hand is what counts
    b = _best(st, low)
    st["met"] |= low > 1.20
    return f"lower hand {low:.2f} m  (best {b:.2f})   need > 1.20"


def squat_wave(m, st):
    high = max(m["lh"][2], m["rh"][2])
    counted = high if m["pelvis"] < 0.62 else 0.0
    b = _best(st, counted)
    st["met"] |= counted > 1.10
    return f"pelvis {m['pelvis']:.2f}, higher hand {high:.2f}  (best while squatting {b:.2f})   need pelvis < 0.62 AND hand > 1.10"


def twist_wave(m, st):
    high = max(m["lh"][2], m["rh"][2])
    counted = high if abs(m["yaw"]) >= 80 else 0.0
    b = _best(st, counted)
    st["met"] |= counted > 1.20
    return f"turned {abs(m['yaw']):.0f} deg, higher hand {high:.2f}  (best while turned {b:.2f})   need >= 80 deg AND hand > 1.20"


def boss(m, st):
    low = min(m["lh"][2], m["rh"][2])
    counted = low if m["pelvis"] < 0.62 else 0.0
    b = _best(st, counted)
    st["met"] |= counted > 1.10
    return f"pelvis {m['pelvis']:.2f}, lower hand {low:.2f}  (best while squatting {b:.2f})   need pelvis < 0.62 AND both > 1.10"


PROBLEMS = {    # number: (title, instructions, goal-disc height or None, update)
    1: ("WAVE", "Raise the left hand above 1.20 m.", 1.20, wave),
    2: ("SWAY", "Move the pelvis 5 cm to the left and 5 cm to the right.", None, sway),
    3: ("TWIST", "Turn the torso 80 degrees or more.", None, twist),
    4: ("SQUAT", "Pelvis below 0.60 m, then stand back up.", 0.60, squat),
    5: ("BOW", "Tilt the torso 40 degrees or more, then stand back up.", None, bow),
    6: ("CLAP", "Hands closer than 10 cm, 25 cm in front of the pelvis.", None, clap),
    7: ("HANDS UP", "Both hands above 1.20 m at the same moment.", 1.20, hands_up),
    8: ("SQUAT-WAVE", "Pelvis below 0.62 m with a hand above 1.10 m.", 1.10, squat_wave),
    9: ("TWIST-WAVE", "Torso turned 80 degrees or more with a hand above 1.20 m.", 1.20, twist_wave),
    10: ("BOSS", "Pelvis below 0.62 m with BOTH hands above 1.10 m.", 1.10, boss),
}


# --- 3. reading moves.py as data -----------------------------------------------------------
#
# moves.py LOOKS like Python, but it is never imported or run: `ast.parse`
# turns it into a syntax tree and `_value` below evaluates only numbers,
# strings, lists, tuples, dicts (with ** to merge), names defined earlier in
# the file, and + - * / on numbers. Anything else is an error that names the
# line. So a moves file can describe motion, and nothing else -- it cannot
# reach into this file and change the rules.

class MovesError(Exception):
    pass


class Blank:
    """A `...` in moves.py: a number the student has not written yet.

    It is kept as a value, not raised as an error, because a blank inside a
    named pose (LEFT_UP = {...}) is read before MOVES and would otherwise
    break every problem. Only the problems that USE the blank fail, and they
    name its line.
    """

    def __init__(self, lineno):
        self.lineno = lineno


def _blank_lines(obj) -> list:
    """Line numbers of every `...` inside a move or pose."""
    if isinstance(obj, Blank):
        return [obj.lineno]
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, (list, tuple)):
        return sorted({n for v in obj for n in _blank_lines(v)})
    return []


def _refuse_blanks(obj, what):
    lines = _blank_lines(obj)
    if lines:
        where = ", ".join(str(n) for n in lines)
        many = len(lines) > 1
        raise MovesError(f"{what} is not filled in yet: replace the ... on line{'s' * many} "
                         f"{where} of moves.py with {'numbers' if many else 'a number'}")


def _value(node, env):
    if isinstance(node, ast.Constant):
        if node.value is Ellipsis:                      # ... = a blank to fill in
            return Blank(node.lineno)
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise MovesError(f"line {node.lineno}: '{node.id}' is not defined above this line")
        return env[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [_value(e, env) for e in node.elts]
        return items if isinstance(node, ast.List) else tuple(items)
    if isinstance(node, ast.Dict):
        out = {}
        for k, v in zip(node.keys, node.values):
            if k is None:                               # {**OTHER, ...}
                out.update(_value(v, env))
            else:
                out[_value(k, env)] = _value(v, env)
        return out
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _value(node.operand, env)
        if isinstance(v, Blank):                        # -... is still a blank
            return v
        return -v if isinstance(node.op, ast.USub) else +v
    if isinstance(node, ast.Set):                       # {0} -- an easy slip for {}
        raise MovesError(f"line {node.lineno}: {{...}} with no ':' is a set, not a pose -- "
                         f"write {{}} for stand, or {{\"joint\": degrees}}")
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        a, b = _value(node.left, env), _value(node.right, env)
        if isinstance(a, Blank) or isinstance(b, Blank):
            return a if isinstance(a, Blank) else b
        ops = {ast.Add: a.__add__, ast.Sub: a.__sub__, ast.Mult: a.__mul__, ast.Div: a.__truediv__}
        return ops[type(node.op)](b)
    raise MovesError(f"line {node.lineno}: only numbers, lists, dicts and names are allowed here")


def read_moves(path: Path):
    """Return (NAME, MOVES) from a moves file, or raise MovesError saying where it is wrong."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as e:
        raise MovesError(f"line {e.lineno}: {e.msg}") from None
    except OSError as e:
        raise MovesError(f"cannot read {path}: {e.strerror}") from None
    env = {}
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue                                    # the docstring
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "MOVES" and isinstance(node.value, ast.Dict)):
            # One problem at a time: a mistake in MOVES[4] fails problem 4 and
            # nothing else. The error is stored in place of the move, and
            # Attempt raises it when that problem is played.
            env["MOVES"] = {}
            for k, v in zip(node.value.keys, node.value.values):
                if k is None:
                    raise MovesError(f"line {v.lineno}: no ** inside MOVES -- give each problem its own line")
                key = _value(k, env)
                try:
                    env["MOVES"][key] = _value(v, env)
                except MovesError as e:
                    env["MOVES"][key] = e
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            env[node.targets[0].id] = _value(node.value, env)
        else:
            raise MovesError(f"line {node.lineno}: only lines of the form  NAME = value  are allowed")
    if not isinstance(env.get("MOVES"), dict):
        raise MovesError("no MOVES = {...} found")
    return str(env.get("NAME", "?")), env["MOVES"]


# --- 4. the robot: poses, IK, measurements ------------------------------------------------

class Robot:
    """The G1, and the three things the game needs from it.

    resolve(pose)  a moves.py pose -> 29 servo targets (and the full qpos, for
                   showing it frozen). This is where "pelvis" becomes IK.
    measure(data)  the numbers every problem is judged on, one physics step.
    """

    def __init__(self):
        self.model = m = soc4180.load_g1()
        self.key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.stand = m.key_qpos[self.key].copy()
        # Every G1 actuator drives one hinge, so actuator a's target is the
        # angle stored at qpos[act_q[a]]. ctrl and qpos speak the same units.
        self.act_q = m.jnt_qposadr[m.actuator_trnid[:, 0]]
        self.lo, self.hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]

        body = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.pelvis, self.torso = body("pelvis"), body("torso_link")
        # There are no hand sites on the G1 (nsite = 4: two feet, two IMUs), so
        # a "hand" is the origin of the last wrist link.
        self.lhand, self.rhand = body("left_wrist_yaw_link"), body("right_wrist_yaw_link")
        self.site = {s: kin.foot_site_id(m, s) for s in ("left", "right")}

        # IK needs the feet where they stand, level, and a BENT-knee seed: the
        # stand pose has every leg joint at zero, the straight-leg singularity
        # this week's lecture is about (sigma_min 9e-7), and DLS cannot bend a
        # knee out of it. The seed is the week 4 walker's crouch.
        self.scratch = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, self.scratch, self.key)
        mujoco.mj_forward(m, self.scratch)
        self.feet = {s: (self.scratch.site_xpos[i].copy(), self.scratch.site_xmat[i].reshape(3, 3).copy())
                     for s, i in self.site.items()}
        self.seed = self.stand.copy()
        for s in self.site:
            i = kin.leg_qpos_indices(m, s)
            self.seed[i[0]], self.seed[i[3]], self.seed[i[4]] = -0.35, 0.70, -0.35

    def resolve(self, pose: dict):
        """One pose -> (ctrl targets, full qpos, list of warnings).

        1. Start from `stand`: every joint not named keeps its stand angle.
        2. If "pelvis" / "left_foot" / "right_foot" are given, run the week 3
           IK: place the pelvis (upright) at stand + offset, put each foot at
           its stand position + offset, level, and let ik_legs find the twelve
           leg angles by damped least squares.
        3. Then apply the named joints and chains, in degrees -- so a joint you
           name wins over what IK chose for it.
        """
        m, q, notes = self.model, self.stand.copy(), []
        if not isinstance(pose, dict):
            raise MovesError(f"a pose must be a dict, got {pose!r}")
        _refuse_blanks(pose, "this pose")
        ik = {k: np.asarray(pose[k], float) for k in ("pelvis", "left_foot", "right_foot") if k in pose}
        if ik:
            base = self.stand[:3] + ik.get("pelvis", np.zeros(3))
            targets = {s: (self.feet[s][0] + ik.get(f"{s}_foot", np.zeros(3)), self.feet[s][1])
                       for s in ("left", "right")}
            r = kin.ik_legs(m, self.scratch, base, [1, 0, 0, 0], targets,
                            seed_qpos=self.seed, iterations=60, tolerance=1e-5)
            for s in ("left", "right"):
                q[kin.leg_qpos_indices(m, s)] = r[s]
            q[:3] = base
            if r["error"] > 0.01:
                notes.append(f"IK could not reach {dict((k, v.tolist()) for k, v in ik.items())}: "
                             f"{r['error'] * 100:.1f} cm short -- outside the legs' reach")
        for name, v in pose.items():
            if name in ik:
                continue
            try:
                if name in bodies.CHAINS:
                    q[bodies.group_indices(m, name)] = np.radians(np.asarray(v, float))
                else:
                    q[bodies.joint_index(m, name)] = np.radians(float(v))
            except ValueError as e:
                raise MovesError(str(e)) from None
        ctrl = q[self.act_q]
        clipped = np.clip(ctrl, self.lo, self.hi)
        for a in np.flatnonzero(np.abs(clipped - ctrl) > 1e-9):
            notes.append(f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)} "
                         f"{np.degrees(ctrl[a]):.0f} deg is past its limit; the servo stops at "
                         f"{np.degrees(clipped[a]):.0f}")
        return clipped, q, notes

    def measure(self, d) -> dict:
        """What the referee sees, every step.

        tilt  the angle between the torso's own z axis and world up:
              R[2, 2] is the z component of that axis, and its arccos is the angle.
        yaw   the heading of the torso's x axis (where its chest points) in the
              floor plane, from R[0, 0] and R[1, 0]. The pelvis stays facing
              forward, so this is how far the waist has turned.
        """
        R = d.xmat[self.torso].reshape(3, 3)
        return {
            "pelvis": float(d.xpos[self.pelvis][2]),
            "pelvis_x": float(d.xpos[self.pelvis][0]),
            "pelvis_y": float(d.xpos[self.pelvis][1]),
            "tilt": float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))),
            "yaw": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
            "lh": d.xpos[self.lhand].copy(),
            "rh": d.xpos[self.rhand].copy(),
        }


class Attempt:
    """One problem, one move, stepped one physics step at a time.

    The move becomes a list of segments (target ctrl, number of steps). Within
    a segment the servo targets blend from the previous targets with a
    smoothstep, s*s*(3 - 2s): zero speed at both ends, so a pose is reached
    gently instead of with a jolt -- but a SHORT segment is still a fast
    motion, and fast motions throw the robot. The last targets are then held
    for HOLD seconds. `step` returns False when the attempt is over.
    """

    def __init__(self, robot: Robot, number: int, move, data):
        self.robot, self.number, self.data = robot, number, data
        self.title, self.text, self.plane, self.update = PROBLEMS[number]
        self.notes = []
        if isinstance(move, MovesError):                # read_moves could not read this entry
            raise MovesError(f"problem {number}, {move}")
        if not isinstance(move, (list, tuple)) or not move:
            raise MovesError(f"problem {number} has no move yet -- add steps to MOVES[{number}] in moves.py")
        _refuse_blanks(move, f"problem {number}")
        dt = robot.model.opt.timestep
        self.segments = []
        for i, step in enumerate(move):
            if not (isinstance(step, (list, tuple)) and len(step) == 2):
                raise MovesError(f"problem {number}, step {i + 1}: write it as (seconds, {{pose}})")
            secs, pose = float(step[0]), step[1]
            if secs <= 0:
                raise MovesError(f"problem {number}, step {i + 1}: seconds must be positive")
            ctrl, _, notes = robot.resolve(pose)
            self.notes += [f"step {i + 1}: {n}" for n in notes]
            self.segments.append((ctrl, max(1, round(secs / dt))))
        self.segments.append((self.segments[-1][0], round(HOLD / dt)))
        self.total_steps = sum(n for _, n in self.segments)

        mujoco.mj_resetDataKeyframe(robot.model, data, robot.key)
        mujoco.mj_forward(robot.model, data)
        self.prev = robot.stand[robot.act_q].copy()
        data.ctrl[:] = self.prev
        self.seg, self.k, self.done_steps = 0, 0, 0
        self.st = {"met": False}
        self.fell = False
        self.gauge = ""

    def step(self) -> bool:
        if self.seg >= len(self.segments):
            return False
        target, n = self.segments[self.seg]
        self.k += 1
        s = self.k / n
        s = s * s * (3 - 2 * s)
        d = self.data
        d.ctrl[:] = self.prev + s * (target - self.prev)
        d.xfrc_applied[:] = 0.0         # no mouse pushes: the referee runs the move as written
        mujoco.mj_step(self.robot.model, d)
        self.done_steps += 1
        m = self.robot.measure(d)
        self.gauge = self.update(m, self.st)
        self.fell |= m["pelvis"] < FALL_Z
        self.last = m
        if self.k >= n:
            self.prev, self.seg, self.k = target, self.seg + 1, 0
        return self.seg < len(self.segments)

    @property
    def seconds(self):
        return self.done_steps * self.robot.model.opt.timestep

    def verdict(self):
        """(passed, reason) once the attempt is over."""
        standing = self.last["pelvis"] > STAND_Z and self.last["tilt"] < STAND_TILT
        if self.fell:
            return False, "FELL"
        if not self.st["met"]:
            return False, "goal not reached"
        if not standing:
            return False, (f"did not end standing (pelvis {self.last['pelvis']:.2f} m, "
                           f"tilt {self.last['tilt']:.0f} deg)")
        return True, "PASS"


# --- 5. grading ------------------------------------------------------------------------------

def scorer_code() -> str:
    """Four letters that change if this file changes.

    A hash of this file's text, line endings normalised so a Windows checkout
    and a Mac checkout of the same file agree. The instructor writes the
    expected code on the board.
    """
    text = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(text).hexdigest()[:4].upper()


def marks_line(results: dict) -> str:
    """1:O 2:X 3:. ...  -- O pass, X fail, . not tried."""
    return "  ".join(f"{n}:{'.' if n not in results else ('O' if results[n] else 'X')}" for n in PROBLEMS)


def grade_headless(robot: Robot, path: Path, only=None) -> int:
    """Play every problem without a window and print the table. Returns the score."""
    try:
        name, moves = read_moves(path)
    except MovesError as e:
        print(f"{path.name}: {e}")
        return 0
    data = mujoco.MjData(robot.model)
    score, results = 0, {}
    print(f"grading {path.name} for {name}")
    for n in (only or PROBLEMS):
        title = PROBLEMS[n][0]
        try:
            att = Attempt(robot, n, moves.get(n), data)
        except MovesError as e:
            results[n] = False
            print(f"  {n:2d} {title:11s} X   {e}")
            continue
        while att.step():
            pass
        ok, why = att.verdict()
        results[n] = ok
        score += ok
        print(f"  {n:2d} {title:11s} {'O' if ok else 'X'}   {why:28s} {att.gauge}")
        for note in att.notes:
            print(f"                     note: {note}")
    print(f"SCORE {score} / {len(PROBLEMS)}    {marks_line(results)}    scorer code {scorer_code()}")
    return score


# --- 6. the viewer: HUD, markers, keys ------------------------------------------------------

BIG = mujoco.mjtFont.mjFONT_BIG
NORMAL = mujoco.mjtFont.mjFONT_NORMAL
TL, TR = mujoco.mjtGridPos.mjGRID_TOPLEFT, mujoco.mjtGridPos.mjGRID_TOPRIGHT
BL, BR = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT

KEYCODES = {48 + (i % 10): i for i in range(1, 11)}     # '1'..'9' -> 1..9, '0' -> 10


def draw_markers(viewer, robot, data, number, met):
    """The goal disc (height problems) and a sphere on each hand."""
    with viewer.lock():
        scn = viewer.user_scn
        scn.ngeom = 0
        plane = PROBLEMS[number][2] if number else None
        px, py = data.xpos[robot.pelvis][:2]
        if plane is not None:
            g = scn.geoms[scn.ngeom]
            rgba = (0.1, 0.9, 0.2, 0.25) if met else (1.0, 0.8, 0.1, 0.25)
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.45, 0.002, 0]),
                                np.array([px, py, plane]), np.eye(3).flatten(), np.array(rgba))
            scn.ngeom += 1
        for b in (robot.lhand, robot.rhand):
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.035, 0, 0]),
                                data.xpos[b].copy(), np.eye(3).flatten(),
                                np.array((0.1, 0.9, 0.2, 0.8) if met else (1.0, 0.3, 0.2, 0.8)))
            scn.ngeom += 1


def pose_as_dict(robot, data) -> str:
    """The frozen pose as a moves.py dict: every joint that differs from stand, in degrees."""
    parts = []
    for a, qi in enumerate(robot.act_q):
        diff = data.qpos[qi] - robot.stand[qi]
        if abs(np.degrees(diff)) > 0.5:
            name = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            j = robot.model.actuator_trnid[a, 0]
            jname = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_JOINT, j).removesuffix("_joint")
            parts.append(f'"{jname}": {np.degrees(data.qpos[qi]):.0f}')
    return "{" + ", ".join(parts) + "}"


def run_viewer(robot: Robot, path: Path) -> int:
    model = robot.model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, robot.key)
    mujoco.mj_forward(model, data)

    # The key callback runs on the viewer's thread (or the terminal reader's),
    # so it only leaves a command here; the loop below does the work.
    state = {"cmd": None, "mode": "edit", "number": 1, "attempt": None, "msg": "",
             "results": {}, "grading": None, "final": None, "kf": -1}

    def on_key(keycode):
        if keycode in KEYCODES:
            state["cmd"] = ("play", KEYCODES[keycode])
        elif keycode == 32:
            state["cmd"] = ("play", state["number"])
        elif keycode == 71:                 # G
            state["cmd"] = ("grade", None)
        elif keycode == 69:                 # E
            state["cmd"] = ("edit", None)
        elif keycode == 75:                 # K
            state["cmd"] = ("keyframe", None)
        elif keycode in (257, 335):         # ENTER
            state["cmd"] = ("print", None)

    def start(number, grading=False):
        """Re-read moves.py and begin one problem. Returns False on an error in moves.py."""
        state["number"] = number
        try:
            name, moves = read_moves(path)
            state["name"] = name
            state["attempt"] = Attempt(robot, number, moves.get(number), data)
        except MovesError as e:
            state["attempt"] = None
            state["msg"] = str(e)
            print(f"[{number}] {e}")
            if grading:
                state["results"][number] = False
            return False
        for note in state["attempt"].notes:
            print(f"[{number}] note: {note}")
        state["msg"] = ""
        state["t0"] = time.perf_counter()
        return True

    def finish():
        att = state["attempt"]
        ok, why = att.verdict()
        state["results"][att.number] = ok
        state["msg"] = why
        print(f"[{att.number} {att.title}] {'PASS' if ok else 'FAIL: ' + why}   {att.gauge}")
        state["attempt"] = None
        if state["grading"] is not None:
            state["grading"].append((att.number, ok))
            next_problem()

    def next_problem():
        """During G: start the next problem, or show the final screen."""
        while True:
            n = len(state["grading"]) + 1
            if n > len(PROBLEMS):
                score = sum(ok for _, ok in state["grading"])
                stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                state["final"] = (score, stamp)
                state["mode"] = "final"
                state["grading"] = None
                print(f"\n=== {state.get('name', '?')}: SCORE {score} / {len(PROBLEMS)}   "
                      f"{marks_line(state['results'])}   scorer code {scorer_code()}   {stamp} ===\n")
                return
            if start(n, grading=True):
                return
            state["grading"].append((n, False))

    def hud():
        att, texts = state["attempt"], []
        name = state.get("name", "")
        if state["mode"] == "final":
            score, stamp = state["final"]
            texts.append((BIG, TL, f"{name}\nSCORE  {score} / {len(PROBLEMS)}\n{marks_line(state['results'])}",
                          ""))
            texts.append((BIG, BL, f"scorer {scorer_code()}   {stamp}", ""))
            return texts
        n = state["number"]
        title, text, _, _ = PROBLEMS[n]
        head = "GRADING  " if state["grading"] is not None else ""
        texts.append((BIG, TL, f"{head}{n}. {title}", ""))
        body = text
        if att is not None:
            body += f"\n{att.gauge}\n{att.seconds:4.1f} s" + ("    GOAL REACHED" if att.st["met"] else "")
        elif state["msg"]:
            body += f"\n>> {state['msg']}"
        if state["mode"] == "edit":
            body += "\nEDIT: drag sliders, ENTER prints the pose, K steps through this move"
        body += "\nkeys: 1-9,0 play   SPACE again   E edit   K pose   ENTER print   G grade"
        texts.append((NORMAL, BL, body, ""))
        texts.append((NORMAL, TR, marks_line(state["results"]), ""))
        return texts

    print(__doc__.split("\n\n")[1])
    print(f"moves file: {path}\nscorer code: {scorer_code()}")
    soc4180.terminal_keys(on_key)
    print("  [terminal] keys dead in the window? press them here instead; 'q' stops.", flush=True)
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    slider_q = robot.act_q
    dt = model.opt.timestep

    with soc4180.launch_viewer(model, data, passive=True, key_callback=on_key) as viewer:
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.2, 145.0, -12.0
        viewer.cam.lookat[:] = (0.0, 0.0, 0.75)
        data.ctrl[:] = robot.stand[slider_q]
        while viewer.is_running() and time.time() < deadline:
            cmd, state["cmd"] = state["cmd"], None
            if cmd is not None and state["grading"] is not None and cmd[0] != "grade":
                cmd = None                                  # nothing interrupts a grading run but G
            if cmd is not None:
                kind, arg = cmd
                if kind == "play":
                    state["mode"] = "play"
                    start(arg)
                elif kind == "grade":
                    state["mode"], state["results"], state["grading"] = "play", {}, []
                    print("\n=== GRADING: all ten problems ===")
                    next_problem()
                elif kind == "edit":
                    state["mode"], state["attempt"], state["kf"] = "edit", None, -1
                    mujoco.mj_resetDataKeyframe(model, data, robot.key)
                    data.ctrl[:] = robot.stand[slider_q]
                    mujoco.mj_forward(model, data)
                elif kind == "keyframe":
                    try:
                        _, moves = read_moves(path)
                        move = moves.get(state["number"]) or []
                        if isinstance(move, MovesError):
                            raise move
                        if not move:
                            raise MovesError(f"problem {state['number']} has no move yet")
                        state["kf"] = (state["kf"] + 1) % len(move)
                        _, q, notes = robot.resolve(move[state["kf"]][1])
                        state["mode"], state["attempt"] = "edit", None
                        data.qpos[:], data.qvel[:] = q, 0.0
                        data.ctrl[:] = q[slider_q]              # the sliders show the pose
                        mujoco.mj_forward(model, data)
                        state["msg"] = f"showing step {state['kf'] + 1} of {len(move)} of move {state['number']}"
                        for note in notes:
                            print(f"  note: {note}")
                    except MovesError as e:
                        state["msg"] = str(e)
                        print(e)
                elif kind == "print" and state["mode"] == "edit":
                    print(f"  pose: {pose_as_dict(robot, data)}")

            att = state["attempt"]
            if att is not None:
                # Real time: run as many physics steps as the wall clock says
                # are due, at most 100 per frame. A slow laptop then plays in
                # slow motion rather than skipping -- the result is identical.
                due = int((time.perf_counter() - state["t0"]) / dt)
                for _ in range(min(max(due - att.done_steps, 0), 100)):
                    if not att.step():
                        finish()
                        break
                if due - att.done_steps > 100 and state["attempt"] is att:
                    state["t0"] = time.perf_counter() - att.done_steps * dt
            elif state["mode"] == "edit":
                # Kinematic: the sliders write ctrl, which only reaches the
                # joints through physics. Frozen, we copy ctrl into qpos by hand.
                data.qpos[slider_q] = data.ctrl
                mujoco.mj_forward(model, data)

            met = bool(att is not None and att.st["met"])
            draw_markers(viewer, robot, data, None if state["mode"] == "final" else state["number"], met)
            viewer.set_texts(hud())
            # MuJoCo's viewer also treats digit keys 0-5 as 'toggle geom group'. The
            # robot's meshes are group 2, so a '2' pressed for this lab would hide it.
            viewer.opt.geomgroup[:3] = 1
            viewer.sync()
            time.sleep(0.01)
    return 0


# --- 7. main ----------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--moves", type=Path, default=HERE / "moves.py", help="the moves file (default: moves.py here)")
    ap.add_argument("--grade", action="store_true", help="grade the moves file headless and print the table")
    ap.add_argument("--problem", type=int, nargs="*", help="with --grade: only these problems")
    ap.add_argument("--joints", action="store_true", help="print the 29 joint names and ranges, in degrees")
    ap.add_argument("--no-viewer", action="store_true", help="never open a window")
    args = ap.parse_args(argv)

    robot = Robot()
    if args.joints:
        for chain, joints in bodies.CHAINS.items():
            print(chain)
            for j in joints:
                jid = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                lo, hi = np.degrees(robot.model.jnt_range[jid])
                stand = np.degrees(robot.stand[robot.model.jnt_qposadr[jid]])
                print(f"  {j.removesuffix('_joint'):24s} {lo:7.0f} .. {hi:4.0f}   stand {stand:5.0f}")
        return 0
    if args.grade or args.no_viewer:
        grade_headless(robot, args.moves, args.problem)
        return 0
    if soc4180.is_colab():
        print("This lab needs a desktop window; run it on your laptop.")
        return 1
    return run_viewer(robot, args.moves)


if __name__ == "__main__":
    raise SystemExit(main())
