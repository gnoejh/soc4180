"""The lab-game machinery of weeks 11-15, shared: an answers file read as data, problems, a score, a scorer code.

Week 3's ``lab_connected.py`` is the model: students edit ONE data file, each
problem is played live in the viewer with a gauge, ``G`` grades them all to a
score out of N, and a four-letter *scorer code* -- a hash of the lab script --
lets the instructor see from across the room that the referee is unchanged.
Week 3 carries its own copy of this machinery (its scorer code is a hash of
that file and must not move); the later weeks share this one.

    answers = read_answers(path)            # dict: name -> value, `...` kept as Blank(lineno)
    need(answers, "STAND")                  # the value, or AnswersError naming the blank lines
    grade_headless(problems, path, lab_file)
    run_viewer(model, data, problems, path, lab_file, draw=...)

An answers file is **parsed, never run**: only assignments of plain values
(numbers, strings, True/False/None, lists, tuples, dicts, names assigned
above, and + - * / on numbers) are allowed. A student's file therefore cannot
change the referee. ``...`` is a blank: it parses, it is kept as a ``Blank``
carrying its line number, and whichever problem needs it refuses to run and
names the line -- so one unfinished answer never breaks another problem.
"""

from __future__ import annotations

import ast
import datetime
import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

__all__ = ["AnswersError", "Attempt", "Blank", "Problem", "grade_headless", "marks_line", "need",
           "read_answers", "run_viewer", "scorer_code"]


# --- 1. reading the answers file as data ----------------------------------------------------

class AnswersError(Exception):
    """Something in the answers file stops a problem: a blank, a typo, a wrong type."""


@dataclass(frozen=True)
class Blank:
    """A `...` left in the answers file, remembered with its line number."""
    lineno: int

    def __repr__(self):
        return f"... (line {self.lineno})"


_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
           ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b}


def _value(node, env, where):
    if isinstance(node, ast.Constant):
        return Blank(node.lineno) if node.value is Ellipsis else node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise AnswersError(f"line {node.lineno}: '{node.id}' is not defined above ({where})")
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_value(e, env, where) for e in node.elts]
    if isinstance(node, ast.Dict):
        out = {}
        for k, v in zip(node.keys, node.values):
            if k is None:                                          # {**OTHER, ...}
                base = _value(v, env, where)
                if not isinstance(base, dict):
                    raise AnswersError(f"line {v.lineno}: ** needs a dict ({where})")
                out.update(base)
            else:
                out[_value(k, env, where)] = _value(v, env, where)
        return out
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _value(node.operand, env, where)
        if isinstance(v, Blank):
            return v
        if not isinstance(v, (int, float)):
            raise AnswersError(f"line {node.lineno}: a sign needs a number ({where})")
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        a, b = _value(node.left, env, where), _value(node.right, env, where)
        for v in (a, b):
            if isinstance(v, Blank):
                return v
            if not isinstance(v, (int, float)):
                raise AnswersError(f"line {node.lineno}: arithmetic only on numbers ({where})")
        return _BINOPS[type(node.op)](a, b)
    raise AnswersError(f"line {getattr(node, 'lineno', '?')}: only plain values are allowed here -- "
                       f"numbers, strings, lists, dicts, names defined above ({where})")


def read_answers(path) -> dict:
    """Every top-level ``NAME = value`` in the file, in order. Docstrings are skipped."""
    path = Path(path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as e:
        raise AnswersError(f"{path.name} line {e.lineno}: {e.msg} -- the file is not valid Python")
    env: dict = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue                                                # a docstring or a bare ...
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
            raise AnswersError(f"{path.name} line {stmt.lineno}: only `NAME = value` lines are allowed")
        name = stmt.targets[0].id
        env[name] = _value(stmt.value, env, name)
    return env


def blanks_in(value) -> list:
    if isinstance(value, Blank):
        return [value.lineno]
    if isinstance(value, dict):
        return sorted({n for v in value.values() for n in blanks_in(v)})
    if isinstance(value, (list, tuple)):
        return sorted({n for v in value for n in blanks_in(v)})
    return []


def need(answers: dict, name: str):
    """The answer called ``name``, with no blanks in it -- or an AnswersError saying which lines."""
    if name not in answers:
        raise AnswersError(f"{name} is missing from the answers file")
    value = answers[name]
    lines = blanks_in(value)
    if lines:
        raise AnswersError(f"{name}: not filled in yet -- the ... on line{'s' if len(lines) > 1 else ''} "
                           f"{', '.join(map(str, lines))}")
    return value


def number(d: dict, key: str, where: str, lo=None, hi=None) -> float:
    """``d[key]`` as a float, checked against a range, with an error a student can act on."""
    if key not in d:
        raise AnswersError(f"{where}: needs \"{key}\"")
    v = d[key]
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise AnswersError(f"{where}: \"{key}\" must be a number, not {v!r}")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise AnswersError(f"{where}: \"{key}\" = {v} is outside {lo} .. {hi}")
    return float(v)


# --- 2. problems and attempts ---------------------------------------------------------------------

class Attempt:
    """One play of one problem. Subclasses implement ``step`` (one decision) and ``verdict``.

    ``model`` / ``data`` are what the viewer draws; ``dt`` is the simulated
    seconds one ``step`` advances (for real-time pacing); ``gauge`` is the live
    line shown under the problem; ``notes`` are printed once at the start.
    """

    model = None
    data = None
    dt = 0.02
    gauge = ""

    def __init__(self):
        self.notes: list[str] = []
        self.met = False

    def step(self) -> bool:
        """Advance one decision. Return False when the attempt is over."""
        raise NotImplementedError

    def verdict(self) -> tuple[bool, str]:
        raise NotImplementedError

    def draw(self, scn):
        """Add markers to the viewer's ``user_scn`` (optional)."""

    def images(self):
        """Pictures to overlay on the viewer: a list of (MjrRect, HxWx3 uint8 image), or None (optional)."""
        return None


@dataclass
class Problem:
    title: str
    text: str
    make: Callable[[dict], Attempt]            # answers -> Attempt (may raise AnswersError)
    extra: dict = field(default_factory=dict)


# --- 3. grading -----------------------------------------------------------------------------------

def scorer_code(lab_file) -> str:
    """Four letters that change if the lab script changes (line endings normalised)."""
    text = Path(lab_file).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(text).hexdigest()[:4].upper()


def marks_line(problems: dict, results: dict) -> str:
    """1:O 2:X 3:. ...  -- O pass, X fail, . not tried."""
    return "  ".join(f"{n}:{'.' if n not in results else ('O' if results[n] else 'X')}" for n in problems)


def _start(problems, n, path):
    answers = read_answers(path)
    return problems[n].make(answers), answers.get("NAME", "?")


def grade_headless(problems: dict, path, lab_file, only=None) -> int:
    """Play every problem without a window, print one line each and the score. Returns the score."""
    score, results = 0, {}
    print(f"grading {Path(path).name}")
    for n in (only or problems):
        p = problems[n]
        t0 = time.time()
        try:
            att, name = _start(problems, n, path)
        except AnswersError as e:
            results[n] = False
            print(f"  {n:2d} {p.title:12s} X   {e}", flush=True)
            continue
        for note in att.notes:
            print(f"                     note: {note}")
        while att.step():
            pass
        ok, why = att.verdict()
        results[n] = ok
        score += ok
        print(f"  {n:2d} {p.title:12s} {'O' if ok else 'X'}   {why:34s} {att.gauge}   ({time.time() - t0:.0f} s)",
              flush=True)
    print(f"SCORE {score} / {len(problems)}    {marks_line(problems, results)}    scorer code {scorer_code(lab_file)}")
    return score


# --- 4. the viewer ------------------------------------------------------------------------------

def run_viewer(problems: dict, path, lab_file, *, idle_model, idle_data, keys_help="", on_key=None,
               camera=None, title="") -> int:
    """Play problems live: 1..9 (0 = 10) play one, SPACE again, G grade all, ENTER print the gauge.

    ``idle_model``/``idle_data`` are shown before any problem is played. Each
    attempt brings its own model and data; the viewer is reopened when the
    model changes (weeks whose problems use different scenes). ``on_key`` gets
    any other key code first and may return True to swallow it.
    """
    import mujoco

    from . import launch_viewer, terminal_keys

    BIG, NORMAL = mujoco.mjtFont.mjFONT_BIG, mujoco.mjtFont.mjFONT_NORMAL
    TL, TR, BL = (mujoco.mjtGridPos.mjGRID_TOPLEFT, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                  mujoco.mjtGridPos.mjGRID_BOTTOMLEFT)
    digits = {48 + (i % 10): i for i in range(1, 11) if i in problems}
    state = {"cmd": None, "number": min(problems), "attempt": None, "msg": "", "results": {},
             "grading": None, "final": None, "name": ""}

    def key(keycode):
        if on_key is not None and on_key(keycode):
            return
        if keycode in digits:
            state["cmd"] = ("play", digits[keycode])
        elif keycode == 32:
            state["cmd"] = ("play", state["number"])
        elif keycode == 71:
            state["cmd"] = ("grade", None)
        elif keycode in (257, 335):
            state["cmd"] = ("print", None)

    def start(n, grading=False):
        state["number"] = n
        try:
            att, state["name"] = _start(problems, n, path)
        except AnswersError as e:
            state["attempt"], state["msg"] = None, str(e)
            print(f"[{n}] {e}")
            if grading:
                state["results"][n] = False
            return False
        for note in att.notes:
            print(f"[{n}] note: {note}")
        state["attempt"], state["msg"], state["t0"], state["steps"] = att, "", time.perf_counter(), 0
        return True

    def finish():
        att, n = state["attempt"], state["number"]
        ok, why = att.verdict()
        state["results"][n] = ok
        state["msg"] = ("PASS  " if ok else "FAIL  ") + why
        print(f"[{n} {problems[n].title}] {'PASS' if ok else 'FAIL: ' + why}   {att.gauge}")
        state["attempt"] = None
        if state["grading"] is not None:
            state["grading"].append((n, ok))
            advance()

    def advance():
        while True:
            left = [n for n in problems if n not in {m for m, _ in state["grading"]}]
            if not left:
                score = sum(ok for _, ok in state["grading"])
                stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                state["final"], state["grading"] = (score, stamp), None
                print(f"\n=== {state['name']}: SCORE {score} / {len(problems)}   "
                      f"{marks_line(problems, state['results'])}   scorer code {scorer_code(lab_file)}   {stamp} ===\n")
                return
            if start(left[0], grading=True):
                return
            state["grading"].append((left[0], False))

    def hud():
        if state["final"] is not None and state["attempt"] is None and state["grading"] is None:
            score, stamp = state["final"]
            return [(BIG, TL, f"{state['name']}\nSCORE  {score} / {len(problems)}\n"
                              f"{marks_line(problems, state['results'])}", ""),
                    (BIG, BL, f"scorer {scorer_code(lab_file)}   {stamp}", "")]
        n = state["number"]
        head = "GRADING  " if state["grading"] is not None else ""
        att = state["attempt"]
        body = problems[n].text
        if att is not None:
            body += f"\n{att.gauge}" + ("    GOAL REACHED" if att.met else "")
        elif state["msg"]:
            body += f"\n>> {state['msg']}"
        body += "\nkeys: " + " ".join(str(k % 10) for k in problems) + " play   SPACE again   G grade   " + keys_help
        return [(BIG, TL, f"{head}{n}. {problems[n].title}", ""), (NORMAL, BL, body, ""),
                (NORMAL, TR, f"{title}\n{marks_line(problems, state['results'])}", "")]

    print(f"answers file: {path}\nscorer code: {scorer_code(lab_file)}")
    terminal_keys(key)
    print("  [terminal] keys dead in the window? press them here instead; 'q' stops.", flush=True)
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)

    model, data = idle_model, idle_data
    while time.time() < deadline:
        reopen = False
        with launch_viewer(model, data, passive=True, key_callback=key) as viewer:
            if camera is not None:
                camera(viewer)
            while viewer.is_running() and time.time() < deadline:
                cmd, state["cmd"] = state["cmd"], None
                if cmd is not None and state["grading"] is not None and cmd[0] != "grade":
                    cmd = None
                if cmd is not None:
                    kind, arg = cmd
                    if kind == "play":
                        state["final"] = None
                        start(arg)
                    elif kind == "grade":
                        state["results"], state["grading"], state["final"] = {}, [], None
                        print(f"\n=== GRADING: all {len(problems)} problems ===")
                        advance()
                    elif kind == "print" and state["attempt"] is not None:
                        print(f"  [{state['number']}] {state['attempt'].gauge}")
                att = state["attempt"]
                if att is not None and att.model is not model:
                    model, data, reopen = att.model, att.data, True
                    break
                if att is not None:
                    # Real time when the laptop keeps up, slow motion when it does not.
                    due = int((time.perf_counter() - state["t0"]) / att.dt)
                    for _ in range(min(max(due - state["steps"], 0), 20)):
                        state["steps"] += 1
                        if not att.step():
                            finish()
                            break
                    if state["attempt"] is att and due - state["steps"] > 20:
                        state["t0"] = time.perf_counter() - state["steps"] * att.dt
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    if state["attempt"] is not None:
                        state["attempt"].draw(viewer.user_scn)
                pictures = state["attempt"].images() if state["attempt"] is not None else None
                if pictures:
                    viewer.set_images(pictures)
                else:
                    viewer.clear_images()
                viewer.set_texts(hud())
                viewer.opt.geomgroup[:3] = 1        # digits 0-5 also toggle geom groups; keep the robot visible
                viewer.sync()
                time.sleep(0.005)
            if not reopen:
                break
    return 0


def sphere(scn, pos, rgba, radius=0.04):
    """Append a sphere marker to a viewer scene."""
    import mujoco

    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                        np.asarray(pos, float), np.eye(3).flatten(), np.asarray(rgba, float))
    scn.ngeom += 1


def arrow(scn, origin, vector, rgba, radius=0.012):
    """Append an arrow from ``origin`` along ``vector`` (its length is the arrow's)."""
    import mujoco

    v = np.asarray(vector, float)
    n = float(np.linalg.norm(v))
    if scn.ngeom >= scn.maxgeom or n < 1e-9:
        return
    quat, mat = np.zeros(4), np.zeros(9)
    mujoco.mju_quatZ2Vec(quat, v / n)
    mujoco.mju_quat2Mat(mat, quat)
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_ARROW, np.array([radius, radius, n]),
                        np.asarray(origin, float), mat, np.asarray(rgba, float))
    scn.ngeom += 1
