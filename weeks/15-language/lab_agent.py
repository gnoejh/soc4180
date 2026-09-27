"""Week 15 lab game, on your laptop: five instructions, one small agent, a score out of 5.

    uv run weeks/15-language/lab_agent.py          # torch: uv sync --extra rl

YOU EDIT ONLY `instructions.py`, next to this file. The robot is given a
sentence. soc4180.agent turns it into a PLAN of skills by keywords (parse),
and runs the plan (run): each skill moves the robot under physics and then
CHECKS, with a sensor, whether it worked -- the colour of the pixels it is
looking at, the height of a hand, whether the pelvis is still up. The plan
stops at the first failed check.

The skills (soc4180.agent.SKILLS):

    ("look", words)         turn the waist until the network (week 15's HeatNet)
                            puts what the words name in the middle of the picture
    ("search", words)       look; if the check fails, turn the waist and look again
    ("raise", "left"|"right")   one arm overhead
    ("raise_both",)         both arms at once
    ("lower", "left"|"right")   one arm back down

You write what the keyword parser cannot know: words it has never seen
(SYNONYMS), whole plans for phrases (PROGRAMS), how to search (SEARCH), and
how hard to turn (LOOK_GAIN).

    1 ... 5        run instruction 1 ... 5 (the plan runs, then plays back live)
    SPACE          the current problem again
    G              GRADE: all five; the score stays on screen
    ENTER          print the live gauge in this terminal
    Keys go to the MuJoCo window -- click it first. If no key works there,
    press the same keys in this terminal (single keys, no enter). `q` quits.

    uv run weeks/15-language/lab_agent.py --grade --no-viewer
    uv run weeks/15-language/lab_agent.py --say "look at the red ball and raise your left hand"

Things measured with this file's own code before it was written (--grade):

- "look at the green ball": parsed to one look; the waist turns +10 deg and
  the check finds green where the network points.
- "face the crimson ball": the parser finds no colour it knows and the plan
  fails at once; the network, asked anyway, points at random (week 15's
  table: 31 deg error for a word it never saw).
- "raise both hands": parsed to raise_both, and the robot falls on its face
  -- week 3's finding, found again. Left then right: both hands at 1.36 m.
- "find the yellow ball" with yellow 110 deg to the left, out of the
  picture: look fails its colour check; search turns +60, -60, +120 and
  finds it after 3 turns.

Everything in this file is complete and explained. You are not meant to edit
it: it is the referee, and the grading screen shows a 4-letter code computed
from it so the instructor can see it is unchanged.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np

import soc4180
from soc4180.game import AnswersError, Attempt, Problem, grade_headless, need, number, run_viewer

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE / "checkpoints" / "heatnet.pt"
BEHIND = {"red": (0.6, -0.2, 1.6), "green": (0.2, 0.1, 1.8), "blue": (-0.2, -0.1, 1.5),
          "yellow": (math.radians(110), 0.0, 1.8)}
_WORLD = {}


def world():
    """Built once: the body in the four-ball scene, and the eyes (HeatNet from the shipped checkpoint)."""
    if not _WORLD:
        from soc4180 import agent

        body = agent.Body()
        _WORLD.update(agent=agent, body=body, eyes=agent.Eyes(body, CHECKPOINT))
    return _WORLD


def settings_from(a):
    """The skill settings the answers control."""
    gain = number({"LOOK_GAIN": need(a, "LOOK_GAIN")}, "LOOK_GAIN", "LOOK_GAIN", 0.0, 5.0)
    search = need(a, "SEARCH")
    if not isinstance(search, dict):
        raise AnswersError("SEARCH must be a dict like {\"step_deg\": 30, \"tries\": 2}")
    step = number(search, "step_deg", "SEARCH", 5.0, 180.0)
    tries = int(number(search, "tries", "SEARCH", 0, 10))
    return {"look": {"gain": gain}, "search": {"step_deg": step, "tries": tries}}


def synonyms_from(a):
    syn = need(a, "SYNONYMS")
    if not isinstance(syn, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in syn.items()):
        raise AnswersError("SYNONYMS must map words to words, like {\"scarlet\": \"red\"}")
    return syn


def programs_from(a):
    progs = need(a, "PROGRAMS")
    if not isinstance(progs, dict):
        raise AnswersError("PROGRAMS must map phrases to plans, like {\"hands up\": [[\"raise\", \"left\"]]}")
    for phrase, plan in progs.items():
        if not isinstance(plan, list) or not all(isinstance(s, list) and s and isinstance(s[0], str) for s in plan):
            raise AnswersError(f"PROGRAMS[\"{phrase}\"]: a plan is a list of steps, each a list like [\"raise\", \"left\"]")
    return progs


class AgentAttempt(Attempt):
    """Parse the instruction, run the plan headless (it takes a few seconds), then play it back live."""

    def __init__(self, instruction, answers, *, balls=None, final_check=None):
        super().__init__()
        w = world()
        self.agent, self.body, self.eyes = w["agent"], w["body"], w["eyes"]
        self.model, self.data = self.body.model, self.body.data
        self.instruction, self.balls, self.final_check = instruction, balls, final_check
        self.settings = settings_from(answers)
        self.plan = self.agent.parse(instruction, synonyms_from(answers), programs_from(answers))
        self.dt = 17 * self.model.opt.timestep                  # one recorded frame
        self.results, self.frames, self.k, self.done_running = None, [], 0, False
        self.gauge = f"\"{instruction}\"  ->  plan {self.plan}"
        self.notes.append(f"\"{instruction}\" -> {self.plan}")

    def step(self):
        if not self.done_running:
            self.body.reset(self.balls)
            self.results = self.agent.run(self.body, self.eyes, self.plan, **self.settings)
            self.frames = list(self.body.frames)
            self.final = self.data.qpos.copy()
            self.done_running = True
            return bool(self.frames)
        if self.k >= len(self.frames):
            self.data.qpos[:] = self.final
            mujoco.mj_forward(self.model, self.data)
            return False
        self.data.qpos[:] = self.frames[self.k]
        self.data.mocap_pos[:] = self.body.balls
        mujoco.mj_forward(self.model, self.data)
        self.k += 1
        done = sum(1 for r in self.results if r.ok)
        self.gauge = (f"\"{self.instruction}\"   plan {self.plan}   steps passed {done} of {len(self.plan)}"
                      + (f"   STOPPED: {self.results[-1].message}" if self.results and not self.results[-1].ok else ""))
        self.met = self.results is not None and len(self.results) == len(self.plan) and all(r.ok for r in self.results)
        return True

    def verdict(self):
        if self.results is None:
            return False, "did not run"
        if not self.plan:
            return False, "the plan is empty"
        bad = next((r for r in self.results if not r.ok), None)
        if bad is not None:
            return False, bad.message
        if len(self.results) < len(self.plan):
            return False, "the plan did not finish"
        if self.final_check is not None:
            return self.final_check(self.body)
        return True, "; ".join(r.message for r in self.results)


def both_up(body):
    zs = [float(body.data.xpos[body.hand[s]][2]) for s in ("left", "right")]
    if not body.standing:
        return False, "not standing at the end"
    if min(zs) < 1.2:
        return False, f"at the end the hands are at {zs[0]:.2f} and {zs[1]:.2f} m (both need 1.2)"
    return True, f"both hands up ({zs[0]:.2f}, {zs[1]:.2f} m), still standing"


def two(instr_a, instr_b):
    """A problem that must pass two instructions."""
    class Two(Attempt):
        def __init__(self, answers):
            super().__init__()
            self.parts = [AgentAttempt(instr_a, answers), AgentAttempt(instr_b, answers)]
            self.i = 0
            self.model, self.data, self.dt = self.parts[0].model, self.parts[0].data, self.parts[0].dt
            self.notes = self.parts[0].notes + self.parts[1].notes

        def step(self):
            while self.i < 2:
                if self.parts[self.i].step():
                    self.gauge = self.parts[self.i].gauge
                    self.met = all(p.met for p in self.parts)
                    return True
                self.i += 1
            return False

        def verdict(self):
            for p in self.parts:
                ok, why = p.verdict()
                if not ok:
                    return False, f"\"{p.instruction}\": {why}"
            return True, "both instructions done"
    return Two


PROBLEMS = {
    1: Problem("LOOK", "\"look at the green ball\"  (LOOK_GAIN)",
               lambda a: AgentAttempt("look at the green ball", a)),
    2: Problem("NEW WORDS", "\"face the crimson ball\" and \"look at the azure one\"  (SYNONYMS)",
               lambda a: two("face the crimson ball", "look at the azure one")(a)),
    3: Problem("BOTH HANDS", "\"raise both hands\" -- and still be standing  (PROGRAMS)",
               lambda a: AgentAttempt("raise both hands", a, final_check=both_up)),
    4: Problem("SEARCH", "\"find the yellow ball\" -- it is 110 deg to the left, out of sight  (SEARCH)",
               lambda a: AgentAttempt("find the yellow ball", a, balls=BEHIND)),
    5: Problem("ERRAND", "\"look at the blue ball, then raise your right hand, then find the yellow ball\"",
               lambda a: AgentAttempt("look at the blue ball, then raise your right hand, then find the yellow ball",
                                      a, balls=BEHIND)),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--instructions", type=Path, default=HERE / "instructions.py", help="the answers file")
    ap.add_argument("--grade", action="store_true")
    ap.add_argument("--problem", type=int, nargs="*")
    ap.add_argument("--say", default=None, help="run one instruction of your own with your answers file")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(argv)
    try:
        import torch
    except ImportError:
        print("This lab needs torch: run `uv sync --extra rl`, then try again.")
        return 1
    torch.set_num_threads(4)
    if args.say:
        from soc4180.game import read_answers
        att = AgentAttempt(args.say, read_answers(args.instructions))
        print(att.notes[0])
        while att.step() and not att.done_running:
            pass
        print("PASS" if att.verdict()[0] else "FAIL", "--", att.verdict()[1])
        return 0
    if args.grade or args.no_viewer:
        grade_headless(PROBLEMS, args.instructions, __file__, args.problem)
        return 0
    if soc4180.is_colab():
        print("This lab needs a desktop window; run it on your laptop.")
        return 1
    print(__doc__.split("\n\n")[1])
    w = world()

    def camera(viewer):
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.4, 200.0, -15.0
        viewer.cam.lookat[:] = (0.5, 0.0, 0.9)
    return run_viewer(PROBLEMS, args.instructions, __file__, idle_model=w["body"].model, idle_data=w["body"].data,
                      camera=camera, title="week 15: language")


if __name__ == "__main__":
    raise SystemExit(main())
