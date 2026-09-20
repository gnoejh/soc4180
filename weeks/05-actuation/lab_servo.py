"""Week 5 lab, on your laptop: open the servos, then replace them with your own.

    uv run weeks/05-actuation/lab_servo.py

A window opens with the G1 walking the week 4 gait in real time. Every leg
joint carries a coloured sphere: GREEN is a motor loafing, RED is one at the
torque you set as the reference (the torque limit if there is one, else
100 N m). Watch which joints go red, and when.

    1 2 3 ...              switch to that entry of SERVOS and restart
    R                      restart the current entry
    H                      hand the servos over to YOUR torque law (see below)
    SPACE                  pause / resume
    ENTER                  sag, peak torque, the loudest joint, and how far your
                           torque law is from MuJoCo's
    double-click a body, then ctrl-drag    push the robot
    left-drag / right-drag / wheel   orbit / pan / zoom
    Keys go to the MuJoCo window, not the terminal -- click the window first.
    If no key does anything in the window, press the same key in this terminal
    instead -- single keys, no enter. `q` stops it.

MuJoCo's position actuator is a PD controller written into the model file:

    tau = kp * (ctrl - q) - kv * qdot

`torque_from_pd` below is that law, complete and explained. ENTER prints the
largest gap between it and `data.actuator_force` over the last second: around
1e-12, unless a torque limit is on, when MuJoCo clips and the law does not --
explain the gap you see.

Then press H. The model's gains are zeroed and the function IS the servo: its
torques go straight into `qfrc_applied`, at SERVO_HZ. Right, nothing changes.
Break it -- drop a term, flip a sign -- and the robot collapses: the week 0 rag
doll, because a servo with the wrong law is no servo. `servo.py` in this
folder measures one servo's step response and the walk under new gains.

SERVO_HZ is 1000, not the 500 the physics normally runs at, and that is a
lesson in itself: a spring this stiff, applied *explicitly* between steps,
is unstable at 500 Hz (it is week 1's exercise 14 again). MuJoCo's own servo
survives 500 Hz only because the engine integrates it implicitly. Set
SERVO_HZ = 500 and press H to see the difference.

Experiments, in order, and what to show the instructor:

1. ENTER while walking: the gap is near 1e-12. Say what the two terms of
   the law do (`servo.py` measures the knee's step response: kp x4 reaches
   1 % in 0.036 s instead of 0.13 s, with 2 % overshoot and 4x the torque;
   kv/4 overshoots 4.6 %; kv x4 takes 0.6 s).
2. Press H. The walk must continue exactly as before. Then break the law on
   purpose -- drop the damping term -- and press R then H. Describe what you see
   and name it (week 5 calls it ringing).
3. Press 2 (torque limit 50 N m). Which joints go red first, in which phase of
   the step, and what happens next? Then 3 (55 N m).
4. Press 4 and 5 (half and double stiffness). One sinks, one falls: say which,
   and why doubling kp does not simply make the robot "more accurate".
5. Set SERVO_HZ = 500, press H, and explain what you see using week 1.
6. Add an entry to SERVOS with the lowest kp that still walks eight steps.
   ENTER prints the mean sag; plot sag against kp on paper.
"""

from __future__ import annotations

import os
import time

import mujoco
import numpy as np

import soc4180
from soc4180.walking import GaitParams, WalkingController

# name, kp scale, kv scale, torque limit (N m, or None). Keys 1-9. Add your own.
SERVOS = [
    ("as shipped: kp 500, no limit",   1.0, 1.0, None),
    ("torque limit 50 N m",            1.0, 1.0, 50.0),
    ("torque limit 55 N m",            1.0, 1.0, 55.0),
    ("half stiffness",                 0.5, 1.0, None),
    ("double stiffness",               2.0, 1.0, None),
    ("quarter damping",                1.0, 0.25, None),
    # add your own below
]

REFERENCE_TORQUE = 100.0     # N m that paints a sphere fully red when there is no limit
SERVO_HZ = 1000              # rate of YOUR servo loop after H (500 is unstable: try it)
CONTROL_DT = 0.002           # the walker's targets update at 500 Hz regardless


# --- 1. the servo law ---------------------------------------------------------------

def torque_from_pd(kp, kv, ctrl, q, qdot):
    """The position servo, as arrays over all 29 actuators: tau = kp (ctrl - q) - kv qdot.

    kp, kv   gains, one per actuator          (from soc4180.gains)
    ctrl     the commanded angles             (data.ctrl)
    q, qdot  the joint angle and velocity each actuator drives

    The first term is a spring pulling the joint toward its target; the
    second is a damper resisting motion. With kv = 2 sqrt(kp M) the pair is
    critically damped, which every G1 leg joint is (zeta = 1.00 on all twelve).
    """
    return kp * (ctrl - q) - kv * qdot


# --- 2. drawing, and one run of one servo setting --------------------------------------

def sphere(geom, pos, rgba, radius):
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0, 0], dtype=float),
                        np.asarray(pos, dtype=float),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=float))


class Run:
    """One run of the walk under one (kp, kv, limit) setting.

    `scale_gains` rewrites the model's gainprm/biasprm, `set_torque_limit`
    its forcerange. `step()` is one 2 ms controller tick: while MuJoCo drives
    the servos it is one `mj_step`; after H it is SERVO_HZ/500 sub-steps in
    which `torque_from_pd` is written into `qfrc_applied` before each one.
    """
    base_gain = None
    base_bias = None
    qadr = None
    dadr = None

    def __init__(self, model, data, name, kp_scale, kv_scale, limit):
        self.name, self.limit = name, limit
        self.model, self.data = model, data
        model.actuator_gainprm[:] = self.base_gain
        model.actuator_biasprm[:] = self.base_bias
        model.opt.timestep = CONTROL_DT
        self.kp, self.kv = soc4180.scale_gains(model, kp_scale, kv_scale)
        soc4180.set_torque_limit(model, limit)
        self.controller = WalkingController(model, GaitParams(n_steps=8))
        fresh = self.controller.initial_data()
        mujoco.mj_resetData(model, data)
        data.qpos[:] = fresh.qpos
        mujoco.mj_forward(model, data)
        self.hand = False
        self.sag, self.gap, self.peak = [], [], np.zeros(model.nu)
        self.tau = np.zeros(model.nu)
        self.fell = False
        print(f"\n[{name}]  kp {self.kp[0]:.0f}  kv x{kv_scale}  limit {limit}")

    def my_torque(self):
        d = self.data
        mine = torque_from_pd(self.kp, self.kv, d.ctrl, d.qpos[self.qadr], d.qvel[self.dadr])
        if mine is None:
            return None
        mine = np.asarray(mine, float).reshape(self.model.nu)
        if self.limit is not None:
            mine = np.clip(mine, -self.limit, self.limit)
        return mine

    def step(self):
        """Advance one controller tick (2 ms): one physics step, or the servo loop."""
        m, d, c = self.model, self.data, self.controller
        if d.time >= c.total_time:
            return
        d.ctrl[:] = c.control(d.time)
        if not self.hand:
            mine = self.my_torque()
            mujoco.mj_step(m, d)
            self.tau = np.abs(d.actuator_force)
            if mine is not None:
                self.gap.append(float(np.abs(mine - d.actuator_force).max()))
                self.gap = self.gap[-500:]
        else:
            n_sub = max(int(round(CONTROL_DT * SERVO_HZ)), 1)
            for _ in range(n_sub):
                mine = self.my_torque()
                d.qfrc_applied[:] = 0.0
                if mine is not None:
                    d.qfrc_applied[self.dadr] = mine
                mujoco.mj_step(m, d)
            self.tau = np.abs(mine) if mine is not None else np.zeros(m.nu)
            if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number and not self.fell:
                print("  the simulation blew up (BADQACC): an explicit spring at this rate "
                      "is unstable -- week 1, exercise 14")
        self.peak = np.maximum(self.peak, self.tau)
        if d.time > c.params.settle_time:
            self.sag.append(c.targets_at(d.time)[0][2] - d.qpos[2])
        if d.qpos[2] < 0.5 and not self.fell:
            self.fell = True
            print(f"  FELL at t = {d.time:.2f} s after {d.qpos[0]:+.2f} m")

    def hand_over(self):
        self.hand = not self.hand
        m = self.model
        if self.hand:
            m.actuator_gainprm[:, 0] = 0.0
            m.actuator_biasprm[:, 1:3] = 0.0
            m.opt.timestep = 1.0 / SERVO_HZ
            print(f"  H: MuJoCo's servos are OFF; torque_from_pd is the servo now, at {SERVO_HZ} Hz")
        else:
            m.actuator_gainprm[:, 0] = self.kp
            m.actuator_biasprm[:, 1] = -self.kp
            m.actuator_biasprm[:, 2] = -self.kv
            m.opt.timestep = CONTROL_DT
            self.data.qfrc_applied[:] = 0.0
            print("  H: MuJoCo's servos are back")

    def report(self):
        d = self.data
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                 for i in range(self.model.nu)]
        loud = int(np.argmax(self.peak))
        sag = f"{np.mean(self.sag) * 1000:+.1f} mm" if self.sag else "n/a"
        gap = ("(torque_from_pd not written)" if not self.gap
               else f"{max(self.gap):.1e} N m over the last second")
        print(f"  t = {d.time:5.2f} s  x {d.qpos[0]:+.3f} m  pelvis z {d.qpos[2]:.3f}"
              f"  mean sag {sag}  {'FELL' if self.fell else 'upright'}"
              f"  servos: {'yours at %d Hz' % SERVO_HZ if self.hand else 'MuJoCo'}")
        print(f"  peak torque {self.peak[loud]:.1f} N m at {names[loud]}"
              f"   your law vs MuJoCo: {gap}")


# --- 3. the loop --------------------------------------------------------------------------

def main() -> int:
    if soc4180.is_colab():
        print("This lab needs a desktop window; run it on your laptop.")
        return 1

    model = soc4180.load_g1()
    data = mujoco.MjData(model)
    Run.base_gain = model.actuator_gainprm.copy()
    Run.base_bias = model.actuator_biasprm.copy()
    Run.qadr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    Run.dadr = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    leg = [a for a in range(model.nu)
           if any(k in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
                  for k in ("hip", "knee", "ankle"))]
    joint_body = model.jnt_bodyid[model.actuator_trnid[:, 0]]

    state = {"i": 0, "run": Run(model, data, *SERVOS[0]), "paused": False, "print": False}

    def on_key(keycode):
        if 49 <= keycode <= 57 and keycode - 49 < len(SERVOS):
            state["i"] = keycode - 49
            state["run"] = Run(model, data, *SERVOS[state["i"]])
        elif keycode == 82:                                  # 'R'
            state["run"] = Run(model, data, *SERVOS[state["i"]])
        elif keycode == 72:                                  # 'H'
            state["run"].hand_over()
        elif keycode == 32:
            state["paused"] = not state["paused"]
        elif keycode in (257, 335):
            state["print"] = True

    def draw(viewer, run):
        ref = run.limit if run.limit is not None else REFERENCE_TORQUE
        with viewer.lock():
            scn = viewer.user_scn
            n = 0
            for a in leg:
                f = float(min(run.tau[a] / ref, 1.0))
                sphere(scn.geoms[n], data.xpos[joint_body[a]], (f, 1 - f, 0.1, 0.8), 0.015 + 0.02 * f)
                n += 1
            scn.ngeom = n

    print("\n\n".join(__doc__.split("\n\n")[2:6]))
    soc4180.terminal_keys(on_key)
    print("  [terminal] keys dead in the window? press them here instead; 'q' stops.",
          flush=True)
    deadline = time.time() + float(os.environ.get("SOC4180_AUTOCLOSE") or 1e12)
    with soc4180.launch_viewer(model, data, passive=True, key_callback=on_key) as viewer:
        while viewer.is_running() and time.time() < deadline:
            wall = time.time()
            run = state["run"]
            if not state["paused"]:
                run.step()
            if state["print"]:
                state["print"] = False
                run.report()
            draw(viewer, run)
            viewer.sync()
            lag = CONTROL_DT - (time.time() - wall)
            if lag > 0:
                time.sleep(lag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
