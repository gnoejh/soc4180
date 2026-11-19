# Project catalogue

Fifteen starting projects (P1-P13, P15, P16) plus P14, your own. Each one gives:

- **Question**: a starting question. Narrow it, and change the numbers.
- **Start from**: the course script or package code that already does most of the work.
- **Measured**: what the course has already measured, so you start from a known baseline.
  These numbers came from the scripts named, on the course machine. Re-measure them on
  your laptop in week 12; with falls, only the margins carry across machines.
- **Twin**: a fair comparison.
- **Stretch**: where to go if the first answer comes quickly.
- **Trap**: the mistake we already know about.

Difficulty: ● laptop, a few minutes per run · ●● needs care or an hour of runs ·
●●● learning or heavy compute, plan the budget in week 12.

Week numbers are the lectures: 4 walking, 5 actuation, 6 sensing, 7 MDPs, 7b tables,
8 learning to act, 9 scale and robustness, 10 planning and imitation, 11 pixels and
language.

---

## P1. Walk farther ●

**Question.** How far and how fast can the analytic LIPM walker go before it falls,
and which gait parameter limits it?

- **Start from.** `weeks/04-walking/walk.py` (`--step-length --step-time --step-height
  --double-support --com-height --pelvis-height --steps`), `soc4180.walking.GaitParams`.
- **Measured (week 4).** Defaults: 0.989 m in 8.8 s. A rushed step falls at 3.2 s, no
  double support at 2.0 s. The defaults came from a sweep, and several nearby settings
  fall.
- **Twin.** The default gait, with one parameter changed.
- **Stretch.** Close the loop: feed the measured capture point back into the next
  footstep (`ξ = x + ẋ/ω`, week 4).
- **Trap.** Seeding IK from `stand` (a singular, straight leg). Seed from the crouch,
  as `WalkingController.nominal` does.

## P2. The walker on hard ground ●●

**Question.** Which disturbance does the open-loop walker tolerate least: low gravity,
ice, a slope, a payload, or weak motors?

- **Start from.** `walk.py --gravity --friction`. Add slopes and payloads with `MjSpec`
  (week 1). Weaken the motors with `weeks/05-actuation/servo.py --walk --limit
  --kp-scale`.
- **Measured (weeks 4-5).** Moon gravity (1.62) falls at 5.3 s. Ice (friction 0.2)
  changes nothing for this gait. With every actuator limited to 50 N·m it falls; at
  55 N·m it walks. Half and double stiffness both fall.
- **Twin.** Flat floor, Earth gravity, the same gait.
- **Stretch.** Find the smallest change to `GaitParams` that restores walking under the
  worst disturbance.
- **Trap.** The controller is not told about the disturbance (on purpose). Say whether
  your fix is allowed to know.

## P3. Balance from the robot's own senses ●●

**Question.** The ankle strategy survives 75 N when it reads the *true* tilt. Does it
still survive when it reads the tilt from a noisy IMU through week 6's complementary
filter?

- **Start from.** [`templates/project_template.py`](templates/project_template.py)
  (it already runs the ankle strategy against hold), `weeks/06-sensing/imu.py`
  (`--noise --alpha --sweep`), `soc4180.estimation`.
- **Measured (weeks 6-7).** Hold falls at 70 N and the ankle strategy at 80 N. On the
  noisy walk the filter's error is 1.8° at α 0.995, with a minimum of 1.15° near
  α 0.998. `G1WalkEnv`'s gravity comes from the true pelvis orientation, not from a
  filter.
- **Twin.** The same controller fed the true gravity vector.
- **Stretch.** Sweep α and the gyro bias. Is the best α for balance the best α for
  estimation?
- **Trap.** The torso IMU and the pelvis differ by up to 0.70° while moving, so compare
  against the frame the filter estimates.

## P4. Reach while standing ●●

**Question.** How far can the G1 reach with one hand while standing on its servos, and
is the limit kinematic (IK cannot get there) or balance (the centre of mass leaves the
feet)?

- **Start from.** `soc4180.kinematics` (damped least-squares IK), `soc4180.bodies`
  (`CHAINS`, `set_pose`), week 3's `lab_connected.py` (which poses fall under gravity).
- **Measured (weeks 2-4).** The foot is 0.170 m long (heel −0.05, toe +0.12 m). There
  are no hand sites: use `xpos` of `left_wrist_yaw_link`. The leg's reach runs from
  0.041 to 0.641 m.
- **Twin.** The same targets solved by arm-only IK against arm + waist, or with and
  without a centre-of-mass row.
- **Stretch.** Lean the pelvis (a whole-body task) to extend the reach.
- **Trap.** `set_pose` pins the pelvis, so a pose that looks fine kinematically can
  fall under physics. Test every reach with `mj_step`.

## P5. Push recovery by planning ●●

**Question.** MPPI survives a 300 N shove by stepping, then walks about 3 m sideways,
because nothing in its cost rewards stopping. Can a cost term make it recover *and*
stop?

- **Start from.** `weeks/10-planning-and-imitation/lab.py`, part 1: its `cost()`,
  `SAMPLES / HORIZON / SIGMA / TEMPERATURE` and `--push`; `soc4180.planning.MPPI`.
- **Measured (week 10).** Hold falls at 70 N. MPPI with 64 samples × 0.4 s stands at
  150 and 300 N and drifts about 3 m. Two samples fall at 150 N. 16 samples with σ 0.25
  stand at 300 N. About 115 ms per decision on 16 threads (0.35x real time). Week 10's
  `lab.py` as shipped: stands at 150 N, feet moving 3.04 m in total, 81-98 ms per decision.
- **Twin.** The lecture's cost, the same push, the same seed.
- **Stretch.** Pushes from the front and back, not only sideways.
- **Trap.** `xfrc_applied` is not part of the planner's state: it never sees the push
  coming. Keep it that way, or say so.

## P6. New tasks for the planner ●●

**Question.** Which whole-body tasks can MPPI do with a cost alone (squat, raise a
hand, stand on one foot, look at a point), and which weights does each need?

- **Start from.** week 10's `lab.py` `cost()` and `planning_model()` (it adds
  sensors for the feet, hands, head and CoM; read them with `r.foot()`, `r.hand()`,
  `r.com`). The old crane and squat costs are in git:
  `git show 7e79ce0:weeks/11-planning/plan.py`.
- **Measured (week 10).** Squat with a height weight of 20 never moves (every sample
  that goes down also tilts: a local minimum). With weight 100 it reaches 0.503 m. A
  first-draft crane passed at one weight and failed at half and double.
- **Twin.** The same task at the neighbouring weights (×0.5, ×2). A result is only
  robust if both neighbours agree.
- **Stretch.** Chain two tasks (squat, then raise a hand) by switching costs.
- **Trap.** Knife-edge weights. Report the range that works, not one value.

## P7. A planner with a wrong model ●●

**Question.** How wrong can MPPI's internal model be before it stops saving the robot,
and is mass, friction or latency the error that hurts most?

- **Start from.** week 10's `lab.py`: give `planning.MPPI` its own copy of the
  model (`planning.planning_model()`) and change only that copy (mass, friction, a
  delay).
- **Measured (week 10).** A −10 kg model falls at 150 N. +10 and +20 kg stand at 150 N
  and fall at 300 N. +40 kg falls at 150 N.
- **Twin.** The exact model.
- **Stretch.** More samples or a longer horizon: do they buy back the model error?
- **Trap.** Changing the robot instead of the planner's copy. Only the model should be
  wrong.

## P8. Learned push recovery: randomisation that helps ●●●

**Question.** Which domain randomisation makes the PPO push-recovery policy survive
which test (bigger pushes, other directions, a heavier robot)?

- **Start from.** `weeks/09-scale-and-robustness/lab.py` (`WORLDS`, `PUSHES`,
  `--any-direction`) and the shipped checkpoints `checkpoints/push_nominal.zip`,
  `push_random.zip`. To retrain: `soc4180.envs.G1PushEnv(world=...)` with SB3's
  `PPO` and `SubprocVecEnv` (the old training script: `git show
  7e79ce0:weeks/12-robustness/robust.py`).
- **Measured (week 9).** Each checkpoint is 3 M steps, about 16 minutes on 36 cores
  (hours on a laptop). Week 9's `lab.py` table (10 episodes): in the nominal world PPO
  random survives 100 N 9 of 10, PPO nominal 80 N; with +15 kg PPO random survives
  every push to 120 N while holding falls with no push; on ice PPO random survives
  100 N 10 of 10. A privileged teacher was *worse* in the nominal world (3 of 10 at 60 N).
- **Twin.** The nominal policy on the same test.
- **Stretch.** One new randomisation (mass, friction, servo gain), trained small, then
  tested.
- **Trap.** Evaluate with pushes of exact size and direction. Random-direction pushes
  are harder (front and back) and once made randomisation look useless.

## P9. Reward design for stepping: diagnose, don't promise ●●●

**Question.** Why does PPO on `G1WalkEnv` stand still, and which reward change moves
the optimum toward stepping without the robot falling?

- **Start from.** `G1WalkEnv(reward_weights=...)` and `DEFAULT_REWARD` in
  `soc4180.envs`, with SB3's `PPO`; week 8's `lab.py` `reward()` for a small first
  version (the old ablation script: `git show 7e79ce0:weeks/09-reward/shape.py`).
- **Measured (week 8).** Walking at the target is worth 1250, standing 776, a lunge and
  fall 52. At 30k steps every variant stands. `stand_still` and `air_time` *together*
  break standing: 224 steps, then it falls backwards (−0.524 m, feet up 0.08). A
  per-step alive bonus is what makes standing profitable.
- **Twin.** The default reward with the same budget and seed.
- **Stretch.** A potential-based term (`--extra potential`): does it change the optimum
  at all?
- **Trap.** Grading convergence. The result is the diagnosis: reward by term, survival
  under exploration noise, the policy's std.

## P10. Distil the expert ●●●

**Question.** Which controllers can a small network copy from what the robot senses,
and does DAgger rescue the ones that cannot be copied?

- **Start from.** week 10's `lab.py`, part 2 (`EXPERT`, `INPUTS`, `fit`), and
  `soc4180.imitation` (`Student`, `dagger`, `shove_test`) for a network and DAgger.
- **Measured (week 10).** The rule: the expert's action must be a function of what the
  student sees, deterministic, and closed loop. The walker is open loop (indexed by
  time): clock + time walks, and adding sensors lowers the loss yet falls. The PPO
  policy clones from its own 42 numbers. MPPI's student falls at 60 N, where doing
  nothing survives, and DAgger makes the loss *rise*. DAgger results fluctuate between
  rounds (3 → 19 → 16), so quote the final round over 20 episodes. Week 10's `lab.py`
  (a linear copy, least squares): the ankle rule copied from tilt alone equals the
  expert (stands at 60 and 75 N); from 29 inputs, with a 0.003 rad fit error, it falls
  at 60 N. MPPI copied from tilt alone stands at 60 and 75 N, from 29 inputs at none.
- **Twin.** The same student and data, with one feature set changed.
- **Stretch.** Make MPPI copyable (fixed noise, more samples, a warm start) and test the
  rule.
- **Trap.** A lower imitation loss is not a better robot. Measure falls, not the loss.

## P11. Eyes on the ball ●●

**Question.** Can the G1 find a coloured ball with its head camera and turn (or point)
at it, and which visual change breaks it: colour, light or distance?

- **Start from.** `weeks/11-pixels-and-language/lab.py` (`find_ball`, `GAIN`,
  `--dim`), and `soc4180.vision` (`render_dataset`, `BallNet`) for a learned
  detector. The camera is on `torso_link` and turns with the waist.
- **Measured (week 11).** `lab.py`'s colour detector centres every ball within 0.3°,
  but with a loose threshold (0.6) or dim light (`--dim 0.3`) it says "done" while up to
  39-41° off. A small CNN (`soc4180.vision.BallNet`) reached 1.0° on its test images and
  27.8° on an unseen colour. Re-measure with `torch.set_num_threads(4)`: the thread count
  changes the numbers.
- **Twin.** The network trained without randomisation, on the same test images.
- **Stretch.** Point a hand at the ball with IK (P4's tools) from the camera's estimate
  alone.
- **Trap.** After moving a mocap ball, call `mj_forward` before rendering, or the first
  frame shows it at its old place.

## P12. Words to actions ●●

**Question.** Extend the verifying agent with a new skill or a new word, and measure
when it should refuse.

- **Start from.** `weeks/11-pixels-and-language/lab.py` (`colour_of`, `obey`,
  `--say`) and `soc4180.agent` (`Body`, the skills `raise_arm`, `parse`, `run`).
- **Measured (week 11).** The 20k-parameter model is 2.0° on trained phrasings and 2.1°
  on new ones, and about 31° (chance) for an unseen word ("crimson"). Absent balls still
  get confident peaks (up to 0.98), so the agent checks the pixel colour where the
  network points. Week 11's `lab.py` refuses "crimson" and a hidden ball; with a loose
  colour threshold it "sees" the hidden ball 149° away and says "done".
- **Twin.** The agent without its verification step.
- **Stretch.** A two-step instruction ("look at the red ball, then raise the hand on
  that side").
- **Trap.** Trusting the network's confidence. Measure refusals on scenes without the
  ball.

## P13. A table that balances ●●

**Question.** Can tabular Q-learning, with the G1's tilt cut into bins, learn an ankle
rule that survives pushes as well as week 7's hand-designed ankle strategy?

- **Start from.** `soc4180.tabular`, `weeks/07b-tabular/qlearn.py --task balance
  --bins`, the [template](templates/project_template.py) as the push test.
- **Measured (weeks 7-7b).** Value iteration on the balance MDP gives J = 72.12. A table
  learned from samples is an inverted U in its size (19.1 / 51.1 / 65.5 / 66.1 / 53.3 at
  5 / 11 / 21 / 41 / 81 bins). A coarse pendulum table learns and then forgets, so keep
  the best (`KeepBest`).
- **Twin.** The ankle strategy at gains (2, 2).
- **Stretch.** Which observation does the table need? Tilt only, or tilt and rate
  (week 7b: angle only is not Markov)?
- **Trap.** `BalanceLearner.advance` has a known defect: `np.add.at` applies many
  updates to one entry per tick, so the 81-bin number is mostly the defect. Write your
  own update, one α step per entry per tick.

## P15. Choreograph from snapshots: how fast can the G1 dance? ●●

**Question.** You compose a dance from named whole-body snapshots. How fast can it be
played before the G1 falls, and which rule of choreography decides it?

- **Start from.** The lab `templates/actions_demo.py` (try every action and snapshot
  first), then the default solution `examples/p15_dance.py`, which is a complete project:
  question, prediction, twin, sweep, `p15_results.csv`, `CHECK`. Library:
  `soc4180.actions` (`G1.play(dance, tempo)`, `G1.check(dance)`, `SNAPSHOTS`).
- **Measured (the default solution, 25 tempos, 37 s on 36 cores).** The sample dance
  (18 snapshots, 19 s at tempo 1) stands at every tempo from 0.5 to 1.35 and falls from
  1.4. Its twin -- one glide changed so an arm swings out while a knee lifts -- falls at
  every tempo from 0.5 to 1.05, stands at 1.1-1.3 and falls from 1.35. The library
  measured too: a weight shift onto one foot stands in 1.0 s and falls in 0.8 s; squat
  depth 0.18 m stands, 0.21 falls; the soles are 5-6 cm wide.
- **Twin.** Your dance with one snapshot or one duration changed.
- **Stretch.** Find the rule behind your fastest tempo: which glide falls first, and does
  giving it more time (and the rest less) raise the limit? That is P16 by hand.
- **Trap.** Reading "falls at tempo x" off one run near the edge. Sweep the tempo in
  steps of 0.05 and report the range that stands. And slower is not always safer: the
  twin falls slow and stands fast, because the creep toward the sole's edge needs time.

## P16. Learn to dance faster ●●●

**Question.** Keep a dance's length fixed at a tempo where it falls. Can the G1 learn
how to split that time between its glides so that it stands -- and what does it learn
to spend the time on?

- **Start from.** `examples/p16_learn_timing.py` (default solution; reads P15's dance).
  The learner is week 8's antithetic parameter-space policy gradient, unchanged, on 18
  numbers (one per glide); the reward is the fraction of the dance done before a fall,
  plus 1 for standing.
- **Measured (seed 0, 30 updates of 8 pairs, 219 s on 36 cores; expect 15-30 minutes
  on a laptop; the run repeats exactly).** At target tempo 1.5, the default split falls; the learned
  split stands from update 18 on (14 of the 16 tries of update 30 stand). Tested: the
  default split stands up to 1.35, the learned one at every tested tempo from 1.0 to 1.5.
  It moved time to the weight shifts onto one foot (1.00 -> 2.33 s and 1.00 -> 1.55 s)
  and to the arms-up sways, and took it from the waves, twists and returns.
  Two first attempts failed, and both are worth knowing: lr 1.0 found splits that
  stood, then jumped into a split that rushes everything (stuck at 93 % of the dance);
  target 1.6 with lr 0.3 crept to 90 % in 25 updates and never stood.
- **Twin.** The same dance and length with the default split (`--updates 0`).
- **Stretch.** Reward smoothness too (the `tracking` number), or learn the balance
  gains together with the timing. Full reinforcement learning of the joints (PPO
  tracking the snapshots) is the research version; it needs a GPU and hours -- ask
  first.
- **Trap.** Partial credit can teach the robot to survive longer instead of finishing:
  watch whether the number of tries that *stand* rises, not only the mean reward. Keep
  the best theta (week 7b's KeepBest): the learner can find a good split and lose it.

## P14. Your own proposal

Any question about the simulated G1 that follows the method: measurable, predicted, with
a twin and a check, and runnable on a laptop with the lock file. Bring it to the week 12
clinic. It is approved when the question, the twin and the check rule fit on one line
each.
