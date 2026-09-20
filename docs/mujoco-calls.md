# MuJoCo, one call at a time: where each idea of the course meets the API

Every week's lab scripts call MuJoCo directly. This page maps each idea to the
call that implements it, the array it reads or writes, and the script that
shows it in use. Read it beside the code; the scripts are the explanation.

Two conventions run through everything:

- **No `x` prefix means "declared in the MJCF, relative to the parent"; an `x`
  prefix means "computed, in world coordinates".** `body_pos` vs `xpos`,
  `site_pos` vs `site_xpos`. `mj_forward` fills every `x` quantity from `qpos`.
- **`qpos` and `qvel` are different arrays with different indices.** The
  floating pelvis takes 7 numbers of `qpos` (position + quaternion) and 6 of
  `qvel`, so every joint's `qvel` index is its `qpos` index minus one. Jacobians
  and forces live in `qvel` space; angles live in `qpos`.

## The model and the state

| Idea | Call / array | Week, script |
| --- | --- | --- |
| Compile MJCF text into a model | `mujoco.MjModel.from_xml_string(xml)`; errors are precise, read them | 01 `mjcf_run.py`, `lab_mjcf.py` |
| Load the G1 | `soc4180.load_g1()` (Menagerie, pinned) | every week |
| A state, set to a keyframe | `soc4180.keyframe_data(model, "stand")` = `MjData` + `mj_resetDataKeyframe` | 00 `stack.py` |
| The state arrays | `data.qpos` (nq), `data.qvel` (nv), `data.ctrl` (nu) | 02b `anatomy.py` prints the map |
| Which slot is which joint | `model.jnt_qposadr[j]`, `model.jnt_dofadr[j]`, `mj_name2id(model, mjOBJ_JOINT, name)` | 02b `anatomy.py`, 01 `mjcf_run.py` |
| An actuator's joint | `model.actuator_trnid[:, 0]`, then `jnt_qposadr` of it | 02 `lab_viewer.py` (sliders), 05 `servo.py` |
| Joint limits | `model.jnt_range[j]` | 03 `reach.py` (the clip) |
| Physics options | `model.opt.timestep`, `model.opt.gravity`, `model.opt.disableflags` | 00 `stack.py`, 01 `mjcf_run.py` |
| Switch every servo off | `model.opt.disableflags \|= mjDSBL_ACTUATION` | 00 `stack.py --layer limp` |
| Many robots in one model | `MjSpec.from_string`, `MjSpec.from_file(str)`, `worldbody.add_frame()`, `frame.attach_body(child.worldbody.first_body(), "r0_", "")`, `compile()` | 10 `many.py`, `lab_many.py` |

## Kinematics: positions from angles

| Idea | Call / array | Week, script |
| --- | --- | --- |
| Forward kinematics (MuJoCo's) | `mujoco.mj_forward(model, data)`: `qpos` → `xpos`, `xmat`, `site_xpos`, `site_xmat` | 02 `fk.py` step 1 |
| Forward kinematics (by hand) | `model.body_pos`, `model.body_quat`, `model.jnt_axis`, `model.jnt_pos`, `model.site_pos`; `mju_quat2Mat`, `mju_axisAngle2Quat` | 02 `fk.py` step 2, `lab_viewer.py chain_fk` |
| Where the foot is | `data.site_xpos[kin.foot_site_id(model, "left")]` | 02, 03 |
| Which way the foot faces | `data.site_xmat[site].reshape(3, 3)` | 03 `reach.py` (the level-foot target) |
| Pose a named part | `soc4180.set_pose(model, data, left_elbow=1.2)`; chains in `bodies.CHAINS` | 02b `anatomy.py`, `lab_body.py` |
| Mirror a pose | `soc4180.mirror(dict)`: roll and yaw flip sign, pitch does not | 02b |

## Differential kinematics and IK

| Idea | Call / array | Week, script |
| --- | --- | --- |
| The site Jacobian | `mujoco.mj_jacSite(model, data, jac_p, jac_r, site)` writes into two 3 × nv arrays; pick the leg's columns with `kin.leg_dof_indices` | 03 `reach.py` step 3, `lab_ik.py Leg.jacobian` |
| The pose error a step must close | `kin.pose_error(data, site, target_pos, target_mat)`: 3 metres + 3 radians (a rotation vector) | 03 |
| One damped step | `J.T @ np.linalg.solve(J @ J.T + lam**2 * I, err)` | 03 `dls_step` |
| Singular values, condition number | `np.linalg.svd(J, compute_uv=False)` | 03 `reach.py --jacobian` |
| Apply the step, inside the limits | `data.qpos[idx] = np.clip(data.qpos[idx] + dq, lo, hi)`; then `mj_forward` | 03 |

## Dynamics: forces, contacts, servos

| Idea | Call / array | Week, script |
| --- | --- | --- |
| One physics step | `mujoco.mj_step(model, data)`: reads `ctrl`, computes forces and contacts, integrates by `opt.timestep` | 00, 01, 04, 05 |
| The contacts | `data.ncon`, `data.contact[i].pos`, `.frame`; `mujoco.mj_contactForce(model, data, i, wrench)` (contact frame, normal first) | 00 `stack.py` (ΣF = mg), 04 `walk.py` (ZMP) |
| The position servo's law | `tau = kp (ctrl − q) − kv q̇`; stored as `actuator_gainprm[a, 0] = kp`, `actuator_biasprm[a, 1:3] = (−kp, −kv)`; the engine's answer is `data.actuator_force` | 05 `servo.py`, `lab_servo.py` |
| Change the gains, set a limit | `soc4180.scale_gains(model, kp_scale, kv_scale)`, `soc4180.set_torque_limit(model, N)` → `actuator_forcerange` | 05 |
| The mass matrix | `mujoco.mj_fullM(model, data, M)` (dense nv × nv; 3.12 takes `data`); `M[dof, dof]` is a joint's inertia; ζ = kv / 2√(kp M) | 05 `servo.py` step 2 |
| Apply your own torque | `data.qfrc_applied[dof] = tau` before `mj_step` (unstable at 500 Hz for a kp = 500 spring; fine at 1 kHz) | 05 `lab_servo.py` `H` |
| Push the robot | `data.xfrc_applied[body, :3] = force` (world frame) | 07 `env_run.py --push` |
| The integrator gave up | `data.warning[mjWARN_BADQACC].number` > 0: the state was reset, `qpos` is not a result | 01 `mjcf_run.py --timestep 0.02` |

## Sensing

| Idea | Call / array | Week, script |
| --- | --- | --- |
| The IMUs | `nsensor = 4` is two gyro + accelerometer pairs; `soc4180.read_imu(model, data, "torso")` picks six numbers out of `data.sensordata` | 06 `imu.py`, `lab_imu.py` |
| Tilt from the accelerometer | `atan2(ay, az)`, `atan2(−ax, hypot(ay, az))` — gravity is wherever the specific force points | 06 |
| The truth (which a real robot never has) | `data.site_xmat[imu].reshape(3, 3)` | 06 `true_tilt` |
| Gravity in the body frame | `R.T @ (0, 0, −1)`; it is `obs[0:3]` of the environment | 06, 07 |

## Walking (the package, calling the above)

| Idea | Call / array | Week, script |
| --- | --- | --- |
| The gait | `soc4180.GaitParams(...)`, `soc4180.footstep_plan(params)` | 04 `walk.py` step 1 |
| The pendulum | `LIPM.evolve(x0, v0, zmp, t)`; by hand: `zmp + (x0 − zmp) cosh ωt + (v0/ω) sinh ωt` | 04 `predict_com` |
| Plan → LIPM → IK → 29 targets | `WalkingController(model, params)`, `.initial_data()`, `.control(t)`, `.zmp(data)`, `.last_ik_error` | 04 `walk.py` step 3 |

## Learning (gymnasium and stable-baselines3, on top of the above)

| Idea | Call / array | Week, script |
| --- | --- | --- |
| The MDP written down | `soc4180.envs.G1WalkEnv(action_scale, control_hz, reward_weights)`; `observation_space`, `action_space`, `reward_weights`, `max_steps`, `control_dt` | 07 `env_run.py` |
| One decision | `obs, r, terminated, truncated, info = env.step(a)`; `info` carries every reward term | 07, 08, 09 |
| The week 4 walker as actions | `soc4180.envs.walker_actions(env, controller, t)` | 07 |
| A policy and its sampling | `PPO("MlpPolicy", env, policy_kwargs=dict(log_std_init=…))`, `predict(obs, deterministic=True)`, `policy.log_std`; a sample is `mean + exp(log_std) · N(0, 1)`, clipped | 08 `train.py`, `lab_train.py` |
| Training with progress | `agent.learn(total_timesteps, callback=BaseCallback)`, `ep_info_buffer` | 08, 09 |
| Change the reward without touching the env | `gym.Wrapper.step` adding a term; potential-based `γΦ(s′) − Φ(s)` | 09 `shape.py`, `lab_reward.py` |
| Throughput | `multiprocessing.get_context("spawn").Pool(n).map(...)`; robot-steps per second | 10 `many.py` |

## Showing it

| Idea | Call / array | Week, script |
| --- | --- | --- |
| The interactive window (lab scripts only, never a notebook cell) | `soc4180.launch_viewer(model, data, passive=True, key_callback=fn)`; `viewer.sync()`, `viewer.is_running()`, `viewer.lock()` | every lab |
| Keys from the terminal too | `soc4180.terminal_keys(fn)` feeds the same callback | every lab |
| Markers | `viewer.user_scn.geoms[i]` with `mujoco.mjv_initGeom(geom, mjGEOM_SPHERE / BOX / ARROW, size, pos, mat, rgba)`; `mju_quatZ2Vec` for an arrow's direction | 00, 03, 04, 06, 07 |
| Close by itself (the smoke test) | `SOC4180_AUTOCLOSE=<seconds>` | `scripts/check_labs.py` |
| Replay a recorded motion | write `qpos`, `mj_forward`, `sync()` — no physics re-run | every pipeline script's last step |
| Video for the notebook | `soc4180.render_rollout(...)`, `render_poses(...)` (EGL / OSMesa chosen at import) | the decks |
