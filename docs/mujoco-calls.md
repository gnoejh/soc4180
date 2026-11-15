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
| Joint limits | `model.jnt_range[j]` | 03 `lab_connected.py --joints` (ranges in degrees) |
| Physics options | `model.opt.timestep`, `model.opt.gravity`, `model.opt.disableflags` | 00 `stack.py`, 01 `mjcf_run.py` |
| Switch every servo off | `model.opt.disableflags \|= mjDSBL_ACTUATION` | 00 `stack.py --layer limp` |
| Many robots in one model | `MjSpec.from_string`, `MjSpec.from_file(str)`, `worldbody.add_frame()`, `frame.attach_body(child.worldbody.first_body(), "r0_", "")`, `compile()` | (old week 10 `many.py`, removed 2026-11-15; git 7e79ce0) |

## Kinematics: positions from angles

| Idea | Call / array | Week, script |
| --- | --- | --- |
| Forward kinematics (MuJoCo's) | `mujoco.mj_forward(model, data)`: `qpos` → `xpos`, `xmat`, `site_xpos`, `site_xmat` | 02 `fk.py` step 1 |
| Forward kinematics (by hand) | `model.body_pos`, `model.body_quat`, `model.jnt_axis`, `model.jnt_pos`, `model.site_pos`; `mju_quat2Mat`, `mju_axisAngle2Quat` | 02 `fk.py` step 2, `lab_viewer.py chain_fk` |
| Where the foot is | `data.site_xpos[kin.foot_site_id(model, "left")]` | 02, 03 |
| Which way the foot faces | `data.site_xmat[site].reshape(3, 3)` | 03 `lab_connected.py` (the planted feet IK holds) |
| Pose a named part | `soc4180.set_pose(model, data, left_elbow=1.2)`; chains in `bodies.CHAINS` | 02b `anatomy.py`, `lab_body.py` |
| Mirror a pose | `soc4180.mirror(dict)`: roll and yaw flip sign, pitch does not | 02b |

## Differential kinematics and IK

| Idea | Call / array | Week, script |
| --- | --- | --- |
| The site Jacobian | `mujoco.mj_jacSite(model, data, jac_p, jac_r, site)` writes into two 3 × nv arrays; pick the leg's columns with `kin.leg_dof_indices` | 03 deck; `soc4180.kinematics.ik_legs`, which `lab_connected.py` calls |
| The pose error a step must close | `kin.pose_error(data, site, target_pos, target_mat)`: 3 metres + 3 radians (a rotation vector) | 03 |
| One damped step | `J.T @ np.linalg.solve(J @ J.T + lam**2 * I, err)` | 03 deck; `ik_legs` |
| Singular values, condition number | `np.linalg.svd(J, compute_uv=False)` | 03 deck (the singularity cell) |
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
| The MDP as a simulator (samples, not tables) | `soc4180.envs.G1WalkEnv(action_scale, control_hz, reward_weights)`; `observation_space`, `action_space`, `reward_weights`, `max_steps`, `control_dt` | 07 `env_run.py` |
| The MDP written down and solved | numpy arrays for S, A, P, R; value iteration; no MuJoCo call except the replay's `mjv_initGeom` / `mjv_connector` markers | 07 `mdp.py` |
| A table learned from samples | `soc4180.tabular.Learner(env, algo="q" or "sarsa")`: `env.reset(seed)`, `env.step(a)` only, never `P`; `value_iteration(env.unwrapped.P, ...)` for the exact answer on toy-text | 07b `qlearn.py`, `lab_tables.py` |
| A MuJoCo env through gymnasium | `gym.make("InvertedPendulum-v5")`; its `unwrapped.model` / `unwrapped.data` go straight into `launch_viewer(passive=True)`, so ctrl + right-drag pushes the pole the table is balancing | 07b |
| A grid world drawn in MuJoCo | MJCF built as a string (one box geom per tile, a `mocap` body for the agent); move it with `data.mocap_pos`, then `mj_forward` before drawing | 07b `lab_tables.py` |
| One decision | `obs, r, terminated, truncated, info = env.step(a)`; `info` carries every reward term | 07, 08 |
| The week 4 walker as actions | `soc4180.envs.walker_actions(env, controller, t)` | 07 |
| A policy with one parameter, learned from reward | `env.reset`, `env.step`; a shove through `data.xfrc_applied[torso, 1]`; two tries `theta ± sigma·eps` on the same push, `theta += lr·mean((G+ − G−)·eps)/(2 sigma)` | 08 `lab.py` |
| Load a trained policy and act with it | `PPO.load(path, device="cpu")`, `predict(obs, deterministic=True)` | 09 `lab.py` |
| Change the world per episode | `geom_friction[:, 0]`, `body_mass[torso]`, `actuator_gainprm[:, 0]` with `actuator_biasprm[:, 1:3]`; `soc4180.envs.G1PushEnv(world=...).set_world(...)` | 09 `lab.py` |
| A shove of exact size | `G1PushEnv.push_force`, `push_time`; or `data.xfrc_applied[torso, :2]` for 0.2 s | 08, 09, 10 `lab.py` |

## Planning and copying (week 10)

| Idea | Call / array | Week, script |
| --- | --- | --- |
| Add sensors to a model before compiling | `spec.add_sensor(name=, type=mjSENS_FRAMEPOS / mjSENS_SUBTREECOM, objtype=mjOBJ_SITE / mjOBJ_BODY, objname=)`, `spec.compile()`; `soc4180.planning.planning_model()` | 10 `lab.py` (`World`) |
| The whole state as one vector | `mj_stateSize(model, mjSTATE_FULLPHYSICS)`, `mj_getState(model, data, state, spec)` | 10 `planning.MPPI.plan` |
| Many futures, in parallel threads | `mujoco.rollout.rollout(model, [MjData] * nthread, states (K × nstate), ctrl (K × T × nu), persistent_pool=True)` | 10 `planning.MPPI` |
| MPPI | `planning.MPPI(model, ctrl0, actuators, samples, horizon, knot, sigma, temperature)`, `.cost = fn(Rollouts) -> (K,)`, `.plan(data)` | 10 `lab.py` |
| Copy an expert | record (inputs, action) pairs, least squares `np.linalg.solve(ZᵀZ + λI, ZᵀY)` | 10 `lab.py` (`fit`) |

## Pixels and words (week 11)

| Idea | Call / array | Week, script |
| --- | --- | --- |
| A camera on a body | `spec.body("torso_link").add_camera(name=, pos=, xyaxes=, fovy=)`; MuJoCo cameras look down their own −z; `soc4180.vision.camera_model` | 11 `lab.py` |
| A target you move by hand | `spec.worldbody.add_body(mocap=True)`, `data.mocap_pos[i]`, then `mj_forward`; `contype = conaffinity = 0` | 11 `lab.py` (`scene`) |
| What the camera sees | `Renderer.update_scene(data, camera="head")`, `render()` (through `soc4180.render._new_renderer`) | 11 `lab.py` (`obey`) |
| Dim the lights | `model.light_diffuse`, `model.light_ambient` | 11 `lab.py --dim` |

## Showing it

| Idea | Call / array | Week, script |
| --- | --- | --- |
| The interactive window (lab scripts only, never a notebook cell) | `soc4180.launch_viewer(model, data, passive=True, key_callback=fn)`; `viewer.sync()`, `viewer.is_running()`, `viewer.lock()` | every lab |
| Keys from the terminal too | `soc4180.terminal_keys(fn)` feeds the same callback | every lab |
| Markers | `viewer.user_scn.geoms[i]` with `mujoco.mjv_initGeom(geom, mjGEOM_SPHERE / BOX / ARROW, size, pos, mat, rgba)`; `mju_quatZ2Vec` for an arrow's direction | 00, 03, 04, 06, 07 |
| Text on screen (HUD, scoreboard) | `viewer.set_texts([(mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right), ...])` | 03 `lab_connected.py` |
| Close by itself (the smoke test) | `SOC4180_AUTOCLOSE=<seconds>` | `scripts/check_labs.py` |
| Replay a recorded motion | write `qpos`, `mj_forward`, `sync()` — no physics re-run | every pipeline script's last step |
| Video for the notebook | `soc4180.render_rollout(...)`, `render_poses(...)` (EGL / OSMesa chosen at import) | the decks |
