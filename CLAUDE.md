# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Undergraduate course "Robot and AI" (SOC4180): 15 weeks of slides + runnable
labs, taught through MuJoCo simulation. The spine is a **walking humanoid**
(Unitree G1) — classical control first, learned control second.

**Simulation only. No hardware exists.** Never propose a lab that needs a
physical robot.

**Each week is two classes: a lecture from the deck, then a lab on student
laptops.** Students bring laptops with the repo installed (`uv sync`), change
code, watch the G1 respond in the interactive viewer, and demonstrate the result
to the instructor. **Every laptop runs the same environment**: `uv.lock` pins
it and `uv sync` reproduces it, so a lab may only depend on what the lock file
installs (plus `--extra rl` from week 7). Colab is the no-install fallback for
the notebook, not the primary lab environment. A week is not finished until it has a lab-class
artifact students can edit and run locally — week 2's `lab_viewer.py` and week
2b's `lab_body.py` are the pattern, and **every week 0–10 now has one** (see
*Lab scripts* below).

**Week 0 is the day-one stack/vocabulary lecture, taught before Week 1.** Every
week names which of its five layers it belongs to.

## Syllabus (revised)

This replaces the older course description, which covered deep learning broadly
(CNNs, RNNs, Transformers, diffusion, CLIP, agentic systems), computational
neuroscience, MicroDuck and NVIDIA IsaacSim. **Those are deliberately out of
scope.** MicroDuck went because there is no hardware; IsaacSim because it cannot
run on Colab, which the zero-install requirement depends on. Generative and
agentic content returns only where it attaches to the robot, in weeks 14–15.

| Wk | Topic | Deck | Status |
| --- | --- | --- | --- |
| 01 | The five-layer robot stack; MuJoCo and MJCF | `00`, `01` | built; figures + `lab_stack.py` / `lab_mjcf.py` added 2026-09-20, **not yet Colab-tested** |
| 02 | Transforms and forward kinematics | `02` | built, Colab-verified; **has a lab-class script** (`lab_viewer.py`); lecture class only — the lab class is `02b` |
| 02b | The robot as code: body tree, `qpos` map, the package | `02b` | built; **not yet Colab-tested**; **has a lab-class script** (`lab_body.py`) |
| 03 | Inverse kinematics | `03` | built; four figures + `lab_ik.py` added 2026-09-20, **not re-tested on Colab since** |
| 04 | Contact, balance, analytic walking (LIPM/ZMP) | `04` | built; five figures + `lab_walk.py` added 2026-09-20, **not re-tested on Colab since** |
| 05 | Actuation, PD control, and CPG gaits | `05` | built; four figures + `lab_servo.py` added 2026-09-20, **not re-tested on Colab since** |
| 06 | Sensing, state estimation, observation design | `06` | built; three figures + `lab_imu.py` added 2026-09-20, **not re-tested on Colab since** |
| 07 | From control to learning: MDPs and environment design | `07` | built; five figures + `lab_env.py` added 2026-09-20, **not re-tested on Colab since** |
| 08 | Policy gradients and PPO | `08` | built; three figures + `lab_train.py` added 2026-09-20, **not re-tested on Colab since** |
| 09 | Reward shaping and diagnosing failed runs | `09` | built; three figures + `lab_reward.py` added 2026-09-20, **not re-tested on Colab since** |
| 10 | Scaling: GPU-parallel locomotion training | `10` | built; GPU training **runs on A100, untimed**; three figures + `lab_many.py` added 2026-09-20 |
| 11 | Domain randomization and robustness | — | not written |
| 12 | Sim-to-real, measured | — | not written |
| 13 | Perception and imitation | — | not written |
| 14 | Vision-language-action: grounding instructions | — | **blocked** |
| 15 | Agentic robotics: perception, reasoning, action | — | **blocked** |

Capstone presentations occupy the final-exam slot. Weeks 0 and 1 share the first
session — the stack lecture is short, the MuJoCo lab is hands-on. **That session
is now the longest of the course**: week 1 grew from 14 slides to 38 when MJCF
was taught properly, so plan to split it or set part of the MJCF read as
preparation.

**Current state.** Weeks 1–9 are built and confirmed working on Colab, except
week 1's new material (MJCF, conventions, the simulation loop): it renders
cleanly here — 27 cells, 38 slides, `lab.ipynb` regenerated — but has **not been
run on Colab**. Week 2b is new and in the same position: it renders cleanly here
— 14 cells, 21 slides, six generated figures, `lab.ipynb` executed end to end
under `nbclient` — but has **not been run on Colab**. Week 10 is built and its GPU training path now runs on a Colab
A100, after a long series of dependency failures documented below —
but **no run has been timed**, so `num_timesteps = 5M` is a reduction from a
known-too-slow figure rather than a measured one. Weeks 11–15 are designed and
unwritten; 14–15 additionally depend on a trained locomotion policy that does
not yet exist.

**Week 8 facts, measured.** REINFORCE from scratch on CartPole: 60 -> 489 in
173 s, using a **batch of 8 episodes per update** — with one episode per update
identical code reached 260 and 88 on two runs, so batching is required for the
lab to be reliable. PPO solves InvertedPendulum (26 -> 1000, 99 s). On
`G1WalkEnv`, PPO **gets worse**: untrained scores 774.3 (exactly the
hold-the-crouch baseline, because residual actions start near zero), and after
30k steps scores 94.7.

**The cause is exploration noise, NOT reward hacking.** An earlier version of
these slides claimed PPO had found a reward loophole; that was wrong and the
numbers disproved it (94.7 < 774.3, so nothing was exploited). Evaluating the
reward by hand: walking at the target is worth 1250 over an episode, standing
776, lunging-and-falling 52. The reward is fine.

What actually happens: PPO's initial action std is 1.0 on a `[-1, 1]` action
space. The untrained **deterministic** policy survives all 500 steps; the
**stochastic** policy PPO collects data from survives 38. Over 90% of training
experience is a fall. `log_std_init=-2.0` (std 0.135) raises stochastic survival
to ~180 and the same 30k run then scores 776 instead of 95.

Even fixed, it only matches the standing baseline — it has learned not to fall,
not to walk (which is worth 1250). 30k steps is 0.02% of a real run.

Week 8 is ~6 min; it trains four times.

**Week 9 facts, measured.** The reward's *ranking* is correct — walking at target
1250, slow walk 1023, standing 776, lunge-and-fall 51. But the planned six-way
ablation **failed to discriminate**: every variant stood still for all 500 steps
at 30k steps, because none of our five terms changes whether standing is
attractive. The real G1 reward in MuJoCo Playground has **24 terms**, notably
`feet_air_time=+2.0`, `stand_still=-1.0`, and `alive=0.0` with
`termination=-100.0` (a per-step alive bonus is what makes standing profitable).
Adding `stand_still` or `air_time` alone changes nothing; **both together** break
the standing optimum — 224 steps, −0.524 m, feet lifted 0.08 — i.e. it falls over
backwards. Shaping picks which optimum you land in; it does not buy the search.
`G1WalkEnv` now takes `reward_weights`; see `DEFAULT_REWARD` in `envs.py`.

### 10 — scaling

**Measured locally (36-core Windows).** Single env 1308 control steps/s (13,083
physics steps/s), so 150M steps is ~32 hours. `DummyVecEnv` gives **no speedup**
(1173/1240/1250 for 1/4/16 envs) — it steps sequentially, so vectorised is not
parallel. `SubprocVecEnv` does scale (1649/3205/5570 for 2/4/8) but is capped by
core count; a Colab CPU runtime has 2, against a tuned config asking for 8192.

**Read from the library, not memory.** Berkeley Humanoid is 150M timesteps at
8192 envs, Op3 100M. Network: policy (512,256,128) on `state`, value
(512,256,128) on **`privileged_state`** — the week-6 asymmetric actor-critic in
production.

#### The GPU path: five failures, all hit for real

1. `ModuleNotFoundError: mujoco_playground` — not in `soc4180[rl]`; install
   `playground` separately.
2. `AttributeError: type object 'int' has no attribute 'WARP'` — env configs
   default to `impl="warp"` (MJWarp), needing the separate `mujoco-warp`
   package. Use `registry.load(name, config_overrides={"impl": "jax"})`.
3. `'State' object has no attribute 'pipeline_state'` — a playground env is not
   a brax env. Training needs `wrap_env_fn=wrapper.wrap_for_brax_training`.
4. **JAX 0.11 removed two APIs still called by flax
   (`jax.core.get_opaque_trace_state`) and brax (`jax.device_put_replicated`).**
   brax requires only `jax>=0.4.6`, no upper bound, so pip installs a broken
   pair — and **Colab ships 0.11, so it is broken out of the box.**
5. `NameError` in a later cell — caused by putting imports in the same cell as
   the install, so the auto-restart killed it before they ran.

#### The fix: patch, do not pin

Restore the two functions rather than pinning JAX. `soc4180.jax_compat.patch_jax()`
does it, and week 10's notebook writes the same six lines out inline —
deliberately, because a cell whose job is repairing a broken environment must not
depend on a package install having succeeded.

Rules, each learned by breaking it:

- **Search for the function, do not assume its home.** JAX's deprecation message
  names `jax.extend.core.get_opaque_trace_state`, and that path **does not exist
  on Colab's build**. Try `jax.extend.core`, then `jax._src.core`, then
  `jax.core`.
- **Never raise from the repair.** If nothing is found, print and continue.
  Assigning unconditionally turned a silent no-op into a hard failure that took
  the notebook down.
- **Report which branch was taken.** `hasattr(jax, "device_put_replicated")` is
  False in a plain interpreter here and True inside the notebook kernel, for
  reasons I never identified. The cell prints what it did so the difference is
  visible instead of mysterious.

An earlier attempt pinned `jax[cuda12]==0.9.2` instead. That works, but Colab
starts every session on JAX 0.11, so it meant a large download **and a forced
restart at the start of every lab**. Do not reintroduce it.

Still required regardless: `impl="jax"` in `registry.load`,
`wrap_env_fn=wrapper.wrap_for_brax_training`, `playground` installed separately
(never `playground[all]`), dependency setup in a cell with no imports, and
**`progress_fn` on `ppo.train`** — brax prints nothing without one.

Rendering this week needs `uv sync --extra rl --extra gpu --extra env` first; a
plain `uv run` re-syncs and drops `mujoco_playground`.

**Confirmed on Colab (A100):** the training cell **runs**. At 15M timesteps /
2048 envs it exceeded 10 minutes, so the default is now 5M with `num_evals=10`.
**Always pass `progress_fn`** — brax prints nothing without one, and a silent
ten-minute cell is indistinguishable from a hang. The first progress line is
slow regardless, because brax XLA-compiles the whole loop before stepping.

**Still uncalibrated:** no Track A run has been timed to completion, so
`num_timesteps=5M` is a reduction from a known-too-slow figure rather than a
measured one. Read seconds-per-step from the progress output and scale.

Ordering matters here: the verification cell must come **before** the training
cell. It originally landed after it, because the troubleshooting slides sat
after the "Running it" slide, so a Run All hit training first.

## Commands

```bash
uv sync                                        # lean env: mujoco + rendering only
uv sync --extra rl                             # + gymnasium, SB3, torch (CUDA on Windows)
uv sync --extra gpu                            # + JAX/MJX/playground (Linux/WSL2)
uv run python -c "import soc4180"              # smoke test
uv run scripts/view.py --walk                  # interactive viewer (laptop/desktop)
quarto render weeks/01-intro/slides.qmd        # -> slides.html + lab.ipynb
uv run python scripts/build_site.py            # rendered decks -> _site/ for Pages
```

There is no test suite or linter configured. Add the tooling before inventing
commands for it.

### Lab scripts, weeks 0–10 (added 2026-09-20)

| Week | Script | Fill-in (returns `None` until written) | What changes on screen when it is right |
| --- | --- | --- | --- |
| 00 | `lab_stack.py` | `my_controller` (29 servo targets) | keys `1/0/2/3/4/G` switch layers on and off; contact-force arrows on the floor |
| 01 | `lab_mjcf.py` | the MJCF string and `TARGET` | the compiler's own errors; droop, `BADQACC` count on ENTER |
| 03 | `lab_ik.py` | `dls_step(J, err, lam)` | the green foot chases the red target inside a translucent reach sphere |
| 04 | `lab_walk.py` | `predict_com` (LIPM closed form) | blue dots inside the white commanded-CoM dots; yellow ZMP sphere |
| 05 | `lab_servo.py` | `torque_from_pd` | joints colour by torque; `H` makes the student's law *the* servo |
| 06 | `lab_imu.py` | `my_filter` (complementary) | three "down" arrows; the green one vertical |
| 07 | `lab_env.py` | `my_policy(obs, t)` | gravity arrow, reward bar, yellow airborne feet, episode lines |
| 08 | `lab_train.py` | `choose_action(mean, std, rng)` | `D` flips deterministic/stochastic; `T` trains in a thread |
| 09 | `lab_reward.py` | `extra_reward` (a wrapper term) | variants trained in a thread; feet-up shown |
| 10 | `lab_many.py` | `predict_rate(n, single)` | N robots in one scene; `P` benchmarks processes |

Conventions, all followed: a list at the top to extend; one fill-in; keys in
the docstring; `soc4180.terminal_keys` as the second route; **restarts reuse
the same `MjData`** (`mj_resetData` + copy `qpos`), because the passive viewer
is bound to one data object — only `lab_many.py` reopens the viewer, since its
model changes; and **`SOC4180_AUTOCLOSE=<seconds>` closes the window by
itself**, so `SOC4180_AUTOCLOSE=6 uv run weeks/NN/lab_x.py < /dev/null` is the
smoke test. Every script passed it on 2026-09-20.

Facts these scripts and the new figures established, each measured:

- **An explicit servo law fed through `qfrc_applied` is unstable at 500 Hz and
  fine at 1 kHz.** Zeroing the G1's gains and applying $k_p(\text{ctrl}-q) -
  k_v\dot q$ by hand blew up (`BADQACC`) after 14 steps at `dt = 0.002`; at
  `dt = 0.001`, `0.0005` and `0.00025` it walks 0.65 m against 0.66 m with the
  model's own servos. MuJoCo's position actuator survives 500 Hz only because
  `implicitfast` integrates its affine bias implicitly. `lab_servo.py`'s `H`
  therefore runs the student's loop at 1 kHz and says so; `SERVO_HZ = 500` is
  a lab step, and it is week 1's exercise 14 from the other side.
- **Every G1 leg joint is critically damped**: $\zeta = k_v / 2\sqrt{k_p M_{ii}}
  = 1.00$ on all twelve against the mass-matrix diagonal at `stand`
  (`mj_fullM(model, data, M)` in mujoco 3.12 — it takes the data object). Arms
  0.7–1.9. That is *why* `kv` varies tenfold while `kp` does not.
- Knee step response, gravity off: `kp x4` reaches 1% in 0.034 s vs 0.116 s
  with 2% overshoot and 4x the torque; `kv / 4` overshoots 9%; `kv x4` is 1%
  short after 0.5 s. Sag vs `kp`: 11.0 mm at 500, 9.4 at 600, 6.0 at 750 (the
  $1/k_p$ law), falls at 400 and at 1000. The knee exceeds 50 N·m for only
  **4.0%** of the walk and that limit still drops the robot.
- Week 6's complementary filter: on the biased-gyro walk, mean error is
  smallest at $\alpha = 0.9985$ ($\tau = 1.32$ s) and final error at 0.9977.
  Both bottom out near 0.998 — an earlier draft claimed they pulled opposite
  ways, and the sweep disproved it. The filter's low-pass time constant is
  $\tau = -\Delta t / \ln\alpha$.
- Week 7's grid, re-measured for the figure: only $\kappa = 1.0$ at 100/200 Hz
  walks (1.05 m); $\kappa = 1.0$ at 50 Hz goes 1.56 m and falls.
- Week 9's "valley": an open-loop stepping family in `G1WalkEnv` (hip pitch
  and knee sinusoids, amplitude 0 → 1) scores *less* than standing at every
  amplitude — first effort, then falling. The reward's optimum is real and no
  route in that family climbs to it.
- **Multi-robot scenes work through `MjSpec`**: `frame = spec.worldbody.add_frame();
  frame.attach_body(child.worldbody.first_body(), "r0_", "")` per robot, then
  `compile()`. Names get the prefix; actuators and `qpos` are contiguous per
  robot in attach order (`nq = 36 K`, `nu = 29 K`). **A free-joint body's
  offset goes in `qpos`, not in the frame's `pos`** — four robots attached at
  different frame positions all sat at `y = 0` and collided, at 2,459
  robot-steps/s against ~13,000 for one robot alone. `MjSpec.from_file` wants
  a `str`, not a `Path`.
- **Timing cells measure the machine.** Week 10's `SubprocVecEnv` numbers came
  out 8 processes < 4 processes when rendered while weeks 8 and 9 were
  training in the background. Never render 8, 9 and 10 concurrently; re-render
  10 on a quiet machine.

### The interactive viewer: lab tool, never a notebook cell

`scripts/view.py` opens MuJoCo's interactive viewer — orbit, pan, zoom, and
ctrl-drag to push the robot. `--walk` runs the week 4 controller live, `--limp`
disables actuation, `--robot`/`--keyframe` reach the other Menagerie humanoids,
`--list` prints what is available, `--static` draws without stepping physics,
and `--pose=A,B,C,D,E,F` places one leg's six angles on `stand`, frozen, and
prints the foot site in world and pelvis frames. **Write `--pose=` with the
equals sign** — a value starting with `-` is otherwise read as an option.

Students use it in lab classes; it is the point of the laptop requirement.
Pushing the robot to see whether a controller survives a disturbance takes
seconds here and is invisible in a rendered video.

**It must never appear as a cell in `lab.ipynb`.** The notebook has to render
headless under Quarto and on the Pages runner, and run on Colab, where there is
no window and `launch_viewer` raises. Viewer work lives in `.py` files under the
week (`weeks/02-transforms/lab_viewer.py`) and in the README's lab section;
the notebook carries the exercises and rendered video. `launch_viewer` takes a
`key_callback` for passive mode; the lab script uses SPACE/arrows to step
through poses and draws its markers through `viewer.user_scn` under
`viewer.lock()`.

**Viewer keys: the callback path is verified, the live failure is not yet
explained.** Reported 2026-09-17: in `lab_body.py` on this machine the mouse
works (orbit, Control sliders) but **no key does anything**. Injecting real
`WM_KEYDOWN` messages into the live GLFW window shows the code is fine --
32/49/77/262/257 all reach `on_key`, and the script steps poses, highlights
chains, mirrors and prints exactly as documented. Ruled out the same way: mouse
position (keys arrive with the cursor over either UI panel, the 3-D view or the
title strip), clicking a widget first (options panel, a Control slider, the
view), and `uv run` (same interpreter, mujoco 3.12.0, glfw 2.10.2). The viewer
window also takes foreground by itself at launch.

Two things are known to swallow a key and are worth excluding first, but neither
matches "nothing at all responds":

- **Focus** -- keys typed at the terminal never reach the viewer.
- **The Korean IME.** While it is composing, Windows sends
  `wParam = VK_PROCESSKEY (229)`, GLFW cannot translate it, and the callback is
  **never called** -- measured: the same `M` arrives as keycode 77 in English
  mode and as nothing in Hangul mode. This kills letters only; digits and arrows
  survive.

**The fix that does not depend on the diagnosis: `soc4180.terminal_keys`.**
A lab class on thirty unknown laptops cannot depend on GLFW receiving a
keystroke, so both lab scripts now start a daemon thread that reads the console
and feeds the *same* `key_callback`. Single keys, no enter, arrows included --
`msvcrt.getwch()` on Windows, `tty.setcbreak` on POSIX, and a whole-line
fallback when stdin is a pipe (which is what makes it testable without a
keyboard). Ctrl-C is forwarded with `interrupt_main()` so the reader cannot
swallow the way out. Two traps, both hit: a space `strip()`s to an empty token
and arrives as ENTER unless handled as a character, and writing `" "`
through a heredoc can halve the backslash and put a real NUL in the source --
hence `chr(0)`/`chr(0xE0)`/`chr(3)`/`chr(27)`.

**Measured on this machine, keys dead in `lab_body.py`:** `keyfocus=MuJoCo`
(so not a focus bug) and `IME_open=False conversion=0x0` (so not the IME) --
both documented suspects are excluded. `keyprobe.py` now reports keyboard focus
via `GetGUIThreadInfo`, which is *not* the foreground window; a window can be
foreground while focus sits elsewhere. Still unexplained: `lab_viewer.py`
(09-16) worked and `lab_body.py` (09-17) does not, with `sim.py` and `uv.lock`
unchanged between them -- run the older script as the control before theorising.

`scripts/keyprobe.py` is the diagnostic. It opens the viewer and, per key press,
prints whether Windows saw the key at all (`GetAsyncKeyState`, focus-independent),
which window was foreground, and whether `key_callback` fired -- which separates
focus, an interceptor between Windows and GLFW, and the keyboard itself. Run it
before theorising.

**In static mode the Control sliders must be wired onto `qpos` by hand.** The
sliders write `ctrl`, and `ctrl` reaches the joints only through `mj_step`; a
kinematic script that just calls `mj_forward` leaves them dead, which is the
first thing a student drags. Both `view.py --static` and the week 2 lab copy
`ctrl` into `qpos[jnt_qposadr[actuator_trnid[:, 0]]]` every tick (every G1
actuator drives one hinge; `ctrlrange == jnt_range`), and copy `qpos` back into
`ctrl` whenever the script sets a pose so the sliders show it.

## Adding a week

The pipeline is proven; follow it rather than improvising.

1. `weeks/NN-slug/slides.qmd`, with the notebook-only Colab header (below) and
   the `<slug>` updated.
2. Put reusable code in `src/soc4180/`, not in the slides. Slides show the idea;
   the package carries anything a later week needs.
3. **Verify a claim before writing it into a slide.** Several assertions in this
   repo were wrong until measured: that the robot falls with `ctrl=0`, that a
   residual was iteration-limited, that a leg segment was 0.194 m long. Run it.
4. `quarto render weeks/NN-slug/slides.qmd` — a broken cell fails the render.
5. Execute the generated notebook standalone with `nbclient` before committing,
   **then dump its figures and look at them**:

   ```bash
   uv run scripts/check_notebook.py run    weeks/NN-slug/lab.ipynb
   uv run scripts/check_notebook.py images weeks/NN-slug/lab.ipynb <dir>
   ```

   `run` compiles every code cell first (magics stripped) and then executes
   them all, `eval: false` included — use `compile` instead of `run` for week
   10, whose GPU cell would otherwise try to train on the CPU. **For any
   `eval: false` cell, the compile check is the only check** — a broken
   f-string reached a student's runtime this way. And a figure that *renders*
   is not a figure that is *right*: on 2026-09-20 five of the first thirty new
   figures came out clipped, overlapping or autoscaled wrongly (matplotlib
   patches do not autoscale the axes; `annotate` does not either), all with a
   green render. Look at the PNGs.
6. Add `weeks/NN-slug/README.md` and a row in the top-level README table.
   The README needs a **lab class** section: what students change, and what
   they should see on screen when it is right.
7. Add the lab-class artifact: a `.py` under the week that students edit and run
   on their laptops (`uv run weeks/NN-slug/<name>.py`), using only what
   `uv.lock` installs. It may open the viewer; `lab.ipynb` may not. Follow
   `weeks/02-transforms/lab_viewer.py`: a list at the top students extend, a
   function they fill in, and a visible on-screen difference between right and
   wrong.
8. **Push, then test it on Colab.** Every environment bug this project hit was
   invisible on Windows: the GL ordering bug, the dependency upgrades that broke
   the runtime, the unguarded renderer. Ask for `soc4180.gl_report()` when
   rendering fails.

   **The badge loads `lab.ipynb` from GitHub, not from the working copy.** So a
   Colab test run before pushing silently exercises the *previous* version of the
   week — it passes, and the change you meant to test was never there. Check
   `git status -sb` for "ahead of origin/main" before trusting a Colab result.

## Authoring model: one source, two outputs

Each week is a single `weeks/NN-*/slides.qmd` that Quarto renders into:

- `slides.html` — self-contained reveal.js deck (**gitignored build artifact**)
- `lab.ipynb` — student notebook (**committed**; Colab loads it from GitHub)

`execute.error: false` in `_quarto.yml` means a broken cell **fails the render** —
verified. Do not set `error: true`; that protection is the point.

Rendered decks are gitignored because reveal.js assets are ~5 MB/week. Only
`lab.ipynb` is committed.

### The decks reach the web through Pages, not through git

Measured, so the gitignore is not re-litigated: the eleven decks are 3.4–5.1 MB
each, **43 MB for the set**. `embed-resources: true` inlines reveal.js, images
and base64 video into a single file, and inlined base64 does not delta-compress
— so committing them would add a fresh full copy of every re-rendered deck to
history, tens of megabytes at a time, permanently. **Do not start committing
`slides.html`, and do not publish through a `gh-pages` branch either** (same
growth, different branch).

`.github/workflows/pages.yml` builds the site on a runner and hands it to
`actions/deploy-pages` as an **artifact**: nothing enters git at all, and
<https://gnoejh.github.io/soc4180/> is stable. `workflow_dispatch` is enabled
deliberately — teaching sometimes happens with no machine but a browser, and the
site must be rebuildable from the Actions tab.

Decisions in that workflow, each with a reason:

- **One `quarto render` per week, in a loop**, not one project-level render. A
  project render is all-or-nothing: week 10 alone failing would publish nothing.
  The loop ships the ten decks that worked; a separate `report` job (which runs
  *after* `deploy`) turns the run red and names the failure. `execute.error:
  false` is untouched — a broken cell still aborts that week.
- **`_freeze` lives in the Actions cache, not in git.** Same speed, no bloat.
  The cache key is deliberately in two parts: the environment hash
  (`src/soc4180/**`, `uv.lock`) is the *prefix*, the week hash is the suffix.
  **Quarto's freeze keys on the `.qmd` alone and does not notice that the
  package changed underneath it**, so restoring a cache across a package change
  would publish stale results. Splitting the key means a package change restores
  nothing and re-executes everything, while adding a week reuses the other weeks
  and still saves the new one. Do not collapse it back to a single `freeze-`
  restore-key.
- **`libosmesa6` is installed explicitly.** The runner has no GPU, so `_gl`
  wants osmesa, and without the library it correctly sets *no* backend and
  rendering raises. The workflow prints `gl_report()` before rendering so the
  branch taken is visible.
- **`--extra gpu` is synced**, because week 10's *executed* cells import
  `mujoco_playground`.
- `scripts/build_site.py` never fails; the render step owns failure. It marks a
  missing deck "Slides unavailable" on the index rather than hiding the week.

Repository setting required once: *Settings → Pages → Source = GitHub Actions*.

### Install unconditionally, before the first import

**`try: import soc4180 / except ImportError: %pip install ...` is wrong for this
repo.** The package changes between labs, so on any runtime with an older copy
the import succeeds, the install is skipped, and a function added later is
missing — surfacing as an `AttributeError` for something that plainly exists in
the repository. This bit for real with `soc4180.jax_compat` in week 10.

Use instead, as the first thing in the notebook:

```python
%pip install -q --upgrade "soc4180[rl] @ git+https://github.com/gnoejh/soc4180.git"
import soc4180
```

Upgrade **before** importing: pip rewrites files on disk and cannot replace a
module the interpreter has already loaded. Weeks 0–9 still use the conditional
form; they are stable, but any week that gains new package features should be
switched.

**Prefer the Colab-guarded variant** (introduced in week 2b, see *The whole body
(week 2b)*): the bare form above also installs during a local `quarto render`,
putting the last *committed* package over the editable venv — so a week that uses
package code written in the same commit cannot render. Wrapping it in
`if importlib.util.find_spec("google.colab") is not None:` keeps the unconditional
upgrade on Colab and makes it a no-op locally.

### Standing pattern: every week starts with a notebook-only header

Immediately after the YAML frontmatter, before the first slide:

````
::: {.content-visible when-format="ipynb"}
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/<slug>/lab.ipynb)

**Before you start, two things:**

1. **Runtime -> Change runtime type -> T4 GPU.** ...
2. **File -> Save a copy in Drive.** ...
:::
````

`content-visible when-format="ipynb"` keeps it out of the deck — verified: the
badge appears in `lab.ipynb` and not in `slides.html`. Both notices matter:
without the GPU runtime nothing renders, and a GitHub-opened Colab notebook is
unsaved, so student work vanishes on tab close. Remember to update `<slug>`.

**Never edit `lab.ipynb` directly** — it is generated. Edit `slides.qmd` and
re-render. In particular do not use Colab's *Save a copy in GitHub* on these
notebooks; it would commit executed outputs over the generated file.

### Verified Quarto behaviours (do not re-litigate)

- `#| eval: false` **does** survive into `lab.ipynb` as a genuine unexecuted code
  cell. No post-render cell injector is needed.
- **…which means a student's *Run All* executes it.** `eval: false` stops *Quarto*
  from running the cell; it does not stop Colab. So an illustrative snippet with
  undefined names is a `NameError` waiting for the first student who runs the
  notebook top to bottom, and `execute.error: false` will never catch it because
  the cell is never executed during render. Week 2 shipped one for exactly one
  commit. **Either make the cell genuinely runnable, or make it a plain fenced
  ```` ```python ```` block that is not a cell at all.** Week 10's is the only
  legitimate use in the repo: an opt-in GPU training cell that carries its own
  imports. Catch these by executing the generated notebook with `nbclient`, which
  runs every cell regardless of `eval`.
- The Week-1 setup cell uses `try: import soc4180 / except ImportError: %pip install`
  rather than a bare `!pip install`, so the notebook is safe to run locally *and*
  installs on Colab. Keep that shape. This is the one place it outranks *Install
  unconditionally, before the first import* below: week 1 must stay renderable
  without reaching GitHub, so **week 1 slides may not depend on new package
  code** — see *MJCF (week 1)*.
- **Video survives into `slides.html` (embedded base64) but is stripped from
  `lab.ipynb`.** Students still see it when they *run* the notebook, which is the
  actual Colab workflow. This is accepted, not a bug to chase.

## Environment gotchas

### Two execution targets

**JAX CUDA plugins are Linux-only** (`manylinux` wheels only, no `win_amd64`), so
MJX/Brax GPU training cannot run natively on this Windows machine — that work
belongs on Colab or WSL2. PyTorch CUDA *does* work locally and is verified.
Keep `jax`/`brax`/`playground` in the `gpu` optional extra, never in the default
dependency set.

### Dependencies must not disturb a Colab runtime

`[project.dependencies]` is deliberately small and permissive: `mujoco`,
`mujoco-menagerie`, `mediapy`, `imageio-ffmpeg`, `numpy>=1.24`. That is exactly
what the package imports.

**Do not add torch, gymnasium or SB3 back to the required set.** Pinning
`torch>=2.14` and `numpy>=2.5.2` there broke a live Colab session: pip upgraded
both, which broke the preinstalled `torchvision` (wants `torch==2.11`) and
`numba` (wants `numpy<2.3`), and forced a kernel restart — for libraries the
package never imports. They live in the `rl` extra now. Exact local versions are
pinned in `uv.lock`, which is the right place for them.

### torch resolves from two sources

PyPI ships CPU-only Windows wheels, so `torch` comes from the `cu130` index on
Windows and from PyPI (already CUDA-enabled) on Linux:

| Platform | torch | Source |
| --- | --- | --- |
| Windows | `2.14.0+cu130` | download.pytorch.org |
| Linux / macOS | `2.14.0` | PyPI |

- **Do not remove the `sys_platform == 'win32'` marker** in `[tool.uv.sources]`.
- **`uv add torchvision`/`torchaudio` bare is wrong on Windows** — they need the
  same marked source, or you get CPU-only builds mismatched against CUDA torch.
- Keep `explicit = true` on the index so only named packages use it.

### Korean Windows locale (cp949)

The system Python defaults to **cp949**, not UTF-8. This bites constantly:

- Always pass `encoding="utf-8"` to `read_text`/`write_text`/`open`. Omitting it
  raises `UnicodeDecodeError` on any file containing an em dash.
- Prefix scripts that print non-ASCII with `PYTHONIOENCODING=utf-8`, or they die
  with `UnicodeEncodeError` on the console.

### ffmpeg

`mediapy` shells out to a **system** ffmpeg, which Colab has and Windows does
not. `render.py` resolves this at import by pointing mediapy at the binary
bundled with `imageio-ffmpeg`. Do not remove that fallback.

### Rendering locally: two flags that are not optional

Quarto is installed (winget `Posit.Quarto`, 1.10.18). Two things bite on this
machine, both every time:

- **Do not invoke it as `& "C:\Program Files\Quarto\bin\quarto.cmd"`.** The
  launcher mangles its own path at the space and dies with
  `Module not found "file:///W:/soc4180GH/Files/Quarto/bin/tools/x86_64/deno.exe"`
  — note it resolved `Files/Quarto` against the *working directory*. Put the bin
  directory on `PATH` and call `quarto` bare.
- **Set `QUARTO_PYTHON` to the project venv.** Quarto otherwise picks a system
  Python with no `nbformat` and stops with *"There is an unactivated Python
  environment in .venv. Did you forget to activate it?"*

So, from PowerShell:

```powershell
$env:PATH = "C:\Program Files\Quarto\bin;$env:PATH"
$env:QUARTO_PYTHON = "W:\soc4180GH\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = 'utf-8'
quarto render weeks/01-intro/slides.qmd
```

**When quarto is unavailable** (another machine, or a broken install), verify a
week the equivalent way: pull every `{python}` cell out of the `.qmd`, run them in
order through `nbclient`, and `compile()` the `eval: false` ones. That catches
what `execute.error: false` catches during a render — but it does **not** produce
`lab.ipynb`, and a stale committed notebook is what students actually open.

### Re-rendering a week is never a no-op in git

- **Pandoc assigns fresh random cell ids on every render**, so `lab.ipynb` shows a
  diff even when nothing changed. That is expected, not drift. To see whether
  anything real moved, filter them out:

  ```bash
  git diff -- weeks/NN-slug/lab.ipynb | grep "^[-+]" | grep -v '"id"'
  ```

- **Never leave `lab.ipynb` open in the VS Code notebook editor.** It re-serialises
  the JSON on save — `id` moves above `metadata`, `name`/`output_type` swap — and
  that is exactly where the unexplained churn in 00/01/04/09 came from. After
  a render the editor offers Save or Ignore: **Ignore, always.** Save writes the
  editor's stale buffer over the notebook quarto just generated, silently
  reverting a week for every student who opens the badge.

- **Cost, measured re-rendering all four:** 00, 01 and 04 take seconds;
  **09 takes 746 s**, because it trains four 40k-step PPO runs on CPU.

- **They reproduce bit-identically on this machine.** 04's walking numbers and
  09's entire ablation table came back unchanged — including `+ both` at 224
  steps, −0.524 m, feet up 0.08 — with a fixed seed and `device="cpu"`. The only
  thing that moved was the wall-clock seconds the 09 cell prints about itself.
  So re-rendering is safe to do freely: **a changed number means a real change.**

### MUJOCO_GL ordering (this has already broken once)

MuJoCo resolves its render backend from `MUJOCO_GL` at `import mujoco` time.
Setting it afterwards does nothing — the process is stuck, and on a headless
machine that means GLFW dying on a missing X11 `DISPLAY`.

Selection therefore lives in **`src/soc4180/_gl.py`**, which imports nothing from
mujoco, and `__init__.py` imports it **first**. Do not move that import, and do
not let any module that imports mujoco be imported ahead of it.

**The original bug:** `__init__.py` imported `.models` first, and `models.py`
does `import mujoco` at module level — so the backend was chosen *after* mujoco
had already picked GLFW. It passed on Windows (the default backend renders
offscreen fine) and failed on Colab with
`an OpenGL platform library has not been loaded into this process`.

Regression test — this must stay true:

```bash
uv run python -c "import soc4180; from soc4180._gl import MUJOCO_WAS_PREIMPORTED; assert not MUJOCO_WAS_PREIMPORTED"
```

Backend choice (verified against simulated conditions):

| Environment | Backend |
| --- | --- |
| Colab / headless Linux **with** NVIDIA device | `egl` (writes the NVIDIA EGL ICD file if missing) |
| Colab / headless Linux **without** GPU | `osmesa` |
| Windows, macOS | MuJoCo default |
| `MUJOCO_GL` already set | respected, always |

**Colab needs a GPU runtime for rendering.** The `osmesa` fallback requires
`libosmesa6`, which Colab images do not reliably ship, so a CPU runtime cannot
render even though the physics runs fine. Week READMEs say to pick a T4.

**Every renderer is created through `render._new_renderer`.** It checks
`GL_UNAVAILABLE` and wraps construction failures with an actionable message. Do
not call `mujoco.Renderer` directly anywhere else — `render_poses` was added
without the guard and reproduced the raw
`FatalError: an OpenGL platform library has not been loaded` on Colab, which is
exactly the message the guard exists to prevent.

`soc4180.gl_report()` prints what selection saw (colab, NVIDIA device node,
OSMesa, DISPLAY, MUJOCO_GL, chosen backend). Ask for it first when someone
reports a rendering problem.

When neither EGL nor OSMesa is usable, `_gl` sets **no** environment variable and
records `GL_UNAVAILABLE` instead; `render_rollout` raises that message. This is
deliberate: exporting `PYOPENGL_PLATFORM=osmesa` when OSMesa is absent makes
`import OpenGL` itself die with a bare
`AttributeError: 'NoneType' object has no attribute 'glGetError'`, ten frames
deep and impossible to act on. Physics still works on a CPU runtime; only
rendering fails, and it fails with instructions.

## `ctrl = 0` is not an uncontrolled robot

The G1's actuators are **position servos with gain 500**. Setting `ctrl = 0`
commands every joint to angle zero, and the servos hold that straight-legged
stance — the robot does **not** fall. Verified: torso stays at 0.792 m.

To simulate a robot with no controller you must disable actuation entirely, via
`soc4180.actuation_disabled(model)` (a context manager that restores the flag).
Then the torso drops 0.790 m → 0.134 m.

This bit once already: an early draft of the Week 1 slides asserted the robot
collapsed when it demonstrably did not. **Any slide claiming the robot falls must
disable actuation.** The distinction is now core Week 0 teaching material.

## Robot models

`mujoco-menagerie` is a **pip package** (~28 KB wheel, downloads models lazily and
pins its own upstream commit) — not a git clone. Use `soc4180.load_g1()` /
`load_robot()`. Pinning `mujoco-menagerie` in `pyproject.toml` pins the models.

- G1: `nq=36 nv=35 nu=29`, 500 Hz timestep, one `stand` keyframe. BSD-3-Clause.
  `nq > nv` because the floating base uses a quaternion — that is week-0/1
  teaching material, not a bug.
- Sensors: `nsensor = 4` is **two IMUs**, not four sensors — a gyro +
  accelerometer pair at `imu_in_torso` and another at `imu_in_pelvis`, 3 axes
  each, 12 numbers per step. There are **no cameras, joint-torque sensors, or
  foot contact sensors**; add them to the MJCF if a lab needs them.
- An accelerometer measures *specific force*, so it senses gravity and reads
  ~9.81 m/s² at rest. That is what makes it a tilt sensor, and why policies
  observe the gravity vector in body frame rather than absolute pose. IMUs give
  no position — double integration drifts without bound.
- Entry points: `g1`, `g1_mjx`, `g1_with_hands`, plus matching `scene*` variants.
  Scenes include floor and lighting; bare robot entries do not.
- 11 humanoids available; Cassie is categorised `biped`, not `humanoid`.
- Known upstream gotcha: `assets()` raises for `robotis_op3` (duplicate mesh
  basenames). Use `load()`/`path()` instead.

## MJCF (week 1)

Week 1 reads the G1's real MJCF section by section and then has students write a
working one. Week 0 teaches the same robot from the *compiled* side
(`model.nbody`, `actuator_gainprm`) and shows no XML at all; the two are
deliberately complementary, so do not move material between them without
preserving that split.

**The `mjcf()` section-printer lives in the slide, not in the package.** That
breaks the usual "reusable code goes in `src/soc4180/`" rule, on purpose: moving
it into the package would force week 1's setup cell to the unconditional-upgrade
form, and a local `quarto render` would then install the *committed* package from
GitHub — which would not yet contain the new module. Week 1 keeps the
`try: import soc4180 / except ImportError` shape for exactly this reason. If a
later week needs the helper, promote it and switch week 1's cell in the same
commit.

Measured against the pinned G1, and used on slides:

- `<option>` sets only `integrator="implicitfast"`. **`timestep = 0.002` and
  gravity are MuJoCo's defaults, not the file's** — an MJCF is a diff against a
  default robot.
- `kp="500"` appears nowhere near the 29 `<position>` actuators; it lives in
  `<default class="g1">`, inherited via `childclass="g1"` on the pelvis.
  `inheritrange="1"` is why `actuator_ctrlrange == jnt_range`.
- 72 geoms: 1 floor, **35 in group 2** (`contype=0 conaffinity=0`, invisible to
  physics) and **36 in group 3** (what actually collides). Each foot touches
  ground through **four spheres of 5 mm radius**, and `friction="0.6"` is on
  exactly those 8 geoms; everything else is at MuJoCo's default 1.0.
- The `stand` keyframe's 36 numbers are 3 position + 4 quaternion + 29 joints.
  Only the arms are bent; **every leg joint is exactly zero**, which is the
  week-4 singularity, visible in week 1.
- Geoms may be unnamed — the foot spheres are — but ids always exist.

Week 1 also teaches the conventions and the simulation loop, all measured:

- **roll/pitch/yaw are x/y/z**, and on the G1 the names are not decoration: every
  one of the 29 named hinges has the axis its name claims — checked in a slide
  cell, zero mismatches. `+z` up (gravity is `-z`), `+y` the robot's left (hips at
  `y = ±0.064`), so `+x` is forward. Students arrive not knowing these words; the
  deck defined joint names for nine weeks without ever defining *pitch*.
- **Standing, `ncon = 8`** — the four foot spheres each side — and the normal
  forces sum to **327.1 N against a weight of 327.1 N**. A free physics check that
  needs no ground truth.
- **~16,000 physics steps/s single-core here, about 32x real time.** The slide
  measures it live rather than hardcoding it, so the number is the student's own.

### Week-1 demo facts, each learned by getting it wrong first

- **A `<freejoint/>` cannot share a body with a hinge**: the compiler refuses with
  *"more than 6 dofs in body"*. The fix is a parent body carrying the freejoint.
  This is now an exercise, not an accident.
- **Zero is also gravity's rest position** for a pendulum leg hanging down, so a
  servo commanded to zero "succeeds" at any gain — a `kp` sweep from there proves
  nothing. The demo therefore commands a *bent* pose (`TARGET = [1.2, -0.8]`),
  which gravity fights; at `kp=200` the hip holds with −0.08 rad of droop.
- **Test sensitivity along a direction that is not a symmetry.** Perturbing
  `qpos[0]` (x) before a fall amplifies 1x, because translating along an infinite
  flat floor is an exact symmetry — the run is the same trajectory, shifted. It
  looks like proof that the fall is *not* chaotic, and it is not. Nudge a **joint
  angle** instead: `1e-12` rad at the knee grows ~100,000x in 3 s, and `1e-6` rad
  puts the robot in a different heap. Same-machine reruns are still bit-identical
  (max diff exactly 0.0), which is why the rule is *assert direction and
  magnitude, never a float from a fall*.
- **A stiff servo plus a coarse timestep fails silently.** At `dt=0.02` with
  `kp=200`, MuJoCo raises `mjWARN_BADQACC` and **resets**, so `qpos` comes back
  `[0, 0]` — which reads as success. The same model run passively at `dt=0.02`
  warns about nothing and merely gives a different, wrong trajectory. Both
  failure modes are exercises.

## Cross-platform reproducibility

Verified on Colab T4 against local Windows:

| Demo | Windows | Colab T4 |
| --- | --- | --- |
| Servos holding `stand`, 3 s | 0.792 m | 0.792 m |
| Actuation disabled, 3 s | 0.134 m | 0.142 m |

Stable equilibria reproduce exactly across platforms; **chaotic trajectories do
not**. A collapsing robot amplifies floating-point and contact-solver
differences over 1500 steps, so the same seed gives different final numbers on
different hardware.

Consequence: **never assert exact float values from a fall, and never autograde
one.** Assert direction and magnitude instead (`fell more than 0.5 m`,
`stayed above 0.7 m`). `set_seed` makes a run repeatable on one machine; it does
not make it identical across machines.

## Kinematics (weeks 2-3)

`fk_foot` composes the leg chain by hand and matches MuJoCo's `site_xpos` to
**~1e-16**. Keep that as the week-2 acceptance test. Two ways to break it:
composing in the wrong order, and scalar-last quaternions — MuJoCo is
`(w, x, y, z)`.

**The joint anchor is not a third way, on this robot.** `a - Rj @ a` in `fk_foot`
is dead code for the G1: **all 30 joints have `jnt_pos = 0`**, and so does every
humanoid in Menagerie (h1, berkeley_humanoid, op3, apollo, booster_t1 — checked).
MJCF authors put the joint at the body origin and carry the offset in `body_pos`.
Keep the term — it is right in general — but do not claim students can break FK
by deleting it, and do not set it as an exercise: an earlier week-2 exercise did
exactly that and the answer was an error of 0.0000. Week 2 now demonstrates the
anchor on a nine-line MJCF with `<joint pos="0.3 0 0">`, where dropping the term
moves the site 0.215 m while MuJoCo holds it exactly still.

**Week 2 carries eight generated figures**, all computed from the measured model
rather than drawn, so a diagram cannot drift from the prose. Two mechanics worth
keeping: `render_poses` is capped at **height 480** by the G1 scene's offscreen
framebuffer (ask for more and it raises before touching GL — and `_new_renderer`
then reports it as "could not create a renderer / check that a GL backend is
available", which mis-describes a pure size error); and the robot shots are
**cropped to `frame[185:445, 105:295]`** because the default free camera is too
far and too frontal to show a bent knee. Only the left leg is posed, leaving the
right leg in shot as a ruler. This makes week 2 the **first week that needs a GL
backend** — it used to run on a CPU Colab runtime and no longer does.

**Week 2 builds up from a planar two-link leg before the 3D chain**, because the
audience is undergraduates with no robotics background. The paper model —
`foot = f(hip_pitch, knee, ankle_pitch)` from sines and cosines, roll and yaw held
at zero — agrees with MuJoCo to **2.3e-06 m**, and the residual is a constant, not
noise: at the zero pose MuJoCo puts the foot at `x = -2.331e-06` rather than
exactly under the hip anchor at `x = 0`. Say so on the slide; it is the difference
between a *model* of the leg and the leg, and the full chain then reaches 1e-16.

**Two different thigh lengths, both correct — do not "fix" either.** Week 2's
planar model needs the **in-plane projection L1 = 0.3366 m**, because the
hip->knee vector is `[0, +0.0541, -0.3366]` and splays 5.4 cm sideways. Week 3's
law of cosines needs the **true 3D length 0.3409 m**. Using 0.3409 in the planar
formula costs 4 mm. The shin is 0.3000 m either way, and the ankle-to-foot-site
drop is 0.0176 m.

**Week 2 has one knob worth knowing about**: `crouch` in the first cell feeds the
three rendered poses, the leg diagram, the Jacobian bar chart, the column-vector
diagram and the singular values — change those six numbers and five figures
redraw. Useful for a live demonstration, and a trap when editing: a change there
is never local. Worth copying as an authoring pattern — one named pose near the
top that the whole second half of a week refers back to.

**Introduce a MuJoCo array before the slide that uses it.** Week 2 reached for
`mj_forward` in its first cell and `site_xmat` 400 lines later, both unexplained,
which made a deck aimed at beginners hard to follow. Two reference slides now sit
straight after the setup cell. The rule worth reusing in any week: **an `x` prefix
means "computed, in world coordinates"; no `x` means "declared in the MJCF,
relative to the parent"** — `body_pos` vs `xpos`, `site_pos` vs `site_xpos`. And
say plainly that **`mj_forward` IS MuJoCo's forward kinematics**: it reads `qpos`
and fills every `x` quantity, which is what makes "write your own and compare"
meaningful rather than arbitrary.

The same slide pair exposes a genuine hazard worth keeping: `leg_qpos_indices`
returns `7..12` and `leg_dof_indices` returns `6..11` — same six joints, addresses
off by one, because the floating base spends 7 numbers on position and 6 on
velocity. Mixing them is a silent bug. `mj_jacSite` needs its own note too: it
**writes into** the arrays passed to it and returns nothing, and they are `3 x nv`
— every joint in the robot, not just the leg.

**The Jacobian is taught as a derivative, not a black box.** Week 2 measures first
(nudge 0.01 rad, read millimetres), then divides by the nudge and writes
`J = ∂x/∂q` with `J e_j ≈ [f(q + ε e_j) − f(q)] / ε`. The 3×6 finite-difference
matrix at `ε = 1e-6` matches `mj_jacSite` to **3.1e-07**. An ε sweep then shows
that smaller is not better: error falls exactly ten-fold per decade from 1e-1 to
**1e-8 (1.1e-08, the minimum — √machine-ε is 1.49e-08)**, then round-off takes
over and 1e-12 (6.6e-05) is worse than 1e-4 (3.1e-05). The ten-fold-per-decade
stretch is visible proof the approximation is first-order, and the whole curve is
the argument for `mj_jacSite`: MuJoCo differentiates analytically, so there is no
ε to trade.

**Draw Jacobian columns as vectors from a common origin, never on the leg.**
Positions and velocities are different spaces, and on the leg the labels collide
because `hip_pitch` and `ankle_pitch` are nearly parallel. Side view plus top view
works: `hip_roll`/`hip_yaw` are invisible from the side and dominant from above.
Note that **both** ankle joints are missing from the drawing, for different
reasons — `ankle_pitch` is a 0.02 m/rad stub below the draw threshold,
`ankle_roll` is exactly zero. `matplotlib`'s `annotate` does **not** extend the
axis limits, so arrows drawn that way are silently clipped unless the limits are
set from the tips by hand; that cost two renders.

**Say why three compact representations exist before comparing them.** The reason
is the constraint count, not the number count: matrix 9 numbers / 6 constraints,
quaternion 4 / 1, axis-angle and Euler 3 / 0. A matrix you keep multiplying drifts
off SO(3) and needs re-orthonormalising against six rules; a quaternion needs
`q /= norm(q)`. Measured over 100k compositions: matrix `|RRᵀ−I|` reaches 1.4e-12,
quaternion `|q|−1` only 2.7e-13. That is why `qpos` stores the base as 4 numbers.

**The "representation → R only" rule is true for week 2 and false for week 3 — so
teach it with its scope.** Forward kinematics never asks "what quaternion is this
matrix?", but `kinematics.pose_error` runs `mju_mat2Quat` then `mju_quat2Vel`,
because the site Jacobian's angular rows want a 3-vector and **an error must be
something you can scale and add** — half a rotation matrix is not a rotation, half
an axis-angle vector is "go half way". Week 2 demonstrates it live: ask for 2 cm
and 10° and `pose_error` returns exactly `[0.02, 0, 0, 0, 0.1745, 0]`. Do not write
"the reverse conversion is never used"; it is used, and the reason it is used is
the best argument for the compact forms.

**Week 2's rotation representations hang on one worked example**: 120° about
`(1,1,1)/sqrt(3)`, chosen because every form is memorable and hand-checkable — the
quaternion is `(0.5, 0.5, 0.5, 0.5)`, the matrix is a permutation sending
x→y→z→x, and the Euler triple is roll 90 / pitch 90 / yaw 0 in `xyz`.

**Every conversion to `R` is written out and checked** rather than left inside a
MuJoCo call — Rodrigues (2.2e-16), the quadratic quaternion 3x3 (1.1e-16), and
`Rx @ Ry @ Rz` (1.1e-16) — because the matrix is the only form that acts on a
vector. **MuJoCo's lowercase `'xyz'` is intrinsic and equals `Rx @ Ry @ Rz`;
uppercase `'XYZ'` is extrinsic and equals `Rz @ Ry @ Rx`** — verified on a
non-degenerate triple (20, 35, 50), where the reversed product does not match.
Do not check this against a triple with a zero in it: `Rz(0) = I` hides the
difference.

All three failure modes are measured too: `q` vs `-q` differ by exactly `0.0`; the
validated pose's foot quaternion is tilted 7.6° but reads as 174.8° scalar-last
(the down-vector flips from -1.0 to +0.99, while staying unit length and legal);
and at pitch 90° four different (roll, yaw) pairs sharing a sum give a
bit-identical quaternion, with the degeneracy fading slowly — 0.52° apart at
pitch 89°, 5.17° at 80°, 15.36° at 60°.

MuJoCo supplies every conversion (`mju_axisAngle2Quat`, `mju_euler2Quat` with a
`seq` string, `mju_mat2Quat`, `mju_quat2Vel`). Week 2 hand-writes them anyway and
diffs against these — that is the teaching, not duplication — but **package code
must call MuJoCo's**, never a copy of the slide's.

**The `left_foot` site is NOT the sole — do not label it one.** It is a site at
the `left_ankle_roll_link` **origin** (`site_pos = [0,0,0]`), and it sits **3.5 cm
above** the four contact spheres. At `stand` it reads `z = 0.0331` while the foot's
underside is at `z = -0.0019`, i.e. on the floor. Week 2 drew it as "sole" against
a floor line at `z = 0` and the leg looked like it was floating for no reason;
the fix was to draw the real foot underneath (measured from the group-3 geoms:
heel at `x = -0.05`, toe at `+0.12`, so the foot is 0.170 m long and longer in
front, which independently confirms `+x` is forward) and to say why there is a gap.

**A kinematics figure that pins the pelvis will float.** In the week-2 crouch
drawing the pelvis stays at standing height while the leg bends, so the foot rises
**3.7 cm** off the floor. That is correct and is the point — kinematics places a
pose, it does not settle a robot — but it must be *annotated*, or it reads as a
bug. A real crouch lowers the pelvis instead, which is what week 4 does. Note the
crouch's pitch angles sum to zero (-0.35 + 0.70 - 0.35), so the foot stays level.

**The foot site sits exactly on the ankle-roll axis**, so that joint moves the
foot 0.00 mm and the *position* Jacobian has a zero column there; it still changes
orientation, which is why the full 6x6 is non-singular at the crouch. Week 2
introduces the Jacobian as a measured mm-per-0.57-degree table (hip_pitch 6.16,
hip_roll 5.09, hip_yaw 3.17, knee 3.17, ankle_pitch 0.18, ankle_roll 0.00) and
only then names it — the finite differences and `mj_jacSite` agree exactly.

**Leg segment lengths must be measured between joint anchors, not from
`|body_pos|`.** The hip is three separate link bodies whose offsets accumulate,
so `|body_pos|` of `knee_link` gives 0.194 m when the real thigh is **0.341 m**
(shin 0.300 m, so reach is an annulus from 0.041 m to 0.641 m). Getting this
wrong silently produced a reachability table calling 0.50 m unreachable.

The week-3 circle demo tracks to 8.4e-05 m. A taller circle leaves the reachable
set and pins at 2.68e-03 m regardless of iteration count — that invariance is
deliberate teaching material, not a solver defect. Do not "fix" it by raising
`iterations`; it does nothing.

`soc4180.launch_viewer()` opens MuJoCo's interactive desktop viewer (blocking,
or `passive=True` for a handle you can step yourself). It raises on Colab, which
has no window to draw into — that is why labs render video. `brax.io.html.render`
is the inline-interactive option for MJX rollouts.

`soc4180.robot_path()` uses `Robot.xml(entry)`. **Not `Robot.path(entry)`** —
that signature takes a cache and returns the robot's directory, so passing an
entry name raises `AttributeError: 'str' object has no attribute 'resolve'`. The
function was written wrong and unused until someone asked for a viewer path.

`render_poses` draws a sequence of `qpos` without stepping physics — use it for
anything demonstrating kinematics, so the robot does not fall over mid-lesson.

### The whole body (week 2b)

Week 2 teaches one leg and never says which six of the thirty-six numbers it is.
**`02b-robot-as-code` is week 2's lab class**: the same robot read as a data
structure, then `lab_body.py` on the laptop. Week 2's deck stays the lecture and
is unchanged at 46 slides — **do not merge them**, 67 slides is not one session.

`src/soc4180/bodies.py` carries the anatomy: `CHAINS`, `set_pose`,
`joint_index`/`dof_index`, `group_indices`, `chain_bodies`, `mirror`, `describe`.

Measured against the pinned G1, and used on slides:

- **Five chains off one pelvis**: left leg 6, right leg 6, waist 3, left arm 7,
  right arm 7 = **29 = `nu`**. Each is **contiguous in `qpos`** in root-to-tip
  order, and together they cover slots 7..35 exactly. The arms hang off
  `torso_link`, three waist joints above the pelvis; the legs hang off the pelvis.
- **`qpos` index − `qvel` index = 1 for all 29 joints.** Both indices are valid
  on either array, so confusing them silently addresses the neighbouring joint.
  The strip diagram exists to make that shift visible rather than a footnote.
- **The effect heat map is the best figure in the week.** Nudge one joint 0.1 rad
  from `stand` and measure five landmarks: a leg joint moves one foot and nothing
  else, a waist joint moves **both hands and neither foot**. The blank cells are
  the tree. Within a limb the effect falls root to tip — hip pitch 65 mm, hip roll
  61, knee 32, hip yaw 13, ankle pitch 2. Waist: roll 25, yaw 22, pitch 11 at the
  hands. Shoulder pitch 38, shoulder roll 36, elbow 18, shoulder yaw 6,
  wrist pitch 5.
- **Three joints move their landmark by exactly zero** — `ankle_roll`,
  `wrist_roll`, `wrist_yaw` — because the landmark sits on the joint's own axis.
  They still rotate it (28.6° per 0.5 rad). **Threshold the heat map at 1e-9 mm**:
  `left_wrist_roll` returns 7e-15 mm where `right_wrist_roll` returns exactly 0,
  so an untresholded `> 0` test colours one and not the other and looks like an
  asymmetry that is not there.
- **There are no hand sites.** `nsite = 4` is two feet and two IMUs, so anything
  about a hand must read `xpos` of `left_wrist_yaw_link`. Use `imu_in_torso` for
  the torso, not `torso_link`: that body's origin is on the waist axes and barely
  moves (0.4 mm under `waist_yaw`, 0.0 under roll and pitch), while the IMU site
  moves 4/15/15 mm.
- **Mirroring negates roll and yaw, not pitch.** The MJCF's own `stand` keyframe
  proves it: left arm `[0.2, 0.2, 0, 1.28, ...]`, right arm `[0.2, -0.2, 0, 1.28,
  ...]`. `soc4180.mirror` flips any joint whose name contains `roll` or `yaw`.
- **The G1 is mirror-symmetric to 10 µm, not to machine precision.**
  `left_shoulder_pitch_link` sits at `y = +0.100220` and its twin at
  `y = −0.100210`. That is the *only* asymmetric `body_pos` in the model. Mirrored
  hand positions therefore agree to 1e-5 m, not 1e-16 — do not "fix" the mirror
  code chasing it, and do not assert 1e-16 anywhere near it.
- **`set_pose` pins the pelvis**, so a symmetric squat lifts the feet **0.111 m**
  off the floor. That is week 2's floating-crouch lesson for the whole body, and
  it is annotated, not hidden.

`lab_body.py` follows `lab_viewer.py`'s shape but drives all five chains: keys
`1`–`5` highlight a chain (yellow spheres along `chain_bodies`, white at the
landmark), `M` mirrors, `ENTER` prints the pose back as a pasteable `set_pose`
call. Same `ctrl → qpos` slider wiring as week 2 — in a kinematic script the
sliders are dead otherwise.

**Week 2b introduced a third setup-cell form, and it is the right one from now
on.** The `try/except` form never updates; the bare unconditional upgrade would
install the *committed* package over the editable venv during a local
`quarto render`, which is why week 1 could not use it. Guard it on Colab:

```python
import importlib.util
if importlib.util.find_spec("google.colab") is not None:
    %pip install -q --upgrade "soc4180 @ git+https://github.com/gnoejh/soc4180.git"
import soc4180
```

Always upgrades on Colab, before the first import; installs nothing locally. Use
this for any future week that adds package code. (`%pip` inside an `if` is fine —
IPython transforms magics at any indentation.)

## Walking (week 4)

`kinematics.py` (damped least-squares leg IK) and `walking.py` (LIPM + footstep
gait) make the G1 walk **~1.0 m in 9 s, open loop, with no learning**. Two bugs
cost real time here; do not reintroduce them.

**The `stand` keyframe is a kinematic singularity.** Measured: the leg Jacobian's
smallest singular value is **8.96e-07 at `stand`** (numerical rank 5 of 6) against
**0.0887 at the week-4 crouch** — a factor of ~99,000, and the answer to a week-2
exercise. Every leg joint is exactly zero, i.e. a perfectly straight leg, so the
Jacobian has no direction that shortens it and IK cannot lower the body at all (the knee range is
`[-0.087, 2.880]`, so it also clips immediately). Seed IK from the bent-knee
crouch in `WalkingController.nominal`, never from `stand`.

**Chain the LIPM boundary value problems.** Each step's CoM trajectory must start
where the previous step ended (`_build_segments`). Computing each step absolutely
instead teleports the commanded pelvis backwards by half a stride at every
support exchange — the robot then falls, and it looks like a balance problem
rather than the bookkeeping error it is.

Verified behaviour, useful as regression checks:

- total vertical ground reaction ≈ body weight (~327 N)
- measured ZMP y reaches ±0.26 m against stance feet at ±0.119 m — **outside the
  support polygon**. The LIPM's assumptions fail (point mass, constant height,
  massless legs, stiff servos), not the ZMP criterion. Teach this; do not "fix" it.
- the gait is genuinely sensitive: several nearby `GaitParams` settings fall over.
  Defaults came from a sweep, so re-sweep before changing one.

`render_rollout(..., track="pelvis")` follows a body with the camera. Any
locomotion video needs it — the robot leaves a fixed frame in about two seconds.

## Pedagogical constraints that drive the code

- **A robot must walk by week ~4 using analytic control (LIPM/ZMP), long before
  any RL.** If the first walking robot depended on a policy converging, a failed
  training run would leave the course with no walking robot at all.
- **Every RL lab must ship a pre-trained checkpoint fallback** (`checkpoints.py`),
  so a lab never dead-ends on a non-converging run. No suitable public checkpoints
  exist — they must be trained and hosted.
- **Never grade convergence**; grade the diagnosis.
- G1 is a 29-DOF full humanoid. If training proves too slow, `berkeley_humanoid`
  (12 actuated DOF) and `robotis_op3` are the cheaper fallbacks and also have
  tuned `playground` locomotion envs.
