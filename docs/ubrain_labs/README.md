# Labs

Eleven sessions, each about 90 minutes. Each lab runs one example, explains the law
behind what you see, and ends with exercises: some ask you to predict a number
before you measure it. Labs 4 and 5 need the robot body (`bodies/mujoco`, set up once
with `uv sync` there); every other lab needs only a laptop.

| lab | region | example | needs |
|---|---|---|---|
| [1. Learning from reward](lab1_striatum.md) | striatum | `01_bandit.py` | nothing |
| [2. One-shot memory and its capacity](lab2_hippocampus.md) | hippocampus | `02_memory.py` | nothing |
| [3. Plugging in a trained model](lab3_codecs.md) | codecs + hippocampus | `03_clip_words.py` | `--extra vision` |
| [4. A brain in a body](lab4_walker.md) ([slides](lab4_slides.html)) | cerebellum | `lab_cerebellum.py` | the body |
| [5. Seeing and remembering](lab5_explorer.md) | all of the above | `g1_explorer.py` | the body, `--extra vision` |
| [6. Holding a list](lab6_working_memory.md) | prefrontal buffer | `06_working_memory.py` | nothing |
| [7. From hippocampus to cortex](lab7_consolidation.md) | hippocampus + cortex | `07_consolidation.py` | nothing |
| [8. Noticing](lab8_attention.md) | salience gate + modes | `08_attention.py` | nothing |
| [9. Memory first, habit later](lab9_memory_and_habit.md) | episodic control + striatum | `09_memory_and_habit.py` | nothing |
| [10. One number each](lab10_neuromodulators.md) | noradrenaline, serotonin | `10_neuromodulators.py` | nothing |
| [11. A brain's life](lab11_a_brains_life.md) | save, load, datasheet, monitor | `11_save_and_load.py`, `12_monitor_and_datasheet.py` | nothing |

Labs 1-5 are the core course; labs 6-11 can follow in any order after lab 2 (lab 9
also uses lab 1's striatum). The two game examples (`04_blackjack.py`,
`05_frozenlake.py`, with `--extra game`) take lab 1's striatum into Gymnasium games.

**The habit these labs teach:** write your prediction down before you run. A brain
built from laws can be predicted, and the interesting moments are where your
prediction and the measurement disagree.
