# 14 — Learning from pixels

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/14-vision/lab.ipynb)

| | |
| --- | --- |
| **Layer** | 2 (a camera) feeding 5 (what reads it is learned) |
| **Runtime** | **GPU runtime on Colab** — renders thousands of camera pictures |
| **Wall clock** | ~2 min (two 4000-picture trainings on the CPU) |
| **Needs** | `soc4180[rl]` (torch) |

## Objectives

1. Add a camera and a target with `MjSpec`; render from `camera="head"`.
2. Get labels for free from the simulator; train a convolutional network by regression.
3. Measure the visual sim-to-real gap: a blue ball, a dim room.
4. Domain randomisation for vision: what it fixes, what it costs, what data buys back.
5. Close a loop through the camera (visual servoing) and find week 5's gain lesson again.

## Measured results (`look.py`, 4000 pictures, 10 epochs, `torch.set_num_threads(4)`)

| trained on | red, as trained | blue ball | dim room (30 %) |
| --- | --- | --- | --- |
| red balls only | **1.0°** | 27.8° | 15.2° |
| random colour | 4.1° | **5.0°** | 27.2° |
| random light (30–100 %) | 1.3° | 28.7° | **1.6°** |
| both | 4.4° | 9.4° | 4.2° |
| both, 12,000 pictures, 20 epochs | 4.4° | 6.0° | 3.8° |

Always guessing the average: 26.0°. `BallNet` has 146,370 parameters; 4000 pictures
render in ~7 s and train in ~12 s on the CPU. Thread count changes the float sums and
so the last digits — every script fixes it at 4.

Turning the waist toward a ball 35° to the left, one correction per 0.1 s: gain 0.3 is
4.5° short at 0.6 s; 0.5–0.8 settle (0.8 peaks at 40°); 1.2 and 1.5 overshoot to 54°
and 60°; **2.2 throws the robot on the floor**, after which the ball leaves the
picture and the network's output is meaningless.

## Lab class: on your laptop

```bash
uv run weeks/14-vision/lab_look.py          # the game: edit eyes.py
uv run weeks/14-vision/lab_look.py --grade --no-viewer
```

Students edit only `eyes.py`: a training recipe per problem (`pictures`, `epochs`,
`colour` "red"/"random", `light` [low, high]) and `GAIN`. The referee renders,
trains (15–20 s, cached per recipe) and tests on 500 new pictures live, with the head
camera in the corner (white + truth, yellow + the network). As shipped 0 / 5; the
reference (`instructor/eyes_solution.py`, sealed) 5 / 5.

| # | Problem | Pass | Reference |
| --- | --- | --- | --- |
| 1 | RED | < 2° | red only: 1.0° |
| 2 | BLUE | < 8° | random colour: 5.0° |
| 3 | DIM | < 3° | light 0.3–1.0: 1.6° |
| 4 | ALL, one network | red < 6°, blue < 12°, dim < 6° | both: 4.4 / 9.4 / 4.2 |
| 5 | TURN | within 3° by 0.6 s, never past 45° | gain 0.8 (0.5 also passes) |

### `look.py`

```bash
uv run weeks/14-vision/look.py
uv run weeks/14-vision/look.py --randomise both --train 12000 --epochs 20
uv run weeks/14-vision/look.py --gain 2.2 --no-viewer
```

## Package

`soc4180.vision`: `camera_model`, `place_ball`, `ball_direction`, `render_dataset`,
`BallNet`, `uv_of`/`angles_of_uv`, and week 15's `Vocab`, `HeatNet`, `cell_of`,
`heat_to_uv`.

## Rebuild

```bash
quarto render weeks/14-vision/slides.qmd
```
