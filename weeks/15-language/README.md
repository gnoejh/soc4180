# 15 — Language

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gnoejh/soc4180/blob/main/weeks/15-language/lab.ipynb)

| | |
| --- | --- |
| **Layer** | 5 — behaviour: words, pixels, and an agent |
| **Runtime** | **GPU runtime on Colab** for rendering |
| **Wall clock** | ~2 min (one 3000-picture training on the CPU) |
| **Needs** | `soc4180[rl]` (torch); `checkpoints/heatnet.pt` for the lab |

## Objectives

1. Place a course-scale vision-language-action model next to RT-2 / OpenVLA / π0.
2. Turn words into a bag of words and pixels into keys; match them with one attention
   step (key · query / √C) and read *where* from the heat map.
3. Measure what it understood: new phrasings work, an unknown colour word is chance —
   and why that is not understanding.
4. Show that a network cannot check itself (confident when the ball is absent), and
   verify with a different sense.
5. Build an agent: sentence → plan of skills → physics → a sensor check after each step.

## Measured results (`ground.py`, 3000 pictures, 10 epochs)

| instructions | error | within 5° |
| --- | --- | --- |
| phrasings it was trained on | 2.0° | 99 % |
| new phrasings, known colours | 2.1° | 99 % |
| "crimson" (never seen) | 31.0° | 7 % |
| no colour word | 31.0° | 6 % |

`HeatNet` has 20,032 parameters and trains in ~30–45 s on the CPU. Heat-map peak, 200
pictures: named ball present — mean 0.88, lowest 0.42; **named ball behind the robot —
mean 0.60, highest 0.98**. `look at the blue ball` turns the waist −10.6° and centres
blue (−1.1°).

Agent (`lab_agent.py`, `soc4180.agent`): "raise both hands" parses to `raise_both` and
**falls** (week 3's finding); left then right puts both hands at 1.36–1.37 m. "find
the yellow ball" with yellow 110° to the left: `look` fails its colour check, `search`
turns +60, −60, +120 and finds it after 3 turns.

## Lab class: on your laptop

```bash
uv run weeks/15-language/lab_agent.py          # the game: edit instructions.py
uv run weeks/15-language/lab_agent.py --grade --no-viewer
uv run weeks/15-language/lab_agent.py --say "look at the red ball and raise your left hand"
```

Students edit only `instructions.py`: `SYNONYMS` (word → a word the robot knows),
`PROGRAMS` (phrase → a plan), `SEARCH` (`step_deg`, `tries`) and `LOOK_GAIN`. Each
instruction is parsed, run under physics with checks, then played back. As shipped
0 / 5; the reference (`instructor/instructions_solution.py`, sealed) 5 / 5 in seconds.

| # | Instruction | Reference |
| --- | --- | --- |
| 1 | "look at the green ball" | `LOOK_GAIN` 0.8 |
| 2 | "face the crimson ball", "look at the azure one" | `crimson → red`, `azure → blue` |
| 3 | "raise both hands" (and still stand) | `both hands → [raise left, raise right]` |
| 4 | "find the yellow ball" (110° left) | `SEARCH` 60°, 4 tries |
| 5 | look, raise, then find | all of the above |

`checkpoints/heatnet.pt` (84 kB) is `ground.py --save` with the default settings.

### `ground.py`

```bash
uv run weeks/15-language/ground.py
uv run weeks/15-language/ground.py --say "look at the green ball"
uv run weeks/15-language/ground.py --save checkpoints/heatnet.pt --no-viewer
```

## The LLM planner

The deck shows the Anthropic tool-use loop with the four skills as tools, as a plain
code block — **not a cell**: it needs an API key and the `anthropic` package, which
`uv.lock` does not install. It is exercise 5.

## Rebuild

```bash
quarto render weeks/15-language/slides.qmd
```
