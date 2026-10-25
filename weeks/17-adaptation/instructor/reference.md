# Week 17 worksheet: reference results (instructor only)

Measured 2026-10-25 on the instructor's machine (Windows, 36 cores), ubrain 0.14.0.dev1
(vendor/ubrain), body digest 4815eedf56d0. `lab_adapt.py` defaults unless stated: wind
15 N (left +), 0.5 m/s, a 200-tick calm settle walk before the schedule. The body runs
in lockstep, so a run is bit-identical on one machine; a run that ends in a fall is
chaotic and differs between machines, so judge falls by "falls / does not fall".
"Turn at the end" = the mean of the last 5 strides of the block.

## Part A

| Row | Command | Turned in the wind | Turn at the end | Turned after | Point |
| --- | --- | --- | --- | --- | --- |
| A1 | `--learner off` | +230.1° | +0.173 rad/s | +2.4° | turns all the way; nothing to undo after |
| A2 | `--learner single` | +44.0° | +0.001 | **−43.4°** | integrator: turn to zero, mirror-image after-effect |
| A2 | `--learner dual` | +69.7° | +0.036 | **−24.6°** | leaky: a steady leftover turn, smaller after-effect |
| A3 | `--wind 10 --learner off` | +113.6° | +0.084 | −1.0° | |
| A3 | `--wind 10 --learner single` | +21.8° | +0.001 | −21.3° | counter-turn −0.091 |
| A3 | `--wind 10 --learner dual` | +34.1° | +0.018 | −12.3° | leftover halves with the wind |
| A5 | `--wind -15 --learner off` | −180.8° | −0.135 | −7.7° | the robot is not symmetric (+230 vs −181) |
| A5 | `--wind -15 --learner single` | −30.9° | −0.001 | +31.0° | |
| A5 | `--wind -15 --learner dual` | −53.1° | −0.030 | +16.7° | |
| A6 | `--wind 20 --learner off` | falls in stride 8 of the wind | | | |
| A6 | `--wind 20 --learner dual` | falls in stride 11 | | | the counter-turn was −0.21 when it fell |
|    | `--wind 25`, either | falls in stride 5 | | | 30 N: stride 4 |

Adaptation speed (A3, from the stride list): the turn peaks in stride 2 of the wind and
halves 3 strides later at 10 N (both learners), 3 (dual) or 4 (single) at 15 N. A
linear learner's time course does not depend on the size of the disturbance.

Answers worth having ready:

- Q1: without learning the turn even grows during the block (0.13 rad/s in the first 3
  strides, 0.17 in the last 5); the walking policy does not correct heading.
- Q2/Q3: single `u <- u + eta e` stops changing only when e = 0 (integral action, week
  5's I term). Dual states forget (a < 1), so they settle where learning per tick equals
  forgetting per tick, which needs a non-zero error; that error is proportional to the
  disturbance (0.018 at 10 N, 0.036 at 15 N), as week 5's sag tau_g / k_p is to the load.
- Q5: the after-effect measures what is still being SENT when the wind stops, i.e. how
  much was learned and kept; the integrator keeps everything (−43 vs +44), the leaky
  learner already lets part go. Bigger is not "better", it is "less forgetful".
- Q7: the counter-turn corrects the heading (yaw rate); it does nothing for balance.

## Part B

| Row | Command | Result | Point |
| --- | --- | --- | --- |
| B1 | `--schedule savings --learner single` | first 10 strides: +36.6° then +36.3° | no savings |
| B1 | `--schedule savings --learner dual` | first 10 strides: +34.0° then +31.4° | small savings (slow state) |
| B2 | `--blocks "5:0,30:15,6:0,20:15" --learner single` | +36.6° then +22.5°; counter-turn after the break −0.114 | NOT savings: never washed out |
| B2 | same, `--learner dual` | +34.0° then +28.2°; counter-turn after the break −0.061 | leftover again |
| B3 | `--schedule recovery --learner single` | 6 strides of counter-wind; clamp: counter-turn ends +0.015, turned +10.5° | no recovery |
| B3 | `--schedule recovery --learner dual` | 4 strides of counter-wind; clamp: counter-turn ends −0.027, turned −25.2° | spontaneous recovery |
| B4 | `--vx 0.3 --learner off` | +168.6° in the wind, turn at the end +0.126 | |
| B4 | `--vx 0.3 --learner single` | +34.5°, end +0.001, after −34.3° | adapts as well |
| B4 | `--vx 0.3 --learner dual` | +53.7°, end +0.028, after −20.1° | |

- Q8: dual, slightly (34.0 → 31.4 in the first 10 strides). The slow state (keeps 99.5 %
  per stride) still holds part of the first wind after the 30-stride wash-out.
- Q9: no: after 6 calm strides the single learner's counter-turn is still −0.114 (62 % of
  its −0.184): behaviour never returned to baseline, so the "fast relearning" is leftover.
- Q10: after unlearning, the dual learner's fast state is pushed the other way and the
  slow state still holds the wind; they cancel (u ≈ 0). Clamped, the fast state forgets
  in a few strides (0.9 per stride) and the slow one barely (0.995), so u drifts back
  toward the slow state's sign. The single learner has one state: frozen at ≈ 0.
- Q11: the dual learner's two states together learn faster per tick than the single rate
  (0.007 + 0.0008 against 0.005), and its fast state was already forgetting, so it gets back
  to zero sooner: 4 strides against 6.
- Q12: at 0.3 m/s the turn is smaller per second (0.126 vs 0.173 rad/s without learning)
  but larger per metre walked (0.42 vs 0.35 rad/m); adaptation works the same way.
- Q13: a robot in changing winds wants to let go quickly (dual: the fast state); one that
  meets the same wind daily benefits from keeping it (single, or the dual's slow state).

