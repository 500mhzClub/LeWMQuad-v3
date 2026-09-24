# Frozen models on four development mazes: complete result

All eight fixed JEPA/direct missions completed, including both failures. With
the preceding eight supervised/pose-command missions, the same four layouts
now have sixteen recorded outcomes. No model or controller was tuned within
the JEPA/direct batch, and no failed attempt was replaced.

| Model | Verified goals | Verified round trips | Disallowed contacts |
| --- | ---: | ---: | ---: |
| JEPA | 3/4 | 3/4 | 0 |
| Direct prediction | 4/4 | 3/4 | 0 |
| Supervised rollout | 4/4 | 4/4 | 0 |
| Pose/command predictor | 4/4 | 4/4 | 0 |

This batch does not demonstrate a JEPA or neural-model advantage. The learned
supervised controller does demonstrate four successful goal-and-return missions
on layouts constructed after training and controller development. The simpler
pose/command predictor matches that success count with identical perception,
mapping, memory, action primitives and safeguards.

| Layout | JEPA | Direct | Supervised rollout | Pose/command |
| --- | --- | --- | --- | --- |
| 0 | Round trip, 157.98 s | Goal only; budget exhausted, 480.80 s | Round trip, 160.92 s | Round trip, 445.28 s |
| 1 | Round trip, 177.48 s | Round trip, 219.60 s | Round trip, 204.88 s | Round trip, 199.28 s |
| 2 | Round trip, 159.98 s | Round trip, 186.78 s | Round trip, 148.42 s | Round trip, 176.06 s |
| 3 | Visual tracking failure; 110.8 s camera recording | Round trip, 228.40 s | Round trip, 201.68 s | Round trip, 237.22 s |

Successful/budget durations are recorded simulation durations, not host real
time. The failed JEPA run has no normal completion-duration field; its camera
span is reported explicitly. Every counted arrival passed the same independent
physical check: within 40 mm for one second, zero requested commands throughout,
and measured 100-ms speed below 50 mm/s. Native poses and geometry were used
only for evaluation.

## What failed

Direct/layout 0 spent a long interval near the goal selecting hold. During
frames 1000–4199, 746/751 plans were on time, 722 selected hold, and hold was
the original highest-utility action for 698 of those choices. All six candidate
forecast paths passed clearance on those 722 hold plans. At frame 2000, hold
ranked highest without the predictive-arrival override being eligible; stopping
projection also admitted every candidate. This supports a forecast/scoring
preference explanation for that near-goal delay. The goal eventually passed
physical verification, but the return exhausted its budget. A separate late
return interval, frames 4500 onward, had 0/76 plans on time. These are distinct
observed problems; neither establishes a counterfactual outcome under a fix.

JEPA/layout 3 lost measured visual pose before any arrival. From frames 500
through 1108, its 152 plans selected 149 turns and three holds, with no
translation choices; 142 plans were on time. Across the run, six visual-recovery
triggers covered 98 plans. At frame 1100, the preferred right turn failed the
0.48-m forecast reserve while a left turn passed; frame 1104 instead requested
a right turn toward a previously supported view. The system then lost tracking.
The recorded path was only 3.049 m, and final physical goal distance was 1.608 m.

Exact public-sensor replay reproduced all 1,105 recorded raw poses and the
failure at frame 1105. Neither camera supplied an accepted current pose;
primary references lacked sufficient rigid matches and the auxiliary chain
also failed its acceptance checks. There was no old-view revisit attempt on
that frame under the four-frame cadence. These observations locate the failure;
they do not show that changing cadence or recovery would complete navigation.
Keep this full sensor recording for targeted follow-up.

## Forecasts and interpretation

The pose/command predictor had lower same-window XY forecast RMSE than the
neural prediction on all sixteen executed trajectories. These overlapping
700-ms windows are diagnostics, not independent missions or outcomes under an
alternative policy. In particular, low error on a largely stationary failed
trajectory must not be interpreted as superior navigation.

Mission duration is also not an isolated forecast-quality measure. Besides the
direct failure's late-return interval, the earlier pose/command layout-0 run
missed 690/693 planning deadlines during its long return pause. Simulator lag
and planning deadlines remain material limitations. Do not summarize only the
successful JEPA runs as evidence that JEPA is faster or better.

The models share the matched training setup, but there is only one training
seed and one execution per model/layout. All layouts belong to one procedural
maze family. The controller was developed using the supervised condition, and
the layouts were already exposed through supervised/pose-command execution
before the frozen JEPA/direct comparison. Thus this is development evidence,
not a final evaluation or complete causal attribution of training method.
Sensing uses synthetic 2-mm depth noise and ideal gyro. No realistic-sensor,
host-real-time, hardware or broader environment-type generalization claim is
established. Broader environment-type tests remain deferred at the user's request.

## Next scientific work

Compare the current successful controller against matched instantaneous-ranking
and prediction-off controls, keeping the current perception and memory package.
Distinguish removal of prediction-based ranking from removal of all predicted
feasibility/recovery rules; the former still uses a world model, while the latter
changes a larger controller package. Reuse the existing baseline implementations
and retain their known limitations rather than treating an intentionally weak
controller as evidence that learning is necessary.

Use the retained JEPA failure for a bounded perception/recovery diagnosis before
making a new controller claim. Memory attribution, independent replication,
realistic sensing/timing and bounded hardware evidence remain outstanding. The
broad goal is active and incomplete.

## Evidence

- Execution journal: `docs/go2_persistent_visual_learning_comparison_2026-09-16.md`.
- Frozen plan: `docs/go2_persistent_visual_learning_plan_2026-09-16.json`.
- Aggregate: `go2_persistent_visual_learning_complete_v1_attempt_001/result.json`.
- Four-arm comparisons and inspected PNG/SVG figures:
  `go2_persistent_visual_learning_comparison_layout00_v1_attempt_001` through
  `go2_persistent_visual_learning_comparison_layout03_v1_attempt_001`.
- Exact failure replay:
  `go2_persistent_visual_learning_jepa03_failure_cadenced_replay_v1_attempt_001`.
- Per-failure diagnoses remain in their original mission roots. Completed,
  diagnosed depth was retired under the standing policy; every outcome and
  non-depth record remains. JEPA/layout-3 depth is retained in full.
