# Dense forecasts across the native planner horizon

The fixed two-arm fit and evaluation are complete. Visual forecasting improves over action-blind prediction and persistence through 800 ms, while 500-ms dense error stays essentially unchanged from the retained parent. The current motion readout and float32 inference cost remain limitations for navigation. No new closed-loop trial has run.

Training session 66149 exited 0 after 1,760 updates per arm and 1,863.7 seconds (31.1 minutes). The encoder stayed frozen. Evaluation session 87270 exited 0 after 104.7 seconds. Final weights are retained; no dense feature cache was written to disk.

## Visual prediction on exposed transfer branches

Eighteen branches form six shared-history groups across two already exposed geometries. These are not eighteen independent mazes. MSE uses normalized dense features; lower is better. Actions are identical through 300 ms, so action retrieval is not applicable before 400 ms.

| Horizon (ms) | Action MSE | Action-blind MSE | Persistence MSE | Action retrieval | Scene–action interaction error / zero-interaction error |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.2222 | 0.2587 | 0.2568 | N/A | N/A |
| 200 | 0.2179 | 0.2655 | 0.2494 | N/A | N/A |
| 300 | 0.2208 | 0.2759 | 0.2619 | N/A | N/A |
| 400 | 0.2528 | 0.3058 | 0.3494 | 17/18 | 0.968 |
| 500 | 0.2475 | 0.3209 | 0.3878 | 18/18 | 0.877 |
| 600 | 0.2348 | 0.3147 | 0.4001 | 18/18 | 0.869 |
| 700 | 0.2343 | 0.3121 | 0.4018 | 18/18 | 0.880 |
| 800 | 0.2364 | 0.3116 | 0.4025 | 18/18 | 0.896 |

At 700 ms, dense error is 24.9% below action-blind prediction and 41.7% below persistence. Action retrieval is 18/18 from 500 through 800 ms. At 500 ms, MSE is 0.247465 versus the supplemented parent’s 0.247789 (0.13% lower), with both at 18/18 retrieval. This is preservation of dense performance, not evidence of an improved physical controller.

Double centering removes scene-only and scene-independent action effects. At 700 ms, aggregate interaction error is 12.0% below the zero-interaction reference. The per-prefix ratios are 0.743 after hold, 1.063 after left-turn and 0.890 after right-turn; the left-turn case does not improve. This is evidence of some scene-dependent visual action prediction, not geometry-specific understanding, collision prediction or a JEPA encoder-objective advantage.

Before 400 ms, there is no true action-dependent target variation. Tiny floating-point centering residuals make the unguarded within-scene centered-effect ratios in the raw report uninterpretable at 100–300 ms; do not treat those ratios as scientific effects. The double-centered diagnostic explicitly leaves negligible-energy ratios undefined, and identical-prefix predictions are broadcast exactly.

## Frozen motion decoder

The same readout was trained only on actual 500-ms feature pairs. Other horizons test its temporal transfer; no head was refitted. The observed-future column is an oracle diagnostic, unavailable online.

| Horizon (ms) | Action XY RMSE (mm) | Action-blind XY RMSE (mm) | Observed-future XY RMSE (mm) | Command-history XY RMSE (mm) | Action yaw RMSE (°) | Command-history yaw RMSE (°) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 10.456 | 11.899 | 11.015 | 0.582 | 1.995 | 0.121 |
| 200 | 10.290 | 10.777 | 10.127 | 1.080 | 1.923 | 0.323 |
| 300 | 10.564 | 10.969 | 10.175 | 2.033 | 1.938 | 0.427 |
| 400 | 11.660 | 10.212 | 10.086 | 4.304 | 1.753 | 0.554 |
| 500 | 9.805 | 7.972 | 10.064 | 5.113 | 2.024 | 0.588 |
| 600 | 9.758 | 8.033 | 10.087 | 6.295 | 2.073 | 0.732 |
| 700 | 10.421 | 8.761 | 9.955 | 7.269 | 2.152 | 0.867 |
| 800 | 11.124 | 9.664 | 10.169 | 8.232 | 2.241 | 0.937 |

The 500-ms motion decode regresses from the retained parent’s 9.067 mm / 1.646° to 9.805 mm / 2.024°, despite stable dense MSE. The oracle decoder remains weak, so the result cannot establish that motion information is absent from the visual features. All matched physical motions are exactly equal across paired scenes, at all eight horizons, and there are no contacts in this panel. Obstacle-dependent physical dynamics are therefore untested.

The established 0.25-m point goals at −45/0/+45° were also scored over the runtime’s 300→700-ms commitment interval. Mean transfer physical action regret is:

| Forecast | Mean regret (mm) |
|---|---:|
| action | 2.182 |
| no_future_action | 2.009 |
| persistence | 2.009 |
| observed_future | 2.926 |
| zero_motion | 2.009 |
| command_history | 1.654 |

This uses three recorded pulse branches, not all six runtime actions, and includes no path-feasibility test. It provides no evidence that the current decoded forecasts choose better physical waypoints. Modest error differences alone are not a requirement to beat before an informative closed-loop comparison; they must remain visible when interpreting its outcome.

## Inference timing and next experiment

Three measurements of native preprocessing, a three-image encoder batch, six candidate tapes at eight horizons, frozen motion decoding and CPU output copy took 2.528 / 2.513 / 2.586 seconds. Excluding the first repetition, the median is **2.550 seconds**. Encoding took approximately 0.64–0.70 s; prediction and decoding took 1.88 s. The measurement uses float32 without autocast and excludes sensing, tracking, mapping, routing and clearance. It exceeds the old 300-ms dispatch budget by 8.5×. No real-time claim is supported.

The next step is an explicit dense-model navigation integration, retaining shared observed mapping and physical arrival evaluation. Inference timing must be addressed through measured acceleration or an explicitly untimed, matched simulation treatment. Four prospective same-family mazes are already fixed in `go2_dense_world_model_maze_inventory_2026-09-18.json`; none has been run. The larger goal remains incomplete.

Raw evidence: `go2_horizon_dense_predictor_result_2026-09-18.json`, `go2_horizon_dense_predictor_evaluation_2026-09-18.json`, and `go2_horizon_commit_waypoint_diagnostic_2026-09-18.json`.
