# Wrong-direction diagnosis on fresh task 01

Status: **COMPLETE**. Five native alternatives and GPU scoring exited 0.
This diagnoses the first
500-ms decision of fresh-comparison case 2; it is not another navigation trial.
The previous full fresh comparison remains 0/4 final arrivals for both methods.

| Action | Forecast learned cost | Actual-image learned cost | Physical cost | Actual XY cm | Actual heading degrees |
|---|---:|---:|---:|---:|---:|
| Hold | 168.671 | 155.858 | 181.798 | 32.358 | 40.453 |
| Forward | 150.214 | 152.618 | 142.825 | 26.352 | 40.461 |
| Left arc | 115.073 | 110.892 | 196.293 | 27.682 | 52.683 |
| Right arc | 131.536 | 115.326 | **108.923** | 26.427 | 27.959 |
| Left turn | **92.616** | **110.548** | 225.247 | 32.099 | 52.609 |
| Right turn | 138.272 | 125.237 | 147.858 | 32.290 | 28.335 |

The learned metric selects left-turn even when supplied actual future images.
Thus its scoring error is sufficient to explain this wrong-direction choice;
perfect successor features would not fix that choice under the same metric.
The predictor exaggerates the left-turn preference but does not create it.
Physical cost is squared planar distance / 3 cm plus squared wrapped world-yaw
difference / 5 degrees, summed. Native relative-frame heading is also reported
separately above. Raw dense image-goal MSE selects right-turn, which turns the
correct way but is not the best physical cost among the six actions.

Mean future dense-feature MSE is 0.348247 versus persistence 0.628064, a 44.6%
reduction. Correct-action retrieval is 5/6 (left-turn misses); hold prediction
is worse than persistence. Useful action-sensitive forecasting therefore
coexists with failed goal scoring. This one state does not establish a general
predictor advantage, and forecasting errors still matter for other costs.

All five branches reproduced the full native departure prefix and the three
context RGB images exactly, then completed without contact. All six actual
command tapes match their forecast inputs. Reconstruction reproduced the
original six costs within 1e-6 before any successor RGB was encoded.
Each native alternative took 8.29–8.58 seconds; GPU scoring took 6.13 seconds.
Native handles: 29758, 14164, 55184, 9673, 66835; scoring handle: 14078.

A separate fixed-cost diagnostic also completed (session 73007, exit 0):
`go2_signed_goal_cost_diagnostic_2026-09-17.json`. Reuse the signed goal readout
already present in the direct-feedback baseline and arrival detector. Its
squared normalized pose estimate on actual successor images chooses right-arc,
matching physical progress. On predicted successor features it chooses forward:
forecast costs are 111.96 forward versus 130.13 right-arc, while actual-image
costs are 113.58 forward versus 78.46 right-arc. This avoids the wrong-direction
preference but still exposes imperfect forecast/readout transfer. No training
or thresholds changed. The subsequent four-task cost-only prospective pilot
is complete: `go2_signed_pose_goal_pilot_2026-09-17.md`. It remains 0/4 final
arrivals and introduces one contact, despite reducing the large right-task
heading errors. These are exposed development cases, not a new independent
test. A further training-only diagnostic measures the missing cross-trajectory
supervision directly: `go2_goal_metric_turn_separation_2026-09-17.md`.

The controller originally selected left-turn despite a rightward goal. Freeze
that original six-action cost vector, model, learned goal metric and supplied
goal image. Reproduce the initial ten quiet ticks and execute hold, forward,
left-arc, right-arc and right-turn for five ticks each. Reuse the recorded
left-turn successor rather than count a repeated factual outcome as new data.
All six outcomes must match their candidate post-limiter command tapes.

Compare predicted learned goal cost against actual successor-image learned
cost, raw feature goal MSE, and independently measured XY/heading goal cost.
Reconstruct predictions using only causal context before encoding any actual
successor, and reproduce the original cost vector. Also report common-space
forecast error versus persistence and cross-action retrieval. These distinguish
an inaccurate visual forecast from a scoring function that ranks even actual
images incorrectly. A one-state diagnostic cannot establish navigation success
or the optimal long-horizon action.

Native alternatives use the RGB-only fresh-layout session, the same physical
seed/gait/controller gains, and exact native pose/joint/command plus context-RGB
matching. Full prospective RGB, physics, commands and failures are retained;
unused depth is not recorded. No weights or thresholds change.

Before launch: 72 GiB RAM available, no competing experiment, GPU utilization
11%. Two CPU-native workers use cores 4-7 and 8-11; GPU scoring follows native
collection. The dedicated volume had about 585 MiB free and met a 512-MiB
reserve plus 48-MiB measured-size allowance for five short RGB-only records.

Plan: `go2_fresh_goal_direction_plan_2026-09-17.json`.
Result: `go2_fresh_goal_direction_result_2026-09-17.json`.
Runner/scorer: `scripts/diagnose_go2_fresh_goal_direction_development.py`.
Output: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_fresh_goal_direction_v1_attempt_001`.
