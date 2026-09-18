# Learned candidate-rollout selection off

Run one fixed supervised seed-2026091001 mission on exposed shared-recovery
layout 0, then layout 1. Compare all outcomes with the completed original
forecast-ranking controller and the instantaneous-ranking controller that kept
predictive gates. No model, fit, layout or threshold is tuned between runs.

The off control uses the existing instantaneous distance/heading utilities
and the original reactive controller's current-position stored-clearance test
(0.45 m nominal radius). It skips the entire learned-forecast selection chain:
predicted paths/reserves, predicted recovery, forecast arrival overrides and
planned stopping projection. No learned rollout value selects an action.

The method trace identifies `_select_clear_prediction` as the common chain for
these consumers; the outer off mixin bypasses it without reading predictions.
It reads only the observed-map waypoint, current scan error and current-position
clearance. Routing-memory and terminal-pulse wrappers still run around action
selection. Camera-view geometry uses observed map/pose; mission arrivals use
measured poses and actual preceding command receipts. Actual translation-veto
and missing-primary view recovery remain measured mechanisms.

The model, correction and original scorer still compute as a partial workload
control, and model-output validity remains checked. Their scores are discarded.
Thus this disables learned candidate-rollout selection, not all model execution
or all geometric projection: observed camera-view planning and the actual
dispatch requested-speed stopping projection remain unchanged. Total computation
is not identical because forecast-dependent selection checks are skipped.

Keep the same sensor noise, tracker, floor/obstacle consumers, routing memory,
command windows, actual sensor dispatch safeguards, measured settling and
independent physical arrival evaluation. Both jobs run sequentially on their
original CPU groups. Preserve every failed recording and both fixed outcomes.

Roots: `go2_rollout_selection_off_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001`.
Evaluate physical goal/home arrivals, contacts, actual selection independence,
current-clearance deadlocks and actual veto/recovery exposure. Two exposed
development mazes and one model seed are insufficient for broad reliability,
learning or JEPA superiority, calibrated sensing or hardware claims.

Six focused tests passed: current-clearance hold/view behavior, no access to
forecast values or predictive selection chain, and the previously established
instantaneous-objective checks.

After layout 0 launched, inspection found inherited launch-description flags
still advertising enabled predictive recovery/stopping despite the explicit
off fields. The runtime already bypassed those methods. Preserve the original
launch and record the corrected effective fields in
`launch_annotation_correction.json`; the launcher description is corrected for
layout 1. This is an annotation-only change, with no controller or setting
change between missions. Recorded selection checks verify actual behavior.

Layout 0 exhausted the mission budget with no arrival: 4,805 poses, zero
contacts, maximum pose error 5.571 mm, 1,173/1,200 plans on time and 480.84
simulated seconds. Owner archived everything and exited 0 after 10:57.49,
without swapping. This is a mission failure despite the clean process exit.
All 1,200 decisions matched instantaneous current-clearance feedback, with
computed learned rollouts unused for selection.

There were 940 consecutive hold plans from frame 1,044 to 4,800 because
current stored clearance was below 0.45 m. Stored/native body-centre-wall
clearances were 0.448440/0.461108 m at the first blocked plan and
0.416560/0.429836 m at the end. Last translation request was at 100.58 s;
first blocked planning observation was 105.90 s. No actual translation-veto
view recovery occurred. Tracking continued; map discrepancy cause remains
unisolated and turn-only escape safety is not established. The full failure
recording and `rollout_off_current_clearance_diagnostic_v1.json` remain.

Interpretation constraint: this control replaces forecast feasibility and
recovery, including the 0.03-m predicted translation/turn reserve, with the
original reactive 0.45-m current-clearance rule. It therefore tests the entire
prediction-dependent selection package. The failure cannot isolate forecasting
from the extra clearance margin. A current-clearance comparison retaining the
same action reserve is needed before attributing a difference to future
prediction itself. Do not change the second fixed mission in response.

Layout 1 independently verified both physical arrivals at frames 1,228 / 3,504:
3,506 poses, zero contacts, maximum pose error 7.751 mm, 852/866 plans on time,
350.80 simulated seconds. Owner archived everything and exited 0 after 8:37.84,
without swapping. All 866 decisions matched the off selector. There were no
current-clearance-blocked plans and no actual translation-veto view recoveries.
The prolonged near-home stall resolved and is not a failed mission.

The final approach explains most of its extra duration. From first observed
distance <=0.10 m on the return leg to verified return, the full forecast
reference took 2.7 s, instantaneous ranking with predictive checks took 11.7 s,
and rollout selection off took 172.0 s. The off robot remained within 0.10 m
throughout that interval and applied 152.4 s of pure turns, 4.1 s of translation
and 15.5 s of zero command. It selected 386 right-turn, 19 right-arc and 19
forward plans in that interval. This is an inefficient terminal approach,
not a clearance deadlock or tracking outage. The three runs each record
`rollout_off_terminal_approach_comparison_v1.json`.

The instantaneous-with-checks reference used terminal position-priority overrides
to select forward instead of its preferred left turn on return frames 1,844,
1,860 and 1,864. This supports examining the terminal objective as well as
forecast quality. Different trajectories preclude a same-state counterfactual
navigation claim; the slow off return does not prove learned forecasts are
necessary for efficient arrival.

## Completed comparison

| Controller | Verified round trips | Contacts | Maze 0 / maze 1 simulated seconds |
| --- | --- | --- | --- |
| Forecast ranking and predictive checks | 2 / 2 | 0 | 245.58 / 167.28 |
| Instantaneous ranking and predictive checks | 2 / 2 | 0 | 251.24 / 188.04 |
| Instantaneous ranking and current-clearance feedback | 1 / 2 | 0 | failed at 480.84 / 350.80 |

Both three-way comparisons and inspected PNG/SVG plots are complete. Each
comparison has 172 unchanged common runtime sources and matching non-treatment
settings. The only between-off-run source change was the documented launcher
annotation correction; selection, sensors, thresholds and command guards did
not change. Both new recordings remain full, including the complete failure.
Three older redundant successes' depth retirement reclaimed 5,027,651,584
allocated bytes during this experiment. All failures and non-depth evidence
remain; artifact storage afterward is about 2.67 GB.

Aggregate: `go2_rollout_selection_off_complete_comparison_v1_attempt_001/result.json`.
This completes the specified two-maze learned-rollout-selection-off experiment,
not the broad navigation goal. It supplies a non-predictive selection control
and identifies two costs of removing the engineered predictive package, but
does not isolate learned prediction from clearance reserves and terminal
objective handling. The next baseline should retain the 0.03-m action reserve
using current clearance and use a non-predictive distance-priority terminal
policy. RGB dependence, multiple-layout replication, calibrated sensing/timing
and hardware evidence remain unresolved. No native owner is running.
