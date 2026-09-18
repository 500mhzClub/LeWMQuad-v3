# Stable references: verified outbound arrival, incomplete home approach

The first stable-reference learned native layout-4 run exited 0 and preserved all
4,805 camera pairs and registered poses. RecoveryStorage root:
`go2_stable_reference_jit_floor_cached_fine_goal_lzma_pulse_learned_round_trip_native_layout04_4800_v1_attempt_001`.

Physical evaluation verified outbound arrival at frame 2,436: distance
16.41–21.12 mm throughout the quiet one-second dwell, maximum 100-ms speed
0.02766 m/s, and all requested commands zero. There were no disallowed contacts.
Median/maximum position error was 10.53/15.67 mm. The preceding original-reference
learned run had 38.45/52.80 mm error and a false outbound arrival. These runs took
different closed-loop trajectories; this is live evidence supporting the treatment,
not a repeatability result or a matched-trajectory estimate of its effect.

The robot navigated back near home but exhausted its budget without completing
the home arrival. Final physical home distance was 101.56 mm; trajectory length
was 26.032 m. It had 1,161/1,192 on-time plans. From frame 3,400 onward it selected
227 forward pulses, 26 left arcs and 98 left turns, with no hold selections. Its
nearest observed home distance at a planning boundary was 29.70 mm, at frame 3,796.
It did not satisfy the unchanged observed 20-mm arrival criterion or complete
a verified round trip.

## Executed-forecast and selection diagnosis

`scripts/evaluate_saved_pulse_forecasts_development.py` uses the actual pulse/zero
tail command sequence, actual simulator command timestamps and saved online
forecasts. It scores only selected actions with a complete matching executed
prefix, against evaluator-only native physics. It does not rerun the predictor,
refit the residual or score unexecuted counterfactual outcomes.

In the final approach, 241 selected pulse translations matched the full command
prefix through 700 ms. Raw/corrected XY forecast RMSE at that horizon was
20.21/11.49 mm. For displacement from dispatch to 700 ms, raw/corrected RMSE was
17.81/8.05 mm. The frozen residual helps these executed pulses, but their remaining
error is relevant to a 20-mm target. The complete result is
`saved_pulse_forecast_evaluation_from3400_v1.json` in the run root.

The position-priority override also creates a concrete heading problem. At frame
3,796 the target was mostly sideways in the current body frame, at
[4.89, 29.45] mm. The combined distance/heading selector chose left turn, predicting
heading error improvement from 1.105 to 0.926 rad. Terminal position priority
overrode that with forward: it predicted just 3.71 mm of positional progress
while worsening heading error to 1.484 rad. The forward endpoint remained about
29.42 mm from the target. This demonstrates the selection mechanism; it does not
show what the unexecuted left turn would physically have achieved.

## Next native intervention

`ArrivalEntryTerminalPriorityRuntime` retains ordinary heading-guided selection
unless a terminal translation override predicts entering the existing arrival
disk at the 700-ms scoring endpoint. It restores the previously accepted action
when that condition fails, while preserving subsequent predictive holds or
clearance recoveries. Ordinary aligned translation remains available. Mission
criteria, clearance limits, action amplitudes, pulse durations and timing are
unchanged. This tests the heading-selection diagnosis without loosening arrival
requirements or assuming counterfactual outcomes.

The implementation is `lewm/arrival_entry_terminal_priority_development.py`,
selected by `--arrival-entry-priority` in the stable-reference native launcher.
The completed trial's original launcher was preserved under `source_witnesses/`
after checking its captured source hash. The saved-selection comparison is
`arrival_entry_priority_saved_selection_v1.json`. A fresh native trial is needed
to establish any navigation benefit.

Two focused tests passed: restore heading for a sideways goal while allowing a
predicted arrival, and preserve later holds plus ordinary aligned translations.
The saved-selection comparison changes 332 plans overall, including 247 in the
final home approach; frame 3,796 changes from forward back to left turn. No
counterfactual physical benefit is inferred.

The fresh native test was launched with `--layout-index 4 --arrival-entry-priority`
after the predecessor and diagnostic processes exited. Root:
`go2_arrival_entry_priority_stable_reference_jit_floor_cached_fine_goal_lzma_pulse_learned_round_trip_native_layout04_4800_v1_attempt_001`.
One native scene uses the existing worker layout. Its result is pending.

## Arrival-entry priority native result: verified round trip

The owner (handle 4914, PID 3360885) exited 0 and saved all 4,414 camera pairs
and poses. Physical evaluation verified both arrivals:

| Phase | Frame | Physical distance throughout dwell | Maximum 100-ms speed |
| --- | ---: | ---: | ---: |
| Outbound | 3,475 | 14.98–17.28 mm | 0.00630 m/s |
| Return | 4,412 | 14.34–17.78 mm | 0.01223 m/s |

Both one-second dwells had all requested commands zero. There were zero
disallowed contacts, median/maximum position error was 3.90/8.29 mm, and final
physical home distance was 17.41 mm. Timed execution was 442.59 wall seconds and
441.60 simulated seconds; path length was 22.555 m. Plans were on time for
1,008/1,096 decisions (91.97%). The successful run remains ideal-sensor,
shared-host measured-simulation evidence, not hardware or host-real-time
qualification.

The new terminal gate actually changed 11 decisions: six outbound and five
return. These are saved in `arrival_entry_priority_activation_v1.json`. The
physical evaluations and summary are in the run root. Different trajectories
and a single successful attempt do not establish the gate's causal benefit or
repeatability. Preserve this complete recording for subsequent scientific use.

The next comparison uses the same stable-reference launcher with `--reactive`
and layout 4, retaining the same sensing, map/routing, mission, timing and pulse
settings. The arrival-entry gate belongs to the predictive selector and is
absent from the reactive arm. JEPA training and memory contributions remain
separate outstanding comparisons.

That matched reactive trial has now completed with no arrivals and no contacts.
The paired result and the next unchanged layout-6 transfer trial are recorded in
`go2_stable_reference_matched_navigation_result_2026-09-14.md`.
