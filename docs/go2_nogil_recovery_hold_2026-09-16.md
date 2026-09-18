# Prompt recovery hold on the exposed fresh-maze failure

The completed 20-run replication is unchanged. All five controllers lost visual
tracking on its layout 0 during the initial panorama, before translation.
The retained JEPA failure gives a concrete timing hypothesis: frame 357
(37.20 s) triggered weak-view recovery; its tracking stage completed at 37.344 s.
The already-committed left turn continued through 37.70 s. Recovery from planning
frame 360 first requested a right turn at 37.80 s, after failed tracking frame
361 had been acquired at 37.60 s. The applied rate-limited command was still
leftward at the first right-turn request. Earlier cancellation could help, but
has not yet been shown to prevent this failure or complete navigation.

Run one full-budget JEPA mission on this now-exposed layout using the existing
`VisualRecoveryDispatchHoldRuntime`, composed with the unchanged direct-stage
profiling runtime. This cancels pre-trigger commitments at actual registration
publication and rejects pre-trigger plans that finish later. The commitment
ledger records actual cancellation time and still rejects a forecast if its
assumed command prefix was interrupted. Post-trigger recovery plans retain the
existing prediction, clearance and sensor gates. No survey view is declared
observed by this intervention.

Keep the completed replication's layout, appearance/physics seeds, frozen model,
six candidates, 0.8-s forecast, 300-ms deadline, 20-ms added publication delay,
500-Hz physics, sensor noise, camera drawing treatment and layout-0 CPU group.
Tracking thresholds, local reference selection, survey policy and arrival
requirements remain unchanged. Record recovery publications and actual cancelled
windows; an unexercised intervention cannot establish a repair. Count every
failure. Passing the former failure frame alone is not navigation success.

Launcher: `scripts/run_go2_nogil_recovery_hold_development.py`.
Output: `go2_nogil_recovery_hold_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`.
Evaluate with `--evaluate` after owner exit and native persistence. The original
JEPA comparison root remains the reference and retains its complete failure
recording. A successful exploratory run would require matched control evidence
and further prospective validation; it would not change the completed batch's
2/4 JEPA result.

No other native job is active. Artifact headroom is about 6.4 GiB; one native
mission runs at a time because concurrent work would change measured deadlines.
No expanded-candidate or broader environment-type experiment is included.

## Result: cancellation alone did not repair navigation

The owner exited with visual-tracking loss after 346 acquired frames. Physical
evaluation found zero contacts and no arrivals. Planning met 82/85 deadlines.
Six camera-cadence recovery publications each cancelled an older turn window;
no nonzero pre-trigger command was requested after the corresponding recovery
publication. The intervention was exercised, unlike the earlier exposed-layout
trial. Nevertheless, the initial survey remained incomplete at its fourth view
and there were zero translation requests. The final recovery itself selected
left turns despite a rightward preferred view, under the existing clearance
logic. Earlier command cancellation alone is not sufficient. Keep the full raw
failure and all source/timing/treatment records; this is an additional failed
experiment, not a replacement for any original comparison outcome.

Nine existing commitment/pulse tests passed before launch. A focused composed-
runtime exercise also checked cancellation at publication time, preservation of
the original forecast prefix, rejection of an interrupted prefix and rejection
of an old plan that finished after publication. Runtime result:
`recovery_hold_treatment_readout_v1.json` in the output root.
