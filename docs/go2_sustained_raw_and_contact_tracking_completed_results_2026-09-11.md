# Completed sustained raw replay and contact tracking recovery

Two prospective replay checks completed. They establish reproducible candidate
boundaries on original public observations. Neither executes the changed
command or establishes continued navigation.

## Sustained-turn raw controller replay

The original full-JEPA model was replayed in two fresh controller instances
through all 407 observations. All 404 compared model forecasts, original
decisions, public inputs, and retained map/contact state matched. Candidate
selections matched the earlier saved-prefix calculation. The only changed
requested command is frame 406: original hold `[0, 0, 0]` becomes continued
left turn `[0, 0, 0.45]`. The actual model state remained unchanged. No
observation after this changed request was consumed, and the request was not
executed in physics.

Raw result SHA-256:
`977d00f6774d68c1b86d32d4e903650e13b89ced6cad34b43e0bd935b1a33e2d`.
Launch SHA-256:
`6f68017be0f68198ab08af97fb4d90f33ebbdc2b1b56f8da123e8b34b4dd89df`.
Root `go2_sustained_hold_reorientation_maze02_prefix_v1_attempt_001`.
1972 original source paths; reported execution wall time 3226.771 seconds.
Original PID 2813368 ended; session 87754 exited 0.

The separate V1 completion checker failed before verification with
`KeyError: 'source_sha256'`: the execution receipt stores a source count,
while its hash-bound launch stores the full table. The failure is retained in
`docs/go2_sustained_hold_reorientation_raw_completion_checker_v1_failure_2026-09-11.json`,
SHA-256 `4d2529bee2171a1e655cee747a313514e3a257029797f556e86857e4809571fb`.
Session 44980 exited 1. No raw experiment was repeated or modified.

V2 uses the actual table from the exact original launch after verifying its
hash, owner, boot, source count and inclusion in the verifier closure. It then
runs the original complete verification logic. Five regression tests passed
in 2.23 seconds, session 87611, exit 0.

Completion verification:
`docs/go2_sustained_hold_reorientation_raw_completion_verification_v2_2026-09-11.json`,
SHA-256 `3f155aac01bbd46033ba66ee84af0918ce208449def0358aa83970cc02b670ed`.
Verifier session 46776 exited 0 with 2036 source paths. It reconstructed the
original input admission, all saved original/candidate comparisons, forecast
comparison population and all 407 actual public-packet fingerprints, then
rechecked the output artifacts. It did not rerun neural inference or full
training ancestry. The sustained native waiter remains responsible for its
one fresh-physics execution after the original budget queue completes.

## Contact-worker tracking-history replay

Both observers started at frame zero with the actual gyro and camera history.
All 562 original visual evidence records were reproduced completely. The
existing direct-flow candidate matched the original on the first 561
observations and first changed at frame 561, where the original observer
failed and the candidate admitted a current auxiliary-camera pose.

The selected reference was frame 560. The candidate used 21 rigid inliers
across seven reference and seven current image regions, with residual RMS
0.0005187962375573326 m and gyro disagreement 0.0014237361934719007 rad.
Rigid, temporal and bridge thresholds remained unchanged. No pose/reference
history reset occurred. There were no qualified original measurements to
conflict-check at the failed boundary. The primary-camera attempts still
failed with ten matches. Replay stopped immediately at the changed evidence.

Observer result SHA-256:
`116ed76d30cf1e9bcbcca01e91618f1a95cfa3c949ec7cd79ec5c6fe5eb56f90`.
Launch SHA-256:
`9ee4ddd13e6672619c4fabb29fcacf9831fbea215377915a2151d6e25da76e62`.
Root `go2_contact_anchored_direct_flow_observer_prefix_v1_attempt_001`.
2045 source paths. Original PID 2819628 ended; session 52711 exited 0.
Reported wall time 566.153 seconds includes waiting for the original CPU
replay and must not be interpreted as observer-only processing time.

Completion recheck:
`docs/go2_contact_anchored_direct_flow_observer_completion_2026-09-11.json`,
SHA-256 `6c5e6efe9839beb723d42a9bf53b1e57c980ef9c9a39acfc10855afa0bedce2b`.
Session 1368 exited 0. All result/source/artifact bindings and original worker
inputs were checked, all 562 stored comparisons were reconstructed against
the actual original decision rows, and the first-change boundary and report
were rechecked. This completion recheck did not repeat the observer inference
or recompute public-packet fingerprints; the original replay did both input
checks and actual current-pose contract validation during execution.

This is pose acceptance under the existing development gates, not a calibrated
pose-error bound or a navigation result. Floor registration, mapping, model
forecasting and command selection were not replayed for the recovered pose.

## Next work

The full CPU replay slot is now free. Prepare the contact-scoring controller
with only the existing direct-flow observer substituted. Replay both complete
controllers with the original assigned full-supervised model through the
first changed requested command, requiring all original decisions and model
forecasts to reproduce and comparing candidate visual evidence to this fixed
observer result. Stop at the first command boundary; do not consume the old
terminal drain as candidate future motion. Preserve negative floor/planner
outcomes if the recovered pose cannot support continued control.

The original contact parent and contact/tracking/budget waiters were still
active at the last check. Do not alter or restart their frozen executions.
The sustained native waiter PID 2817601 remains behind the complete original
queue; its raw prerequisite has now completed. Any further contact-plus-flow
native follow-up must remain separate from already frozen runs.

No verified round trip, independent-layout advantage, real-time qualification
or hardware result has been added. The overall goal remains active.
