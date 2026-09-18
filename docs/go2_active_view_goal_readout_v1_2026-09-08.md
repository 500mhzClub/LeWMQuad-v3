# Active-view goal readout V1

Admit the exact complete result of `go2_active_view_goal_probe_v1_attempt_001`
by externally supplied SHA-256 and verify every bound source/input/artifact.
Create `go2_active_view_goal_readout_v1_attempt_001` exclusively. Preserve actual
goals, contact/controller stops, all visibility failures, full sensor/model replay
status, accepted-pose errors and stage timing. Report mode transitions, model
forecast counts, measured heading range, original map receipts, selected scan
headings/waypoints, completed active command intervals and their nonzero subset.
Distinguish a waypoint proposal or zero hold from executed nonzero waypoint motion.
Retain terminal drain separately through the original collection and command tape.

This fixed descriptive readout performs no native execution, model selection,
training, parameter changes, phase relabeling, retry or outcome modification.
No successful view or intermediate waypoint replaces the mission goal. Verify
all input and source bindings again before writing the terminal readout result.
