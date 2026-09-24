# Saved-JSON identity adapter for the measured-settling mission prefix

The original saved-mission comparison failed before processing frame0 because
JSON stores episode identities as lists while the strict live evidence accessor
requires tuples. Preserve that failed attempt and all its sources.

This separately named comparison keeps the complete scientific scope and
stopping rule of go2_measured_settling_mission_prefix_v1_2026-09-09.md. Restore
only the two explicit episode identities using the existing tested readout
json_identity helper before calling the unchanged registered-pose accessor.
Reject malformed/boolean/negative/wrong-episode identities. Copy dictionaries
instead of mutating saved evidence. No pose values, fit witnesses, clocks,
commands, mission thresholds or future-decision stopping rules change.

New output go2_measured_settling_mission_json_prefix_v1_attempt_001. Bind the
same saved input bytes and inherited frozen source plus the new helper, script,
protocol and focused tests. Resource admission, before/after validation,
one-process concurrency and all qualification limitations remain as specified
in the original protocol. The original failure is infrastructure evidence;
it is not a candidate navigation outcome.
