# Supervised comparison: completed raw audit, negative navigation result

The 8,014-decision supervised-rollout collection exhausted its navigation
budget without either arrival. It recorded no translation requests after
settling, less than 1 cm maximum native displacement and no contacts.
Complete score analysis attributes the holds to the original contact penalty
outweighing forward progress despite all six candidates being feasible.

The completed audit passes raw sensor reconstruction, auxiliary RGB
reconstruction, model-command replay, command audit, unchanged model weights
and strict physical visibility. There are no hard measurement failure frames.
This strengthens the negative scientific result; it does not establish a
successful round trip.

The runner subsequently failed its final source check. Its inherited manifest
included 2,689 historical development paths, of which exactly one changed:
`lewm/commitment_contact_controller_development.py`. That file implements the
separate contact-horizon experiment; the original supervised controller uses
`StopConditionedForecastController` and its unchanged selector. The saved raw
audit and readout are retained, as is the terminal `failure.json`; no synthetic
replacement `result.json` was written. Exact artifact hashes are in the paired
JSON report.

All seven old waiters stopped when that predecessor lacked operational
completion. A first continuation launch also stopped before creating a native
attempt because the historical environment check contained the same source
binding. Those queue receipts remain preserved.

For new independent runs, the runner now binds the current source versions
at launch and explicitly records differences from the stopping-trial manifest.
Its environment check uses these same source bindings; native binaries, model
inputs, geometry and environment checks remain in place. End-of-run checks
still compare against this run's launch versions. Historical files are not
rewritten.

The second continuation started the first direct-model native attempt:
queue PID 3196768 (creation 1789286098.74), native PID 3196770 (creation
1789286098.79). The remaining fixed order is nominal, contact-horizon JEPA,
contact-horizon supervised, layout-01 JEPA, layout-01 supervised, then layout-00
current-planning memory control. Collection and audit remain serial. No prior
native attempt is retried. The queue receipt is
`go2_independent_comparisons_continuation_v2_2026-09-13.json`.
